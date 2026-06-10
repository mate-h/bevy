//! Display encoding: the gamut-transform + transfer-encoding stage of the
//! separated display pipeline (tone map → gamut transform → transfer
//! encoding).
//!
//! The tone-mapping pass outputs *display-linear* color in the working
//! primaries, scaled so `1.0` = paper white; UI then composites in that same
//! space. This pass — scheduled after the UI pass and before the upscaling
//! blit — converts that buffer into the display's signal: a 3×3 gamut
//! transform from the working primaries to the display primaries, an
//! out-of-gamut handling step, and the display transfer function (OETF).
//!
//! **Plain SDR targets never run this pass.** For the default
//! [`DisplayTarget::SDR_SRGB`](bevy_window::DisplayTarget) (and any other
//! target whose transfer is [`DisplayTransfer::Srgb`]), the exact sRGB OETF is
//! applied for free by the hardware on the upscaling blit's `*UnormSrgb`
//! writeback, exactly as before this pass existed; such views never receive a
//! [`ViewDisplayEncodingPipeline`] and the node early-returns without touching
//! the GPU. The shader-side transfer functions (scRGB / PQ) activate only for
//! HDR transfers.
//!
//! Surface selection (`create_surfaces` in `bevy_render::view::window`)
//! negotiates an `Rgba16Float` scRGB-linear swapchain when a window's
//! [`DisplayTarget`](bevy_window::DisplayTarget) requests
//! [`DisplayTransfer::ScRgbLinear`] and the backend offers it (macOS/iOS
//! Metal, Windows Vulkan, Wayland Vulkan); the encoded output of this pass is
//! then blitted to the surface unchanged (no hardware sRGB encode — float
//! surfaces have no sRGB view). When the backend cannot fulfil the requested
//! transfer, the view's **resolved** display target degrades to plain SDR and
//! this pass never runs, so the predicate below always reflects what the
//! surface can actually show. PQ/HLG surfaces remain unreachable until wgpu
//! gains color-space control (<https://github.com/gfx-rs/wgpu/issues/2920>);
//! the pass itself is transfer-agnostic and ready.

use crate::FullscreenShader;
use bevy_app::{App, Plugin};
use bevy_asset::{embedded_asset, load_embedded_asset, AssetServer, Handle};
use bevy_camera::CompositingSpace;
use bevy_ecs::prelude::*;
use bevy_log::{info_once, warn_once};
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        binding_types::{sampler, texture_2d, uniform_buffer},
        *,
    },
    renderer::RenderDevice,
    view::{DisplayTargetUniform, ExtractedView, ViewDisplayTarget},
    GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_shader::Shader;
use bevy_utils::default;
use bevy_window::{DisplayGamut, DisplayTransfer};

use crate::tonemapping::Tonemapping;

mod node;

pub use node::display_encoding;

/// Adds the display-encoding pass (gamut transform + transfer encoding) used
/// by views whose resolved [`DisplayTarget`](bevy_window::DisplayTarget)
/// requests an HDR transfer function.
///
/// The `display_encoding` node itself is registered in the `Core2d` / `Core3d`
/// schedules by their plugins (after the UI pass, before upscaling).
pub struct DisplayEncodingPlugin;

impl Plugin for DisplayEncodingPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "display_encoding.wgsl");

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_gpu_resource::<SpecializedRenderPipelines<DisplayEncodingPipeline>>()
            .add_systems(RenderStartup, init_display_encoding_pipeline)
            .add_systems(
                Render,
                prepare_view_display_encoding_pipelines.in_set(RenderSystems::Prepare),
            );
    }
}

/// Render-world resource holding the display-encoding pass's bind group
/// layout, sampler, and shaders.
#[derive(Resource)]
pub struct DisplayEncodingPipeline {
    layout: BindGroupLayoutDescriptor,
    sampler: Sampler,
    fullscreen_shader: FullscreenShader,
    fragment_shader: Handle<Shader>,
}

/// Initializes [`DisplayEncodingPipeline`] at [`RenderStartup`].
pub fn init_display_encoding_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    fullscreen_shader: Res<FullscreenShader>,
    asset_server: Res<AssetServer>,
) {
    let layout = BindGroupLayoutDescriptor::new(
        "display_encoding_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: false }),
                sampler(SamplerBindingType::NonFiltering),
                // The per-view display-target calibration (paper white, peak),
                // shared with the tonemapping pass via `DisplayTargetUniforms`.
                uniform_buffer::<DisplayTargetUniform>(true),
            ),
        ),
    );

    let sampler = render_device.create_sampler(&SamplerDescriptor::default());

    commands.insert_resource(DisplayEncodingPipeline {
        layout,
        sampler,
        fullscreen_shader: fullscreen_shader.clone(),
        fragment_shader: load_embedded_asset!(asset_server.as_ref(), "display_encoding.wgsl"),
    });
}

/// Specialization key for the display-encoding pipeline.
///
/// `gamut` and `transfer` are the **resolved** values after the prepare-time
/// coercions in [`prepare_view_display_encoding_pipelines`] (HLG → PQ,
/// PQ forces Rec.2020, scRGB forces Rec.709 — scRGB signals are by definition
/// expressed in extended Rec.709/sRGB coordinates — and Display P3 currently
/// falls back to Rec.709), so the pipeline hashes on what is actually
/// encoded, not on what was requested. The only reachable gamut × transfer
/// combinations are therefore `Rec709 × ScRgbLinear` and `Rec2020 × Pq`.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct DisplayEncodingPipelineKey {
    /// Format of the main texture the pass writes to (the
    /// [`post_process_write`](bevy_render::view::ViewTarget::post_process_write)
    /// destination).
    pub target_format: TextureFormat,
    /// The view's [`CompositingSpace`], if any. `Some(Srgb)` / `Some(Oklab)`
    /// main textures hold encoded values that must be decoded back to linear
    /// before gamut/transfer encoding (the same decode the upscaling blit
    /// performs for non-encoded views, which this pass takes over).
    pub source_space: Option<CompositingSpace>,
    /// The resolved display gamut the working-space color is transformed to.
    pub gamut: DisplayGamut,
    /// The resolved transfer function. Only HDR transfers occur here
    /// ([`DisplayTransfer::ScRgbLinear`] or [`DisplayTransfer::Pq`]);
    /// sRGB targets never get this pass.
    pub transfer: DisplayTransfer,
}

impl SpecializedRenderPipeline for DisplayEncodingPipeline {
    type Key = DisplayEncodingPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut shader_defs = Vec::new();

        // Same def names (and semantics) as the upscaling blit, which skips
        // its own decode when this pass ran.
        match key.source_space {
            Some(CompositingSpace::Srgb) => shader_defs.push("SRGB_TO_LINEAR".into()),
            Some(CompositingSpace::Oklab) => shader_defs.push("OKLAB_TO_LINEAR".into()),
            Some(CompositingSpace::Linear) | None => {}
        }

        match key.gamut {
            // Identity: the working space is currently linear Rec.709.
            DisplayGamut::Rec709 => {}
            // Only reachable with the PQ transfer: scRGB coerces its
            // encoding gamut to Rec.709 at prepare time (scRGB signals are
            // definitionally Rec.709-coordinate).
            DisplayGamut::Rec2020 => shader_defs.push("DISPLAY_GAMUT_REC2020".into()),
            // Coerced away at prepare time; unreachable here.
            DisplayGamut::DisplayP3 => unreachable!(
                "DisplayP3 is coerced to Rec709 in prepare_view_display_encoding_pipelines"
            ),
        }

        match key.transfer {
            DisplayTransfer::ScRgbLinear => shader_defs.push("DISPLAY_TRANSFER_SCRGB".into()),
            DisplayTransfer::Pq => shader_defs.push("DISPLAY_TRANSFER_PQ".into()),
            // sRGB never reaches this pass (hardware encode on the blit);
            // HLG is coerced to PQ at prepare time.
            DisplayTransfer::Srgb | DisplayTransfer::Hlg => unreachable!(
                "only HDR transfers (scRGB / PQ) are encoded by the display-encoding pass"
            ),
        }

        RenderPipelineDescriptor {
            label: Some("display_encoding pipeline".into()),
            layout: vec![self.layout.clone()],
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.fragment_shader.clone(),
                shader_defs,
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            ..default()
        }
    }
}

/// The specialized display-encoding pipeline of a view.
///
/// Present **only** on views whose resolved display target requests an HDR
/// transfer function; for everything else (every view on a default SDR sRGB
/// target) the component is absent and the `display_encoding` node returns
/// immediately without recording any GPU work.
#[derive(Component)]
pub struct ViewDisplayEncodingPipeline {
    pipeline_id: CachedRenderPipelineId,
}

/// Specializes the display-encoding pipeline for views that need it and keeps
/// the [`ViewDisplayEncodingPipeline`] marker in sync (inserted for views on
/// HDR-transfer display targets, removed otherwise).
///
/// Applies the D6 prepare-time coercions before keying the pipeline:
/// * [`DisplayTransfer::Hlg`] → [`DisplayTransfer::Pq`]: no wgpu surface can
///   negotiate HLG, and HLG is scene-referred (encoding tone-mapped output
///   with the HLG OETF would double-tone-map). `warn_once!`.
/// * [`DisplayGamut::DisplayP3`] → [`DisplayGamut::Rec709`]: no P3 gamut
///   matrix ships yet (P3 surfaces are unreachable through wgpu). `warn_once!`.
/// * scRGB-linear with a non-Rec.709 gamut → [`DisplayGamut::Rec709`]: scRGB
///   (IEC 61966-2-2) is definitionally encoded against Rec.709/sRGB
///   primaries — every backend declares the `Rgba16Float` surface as
///   extended-sRGB-linear and the OS compositor maps to the panel's physical
///   gamut itself, with wide gamut expressed through out-of-range component
///   values. Encoding Rec.2020 coordinates into it would desaturate the whole
///   frame. Only the *encoding* is coerced; `DisplayTarget::gamut` itself
///   stays user-authored (it still correctly describes the panel). `info_once!`
///   (benign: the authored value is a natural description of an HDR panel).
/// * PQ with a non-Rec.2020 gamut → [`DisplayGamut::Rec2020`]: PQ is
///   canonically Rec.2020. `warn_once!`.
pub fn prepare_view_display_encoding_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<DisplayEncodingPipeline>>,
    encoding_pipeline: Res<DisplayEncodingPipeline>,
    views: Query<(
        Entity,
        &ExtractedView,
        &ExtractedCamera,
        &ViewDisplayTarget,
        Option<&Tonemapping>,
    )>,
) {
    for (entity, view, camera, view_display_target, tonemapping) in &views {
        if !view_display_target.is_hdr_transfer() {
            // sRGB transfer: hardware encode on the upscaling blit, no pass.
            // (Render-world entities are retained, so the component must be
            // actively removed when a view's target stops being HDR.)
            commands
                .entity(entity)
                .remove::<ViewDisplayEncodingPipeline>();
            continue;
        }

        // A camera without an active tone-mapping operator writes unbounded
        // scene-linear values straight into the encoder (D6: defined but
        // almost certainly unintended — PQ of raw scene values clips and
        // distorts).
        if !tonemapping.is_some_and(Tonemapping::is_enabled) {
            warn_once!(
                "A camera with `Tonemapping::None` is rendering to an HDR display target; \
                scene-linear values will be transfer-encoded without tone mapping. \
                Use a tone-mapping operator (e.g. `Tonemapping::GranTurismo7`) on HDR targets."
            );
        }

        let target = view_display_target.resolved;
        let mut transfer = target.transfer;
        let mut gamut = target.gamut;

        if transfer == DisplayTransfer::Hlg {
            warn_once!(
                "`DisplayTransfer::Hlg` is not reachable through wgpu surfaces and HLG is \
                scene-referred (encoding tone-mapped output with the HLG OETF would \
                double-tone-map); encoding with PQ instead."
            );
            transfer = DisplayTransfer::Pq;
        }
        if gamut == DisplayGamut::DisplayP3 {
            warn_once!(
                "`DisplayGamut::DisplayP3` output is not supported yet (no wgpu backend \
                exposes P3 surfaces); leaving colors in Rec.709 primaries."
            );
            gamut = DisplayGamut::Rec709;
        }
        if transfer == DisplayTransfer::ScRgbLinear && gamut != DisplayGamut::Rec709 {
            // scRGB-linear (IEC 61966-2-2) is *definitionally* encoded against
            // Rec.709/sRGB primaries: every backend that negotiates the
            // Rgba16Float surface declares it as extended-sRGB-linear, and the
            // OS compositor performs the mapping to the panel's physical gamut
            // itself. Wide gamut rides scRGB's out-of-range (including
            // negative) component values, never a change of primaries —
            // re-coordinatizing into Rec.2020 here would be interpreted as
            // Rec.709 by the compositor and desaturate every pixel.
            info_once!(
                "scRGB-linear signals are always expressed in (extended) Rec.709/sRGB \
                coordinates (the OS compositor performs the mapping to the panel's gamut); \
                ignoring `DisplayTarget::gamut` ({gamut:?}) for encoding. The field still \
                correctly describes the panel for luminance/metadata purposes."
            );
            gamut = DisplayGamut::Rec709;
        }
        if transfer == DisplayTransfer::Pq && gamut != DisplayGamut::Rec2020 {
            warn_once!(
                "PQ display targets are canonically Rec.2020 (ITU-R BT.2100); coercing \
                `DisplayTarget::gamut` from {gamut:?} to Rec2020 for encoding."
            );
            gamut = DisplayGamut::Rec2020;
        }

        let key = DisplayEncodingPipelineKey {
            target_format: view.target_format,
            source_space: camera.compositing_space,
            gamut,
            transfer,
        };
        let pipeline_id = pipelines.specialize(&pipeline_cache, &encoding_pipeline, key);

        commands
            .entity(entity)
            .insert(ViewDisplayEncodingPipeline { pipeline_id });
    }
}
