//! Display encoding: the gamut-transform + transfer-encoding stage of the
//! separated display pipeline (tone map → gamut transform → transfer
//! encoding).
//!
//! The tone-mapping pass outputs *display-linear* color, scaled so `1.0` =
//! paper white, in per-view source primaries (Rec.709 for every effective
//! operator except `Tonemapping::GranTurismo7` on HDR targets — authored or
//! substituted by the D6 table (`effective_tonemapping`) — which emits its
//! native Rec.2020; see `tonemap_output_gamut`); UI then composites in that
//! same space. This pass — scheduled after the UI pass and before the upscaling
//! blit — converts that buffer into the display's signal: a 3×3 gamut
//! transform from the source primaries to the display primaries, an
//! out-of-gamut handling step (ACES-RGC-style chroma compression when the
//! gamut transform can produce out-of-gamut colors — see
//! [`DisplayGamutCompression`] and the [`gamut_compression`] module — plus an
//! always-on `max(0)` safety clip), and the display transfer function (OETF).
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
//! Surface negotiation (`create_surfaces` in `bevy_render::view::window`)
//! configures the swapchain this pass's output is presented through, using
//! wgpu's surface color-space API (currently Bevy's wgpu fork, pending
//! upstream under <https://github.com/gfx-rs/wgpu/issues/2920>): an
//! `Rgba16Float` extended-sRGB-linear swapchain for
//! [`DisplayTransfer::ScRgbLinear`] (macOS/iOS Metal, Windows Vulkan/DX12,
//! Wayland Vulkan), or an HDR10 swapchain (typically `Rgb10a2Unorm`) for
//! [`DisplayTransfer::Pq`] where the backend and OS advertise it. In both
//! cases the encoded output of this pass is blitted to the surface unchanged
//! (no hardware sRGB encode — these formats have no sRGB view). When the
//! backend cannot fulfil the requested transfer, the view's **resolved**
//! display target degrades (PQ → scRGB → plain SDR, with warnings), so the
//! predicate below always reflects what the surface can actually show.
//! [`DisplayTransfer::Hlg`] requests are fulfilled as PQ/HDR10 at
//! negotiation (HLG is scene-referred; see the coercion notes on
//! [`prepare_view_display_encoding_pipelines`]), so a resolved HLG transfer
//! can only reach this pass through manual (non-window) targets.

use crate::FullscreenShader;
use bevy_app::{App, Plugin};
use bevy_asset::{embedded_asset, load_embedded_asset, AssetServer, Handle};
use bevy_camera::ClearColorConfig;
use bevy_camera::CompositingSpace;
use bevy_ecs::prelude::*;
use bevy_log::{info_once, warn_once};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    camera::ExtractedCamera,
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_resource::{
        binding_types::{sampler, texture_2d, uniform_buffer},
        *,
    },
    renderer::RenderDevice,
    view::{DisplayTargetUniform, ExtractedView, ViewDisplayTarget, ViewTarget},
    working_color_space::WorkingColorSpace,
    GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_shader::Shader;
use bevy_utils::default;
use bevy_window::{DisplayGamut, DisplayTransfer};

use crate::camera_stack::{stack_deferred_views, StackView};
use crate::tonemapping::{tonemap_output_gamut, Tonemapping};

pub mod gamut_compression;
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

        app.register_type::<DisplayGamutCompression>()
            .init_resource::<DisplayGamutCompression>()
            .add_plugins(ExtractResourcePlugin::<DisplayGamutCompression>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_gpu_resource::<SpecializedRenderPipelines<DisplayEncodingPipeline>>()
            .add_systems(RenderStartup, init_display_encoding_pipeline)
            .add_systems(
                Render,
                // Mutates `PipelineCache` (`block_on_render_pipeline`);
                // ordering ambiguities against other pipeline-cache users
                // are ignored, like the upscaling system
                // (see https://github.com/bevyengine/bevy/issues/14770).
                prepare_view_display_encoding_pipelines
                    .in_set(RenderSystems::Prepare)
                    .ambiguous_with_all(),
            );
    }
}

/// Controls how the display-encoding pass handles colors that fall outside
/// the display gamut after its gamut-transform stage.
///
/// The primary handling (per DECISIONS.md D3) is a perceptual,
/// hue-approximate chroma compression toward the achromatic axis in the
/// style of the ACES 1.3 Reference Gamut Compression — see the
/// [`gamut_compression`] module for the algorithm, constants, citations, and
/// the CPU mirror of the shader implementation. The debug fallback is the
/// plain hue-shifting per-channel clip (`max(c, 0.0)`), which always runs
/// after the compression as a final safety (PQ encoding requires
/// non-negative input) and *is* the entire handling when compression is off.
///
/// Out-of-gamut colors can only come out of the gamut stage when it
/// *contracts* — when the pass's input primaries are wider than the resolved
/// display primaries. The pass's input gamut is per-view (see
/// `encoder_input_gamut`): Rec.2020 when the view's effective operator is
/// `Tonemapping::GranTurismo7` on an HDR-transfer target — authored, or
/// substituted for an SDR-only operator by the D6 table
/// (`effective_tonemapping`) — as the operator emits its native Rec.2020
/// display-referred output; Rec.709 otherwise. Under
/// [`DisplayGamutCompression::Auto`] the compression is therefore active for
/// exactly one reachable configuration: GT7 HDR-native Rec.2020 input onto a
/// Rec.709-coordinate scRGB signal (a contraction). Identity transforms
/// (Rec.709 → scRGB, GT7's Rec.2020 → PQ/Rec.2020) and the
/// Rec.709 → Rec.2020 PQ expansion keep the plain clip, which is a no-op for
/// their in-gamut-by-construction inputs.
/// [`DisplayGamutCompression::Always`] forces compression on for every
/// HDR-transfer view (e.g. to exercise or demonstrate the path).
#[derive(Resource, ExtractResource, Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Reflect)]
#[reflect(Resource, Debug, Default, Clone, PartialEq, Hash)]
pub enum DisplayGamutCompression {
    /// Compress exactly when the gamut stage can produce out-of-gamut colors
    /// (a gamut contraction). Identity and expanding transforms keep the
    /// plain clip, which is a no-op for their in-gamut-by-construction
    /// inputs. This is the default.
    #[default]
    Auto,
    /// Always compress on views the display-encoding pass runs for.
    ///
    /// Note that compression is not free for in-gamut colors: channels whose
    /// distance from the achromatic axis exceeds the ACES RGC threshold
    /// (≈ 0.8) are pulled slightly inward to make room for the compressed
    /// out-of-gamut range, so forcing this on a path with no possible
    /// out-of-gamut input desaturates highly saturated colors for no
    /// benefit.
    Always,
    /// Debug fallback (DECISIONS.md D3): replace the compression with the
    /// hue-shifting per-channel clip, for A/B comparison. Pushes the
    /// `DISPLAY_GAMUT_CLIP_DEBUG` shader def instead of
    /// `DISPLAY_GAMUT_COMPRESSION`.
    Clip,
}

/// The resolved per-pipeline out-of-gamut handling of the gamut stage, after
/// [`DisplayGamutCompression`] and the gamut-contraction rule have been
/// applied by [`prepare_view_display_encoding_pipelines`].
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum OutOfGamutHandling {
    /// Only the always-on final `max(c, 0.0)` safety clip; no shader def.
    /// Used when the gamut stage cannot produce out-of-gamut colors.
    Clip,
    /// ACES-RGC-style chroma compression (`DISPLAY_GAMUT_COMPRESSION`),
    /// followed by the safety clip.
    Compress,
    /// The per-channel clip selected explicitly as a debug fallback
    /// (`DISPLAY_GAMUT_CLIP_DEBUG`); behaves like [`Self::Clip`] but
    /// specializes a visibly distinct pipeline.
    ClipDebug,
}

/// The color primaries of the display-encoding pass's *input* (the main
/// texture it reads) for a given view.
///
/// Delegates to [`tonemap_output_gamut`], the single source of truth shared
/// with the tonemapping pipeline's `TONEMAP_OUTPUT_REC2020` def push:
/// Rec.2020 exactly when the view's *effective* operator (after the D6
/// SDR-only-operator substitution, see
/// [`effective_tonemapping`](crate::tonemapping::effective_tonemapping)) is
/// [`Tonemapping::GranTurismo7`] and its resolved transfer is HDR (the
/// operator then emits its native linear Rec.2020 display-referred output
/// with no Rec.709 back-conversion), Rec.709 in every other configuration
/// (Rec.709-fit operators receive a Rec.2020 → Rec.709 conversion at the
/// tone-mapping pass entry under the Rec.2020 working space, so their output
/// is Rec.709 under every [`WorkingColorSpace`]; see
/// `tonemapping_shared.wgsl` / `gt7.wgsl`). Both prepare systems read the
/// same extracted `Tonemapping` and the same `ViewDisplayTarget`, so the def
/// push and this contract always agree within a frame.
///
/// Known limitation (documented, see the release notes and
/// `plans/ui-hdr-rfc.md`): the UI pass composites Rec.709-authored colors
/// into the post-tonemap buffer unconverted, so on GT7-HDR views (Rec.2020
/// buffer) saturated UI colors are reinterpreted in the wider primaries and
/// oversaturate; grays and whites are unaffected (shared D65 white point).
/// Converting UI colors per view needs a per-view key axis on the UI
/// pipelines and is deferred to the UI HDR follow-up.
fn encoder_input_gamut(
    tonemapping: Option<&Tonemapping>,
    view_display_target: &ViewDisplayTarget,
) -> DisplayGamut {
    tonemap_output_gamut(tonemapping, Some(view_display_target))
}

/// Whether transforming from `source` primaries to `display` primaries can
/// produce out-of-gamut (negative-component) colors, i.e. whether the source
/// gamut is strictly wider than the display gamut
/// (Rec.2020 ⊃ Display P3 ⊃ Rec.709).
const fn is_gamut_contraction(source: DisplayGamut, display: DisplayGamut) -> bool {
    const fn coverage_rank(gamut: DisplayGamut) -> u8 {
        match gamut {
            DisplayGamut::Rec709 => 0,
            DisplayGamut::DisplayP3 => 1,
            DisplayGamut::Rec2020 => 2,
        }
    }
    coverage_rank(source) > coverage_rank(display)
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
/// encoded, not on what was requested. With the per-view
/// [`source_gamut`](Self::source_gamut), the reachable
/// source × display × transfer combinations are:
///
/// | source (tonemap output) | display gamut | transfer | gamut stage |
/// |---|---|---|---|
/// | Rec.709 | Rec.709 | scRGB | identity |
/// | Rec.709 | Rec.2020 | PQ | expansion (`DISPLAY_GAMUT_REC2020`) |
/// | Rec.2020 (GT7 HDR) | Rec.709 | scRGB | contraction (`GAMUT_REC2020_TO_REC709`, compression active under `Auto`) |
/// | Rec.2020 (GT7 HDR) | Rec.2020 | PQ | identity |
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
    /// The color primaries of the pass's input — the tonemapping pass's
    /// output gamut for this view (see `encoder_input_gamut` /
    /// [`tonemap_output_gamut`]): Rec.2020 for GT7 (authored or
    /// D6-substituted) on HDR-transfer targets, Rec.709 otherwise.
    pub source_gamut: DisplayGamut,
    /// The resolved display gamut the source color is transformed to.
    pub gamut: DisplayGamut,
    /// The resolved transfer function. Only HDR transfers occur here
    /// ([`DisplayTransfer::ScRgbLinear`] or [`DisplayTransfer::Pq`]);
    /// sRGB targets never get this pass.
    pub transfer: DisplayTransfer,
    /// The resolved out-of-gamut handling of the gamut stage (see
    /// [`DisplayGamutCompression`]). Under the default
    /// [`DisplayGamutCompression::Auto`] this is
    /// [`OutOfGamutHandling::Compress`] exactly when the gamut stage is a
    /// contraction (Rec.2020 source onto a Rec.709 scRGB signal) and
    /// [`OutOfGamutHandling::Clip`] otherwise (identity and expanding stages
    /// cannot produce out-of-gamut colors).
    pub out_of_gamut: OutOfGamutHandling,
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

        match (key.source_gamut, key.gamut) {
            // Identity transforms: a Rec.709 tonemap output onto the
            // Rec.709-coordinate scRGB signal, and GT7's HDR-native Rec.2020
            // output onto a PQ/Rec.2020 signal.
            (DisplayGamut::Rec709, DisplayGamut::Rec709)
            | (DisplayGamut::Rec2020, DisplayGamut::Rec2020) => {}
            // Expansion (in-gamut by construction): Rec.709 tonemap output
            // onto a PQ/Rec.2020 signal.
            (DisplayGamut::Rec709, DisplayGamut::Rec2020) => {
                shader_defs.push("DISPLAY_GAMUT_REC2020".into());
            }
            // Contraction: GT7's HDR-native Rec.2020 output onto the
            // Rec.709-coordinate scRGB signal — the one reachable stage that
            // can produce out-of-gamut colors, for which `Auto` keys in the
            // compression below.
            (DisplayGamut::Rec2020, DisplayGamut::Rec709) => {
                shader_defs.push("GAMUT_REC2020_TO_REC709".into());
            }
            // Coerced away at prepare time (display side); never emitted by
            // the tonemapping pass (source side).
            (DisplayGamut::DisplayP3, _) | (_, DisplayGamut::DisplayP3) => unreachable!(
                "DisplayP3 is coerced to Rec709 in prepare_view_display_encoding_pipelines \
                 and the tonemapping pass never emits DisplayP3"
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

        match key.out_of_gamut {
            // The always-on max(0) safety clip is the entire handling.
            OutOfGamutHandling::Clip => {}
            OutOfGamutHandling::Compress => {
                shader_defs.push("DISPLAY_GAMUT_COMPRESSION".into());
            }
            OutOfGamutHandling::ClipDebug => {
                shader_defs.push("DISPLAY_GAMUT_CLIP_DEBUG".into());
            }
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
/// * [`DisplayTransfer::Hlg`] → [`DisplayTransfer::Pq`]: HLG is
///   scene-referred — encoding tone-mapped (display-referred) output with
///   the HLG OETF would double-tone-map. Window surfaces fulfil HLG requests
///   as PQ/HDR10 at negotiation for the same reason, so this arm is only
///   reachable through manual (non-window) targets. `warn_once!`.
/// * [`DisplayGamut::DisplayP3`] → [`DisplayGamut::Rec709`]: no P3 gamut
///   matrix ships yet. (wgpu's surface color-space API can advertise
///   Display P3 surfaces, but Bevy does not negotiate them until this pass
///   grows a P3 encode path — see `negotiate_surface_format` in
///   `bevy_render::view::window`.) `warn_once!`.
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
///
/// The pass's input gamut is resolved per view through `encoder_input_gamut`
/// (the [`tonemap_output_gamut`] single source shared with the tonemapping
/// pipeline's `TONEMAP_OUTPUT_REC2020` def push): Rec.2020 for
/// [`Tonemapping::GranTurismo7`] on HDR-transfer targets (authored, or
/// substituted for an SDR-only operator by the D6 table —
/// [`effective_tonemapping`](crate::tonemapping::effective_tonemapping)),
/// Rec.709 otherwise.
/// It also resolves the gamut stage's out-of-gamut handling from
/// [`DisplayGamutCompression`]: compression is keyed in exactly when the
/// gamut stage is a contraction (the input primaries are wider than the
/// resolved display primaries — today only GT7's Rec.2020 output onto an
/// scRGB signal) or when forced with [`DisplayGamutCompression::Always`].
pub fn prepare_view_display_encoding_pipelines(
    mut commands: Commands,
    mut pipeline_cache: ResMut<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<DisplayEncodingPipeline>>,
    encoding_pipeline: Res<DisplayEncodingPipeline>,
    views: Query<(
        Entity,
        &ExtractedView,
        &ExtractedCamera,
        &ViewTarget,
        &ViewDisplayTarget,
        Option<&Tonemapping>,
    )>,
    working_color_space: Res<WorkingColorSpace>,
    gamut_compression: Res<DisplayGamutCompression>,
) {
    // Cameras stacked on a shared main texture encode once, on the last
    // camera whose target needs the pass. Encoding per camera would make the
    // later cameras' main passes and upscaling blits composite over (and
    // alpha-blend with) already transfer-encoded signal, which is non-linear
    // and visibly wrong for PQ.
    let deferred = stack_deferred_views(views.iter().map(
        |(entity, _, camera, view_target, view_display_target, _)| StackView {
            entity,
            texture: view_target.main_texture().id(),
            sorted_index: camera.sorted_camera_index_for_target,
            enabled: view_display_target.is_hdr_transfer(),
            composites_fullscreen: matches!(camera.clear_color, ClearColorConfig::None)
                && camera.viewport.is_none(),
        },
    ));

    for (entity, view, camera, _, view_display_target, tonemapping) in &views {
        if deferred.contains_key(&entity) {
            // The finalizing camera encodes the composed buffer; this view
            // must not run the pass. (Render-world entities are retained, so
            // the component must be actively removed.)
            commands
                .entity(entity)
                .remove::<ViewDisplayEncodingPipeline>();
            continue;
        }
        // D1 guidance: HDR display output reaches noticeably wider gamuts
        // when the scene is rendered in the Rec.2020 working space. This is
        // advisory only (the Rec.709 working space remains correct, just
        // gamut-limited); a global axis must never flip automatically
        // because one window went HDR.
        if view_display_target.is_hdr_transfer() && !working_color_space.is_rec2020() {
            warn_once!(
                "A camera is rendering to an HDR display target while the working color \
                space is the default `WorkingColorSpace::Rec709`. Output is correct but \
                limited to the Rec.709 gamut; consider opting into the wide working space \
                with `RenderPlugin {{ working_color_space: WorkingColorSpace::Rec2020, .. }}`."
            );
        }

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
                "A resolved `DisplayTransfer::Hlg` reached the display encoder (a manual \
                render target; window surfaces fulfil HLG as PQ at negotiation). HLG is \
                scene-referred (encoding tone-mapped output with the HLG OETF would \
                double-tone-map); encoding with PQ instead."
            );
            transfer = DisplayTransfer::Pq;
        }
        if gamut == DisplayGamut::DisplayP3 {
            warn_once!(
                "`DisplayGamut::DisplayP3` output is not supported yet (the display \
                encoder ships no P3 gamut matrices, so P3 surfaces are not negotiated); \
                leaving colors in Rec.709 primaries."
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

        // The tonemapping pass's output gamut for this view — per-view, from
        // the same single-sourced predicate that pushes the
        // `TONEMAP_OUTPUT_REC2020` shader def on the tonemapping pipeline.
        let source_gamut = encoder_input_gamut(tonemapping, view_display_target);

        let out_of_gamut = match *gamut_compression {
            DisplayGamutCompression::Auto => {
                if is_gamut_contraction(source_gamut, gamut) {
                    OutOfGamutHandling::Compress
                } else {
                    OutOfGamutHandling::Clip
                }
            }
            DisplayGamutCompression::Always => OutOfGamutHandling::Compress,
            DisplayGamutCompression::Clip => OutOfGamutHandling::ClipDebug,
        };

        let key = DisplayEncodingPipelineKey {
            target_format: view.target_format,
            source_space: camera.compositing_space,
            source_gamut,
            gamut,
            transfer,
            out_of_gamut,
        };
        let pipeline_id = pipelines.specialize(&pipeline_cache, &encoding_pipeline, key);

        // The pass-through upscaling blit for HDR transfers blocks on its
        // own pipeline and presents the main texture as-is, so an unready
        // encoder pipeline would present raw display-linear values — on a
        // PQ swapchain those read as severely distorted signal. Block until
        // the encoder is compiled; this is O(1) once it is, and only ever
        // runs for HDR-transfer views.
        pipeline_cache.block_on_render_pipeline(pipeline_id);

        commands
            .entity(entity)
            .insert(ViewDisplayEncodingPipeline { pipeline_id });
    }
}
