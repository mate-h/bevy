use bevy_app::prelude::*;
use bevy_asset::{
    embedded_asset, load_embedded_asset, AssetServer, Assets, Handle, RenderAssetUsages,
};
use bevy_camera::{Camera, CompositingSpace, TonemappingEnabled};
use bevy_ecs::prelude::*;
use bevy_image::{CompressedImageFormats, Image, ImageSampler, ImageType};
#[cfg(not(feature = "tonemapping_luts"))]
use bevy_log::error;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    extract_resource::{ExtractResource, ExtractResourcePlugin},
    render_asset::RenderAssets,
    render_resource::{
        binding_types::{sampler, texture_2d, texture_3d, uniform_buffer},
        *,
    },
    renderer::RenderDevice,
    texture::{FallbackImage, GpuImage},
    view::{DisplayTargetUniform, ExtractedView, ViewDisplayTarget, ViewTarget, ViewUniform},
    GpuResourceAppExt, Render, RenderApp, RenderStartup, RenderSystems,
};
use bevy_shader::{load_shader_library, Shader, ShaderDefVal};
use bitflags::bitflags;

mod gt7;
mod node;

use bevy_utils::default;
pub use gt7::{
    prepare_gt7_params_uniforms, GranTurismo7Params, Gt7ParamsUniform, Gt7ParamsUniforms,
    Gt7ToneMapping, Gt7ToneMappingCurve, ViewGt7ParamsUniformOffset, GRAN_TURISMO_SDR_PAPER_WHITE,
    GT7_MAX_HDR_PEAK_NITS, GT7_MIN_HDR_PEAK_NITS, REC_2020_TO_REC_709, REC_709_TO_REC_2020,
    REFERENCE_LUMINANCE,
};
pub use node::tonemapping;

use crate::FullscreenShader;

/// 3D LUT (look up table) textures used for tonemapping
#[derive(Resource, Clone, ExtractResource)]
pub struct TonemappingLuts {
    pub blender_filmic: Handle<Image>,
    pub agx: Handle<Image>,
    pub tony_mc_mapface: Handle<Image>,
}

pub struct TonemappingPlugin;

impl Plugin for TonemappingPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "tonemapping_shared.wgsl");
        load_shader_library!(app, "lut_bindings.wgsl");
        load_shader_library!(app, "gt7.wgsl");

        embedded_asset!(app, "tonemapping.wgsl");

        if !app.world().is_resource_added::<TonemappingLuts>() {
            let mut images = app.world_mut().resource_mut::<Assets<Image>>();

            #[cfg(feature = "tonemapping_luts")]
            let tonemapping_luts = {
                TonemappingLuts {
                    blender_filmic: images.add(setup_tonemapping_lut_image(
                        include_bytes!("luts/Blender_-11_12.ktx2"),
                        ImageType::Extension("ktx2"),
                    )),
                    agx: images.add(setup_tonemapping_lut_image(
                        include_bytes!("luts/AgX-default_contrast.ktx2"),
                        ImageType::Extension("ktx2"),
                    )),
                    tony_mc_mapface: images.add(setup_tonemapping_lut_image(
                        include_bytes!("luts/tony_mc_mapface.ktx2"),
                        ImageType::Extension("ktx2"),
                    )),
                }
            };

            #[cfg(not(feature = "tonemapping_luts"))]
            let tonemapping_luts = {
                let placeholder = images.add(lut_placeholder());
                TonemappingLuts {
                    blender_filmic: placeholder.clone(),
                    agx: placeholder.clone(),
                    tony_mc_mapface: placeholder,
                }
            };

            app.insert_resource(tonemapping_luts);
        }

        app.add_plugins(ExtractResourcePlugin::<TonemappingLuts>::default());

        app.add_plugins((
            ExtractComponentPlugin::<Tonemapping>::default(),
            ExtractComponentPlugin::<DebandDither>::default(),
            ExtractComponentPlugin::<GranTurismo7Params>::default(),
        ));

        app.add_systems(PostUpdate, sync_tonemapping_enabled);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_gpu_resource::<SpecializedRenderPipelines<TonemappingPipeline>>()
            .init_gpu_resource::<Gt7ParamsUniforms>()
            .add_systems(RenderStartup, init_tonemapping_pipeline)
            .add_systems(
                Render,
                (
                    prepare_view_tonemapping_pipelines.in_set(RenderSystems::Prepare),
                    prepare_gt7_params_uniforms.in_set(RenderSystems::PrepareResources),
                ),
            );
    }
}

#[derive(Resource)]
pub struct TonemappingPipeline {
    texture_bind_group: BindGroupLayoutDescriptor,
    /// [`Self::texture_bind_group`] plus the per-view
    /// [`DisplayTargetUniform`] at binding 5. Used by pipelines specialized
    /// with [`TonemappingPipelineKeyFlags::DISPLAY_TARGET_UNIFORM`].
    display_target_bind_group: BindGroupLayoutDescriptor,
    /// [`Self::display_target_bind_group`] plus the per-view
    /// [`Gt7ParamsUniform`] at binding 6. Used by pipelines specialized with
    /// [`TonemappingPipelineKeyFlags::GT7_PARAMS_UNIFORM`].
    display_target_gt7_bind_group: BindGroupLayoutDescriptor,
    sampler: Sampler,
    fullscreen_shader: FullscreenShader,
    fragment_shader: Handle<Shader>,
}

/// Optionally enables a tonemapping shader that attempts to map linear input stimulus into a perceptually uniform image for a given [`Camera`] entity.
#[derive(
    Component, Debug, Hash, Clone, Copy, Reflect, Default, ExtractComponent, PartialEq, Eq,
)]
#[extract_component_filter(With<Camera>)]
#[reflect(Component, Debug, Hash, Default, PartialEq)]
pub enum Tonemapping {
    /// Bypass tonemapping.
    None,
    /// Suffers from lots hue shifting, brights don't desaturate naturally.
    /// Bright primaries and secondaries don't desaturate at all.
    Reinhard,
    /// Suffers from hue shifting. Brights don't desaturate much at all across the spectrum.
    ReinhardLuminance,
    /// Same base implementation that Godot 4.0 uses for Tonemap ACES.
    /// <https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl>
    /// Not neutral, has a very specific aesthetic, intentional and dramatic hue shifting.
    /// Bright greens and reds turn orange. Bright blues turn magenta.
    /// Significantly increased contrast. Brights desaturate across the spectrum.
    AcesFitted,
    /// By Troy Sobotka
    /// <https://github.com/sobotka/AgX>
    /// Very neutral. Image is somewhat desaturated when compared to other tonemappers.
    /// Little to no hue shifting. Subtle [Abney shifting](https://en.wikipedia.org/wiki/Abney_effect).
    /// NOTE: Requires the `tonemapping_luts` cargo feature.
    AgX,
    /// By Tomasz Stachowiak
    /// Has little hue shifting in the darks and mids, but lots in the brights. Brights desaturate across the spectrum.
    /// Is sort of between Reinhard and `ReinhardLuminance`. Conceptually similar to reinhard-jodie.
    /// Designed as a compromise if you want e.g. decent skin tones in low light, but can't afford to re-do your
    /// VFX to look good without hue shifting.
    SomewhatBoringDisplayTransform,
    /// Current Bevy default.
    /// By Tomasz Stachowiak
    /// <https://github.com/h3r2tic/tony-mc-mapface>
    /// Very neutral. Subtle but intentional hue shifting. Brights desaturate across the spectrum.
    /// Comment from author:
    /// Tony is a display transform intended for real-time applications such as games.
    /// It is intentionally boring, does not increase contrast or saturation, and stays close to the
    /// input stimulus where compression isn't necessary.
    /// Brightness-equivalent luminance of the input stimulus is compressed. The non-linearity resembles Reinhard.
    /// Color hues are preserved during compression, except for a deliberate [Bezold–Brücke shift](https://en.wikipedia.org/wiki/Bezold%E2%80%93Br%C3%BCcke_shift).
    /// To avoid posterization, selective desaturation is employed, with care to avoid the [Abney effect](https://en.wikipedia.org/wiki/Abney_effect).
    /// NOTE: Requires the `tonemapping_luts` cargo feature.
    #[default]
    TonyMcMapface,
    /// Default Filmic Display Transform from blender.
    /// Somewhat neutral. Suffers from hue shifting. Brights desaturate across the spectrum.
    /// NOTE: Requires the `tonemapping_luts` cargo feature.
    BlenderFilmic,
    /// Despite its name, it is not considered to be neutral.
    /// Highly saturated colors and tends to produce a very high contrast image.
    /// Suffers from significant [Abney shifting](https://en.wikipedia.org/wiki/Abney_effect), and tends to crush grays and desaturated colors.
    /// Designed for e-commerce to faithfully reproduce the colors of brand's logos when used with low brightness grayscale lighting.
    /// See [the KhronosGroup spec](https://github.com/KhronosGroup/ToneMapping/tree/main/PBR_Neutral) for more information.
    KhronosPbrNeutral,
    /// By Polyphony Digital, the operator used in Gran Turismo 7.
    /// Published with the SIGGRAPH 2025 course "Physically Based Tone Mapping in Gran Turismo 7"
    /// (MIT License, Copyright (c) 2025 Polyphony Digital Inc.).
    /// Blends a per-channel filmic curve ("camera-like" highlight skew) with a hue-preserving
    /// `ICtCp` branch (60% hue-preserving / 40% per-channel by default), with a luminance-driven
    /// chroma fade near peak white. Natively peak-luminance aware, designed to drive both SDR and
    /// HDR displays; only the SDR path is wired up today.
    /// Algorithmic: does NOT require the `tonemapping_luts` cargo feature.
    /// Tunable per camera via [`GranTurismo7Params`] (defaults are baked until the HDR
    /// display-target uniform plumbing lands).
    GranTurismo7,
}

impl Tonemapping {
    pub fn is_enabled(&self) -> bool {
        *self != Tonemapping::None
    }
}

/// Keeps the auto-managed [`TonemappingEnabled`] marker (in `bevy_camera`,
/// where [`Tonemapping`] itself is not visible) in sync with each camera's
/// [`Tonemapping`] component: present iff the operator is not
/// [`Tonemapping::None`].
///
/// The marker is what lets the renderer's camera extraction (in
/// `bevy_render`, which cannot depend on this crate) select an `Rgba16Float`
/// intermediate main texture for tone-mapped cameras. Runs in
/// [`PostUpdate`]; changes made to [`Tonemapping`] after that point are
/// picked up the next frame.
pub fn sync_tonemapping_enabled(
    mut commands: Commands,
    changed: Query<(Entity, &Tonemapping, Has<TonemappingEnabled>), Changed<Tonemapping>>,
    mut removed: RemovedComponents<Tonemapping>,
) {
    for (entity, tonemapping, has_marker) in &changed {
        if tonemapping.is_enabled() {
            if !has_marker {
                commands.entity(entity).insert(TonemappingEnabled);
            }
        } else if has_marker {
            commands.entity(entity).remove::<TonemappingEnabled>();
        }
    }
    for entity in removed.read() {
        // The entity may have been despawned entirely.
        if let Ok(mut entity_commands) = commands.get_entity(entity) {
            entity_commands.remove::<TonemappingEnabled>();
        }
    }
}

bitflags! {
    /// Various flags describing what tonemapping needs to do.
    ///
    /// This allows the shader to skip unneeded steps.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub struct TonemappingPipelineKeyFlags: u8 {
        /// The hue needs to be changed.
        const HUE_ROTATE                = 0x01;
        /// The white balance needs to be adjusted.
        const WHITE_BALANCE             = 0x02;
        /// Saturation/contrast/gamma/gain/lift for one or more sections
        /// (shadows, midtones, highlights) need to be adjusted.
        const SECTIONAL_COLOR_GRADING   = 0x04;
        /// The per-view [`DisplayTargetUniform`] is bound (at binding 5) and
        /// the `DISPLAY_TARGET_UNIFORM` shader def is pushed.
        ///
        /// Set when the view's resolved [`ViewDisplayTarget`] differs from
        /// the plain SDR default
        /// ([`DisplayTarget::SDR_SRGB`](bevy_window::DisplayTarget::SDR_SRGB)),
        /// or when an active operator needs the calibration data
        /// (currently: whenever [`Self::GT7_PARAMS_UNIFORM`] is set, so the
        /// GT7 params binding index stays fixed). Views on default SDR
        /// targets never set this flag, keeping their pipelines —
        /// layout, shader defs, and composed shader source — byte-identical
        /// to those produced before `DisplayTarget` existed.
        const DISPLAY_TARGET_UNIFORM    = 0x08;
        /// The per-view [`Gt7ParamsUniform`] is bound (at binding 6) and the
        /// `GT7_PARAMS_UNIFORM` shader def is pushed, replacing the GT7
        /// operator's baked SDR defaults with prepared per-camera values.
        ///
        /// Set when the view's [`Tonemapping`] is
        /// [`Tonemapping::GranTurismo7`] **and** the camera has a
        /// [`GranTurismo7Params`] component. Implies
        /// [`Self::DISPLAY_TARGET_UNIFORM`].
        const GT7_PARAMS_UNIFORM        = 0x10;
        /// The view composites in gamma-encoded sRGB space
        /// ([`CompositingSpace::Srgb`](bevy_camera::CompositingSpace::Srgb)):
        /// main pass shaders write sRGB-encoded values, so this pass decodes
        /// the input to scene-linear before tone mapping and re-encodes the
        /// result, preserving the buffer convention the upscaling blit
        /// expects (`SRGB_TO_LINEAR`). Pushes the `SRGB_COMPOSITING` shader
        /// def.
        const SRGB_COMPOSITING          = 0x20;
        /// The view composites in Oklab space
        /// ([`CompositingSpace::Oklab`](bevy_camera::CompositingSpace::Oklab)).
        /// Like [`Self::SRGB_COMPOSITING`], but decoding/encoding with the
        /// Oklab transforms (the blit's `OKLAB_TO_LINEAR` counterpart).
        /// Pushes the `OKLAB_COMPOSITING` shader def.
        const OKLAB_COMPOSITING         = 0x40;
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TonemappingPipelineKey {
    target_format: TextureFormat,
    deband_dither: DebandDither,
    tonemapping: Tonemapping,
    flags: TonemappingPipelineKeyFlags,
}

impl SpecializedRenderPipeline for TonemappingPipeline {
    type Key = TonemappingPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut shader_defs = Vec::new();

        shader_defs.push(ShaderDefVal::UInt(
            "TONEMAPPING_LUT_TEXTURE_BINDING_INDEX".into(),
            3,
        ));
        shader_defs.push(ShaderDefVal::UInt(
            "TONEMAPPING_LUT_SAMPLER_BINDING_INDEX".into(),
            4,
        ));

        if let DebandDither::Enabled = key.deband_dither {
            shader_defs.push("DEBAND_DITHER".into());
        }

        // Define shader flags depending on the color grading options in use.
        if key.flags.contains(TonemappingPipelineKeyFlags::HUE_ROTATE) {
            shader_defs.push("HUE_ROTATE".into());
        }
        if key
            .flags
            .contains(TonemappingPipelineKeyFlags::WHITE_BALANCE)
        {
            shader_defs.push("WHITE_BALANCE".into());
        }
        if key
            .flags
            .contains(TonemappingPipelineKeyFlags::SECTIONAL_COLOR_GRADING)
        {
            shader_defs.push("SECTIONAL_COLOR_GRADING".into());
        }

        // Views compositing in an encoded space (sRGB / Oklab) need the pass
        // to decode before tone mapping and re-encode afterwards, so the main
        // texture keeps holding encoded values for the rest of the frame
        // (UI pass, upscaling blit). Conditional, so plain scene-linear views
        // keep byte-identical pipelines.
        if key
            .flags
            .contains(TonemappingPipelineKeyFlags::SRGB_COMPOSITING)
        {
            shader_defs.push("SRGB_COMPOSITING".into());
        }
        if key
            .flags
            .contains(TonemappingPipelineKeyFlags::OKLAB_COMPOSITING)
        {
            shader_defs.push("OKLAB_COMPOSITING".into());
        }

        // The display-target / GT7-params uniforms are strictly additive:
        // when the flags are unset (every view on a default SDR_SRGB target
        // without GT7 per-camera params), no defs are pushed and the layout
        // below stays the pre-`DisplayTarget` one, so SDR pipelines remain
        // byte-identical.
        if key
            .flags
            .contains(TonemappingPipelineKeyFlags::DISPLAY_TARGET_UNIFORM)
        {
            shader_defs.push("DISPLAY_TARGET_UNIFORM".into());
        }
        if key
            .flags
            .contains(TonemappingPipelineKeyFlags::GT7_PARAMS_UNIFORM)
        {
            shader_defs.push("GT7_PARAMS_UNIFORM".into());
            shader_defs.push(ShaderDefVal::UInt(
                "GT7_PARAMS_BINDING_INDEX".into(),
                GT7_PARAMS_BINDING_INDEX,
            ));
        }

        match key.tonemapping {
            Tonemapping::None => shader_defs.push("TONEMAP_METHOD_NONE".into()),
            Tonemapping::Reinhard => shader_defs.push("TONEMAP_METHOD_REINHARD".into()),
            Tonemapping::ReinhardLuminance => {
                shader_defs.push("TONEMAP_METHOD_REINHARD_LUMINANCE".into());
            }
            Tonemapping::AcesFitted => shader_defs.push("TONEMAP_METHOD_ACES_FITTED".into()),
            Tonemapping::AgX => {
                #[cfg(not(feature = "tonemapping_luts"))]
                error!(
                    "AgX tonemapping requires the `tonemapping_luts` feature.
                    Either enable the `tonemapping_luts` feature for bevy in `Cargo.toml` (recommended),
                    or use a different `Tonemapping` method for your `Camera2d`/`Camera3d`."
                );
                shader_defs.push("TONEMAP_METHOD_AGX".into());
            }
            Tonemapping::SomewhatBoringDisplayTransform => {
                shader_defs.push("TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM".into());
            }
            Tonemapping::TonyMcMapface => {
                #[cfg(not(feature = "tonemapping_luts"))]
                error!(
                    "TonyMcMapFace tonemapping requires the `tonemapping_luts` feature.
                    Either enable the `tonemapping_luts` feature for bevy in `Cargo.toml` (recommended),
                    or use a different `Tonemapping` method for your `Camera2d`/`Camera3d`."
                );
                shader_defs.push("TONEMAP_METHOD_TONY_MC_MAPFACE".into());
            }
            Tonemapping::BlenderFilmic => {
                #[cfg(not(feature = "tonemapping_luts"))]
                error!(
                    "BlenderFilmic tonemapping requires the `tonemapping_luts` feature.
                    Either enable the `tonemapping_luts` feature for bevy in `Cargo.toml` (recommended),
                    or use a different `Tonemapping` method for your `Camera2d`/`Camera3d`."
                );
                shader_defs.push("TONEMAP_METHOD_BLENDER_FILMIC".into());
            }
            Tonemapping::KhronosPbrNeutral => shader_defs.push("TONEMAP_METHOD_PBR_NEUTRAL".into()),
            Tonemapping::GranTurismo7 => {
                shader_defs.push("TONEMAP_METHOD_GRAN_TURISMO_7".into());
            }
        }
        let bind_group_layout = if key
            .flags
            .contains(TonemappingPipelineKeyFlags::GT7_PARAMS_UNIFORM)
        {
            self.display_target_gt7_bind_group.clone()
        } else if key
            .flags
            .contains(TonemappingPipelineKeyFlags::DISPLAY_TARGET_UNIFORM)
        {
            self.display_target_bind_group.clone()
        } else {
            self.texture_bind_group.clone()
        };

        RenderPipelineDescriptor {
            label: Some("tonemapping pipeline".into()),
            layout: vec![bind_group_layout],
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

/// Binding index of the per-view [`DisplayTargetUniform`] in the tonemapping
/// pass bind group. Only part of the layout (and the shader) when
/// [`TonemappingPipelineKeyFlags::DISPLAY_TARGET_UNIFORM`] is set; the index
/// is hardcoded in `tonemapping.wgsl` under the matching shader def.
pub const DISPLAY_TARGET_BINDING_INDEX: u32 = 5;

/// Binding index of the per-view [`Gt7ParamsUniform`] in the tonemapping pass
/// bind group. Only part of the layout (and the shader) when
/// [`TonemappingPipelineKeyFlags::GT7_PARAMS_UNIFORM`] is set; pushed into
/// `gt7.wgsl` as the `GT7_PARAMS_BINDING_INDEX` shader def so other bind
/// groups can rebind it at a different index later.
pub const GT7_PARAMS_BINDING_INDEX: u32 = 6;

pub fn init_tonemapping_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    fullscreen_shader: Res<FullscreenShader>,
    asset_server: Res<AssetServer>,
) {
    let mut entries = DynamicBindGroupLayoutEntries::new_with_indices(
        ShaderStages::FRAGMENT,
        (
            (0, uniform_buffer::<ViewUniform>(true)),
            (
                1,
                texture_2d(TextureSampleType::Float { filterable: false }),
            ),
            (2, sampler(SamplerBindingType::NonFiltering)),
        ),
    );
    let lut_layout_entries = get_lut_bind_group_layout_entries();
    entries = entries.extend_with_indices(((3, lut_layout_entries[0]), (4, lut_layout_entries[1])));

    let tonemap_texture_bind_group =
        BindGroupLayoutDescriptor::new("tonemapping_hdr_texture_bind_group_layout", &entries);

    // Layout variants that additionally carry the per-view display-target
    // calibration uniform, and the GT7 operator's params uniform on top of
    // that. Kept separate from the base layout so SDR pipelines stay
    // byte-identical to pre-`DisplayTarget` Bevy.
    let display_target_entries = entries.extend_with_indices(((
        DISPLAY_TARGET_BINDING_INDEX,
        uniform_buffer::<DisplayTargetUniform>(true),
    ),));
    let tonemap_display_target_bind_group = BindGroupLayoutDescriptor::new(
        "tonemapping_display_target_bind_group_layout",
        &display_target_entries,
    );

    let display_target_gt7_entries = display_target_entries.extend_with_indices(((
        GT7_PARAMS_BINDING_INDEX,
        uniform_buffer::<Gt7ParamsUniform>(true),
    ),));
    let tonemap_display_target_gt7_bind_group = BindGroupLayoutDescriptor::new(
        "tonemapping_display_target_gt7_bind_group_layout",
        &display_target_gt7_entries,
    );

    let sampler = render_device.create_sampler(&SamplerDescriptor::default());

    commands.insert_resource(TonemappingPipeline {
        texture_bind_group: tonemap_texture_bind_group,
        display_target_bind_group: tonemap_display_target_bind_group,
        display_target_gt7_bind_group: tonemap_display_target_gt7_bind_group,
        sampler,
        fullscreen_shader: fullscreen_shader.clone(),
        fragment_shader: load_embedded_asset!(asset_server.as_ref(), "tonemapping.wgsl"),
    });
}

/// The specialized tonemapping pipeline of a view, plus the
/// [`TonemappingPipelineKeyFlags`] it was specialized with (which the
/// tonemapping node uses to bind the matching optional uniforms).
#[derive(Component)]
pub struct ViewTonemappingPipeline {
    pipeline_id: CachedRenderPipelineId,
    flags: TonemappingPipelineKeyFlags,
}

impl ViewTonemappingPipeline {
    /// The bind group layout this view's pipeline was specialized with
    /// (matching the layout selection in
    /// [`TonemappingPipeline::specialize`](SpecializedRenderPipeline::specialize)).
    fn bind_group_layout<'a>(
        &self,
        pipeline: &'a TonemappingPipeline,
    ) -> &'a BindGroupLayoutDescriptor {
        if self
            .flags
            .contains(TonemappingPipelineKeyFlags::GT7_PARAMS_UNIFORM)
        {
            &pipeline.display_target_gt7_bind_group
        } else if self
            .flags
            .contains(TonemappingPipelineKeyFlags::DISPLAY_TARGET_UNIFORM)
        {
            &pipeline.display_target_bind_group
        } else {
            &pipeline.texture_bind_group
        }
    }
}

pub fn prepare_view_tonemapping_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<TonemappingPipeline>>,
    upscaling_pipeline: Res<TonemappingPipeline>,
    view_targets: Query<(
        Entity,
        &ExtractedView,
        &ViewTarget,
        Option<&Tonemapping>,
        Option<&DebandDither>,
        Option<&ViewDisplayTarget>,
        Option<&GranTurismo7Params>,
    )>,
) {
    for (entity, view, view_target, tonemapping, dither, view_display_target, gt7_params) in
        view_targets.iter()
    {
        let tonemapping = *tonemapping.unwrap_or(&Tonemapping::None);

        // As an optimization, we omit parts of the shader that are unneeded.
        let mut flags = TonemappingPipelineKeyFlags::empty();
        flags.set(
            TonemappingPipelineKeyFlags::HUE_ROTATE,
            view.color_grading.global.hue != 0.0,
        );
        flags.set(
            TonemappingPipelineKeyFlags::WHITE_BALANCE,
            view.color_grading.global.temperature != 0.0 || view.color_grading.global.tint != 0.0,
        );
        flags.set(
            TonemappingPipelineKeyFlags::SECTIONAL_COLOR_GRADING,
            view.color_grading
                .all_sections()
                .any(|section| *section != default()),
        );

        // Views compositing in an encoded space need the pass to decode /
        // re-encode around the operator (see the flag docs). Scene-linear
        // views (`CompositingSpace::Linear` or no component) set neither
        // flag, keeping their key identical to before these flags existed.
        flags.set(
            TonemappingPipelineKeyFlags::SRGB_COMPOSITING,
            view_target.compositing_space == Some(CompositingSpace::Srgb),
        );
        flags.set(
            TonemappingPipelineKeyFlags::OKLAB_COMPOSITING,
            view_target.compositing_space == Some(CompositingSpace::Oklab),
        );

        // The GT7 params uniform is active exactly when the operator is GT7
        // and the camera opted in with a `GranTurismo7Params` component
        // (`prepare_gt7_params_uniforms` uses the same predicate).
        let gt7_uniform_active = tonemapping == Tonemapping::GranTurismo7 && gt7_params.is_some();
        // The display-target uniform is bound for views whose resolved
        // display target is not the plain SDR default, and whenever an
        // operator needs it (currently: the GT7 params uniform path, which
        // keeps binding 5 occupied so the params binding index is stable).
        // Views on default SDR targets push neither flag, so their pipeline
        // key, layout, and shader stay byte-identical to before
        // `DisplayTarget` existed.
        let display_target_uniform_active = gt7_uniform_active
            || view_display_target
                .is_some_and(|view_display_target| !view_display_target.is_plain_sdr_srgb());
        flags.set(
            TonemappingPipelineKeyFlags::DISPLAY_TARGET_UNIFORM,
            display_target_uniform_active,
        );
        flags.set(
            TonemappingPipelineKeyFlags::GT7_PARAMS_UNIFORM,
            gt7_uniform_active,
        );

        let key = TonemappingPipelineKey {
            target_format: view.target_format,
            deband_dither: *dither.unwrap_or(&DebandDither::Disabled),
            tonemapping,
            flags,
        };
        let pipeline = pipelines.specialize(&pipeline_cache, &upscaling_pipeline, key);

        commands.entity(entity).insert(ViewTonemappingPipeline {
            pipeline_id: pipeline,
            flags,
        });
    }
}
/// Enables a debanding shader that applies dithering to mitigate color banding in the final image for a given [`Camera`] entity.
#[derive(
    Component, Debug, Hash, Clone, Copy, Reflect, Default, ExtractComponent, PartialEq, Eq,
)]
#[extract_component_filter(With<Camera>)]
#[reflect(Component, Debug, Hash, Default, PartialEq)]
pub enum DebandDither {
    #[default]
    Disabled,
    Enabled,
}

pub fn get_lut_bindings<'a>(
    images: &'a RenderAssets<GpuImage>,
    tonemapping_luts: &'a TonemappingLuts,
    tonemapping: &Tonemapping,
    fallback_image: &'a FallbackImage,
) -> (&'a TextureView, &'a Sampler) {
    let image = match tonemapping {
        // AgX lut texture used when tonemapping doesn't need a texture since it's very small (32x32x32)
        Tonemapping::None
        | Tonemapping::Reinhard
        | Tonemapping::ReinhardLuminance
        | Tonemapping::AcesFitted
        | Tonemapping::AgX
        | Tonemapping::KhronosPbrNeutral
        | Tonemapping::SomewhatBoringDisplayTransform
        | Tonemapping::GranTurismo7 => &tonemapping_luts.agx,
        Tonemapping::TonyMcMapface => &tonemapping_luts.tony_mc_mapface,
        Tonemapping::BlenderFilmic => &tonemapping_luts.blender_filmic,
    };
    let lut_image = images.get(image).unwrap_or(&fallback_image.d3);
    (&lut_image.texture_view, &lut_image.sampler)
}

pub fn get_lut_bind_group_layout_entries() -> [BindGroupLayoutEntryBuilder; 2] {
    [
        texture_3d(TextureSampleType::Float { filterable: true }),
        sampler(SamplerBindingType::Filtering),
    ]
}

#[expect(clippy::allow_attributes, reason = "`dead_code` is not always linted.")]
#[allow(
    dead_code,
    reason = "There is unused code when the `tonemapping_luts` feature is disabled."
)]
fn setup_tonemapping_lut_image(bytes: &[u8], image_type: ImageType) -> Image {
    let image_sampler = ImageSampler::Descriptor(bevy_image::ImageSamplerDescriptor {
        label: Some("Tonemapping LUT sampler".to_string()),
        address_mode_u: bevy_image::ImageAddressMode::ClampToEdge,
        address_mode_v: bevy_image::ImageAddressMode::ClampToEdge,
        address_mode_w: bevy_image::ImageAddressMode::ClampToEdge,
        mag_filter: bevy_image::ImageFilterMode::Linear,
        min_filter: bevy_image::ImageFilterMode::Linear,
        mipmap_filter: bevy_image::ImageFilterMode::Linear,
        ..default()
    });
    Image::from_buffer(
        bytes,
        image_type,
        CompressedImageFormats::NONE,
        false,
        image_sampler,
        // LUT must be kept in main world for render recovery reasons
        RenderAssetUsages::default(),
    )
    .unwrap()
}

pub fn lut_placeholder() -> Image {
    let format = TextureFormat::Rgba8Unorm;
    let data = vec![255, 0, 255, 255];
    Image {
        data: Some(data),
        data_order: TextureDataOrder::default(),
        texture_descriptor: TextureDescriptor {
            size: Extent3d::default(),
            format,
            dimension: TextureDimension::D3,
            label: None,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        },
        sampler: ImageSampler::Default,
        texture_view_descriptor: None,
        asset_usage: RenderAssetUsages::RENDER_WORLD,
        copy_on_resize: false,
        source_primaries: Default::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_app::{App, Update};

    #[test]
    fn tonemapping_enabled_marker_syncs_with_tonemapping() {
        let mut app = App::new();
        app.add_systems(Update, sync_tonemapping_enabled);

        // Spawning with an active operator inserts the marker.
        let entity = app.world_mut().spawn(Tonemapping::TonyMcMapface).id();
        app.update();
        assert!(app.world().entity(entity).contains::<TonemappingEnabled>());

        // Changing to `None` removes it.
        *app.world_mut()
            .entity_mut(entity)
            .get_mut::<Tonemapping>()
            .unwrap() = Tonemapping::None;
        app.update();
        assert!(!app.world().entity(entity).contains::<TonemappingEnabled>());

        // Changing back to an operator re-inserts it.
        *app.world_mut()
            .entity_mut(entity)
            .get_mut::<Tonemapping>()
            .unwrap() = Tonemapping::GranTurismo7;
        app.update();
        assert!(app.world().entity(entity).contains::<TonemappingEnabled>());

        // Removing `Tonemapping` entirely removes the marker too.
        app.world_mut().entity_mut(entity).remove::<Tonemapping>();
        app.update();
        assert!(!app.world().entity(entity).contains::<TonemappingEnabled>());

        // `Tonemapping::None` from the start never inserts the marker.
        let none_entity = app.world_mut().spawn(Tonemapping::None).id();
        app.update();
        assert!(!app
            .world()
            .entity(none_entity)
            .contains::<TonemappingEnabled>());
    }
}
