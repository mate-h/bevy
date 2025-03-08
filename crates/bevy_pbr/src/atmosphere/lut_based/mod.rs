use bevy_app::{App, Plugin};
use bevy_asset::load_internal_asset;
use bevy_core_pipeline::{
    core_3d::{
        graph::{Core3d, Node3d},
        Camera3d,
    },
    fullscreen_vertex_shader::fullscreen_shader_vertex_state,
};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::With,
    resource::Resource,
    schedule::IntoSystemConfigs,
    system::{Commands, Query, Res, ResMut},
    world::{FromWorld, World},
};
use bevy_math::{Mat4, UVec2, UVec3, Vec3};
use bevy_reflect::Reflect;
use bevy_render::{
    camera::{Camera, ExtractedCamera},
    extract_component::{
        ComponentUniforms, ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin,
    },
    render_graph::{RenderGraphApp, ViewNodeRunner},
    render_resource::{
        binding_types::{
            sampler, storage_buffer, texture_2d, texture_depth_2d, texture_depth_2d_multisampled,
            texture_storage_2d, texture_storage_3d, uniform_buffer,
        },
        BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, BlendComponent,
        BlendFactor, BlendOperation, BlendState, CachedComputePipelineId, CachedRenderPipelineId,
        ColorTargetState, ColorWrites, ComputePipelineDescriptor, DynamicUniformBuffer, Extent3d,
        FragmentState, MultisampleState, PipelineCache, PrimitiveState, RenderPipelineDescriptor,
        Sampler, SamplerBindingType, Shader, ShaderStages, ShaderType, SpecializedRenderPipeline,
        SpecializedRenderPipelines, StorageTextureAccess, TextureDescriptor, TextureDimension,
        TextureFormat, TextureSampleType, TextureUsages,
    },
    renderer::{RenderAdapter, RenderDevice},
    texture::{CachedTexture, TextureCache},
    view::{ExtractedView, Msaa, ViewDepthTexture, ViewUniform, ViewUniforms},
    Render, RenderApp, RenderSet,
};
use tracing::warn;

use crate::{
    atmosphere::{core, AtmosphericScattering},
    GpuLights, LightMeta,
};

use super::{validate_plugin, AtmosphericScatteringSettings, ExtractedAtmosphere};

mod node;

pub mod shaders {
    use bevy_asset::{weak_handle, Handle};
    use bevy_render::render_resource::Shader;

    pub const SKY_VIEW_LUT: Handle<Shader> = weak_handle!("f87e007a-bf4b-4f99-9ef0-ac21d369f0e5");
    pub const AERIAL_VIEW_LUT: Handle<Shader> =
        weak_handle!("a3daf030-4b64-49ae-a6a7-354489597cbe");
    pub const RESOLVE_ATMOSPHERE: Handle<Shader> =
        weak_handle!("09422f46-d0f7-41c1-be24-121c17d6e834");
}

pub struct LutBasedAtmospherePlugin;

impl Plugin for LutBasedAtmospherePlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            shaders::SKY_VIEW_LUT,
            "sky_view_lut.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            shaders::AERIAL_VIEW_LUT,
            "aerial_view_lut.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            shaders::RESOLVE_ATMOSPHERE,
            "resolve_atmosphere.wgsl",
            Shader::from_wgsl
        );

        app.register_type::<Settings>()
            .add_plugins(UniformComponentPlugin::<Uniforms>::default());
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        let device = render_app.world().resource::<RenderDevice>();
        let adapter = render_app.world().resource::<RenderAdapter>();
        if let Err(err) = validate_plugin(device, adapter) {
            warn!("LutBasedAtmospherePlugin not loaded: {err}");
        }

        render_app
            .init_resource::<Layout>()
            .init_resource::<Pipelines>()
            .init_resource::<SpecializedRenderPipelines<Layout>>()
            .add_systems(
                Render,
                (
                    queue_render_sky_pipelines.in_set(RenderSet::Queue),
                    prepare_luts.in_set(RenderSet::PrepareAssets),
                    prepare_bind_groups.in_set(RenderSet::PrepareBindGroups),
                ),
            )
            .add_render_graph_node::<node::LutsNode>(Core3d, node::LutsLabel)
            .add_render_graph_node::<node::ResolveNode>(Core3d, node::ResolveLabel)
            .add_render_graph_edges(Core3d, (node::LutsLabel, node::ResolveLabel))
            .add_render_graph_edges(
                Core3d,
                (Node3d::EndMainPass, node::ResolveLabel, Node3d::Tonemapping),
            );
    }
}

/// This component controls the resolution of the atmosphere LUTs, and
/// how many samples are used when computing them.
///
/// The transmittance LUT stores the transmittance from a point in the
/// atmosphere to the outer edge of the atmosphere in any direction,
/// parametrized by the point's radius and the cosine of the zenith angle
/// of the ray.
///
/// The multiscattering LUT stores the factor representing luminance scattered
/// towards the camera with scattering order >2, parametrized by the point's radius
/// and the cosine of the zenith angle of the sun.
///
/// The sky-view lut is essentially the actual skybox, storing the light scattered
/// towards the camera in every direction with a cubemap.
///
/// The aerial-view lut is a 3d LUT fit to the view frustum, which stores the luminance
/// scattered towards the camera at each point (RGB channels), alongside the average
/// transmittance to that point (A channel).
#[derive(Clone, Reflect, ShaderType)]
#[type_path = "bevy_pbr::atmosphere::LutBasedAtmosphericScatteringSettings"]
pub struct Settings {
    /// The size of the sky-view LUT.
    pub sky_view_lut_size: UVec2,

    /// The number of points to sample along each ray when
    /// computing the sky-view LUT.
    pub sky_view_lut_samples: u32,

    /// The number of points to sample for each slice along the z-axis
    /// of the aerial-view LUT.
    pub aerial_view_lut_samples: u32,

    /// The size of the aerial-view LUT.
    pub aerial_view_lut_size: UVec3,

    /// The maximum distance from the camera to evaluate the
    /// aerial view LUT. The slices along the z-axis of the
    /// texture will be distributed linearly from the camera
    /// to this value.
    ///
    /// units: m
    pub aerial_view_lut_max_distance: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            sky_view_lut_size: UVec2::new(400, 200),
            sky_view_lut_samples: 16,
            aerial_view_lut_size: UVec3::new(32, 32, 32),
            aerial_view_lut_samples: 10,
            aerial_view_lut_max_distance: 3.2e4,
        }
    }
}

#[derive(ShaderType, Clone)]
pub struct AtmosphereTransforms {
    world_from_atmosphere: Mat4,
    atmosphere_from_world: Mat4,
}

#[derive(Component, ShaderType, Clone)]
pub struct Uniforms {
    transforms: AtmosphereTransforms,
    settings: Settings,
}

pub fn prepare_uniforms(
    views: Query<
        (Entity, &ExtractedView, &AtmosphericScatteringSettings),
        (With<Camera3d>, With<AtmosphericScattering>),
    >,
    mut commands: Commands,
) {
    for (entity, view, AtmosphericScatteringSettings::LutBased(settings)) in &views {
        let world_from_view = view.world_from_view.compute_matrix();
        let camera_z = world_from_view.z_axis.truncate();
        let camera_y = world_from_view.y_axis.truncate();
        let atmo_z = camera_z
            .with_y(0.0)
            .try_normalize()
            .unwrap_or_else(|| camera_y.with_y(0.0).normalize());
        let atmo_y = Vec3::Y;
        let atmo_x = atmo_y.cross(atmo_z).normalize();
        let world_from_atmosphere = Mat4::from_cols(
            atmo_x.extend(0.0),
            atmo_y.extend(0.0),
            atmo_z.extend(0.0),
            world_from_view.w_axis,
        );

        let atmosphere_from_world = world_from_atmosphere.inverse();

        let transforms = AtmosphereTransforms {
            world_from_atmosphere,
            atmosphere_from_world,
        };

        commands.entity(entity).insert(Uniforms {
            transforms,
            settings: settings.clone(),
        });
    }
}

#[derive(Resource)]
pub struct Layout {
    sky_view_lut: BindGroupLayout,
    aerial_view_lut: BindGroupLayout,
    resolve_atmosphere: BindGroupLayout,
    resolve_atmosphere_msaa: BindGroupLayout,
    sampler: Sampler,
}

impl FromWorld for Layout {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let sky_view_lut = render_device.create_bind_group_layout(
            "sky_view_lut_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, storage_buffer::<core::Uniforms>(true)),
                    (1, sampler(SamplerBindingType::Filtering)),
                    (2, uniform_buffer::<ViewUniform>(true)),
                    (3, uniform_buffer::<GpuLights>(true)),
                    (4, uniform_buffer::<Settings>(true)),
                    (5, texture_2d(TextureSampleType::Float { filterable: true })), // transmittance lut
                    (6, texture_2d(TextureSampleType::Float { filterable: true })), // multiscattering lut
                    (
                        9,
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ),
        );

        let aerial_view_lut = render_device.create_bind_group_layout(
            "aerial_view_lut_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, storage_buffer::<core::Uniforms>(true)),
                    (1, sampler(SamplerBindingType::Filtering)),
                    (2, uniform_buffer::<ViewUniform>(true)),
                    (3, uniform_buffer::<GpuLights>(true)),
                    (4, uniform_buffer::<Settings>(true)),
                    (5, texture_2d(TextureSampleType::Float { filterable: true })), // transmittance lut
                    (6, texture_2d(TextureSampleType::Float { filterable: true })), // mulitscattering lut
                    (
                        9,
                        texture_storage_3d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ),
        );

        let resolve_atmosphere = render_device.create_bind_group_layout(
            "resolve_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, storage_buffer::<core::Uniforms>(true)),
                    (1, sampler(SamplerBindingType::Filtering)),
                    (2, uniform_buffer::<ViewUniform>(true)),
                    (3, uniform_buffer::<GpuLights>(true)),
                    (5, uniform_buffer::<Settings>(true)),
                    (6, texture_2d(TextureSampleType::Float { filterable: true })), // transmittance lut
                    (8, texture_2d(TextureSampleType::Float { filterable: true })), // sky view lut
                    (9, texture_2d(TextureSampleType::Float { filterable: true })), // aerial view lut
                    (10, texture_depth_2d()), // view depth texture
                ),
            ),
        );

        let resolve_atmosphere_msaa = render_device.create_bind_group_layout(
            "resolve_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, storage_buffer::<core::Uniforms>(true)),
                    (1, sampler(SamplerBindingType::Filtering)),
                    (2, uniform_buffer::<ViewUniform>(true)),
                    (3, uniform_buffer::<GpuLights>(true)),
                    (4, uniform_buffer::<Settings>(true)),
                    (5, texture_2d(TextureSampleType::Float { filterable: true })), // transmittance lut
                    (7, texture_2d(TextureSampleType::Float { filterable: true })), // sky view lut
                    (8, texture_2d(TextureSampleType::Float { filterable: true })), // aerial view lut
                    (9, texture_depth_2d_multisampled()), // view depth texture
                ),
            ),
        );

        let sampler = world.resource::<core::Layout>().sampler.clone();

        Self {
            sky_view_lut,
            aerial_view_lut,
            resolve_atmosphere,
            resolve_atmosphere_msaa,
            sampler,
        }
    }
}

#[derive(Resource)]
pub struct Pipelines {
    sky_view_lut: CachedComputePipelineId,
    aerial_view_lut: CachedComputePipelineId,
}

impl FromWorld for Pipelines {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let layout = world.resource::<Layout>();

        let sky_view_lut = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("sky_view_lut_pipeline".into()),
            layout: vec![layout.sky_view_lut.clone()],
            push_constant_ranges: vec![],
            shader: shaders::SKY_VIEW_LUT,
            shader_defs: vec![],
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        });

        let aerial_view_lut = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("aerial_view_lut_pipeline".into()),
            layout: vec![layout.aerial_view_lut.clone()],
            push_constant_ranges: vec![],
            shader: shaders::AERIAL_VIEW_LUT,
            shader_defs: vec![],
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        });

        Self {
            sky_view_lut,
            aerial_view_lut,
        }
    }
}

#[derive(Component)]
pub struct ResolvePipelineId(CachedRenderPipelineId);

impl ResolvePipelineId {
    pub fn id(&self) -> CachedRenderPipelineId {
        self.0
    }
}

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub struct ResolvePipelineKey {
    pub msaa_samples: u32,
    pub hdr: bool,
}

impl SpecializedRenderPipeline for Layout {
    type Key = ResolvePipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut shader_defs = Vec::new();

        if key.msaa_samples > 1 {
            shader_defs.push("MULTISAMPLED".into());
        }
        if key.hdr {
            shader_defs.push("TONEMAP_IN_SHADER".into());
        }

        RenderPipelineDescriptor {
            label: Some(format!("render_sky_pipeline_{}", key.msaa_samples).into()),
            layout: vec![if key.msaa_samples > 1 {
                self.resolve_atmosphere_msaa.clone()
            } else {
                self.resolve_atmosphere.clone()
            }],
            push_constant_ranges: vec![],
            vertex: fullscreen_shader_vertex_state(),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState {
                count: key.msaa_samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            zero_initialize_workgroup_memory: false,
            fragment: Some(FragmentState {
                shader: shaders::RESOLVE_ATMOSPHERE.clone(),
                shader_defs,
                entry_point: "main".into(),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba16Float,
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::Src1,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::Zero,
                            dst_factor: BlendFactor::One,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                })],
            }),
        }
    }
}

fn queue_render_sky_pipelines(
    views: Query<(Entity, &ExtractedCamera, &Msaa), With<ExtractedAtmosphere>>,
    pipeline_cache: Res<PipelineCache>,
    layout: Res<Layout>,
    mut specializer: ResMut<SpecializedRenderPipelines<Layout>>,
    mut commands: Commands,
) {
    for (entity, camera, msaa) in &views {
        let id = specializer.specialize(
            &pipeline_cache,
            &layout,
            ResolvePipelineKey {
                msaa_samples: msaa.samples(),
                hdr: camera.hdr,
            },
        );
        commands.entity(entity).insert(ResolvePipelineId(id));
    }
}

#[derive(Component)]
pub struct Luts {
    sky_view_lut: CachedTexture,
    aerial_view_lut: CachedTexture,
}

fn prepare_luts(
    atmospheres: Query<
        (Entity, &AtmosphericScatteringSettings),
        (With<Camera3d>, With<AtmosphericScattering>),
    >,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
    mut commands: Commands,
) {
    for (entity, AtmosphericScatteringSettings::LutBased(settings)) in &atmospheres {
        let sky_view_lut = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("sky_view_lut"),
                size: Extent3d {
                    width: settings.sky_view_lut_size.x,
                    height: settings.sky_view_lut_size.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        let aerial_view_lut = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("aerial_view_lut"),
                size: Extent3d {
                    width: settings.aerial_view_lut_size.x,
                    height: settings.aerial_view_lut_size.y,
                    depth_or_array_layers: settings.aerial_view_lut_size.z,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D3,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        commands.entity(entity).insert(Luts {
            sky_view_lut,
            aerial_view_lut,
        });
    }
}

#[derive(Component)]
pub struct BindGroups {
    pub sky_view_lut: BindGroup,
    pub aerial_view_lut: BindGroup,
    pub resolve_atmosphere: BindGroup,
}

fn prepare_bind_groups(
    views: Query<
        (
            Entity,
            &Luts,
            &AtmosphericScattering,
            &Msaa,
            &ViewDepthTexture,
        ),
        With<Camera3d>,
    >,
    atmospheres: Query<&core::Luts, With<ExtractedAtmosphere>>,
    render_device: Res<RenderDevice>,
    core_uniforms: Res<core::UniformsBuffer>,
    lut_based_uniforms: Res<ComponentUniforms<Uniforms>>,
    view_uniforms: Res<ViewUniforms>,
    lights_uniforms: Res<LightMeta>,
    layout: Res<Layout>,
    mut commands: Commands,
) {
    let core_uniforms_binding = core_uniforms
        .binding()
        .expect("Failed to prepare atmosphere bind groups. Atmosphere storage buffer missing");

    let lut_based_uniforms_binding = lut_based_uniforms.binding().expect(
        "Failed to prepare atmosphere bind groups. AtmosphereSettings uniform buffer missing",
    );

    let view_binding = view_uniforms
        .uniforms
        .binding()
        .expect("Failed to prepare atmosphere bind groups. View uniform buffer missing");

    let lights_binding = lights_uniforms
        .view_gpu_lights
        .binding()
        .expect("Failed to prepare atmosphere bind groups. Lights uniform buffer missing");

    for (entity, aux_luts, atmosphere, msaa, view_depth_texture) in &views {
        let Ok(core_luts) = atmospheres.get(atmosphere.0) else {
            continue;
        };

        let sky_view_lut = render_device.create_bind_group(
            "sky_view_lut_bind_group",
            &layout.sky_view_lut,
            &BindGroupEntries::with_indices((
                (0, core_uniforms_binding.clone()),
                (1, &layout.sampler),
                (2, view_binding.clone()),
                (3, lights_binding.clone()),
                (4, lut_based_uniforms_binding.clone()),
                (5, &core_luts.transmittance_lut.default_view),
                (6, &core_luts.multiscattering_lut.default_view),
                (9, &aux_luts.sky_view_lut.default_view),
            )),
        );

        let aerial_view_lut = render_device.create_bind_group(
            "aerial_view_lut_bind_group",
            &layout.aerial_view_lut,
            &BindGroupEntries::with_indices((
                (0, core_uniforms_binding.clone()),
                (1, &layout.sampler),
                (2, view_binding.clone()),
                (3, lights_binding.clone()),
                (4, lut_based_uniforms_binding.clone()),
                (5, &core_luts.transmittance_lut.default_view),
                (6, &core_luts.multiscattering_lut.default_view),
                (9, &aux_luts.aerial_view_lut.default_view),
            )),
        );

        let resolve_atmosphere_layout = if msaa.samples() > 1 {
            &layout.resolve_atmosphere_msaa
        } else {
            &layout.resolve_atmosphere
        };

        let resolve_atmosphere = render_device.create_bind_group(
            "resolve_atmosphere_bind_group",
            resolve_atmosphere_layout,
            &BindGroupEntries::with_indices((
                (0, core_uniforms_binding.clone()),
                (1, &layout.sampler),
                (2, view_binding.clone()),
                (3, lights_binding.clone()),
                (4, lut_based_uniforms_binding.clone()),
                (5, &core_luts.transmittance_lut.default_view),
                (6, &core_luts.multiscattering_lut.default_view),
                (7, &aux_luts.sky_view_lut.default_view),
                (8, &aux_luts.aerial_view_lut.default_view),
                (9, view_depth_texture.view()),
            )),
        );

        commands.entity(entity).insert(BindGroups {
            sky_view_lut,
            aerial_view_lut,
            resolve_atmosphere,
        });
    }
}
