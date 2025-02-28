use bevy_asset::prelude::AssetChanged;
use bevy_core_pipeline::{
    core_3d::Camera3d, fullscreen_vertex_shader::fullscreen_shader_vertex_state,
};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{Changed, Or, With},
    removal_detection::RemovedComponents,
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
    world::{FromWorld, World},
};
use bevy_math::{Mat4, Vec3};
use bevy_render::{
    camera::Camera,
    extract_component::ComponentUniforms,
    render_resource::{binding_types::*, *},
    renderer::{RenderDevice, RenderQueue},
    texture::{CachedTexture, TextureCache},
    view::{ExtractedView, Msaa, ViewDepthTexture, ViewUniform, ViewUniforms},
    Extract,
};

use crate::{GpuLights, LightMeta};

use super::{shaders, Atmosphere, AtmosphereSettings, Planet, ScatteringProfile};

#[derive(ShaderType)]
struct GpuPlanet {
    ground_albedo: Vec3,
    lower_radius: f32,
    lower_radius_sq: f32,
    upper_radius: f32,
    upper_radius_sq: f32,
    space_altitude: f32,
}

#[derive(ShaderType)]
struct GpuAtmosphere {
    profile: ScatteringProfile,
    planet: GpuPlanet,
}

impl From<Planet> for GpuPlanet {
    fn from(planet: Planet) -> Self {
        let lower_radius = planet.radius;
        let upper_radius = lower_radius + planet.space_altitude;

        Self {
            ground_albedo: planet.ground_albedo.to_vec3(),
            lower_radius,
            lower_radius_sq: lower_radius * lower_radius,
            upper_radius,
            upper_radius_sq: upper_radius * upper_radius,
            space_altitude: planet.space_altitude,
        }
    }
}

#[derive(Resource)]
pub struct AtmosphereBuffers {
    atmospheres: DynamicStorageBuffer<GpuAtmosphere>,
    core_lut_settings: DynamicUniformBuffer<AtmosphereSettings>,
}

pub(super) fn extract_atmospheres(
    removed: Extract<RemovedComponents<Atmosphere>>,
    atmospheres: Extract<
        Query<
            (&Atmosphere, &AtmosphereSettings, &Planet),
            Or<(
                Changed<Atmosphere>,
                AssetChanged<Atmosphere>,
                Changed<AtmosphereSettings>,
                Changed<Planet>,
            )>,
        >,
    >,
) {
}

#[derive(Resource)]
pub(crate) struct AtmosphereLayout {
    pub transmittance_lut: BindGroupLayout,
    pub multiscattering_lut: BindGroupLayout,
    pub sky_view_lut: BindGroupLayout,
    pub aerial_view_lut: BindGroupLayout,
    pub sampler: Sampler,
}

#[derive(Resource)]
pub(crate) struct RenderSkyLayout {
    pub render_sky: BindGroupLayout,
    pub render_sky_msaa: BindGroupLayout,
    pub sampler: Sampler,
}

impl FromWorld for AtmosphereLayout {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let transmittance_lut = render_device.create_bind_group_layout(
            "transmittance_lut_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, storage_buffer::<GpuAtmosphere>(true)),
                    (4, uniform_buffer::<AtmosphereSettings>(true)),
                    (
                        // transmittance lut storage texture
                        9,
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ),
        );

        let multiscattering_lut = render_device.create_bind_group_layout(
            "multiscattering_lut_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, storage_buffer::<GpuAtmosphere>(true)),
                    (1, sampler(SamplerBindingType::Filtering)),
                    (4, uniform_buffer::<AtmosphereSettings>(true)),
                    (5, texture_2d(TextureSampleType::Float { filterable: true })), // transmittance lut
                    (
                        // multiscattering lut storage texture
                        9,
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ),
        );

        let sky_view_lut = render_device.create_bind_group_layout(
            "sky_view_lut_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, storage_buffer::<GpuAtmosphere>(true)),
                    (1, sampler(SamplerBindingType::Filtering)),
                    (2, uniform_buffer::<ViewUniform>(true)),
                    (3, uniform_buffer::<GpuLights>(true)),
                    (4, uniform_buffer::<LutBasedAtmosphereSettings>(true)),
                    (5, texture_2d(TextureSampleType::Float { filterable: true })), // transmittance lut
                    (6, texture_2d(TextureSampleType::Float { filterable: true })), // multiscattering lut
                    (
                        // multiscattering lut storage texture
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
                    (0, storage_buffer::<GpuAtmosphere>(true)),
                    (3, sampler(SamplerBindingType::Filtering)),
                    (1, uniform_buffer::<ViewUniform>(true)),
                    (2, uniform_buffer::<GpuLights>(true)),
                    (4, uniform_buffer::<LutBasedAtmosphereSettings>(true)),
                    (4, texture_2d(TextureSampleType::Float { filterable: true })), // transmittance lut
                    (5, texture_2d(TextureSampleType::Float { filterable: true })), // multiscattering lut
                    (
                        // multiscattering lut storage texture
                        9,
                        texture_storage_3d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ),
        );

        let sampler = render_device.create_sampler(&SamplerDescriptor {
            label: Some("atmosphere_sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            lod_max_clamp: 0.0,
            ..Default::default()
        });

        Self {
            transmittance_lut,
            multiscattering_lut,
            sky_view_lut,
            aerial_view_lut,
            sampler,
        }
    }
}

impl FromWorld for RenderSkyLayout {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let render_sky = render_device.create_bind_group_layout(
            "render_sky_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::FRAGMENT,
                (
                    (0, storage_buffer::<GpuAtmosphere>(true)),
                    (1, sampler(SamplerBindingType::Filtering)),
                    (2, uniform_buffer::<ViewUniform>(true)),
                    (3, uniform_buffer::<GpuLights>(true)),
                    (4, uniform_buffer::<LutBasedAtmosphereSettings>(true)),
                    (5, texture_2d(TextureSampleType::Float { filterable: true })), // transmittance lut
                    (6, texture_2d(TextureSampleType::Float { filterable: true })), // multiscattering lut
                    (7, texture_2d(TextureSampleType::Float { filterable: true })), // sky view lut
                    (8, texture_3d(TextureSampleType::Float { filterable: true })), // aerial view lut
                    (9, texture_2d(TextureSampleType::Depth)), //view depth texture
                ),
            ),
        );

        let render_sky_msaa = render_device.create_bind_group_layout(
            "render_sky_msaa_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::FRAGMENT,
                (
                    (0, storage_buffer::<GpuAtmosphere>(true)),
                    (1, sampler(SamplerBindingType::Filtering)),
                    (2, uniform_buffer::<ViewUniform>(true)),
                    (3, uniform_buffer::<GpuLights>(true)),
                    (4, uniform_buffer::<LutBasedAtmosphereSettings>(true)),
                    (5, texture_2d(TextureSampleType::Float { filterable: true })), // transmittance lut
                    (6, texture_2d(TextureSampleType::Float { filterable: true })), // multiscattering lut
                    (7, texture_2d(TextureSampleType::Float { filterable: true })), // sky view lut
                    (8, texture_3d(TextureSampleType::Float { filterable: true })), // aerial view lut
                    (9, texture_2d_multisampled(TextureSampleType::Depth)), //view depth texture
                ),
            ),
        );

        let sampler = world.resource::<AtmosphereLayout>().sampler.clone();

        Self {
            render_sky,
            render_sky_msaa,
            sampler,
        }
    }
}

#[derive(Resource)]
pub(crate) struct AtmosphereLutPipelines {
    pub transmittance_lut: CachedComputePipelineId,
    pub multiscattering_lut: CachedComputePipelineId,
    pub sky_view_lut: CachedComputePipelineId,
    pub aerial_view_lut: CachedComputePipelineId,
}

impl FromWorld for AtmosphereLutPipelines {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let layouts = world.resource::<AtmosphereLayout>();

        let transmittance_lut = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("transmittance_lut_pipeline".into()),
            layout: vec![layouts.transmittance_lut.clone()],
            push_constant_ranges: vec![],
            shader: shaders::TRANSMITTANCE_LUT,
            shader_defs: vec![],
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        });

        let multiscattering_lut =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("multi_scattering_lut_pipeline".into()),
                layout: vec![layouts.multiscattering_lut.clone()],
                push_constant_ranges: vec![],
                shader: shaders::MULTISCATTERING_LUT,
                shader_defs: vec![],
                entry_point: "main".into(),
                zero_initialize_workgroup_memory: false,
            });

        let sky_view_lut = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("sky_view_lut_pipeline".into()),
            layout: vec![layouts.sky_view_lut.clone()],
            push_constant_ranges: vec![],
            shader: shaders::SKY_VIEW_LUT,
            shader_defs: vec![],
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        });

        let aerial_view_lut = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("aerial_view_lut_pipeline".into()),
            layout: vec![layouts.aerial_view_lut.clone()],
            push_constant_ranges: vec![],
            shader: shaders::AERIAL_VIEW_LUT,
            shader_defs: vec![],
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        });

        Self {
            transmittance_lut,
            multiscattering_lut,
            sky_view_lut,
            aerial_view_lut,
        }
    }
}

#[derive(Component)]
pub(crate) struct RenderSkyPipelineId(pub CachedRenderPipelineId);

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub(crate) struct RenderSkyPipelineKey {
    pub msaa_samples: u32,
    pub hdr: bool,
}

impl SpecializedRenderPipeline for RenderSkyLayout {
    type Key = RenderSkyPipelineKey;

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
            layout: vec![if key.msaa_samples == 1 {
                self.render_sky.clone()
            } else {
                self.render_sky_msaa.clone()
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
                shader: shaders::RENDER_SKY.clone(),
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

pub(super) fn queue_render_sky_pipelines(
    views: Query<(Entity, &Camera, &Msaa), With<ScatteringProfile>>,
    pipeline_cache: Res<PipelineCache>,
    layouts: Res<RenderSkyLayout>,
    mut specializer: ResMut<SpecializedRenderPipelines<RenderSkyLayout>>,
    mut commands: Commands,
) {
    for (entity, camera, msaa) in &views {
        let id = specializer.specialize(
            &pipeline_cache,
            &layouts,
            RenderSkyPipelineKey {
                msaa_samples: msaa.samples(),
                hdr: camera.hdr,
            },
        );
        commands.entity(entity).insert(RenderSkyPipelineId(id));
    }
}

#[derive(Component)]
pub struct AtmosphereCoreLuts {
    pub transmittance_lut: CachedTexture,
    pub multiscattering_lut: CachedTexture,
}

#[derive(Component)]
pub struct AtmosphereAuxLuts {
    pub sky_view_lut: CachedTexture,
    pub aerial_view_lut: CachedTexture,
}

pub(super) fn prepare_core_atmosphere_luts(
    atmospheres: Query<
        (Entity, &AtmosphereSettings),
        (With<Planet>, With<Atmosphere>, Changed<AtmosphereSettings>),
    >,
    render_device: Res<RenderDevice>,
    mut commands: Commands,
) {
    for (entity, settings) in &atmospheres {
        let transmittance_lut = render_device.create_texture(&TextureDescriptor {
            label: Some("transmittance_lut"),
            size: Extent3d {
                width: settings.transmittance_lut_size.x,
                height: settings.transmittance_lut_size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let transmittance_lut_view = transmittance_lut.create_view(&TextureViewDescriptor {
            label: Some("transmittance_lut_view"),
            ..Default::default()
        });

        let multiscattering_lut = render_device.create_texture(&TextureDescriptor {
            label: Some("multiscattering_lut"),
            size: Extent3d {
                width: settings.multiscattering_lut_size.x,
                height: settings.multiscattering_lut_size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let multiscattering_lut_view = multiscattering_lut.create_view(&TextureViewDescriptor {
            label: Some("multiscattering_lut_view"),
            ..Default::default()
        });

        commands.entity(entity).insert(AtmosphereCoreLuts {
            transmittance_lut: CachedTexture {
                texture: transmittance_lut,
                default_view: transmittance_lut_view,
            },
            multiscattering_lut: CachedTexture {
                texture: multiscattering_lut,
                default_view: multiscattering_lut_view,
            },
        });
    }
}

pub(super) fn prepare_atmosphere_textures(
    views: Query<(Entity, &AtmosphereMode), With<ScatteringProfile>>,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
    mut commands: Commands,
) {
    for (entity, lut_settings) in &views {
        let transmittance_lut = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("transmittance_lut"),
                size: Extent3d {
                    width: lut_settings.transmittance_lut_size.x,
                    height: lut_settings.transmittance_lut_size.y,
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

        let multiscattering_lut = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("multiscattering_lut"),
                size: Extent3d {
                    width: lut_settings.multiscattering_lut_size.x,
                    height: lut_settings.multiscattering_lut_size.y,
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

        let sky_view_lut = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("sky_view_lut"),
                size: Extent3d {
                    width: lut_settings.sky_view_lut_size.x,
                    height: lut_settings.sky_view_lut_size.y,
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
                    width: lut_settings.aerial_view_lut_size.x,
                    height: lut_settings.aerial_view_lut_size.y,
                    depth_or_array_layers: lut_settings.aerial_view_lut_size.z,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D3,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        commands.entity(entity).insert({
            AtmosphereTextures {
                transmittance_lut,
                multiscattering_lut,
                sky_view_lut,
                aerial_view_lut,
            }
        });
    }
}

#[derive(Resource, Default)]
pub struct AtmosphereTransforms {
    uniforms: DynamicUniformBuffer<AtmosphereTransform>,
}

impl AtmosphereTransforms {
    #[inline]
    pub fn uniforms(&self) -> &DynamicUniformBuffer<AtmosphereTransform> {
        &self.uniforms
    }
}

#[derive(ShaderType)]
pub struct AtmosphereTransform {
    world_from_atmosphere: Mat4,
    atmosphere_from_world: Mat4,
}

#[derive(Component)]
pub struct AtmosphereTransformsOffset {
    index: u32,
}

impl AtmosphereTransformsOffset {
    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }
}

pub(super) fn prepare_atmosphere_transforms(
    views: Query<(Entity, &ExtractedView), (With<ScatteringProfile>, With<Camera3d>)>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut atmo_uniforms: ResMut<AtmosphereTransforms>,
    mut commands: Commands,
) {
    let atmo_count = views.iter().len();
    let Some(mut writer) =
        atmo_uniforms
            .uniforms
            .get_writer(atmo_count, &render_device, &render_queue)
    else {
        return;
    };

    for (entity, view) in &views {
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

        commands.entity(entity).insert(AtmosphereTransformsOffset {
            index: writer.write(&AtmosphereTransform {
                world_from_atmosphere,
                atmosphere_from_world,
            }),
        });
    }
}

#[derive(Component)]
pub(crate) struct AtmosphereBindGroups {
    pub transmittance_lut: BindGroup,
    pub multiscattering_lut: BindGroup,
    pub sky_view_lut: BindGroup,
    pub aerial_view_lut: BindGroup,
    pub render_sky: BindGroup,
}

pub(super) fn prepare_atmosphere_bind_groups(
    views: Query<
        (Entity, &AtmosphereTextures, &ViewDepthTexture, &Msaa),
        (With<Camera3d>, With<ScatteringProfile>),
    >,
    render_device: Res<RenderDevice>,
    layouts: Res<AtmosphereLayout>,
    render_sky_layouts: Res<RenderSkyLayout>,
    samplers: Res<AtmosphereSamplers>,
    view_uniforms: Res<ViewUniforms>,
    lights_uniforms: Res<LightMeta>,
    atmosphere_transforms: Res<AtmosphereTransforms>,
    atmosphere_uniforms: Res<ComponentUniforms<ScatteringProfile>>,
    settings_uniforms: Res<ComponentUniforms<AtmosphereMode>>,

    mut commands: Commands,
) {
    if views.iter().len() == 0 {
        return;
    }

    let atmosphere_binding = atmosphere_uniforms
        .binding()
        .expect("Failed to prepare atmosphere bind groups. Atmosphere uniform buffer missing");

    let transforms_binding = atmosphere_transforms
        .uniforms()
        .binding()
        .expect("Failed to prepare atmosphere bind groups. Atmosphere transforms buffer missing");

    let settings_binding = settings_uniforms.binding().expect(
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

    for (entity, textures, view_depth_texture, msaa) in &views {
        let transmittance_lut = render_device.create_bind_group(
            "transmittance_lut_bind_group",
            &layouts.transmittance_lut,
            &BindGroupEntries::with_indices((
                (0, atmosphere_binding.clone()),
                (1, settings_binding.clone()),
                (13, &textures.transmittance_lut.default_view),
            )),
        );

        let multiscattering_lut = render_device.create_bind_group(
            "multiscattering_lut_bind_group",
            &layouts.multiscattering_lut,
            &BindGroupEntries::with_indices((
                (0, atmosphere_binding.clone()),
                (1, settings_binding.clone()),
                (5, &textures.transmittance_lut.default_view),
                (6, &samplers.transmittance_lut),
                (13, &textures.multiscattering_lut.default_view),
            )),
        );

        let sky_view_lut = render_device.create_bind_group(
            "sky_view_lut_bind_group",
            &layouts.sky_view_lut,
            &BindGroupEntries::with_indices((
                (0, atmosphere_binding.clone()),
                (1, settings_binding.clone()),
                (2, transforms_binding.clone()),
                (3, view_binding.clone()),
                (4, lights_binding.clone()),
                (5, &textures.transmittance_lut.default_view),
                (6, &samplers.transmittance_lut),
                (7, &textures.multiscattering_lut.default_view),
                (8, &samplers.multiscattering_lut),
                (13, &textures.sky_view_lut.default_view),
            )),
        );

        let aerial_view_lut = render_device.create_bind_group(
            "sky_view_lut_bind_group",
            &layouts.aerial_view_lut,
            &BindGroupEntries::with_indices((
                (0, atmosphere_binding.clone()),
                (1, settings_binding.clone()),
                (3, view_binding.clone()),
                (4, lights_binding.clone()),
                (5, &textures.transmittance_lut.default_view),
                (6, &samplers.transmittance_lut),
                (7, &textures.multiscattering_lut.default_view),
                (8, &samplers.multiscattering_lut),
                (13, &textures.aerial_view_lut.default_view),
            )),
        );

        let render_sky = render_device.create_bind_group(
            "render_sky_bind_group",
            if *msaa == Msaa::Off {
                &render_sky_layouts.render_sky
            } else {
                &render_sky_layouts.render_sky_msaa
            },
            &BindGroupEntries::with_indices((
                (0, atmosphere_binding.clone()),
                (1, settings_binding.clone()),
                (2, transforms_binding.clone()),
                (3, view_binding.clone()),
                (4, lights_binding.clone()),
                (5, &textures.transmittance_lut.default_view),
                (6, &samplers.transmittance_lut),
                (9, &textures.sky_view_lut.default_view),
                (10, &samplers.sky_view_lut),
                (11, &textures.aerial_view_lut.default_view),
                (12, &samplers.aerial_view_lut),
                (13, view_depth_texture.view()),
            )),
        );

        commands.entity(entity).insert(AtmosphereBindGroups {
            transmittance_lut,
            multiscattering_lut,
            sky_view_lut,
            aerial_view_lut,
            render_sky,
        });
    }
}
