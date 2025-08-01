use crate::{
    atmosphere::AtmosphereGlobalTransform, Bluenoise, GpuLights, LightMeta, MeshPipeline,
    MeshPipelineViewLayoutKey, MeshPipelineViewLayouts, ShadowSamplers, ViewShadowBindings,
};
use bevy_asset::{load_embedded_asset, Assets, Handle, RenderAssetUsages};
use bevy_core_pipeline::prepass::{
    DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass,
};
use bevy_core_pipeline::{core_3d::Camera3d, FullscreenShader};
use bevy_ecs::query::Has;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{With, Without},
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
    world::{FromWorld, World},
};
use bevy_image::{Image, ToExtents};
use bevy_light::{AtmosphereEnvironmentMapLight, GeneratedEnvironmentMapLight};
use bevy_math::{Mat4, UVec2, Vec3};
use bevy_render::{
    camera::Camera,
    extract_component::{ComponentUniforms, ExtractComponent},
    render_asset::RenderAssets,
    render_resource::{binding_types::*, *},
    renderer::{RenderDevice, RenderQueue},
    texture::{CachedTexture, GpuImage, TextureCache},
    view::{ExtractedView, Msaa, ViewDepthTexture, ViewUniform, ViewUniforms},
};
use bevy_utils::default;

use super::{Atmosphere, AtmosphereSettings};

#[derive(Resource)]
pub(crate) struct AtmosphereBindGroupLayouts {
    pub transmittance_lut: BindGroupLayout,
    pub multiscattering_lut: BindGroupLayout,
    pub sky_view_lut: BindGroupLayout,
    pub aerial_view_lut: BindGroupLayout,
    pub environment: BindGroupLayout,
}

#[derive(Resource)]
pub(crate) struct RenderSkyPipeline {
    pub render_sky: BindGroupLayout,
    pub render_sky_msaa: BindGroupLayout,
    pub fullscreen_shader: FullscreenShader,
    pub fragment_shader: Handle<Shader>,
    pub mesh_view_layouts: MeshPipelineViewLayouts,
}

impl FromWorld for AtmosphereBindGroupLayouts {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let transmittance_lut = render_device.create_bind_group_layout(
            "transmittance_lut_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, uniform_buffer::<Atmosphere>(true)),
                    (1, uniform_buffer::<AtmosphereSettings>(true)),
                    (
                        // transmittance lut storage texture
                        13,
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
                    (0, uniform_buffer::<Atmosphere>(true)),
                    (1, uniform_buffer::<AtmosphereSettings>(true)),
                    (5, texture_2d(TextureSampleType::Float { filterable: true })), //transmittance lut and sampler
                    (6, sampler(SamplerBindingType::Filtering)),
                    (
                        //multiscattering lut storage texture
                        13,
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
                    (0, uniform_buffer::<Atmosphere>(true)),
                    (1, uniform_buffer::<AtmosphereSettings>(true)),
                    (2, uniform_buffer::<AtmosphereTransform>(true)),
                    (3, uniform_buffer::<ViewUniform>(true)),
                    (4, uniform_buffer::<GpuLights>(true)),
                    (5, texture_2d(TextureSampleType::Float { filterable: true })), //transmittance lut and sampler
                    (6, sampler(SamplerBindingType::Filtering)),
                    (7, texture_2d(TextureSampleType::Float { filterable: true })), //multiscattering lut and sampler
                    (8, sampler(SamplerBindingType::Filtering)),
                    (
                        13,
                        texture_storage_2d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                    (14, texture_2d_array(TextureSampleType::Depth)), // directional shadow texture
                    (15, sampler(SamplerBindingType::Comparison)),
                    (
                        16,
                        texture_2d_array(TextureSampleType::Float { filterable: true }),
                    ), // blue noise texture
                    (17, sampler(SamplerBindingType::Filtering)),
                ),
            ),
        );

        let aerial_view_lut = render_device.create_bind_group_layout(
            "aerial_view_lut_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, uniform_buffer::<Atmosphere>(true)),
                    (1, uniform_buffer::<AtmosphereSettings>(true)),
                    (2, uniform_buffer::<AtmosphereTransform>(true)),
                    (3, uniform_buffer::<ViewUniform>(true)),
                    (4, uniform_buffer::<GpuLights>(true)),
                    (5, texture_2d(TextureSampleType::Float { filterable: true })), //transmittance lut and sampler
                    (6, sampler(SamplerBindingType::Filtering)),
                    (7, texture_2d(TextureSampleType::Float { filterable: true })), //multiscattering lut and sampler
                    (8, sampler(SamplerBindingType::Filtering)),
                    (
                        13,
                        texture_storage_3d(
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ), //Aerial view lut storage texture
                    ),
                    (14, texture_2d_array(TextureSampleType::Depth)), // directional shadow texture
                    (15, sampler(SamplerBindingType::Comparison)),
                    (
                        16,
                        texture_2d_array(TextureSampleType::Float { filterable: true }),
                    ), // blue noise texture
                    (17, sampler(SamplerBindingType::Filtering)),
                ),
            ),
        );

        let environment = render_device.create_bind_group_layout(
            "environment_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, uniform_buffer::<Atmosphere>(true)),
                    (1, uniform_buffer::<AtmosphereSettings>(true)),
                    (2, uniform_buffer::<AtmosphereTransform>(true)),
                    (3, uniform_buffer::<ViewUniform>(true)),
                    (4, uniform_buffer::<GpuLights>(true)),
                    (5, texture_2d(TextureSampleType::Float { filterable: true })), //transmittance lut and sampler
                    (6, sampler(SamplerBindingType::Filtering)),
                    (7, texture_2d(TextureSampleType::Float { filterable: true })), //multiscattering lut and sampler
                    (8, sampler(SamplerBindingType::Filtering)),
                    (9, texture_2d(TextureSampleType::Float { filterable: true })), //sky view lut and sampler
                    (10, sampler(SamplerBindingType::Filtering)),
                    (
                        13,
                        texture_storage_2d_array(
                            // output 2D array texture
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                    (14, texture_2d_array(TextureSampleType::Depth)), // directional shadow texture
                    (15, sampler(SamplerBindingType::Comparison)),
                    (
                        16,
                        texture_2d_array(TextureSampleType::Float { filterable: true }),
                    ), // blue noise texture
                    (17, sampler(SamplerBindingType::Filtering)),
                    (18, uniform_buffer::<Mat4>(false)),
                ),
            ),
        );

        Self {
            transmittance_lut,
            multiscattering_lut,
            sky_view_lut,
            aerial_view_lut,
            environment,
        }
    }
}

impl FromWorld for RenderSkyPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let render_sky = render_device.create_bind_group_layout(
            "render_sky_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::FRAGMENT,
                (
                    (0, uniform_buffer::<Atmosphere>(true)),
                    (1, uniform_buffer::<AtmosphereSettings>(true)),
                    (2, uniform_buffer::<AtmosphereTransform>(true)),
                    (3, uniform_buffer::<ViewUniform>(true)),
                    (4, uniform_buffer::<GpuLights>(true)),
                    (5, texture_2d(TextureSampleType::Float { filterable: true })), //transmittance lut and sampler
                    (6, sampler(SamplerBindingType::Filtering)),
                    (7, texture_2d(TextureSampleType::Float { filterable: true })), //multiscattering lut and sampler
                    (8, sampler(SamplerBindingType::Filtering)),
                    (9, texture_2d(TextureSampleType::Float { filterable: true })), //sky view lut and sampler
                    (10, sampler(SamplerBindingType::Filtering)),
                    (
                        11,
                        texture_3d(TextureSampleType::Float { filterable: true }),
                    ), // aerial view lut and sampler
                    (12, sampler(SamplerBindingType::Filtering)),
                    (13, texture_2d(TextureSampleType::Depth)), //view depth texture
                    (14, texture_2d_array(TextureSampleType::Depth)), // directional shadow texture
                    (15, sampler(SamplerBindingType::Comparison)),
                    (
                        16,
                        texture_2d_array(TextureSampleType::Float { filterable: true }),
                    ), // blue noise texture
                    (17, sampler(SamplerBindingType::Filtering)),
                ),
            ),
        );

        let render_sky_msaa = render_device.create_bind_group_layout(
            "render_sky_msaa_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::FRAGMENT,
                (
                    (0, uniform_buffer::<Atmosphere>(true)),
                    (1, uniform_buffer::<AtmosphereSettings>(true)),
                    (2, uniform_buffer::<AtmosphereTransform>(true)),
                    (3, uniform_buffer::<ViewUniform>(true)),
                    (4, uniform_buffer::<GpuLights>(true)),
                    (5, texture_2d(TextureSampleType::Float { filterable: true })), //transmittance lut and sampler
                    (6, sampler(SamplerBindingType::Filtering)),
                    (7, texture_2d(TextureSampleType::Float { filterable: true })), //multiscattering lut and sampler
                    (8, sampler(SamplerBindingType::Filtering)),
                    (9, texture_2d(TextureSampleType::Float { filterable: true })), //sky view lut and sampler
                    (10, sampler(SamplerBindingType::Filtering)),
                    (
                        11,
                        texture_3d(TextureSampleType::Float { filterable: true }),
                    ), // aerial view lut and sampler
                    (12, sampler(SamplerBindingType::Filtering)),
                    (13, texture_2d_multisampled(TextureSampleType::Depth)), //view depth texture
                    (14, texture_2d_array(TextureSampleType::Depth)), // directional shadow texture
                    (15, sampler(SamplerBindingType::Comparison)),
                    (
                        16,
                        texture_2d_array(TextureSampleType::Float { filterable: true }),
                    ), // blue noise texture
                    (17, sampler(SamplerBindingType::Filtering)),
                ),
            ),
        );

        Self {
            render_sky,
            render_sky_msaa,
            fullscreen_shader: world.resource::<FullscreenShader>().clone(),
            fragment_shader: load_embedded_asset!(world, "render_sky.wgsl"),
            mesh_view_layouts: world.resource::<MeshPipelineViewLayouts>().clone(),
        }
    }
}

#[derive(Resource)]
pub struct AtmosphereSamplers {
    pub transmittance_lut: Sampler,
    pub multiscattering_lut: Sampler,
    pub sky_view_lut: Sampler,
    pub aerial_view_lut: Sampler,
    pub blue_noise: Sampler,
}

impl FromWorld for AtmosphereSamplers {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let base_sampler = SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        };

        let transmittance_lut = render_device.create_sampler(&SamplerDescriptor {
            label: Some("transmittance_lut_sampler"),
            ..base_sampler
        });

        let multiscattering_lut = render_device.create_sampler(&SamplerDescriptor {
            label: Some("multiscattering_lut_sampler"),
            ..base_sampler
        });

        let sky_view_lut = render_device.create_sampler(&SamplerDescriptor {
            label: Some("sky_view_lut_sampler"),
            address_mode_u: AddressMode::Repeat,
            ..base_sampler
        });

        let aerial_view_lut = render_device.create_sampler(&SamplerDescriptor {
            label: Some("aerial_view_lut_sampler"),
            ..base_sampler
        });

        let blue_noise = render_device.create_sampler(&SamplerDescriptor {
            label: Some("blue_noise_sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            ..Default::default()
        });

        Self {
            transmittance_lut,
            multiscattering_lut,
            sky_view_lut,
            aerial_view_lut,
            blue_noise,
        }
    }
}

#[derive(Resource)]
pub(crate) struct AtmospherePipelines {
    pub transmittance_lut: CachedComputePipelineId,
    pub multiscattering_lut: CachedComputePipelineId,
    pub sky_view_lut: CachedComputePipelineId,
    pub aerial_view_lut: CachedComputePipelineId,
    pub environment: CachedComputePipelineId,
}

impl FromWorld for AtmospherePipelines {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let layouts = world.resource::<AtmosphereBindGroupLayouts>();

        let transmittance_lut = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("transmittance_lut_pipeline".into()),
            layout: vec![layouts.transmittance_lut.clone()],
            shader: load_embedded_asset!(world, "transmittance_lut.wgsl"),
            ..default()
        });

        let multiscattering_lut =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("multi_scattering_lut_pipeline".into()),
                layout: vec![layouts.multiscattering_lut.clone()],
                shader: load_embedded_asset!(world, "multiscattering_lut.wgsl"),
                ..default()
            });

        let sky_view_lut = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("sky_view_lut_pipeline".into()),
            layout: vec![layouts.sky_view_lut.clone()],
            shader: load_embedded_asset!(world, "sky_view_lut.wgsl"),
            ..default()
        });

        let aerial_view_lut = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("aerial_view_lut_pipeline".into()),
            layout: vec![layouts.aerial_view_lut.clone()],
            shader: load_embedded_asset!(world, "aerial_view_lut.wgsl"),
            ..default()
        });

        let environment = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("environment_pipeline".into()),
            layout: vec![layouts.environment.clone()],
            shader: load_embedded_asset!(world, "environment.wgsl"),
            ..default()
        });

        Self {
            transmittance_lut,
            multiscattering_lut,
            sky_view_lut,
            aerial_view_lut,
            environment,
        }
    }
}

#[derive(Component)]
pub(crate) struct RenderSkyPipelineId(pub CachedRenderPipelineId);

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
pub(crate) struct RenderSkyPipelineKey {
    pub msaa_samples: u32,
    pub dual_source_blending: bool,
    pub mesh_pipeline_view_key: MeshPipelineViewLayoutKey,
}

impl SpecializedRenderPipeline for RenderSkyPipeline {
    type Key = RenderSkyPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut shader_defs = Vec::new();

        if key.msaa_samples > 1 {
            shader_defs.push("MULTISAMPLED".into());
        }
        if key.dual_source_blending {
            shader_defs.push("DUAL_SOURCE_BLENDING".into());
        }

        let dst_factor = if key.dual_source_blending {
            BlendFactor::Src1
        } else {
            BlendFactor::SrcAlpha
        };

        let render_sky_bind_group_layout = if key.msaa_samples == 1 {
            self.render_sky.clone()
        } else {
            self.render_sky_msaa.clone()
        };

        let layout = self
            .mesh_view_layouts
            .get_view_layout(key.mesh_pipeline_view_key);
        let layout = vec![
            layout.main_layout.clone(),
            render_sky_bind_group_layout.clone(),
        ];

        RenderPipelineDescriptor {
            label: Some(format!("render_sky_pipeline_{}", key.msaa_samples).into()),
            layout,
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.fragment_shader.clone(),
                shader_defs,
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba16Float,
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor,
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
                ..default()
            }),
            multisample: MultisampleState {
                count: key.msaa_samples,
                ..default()
            },
            ..default()
        }
    }
}

pub(super) fn queue_render_sky_pipelines(
    views: Query<
        (
            Entity,
            &Msaa,
            Has<NormalPrepass>,
            Has<DepthPrepass>,
            Has<MotionVectorPrepass>,
            Has<DeferredPrepass>,
        ),
        (With<Camera>, With<Atmosphere>),
    >,
    pipeline_cache: Res<PipelineCache>,
    layouts: Res<RenderSkyPipeline>,
    mut specializer: ResMut<SpecializedRenderPipelines<RenderSkyPipeline>>,
    render_device: Res<RenderDevice>,
    _mesh_pipeline: Res<MeshPipeline>,
    mut commands: Commands,
) {
    for (entity, msaa, normal_prepass, depth_prepass, motion_vector_prepass, deferred_prepass) in
        &views
    {
        let mut mesh_pipeline_view_key = MeshPipelineViewLayoutKey::from(*msaa);
        mesh_pipeline_view_key.set(MeshPipelineViewLayoutKey::NORMAL_PREPASS, normal_prepass);
        mesh_pipeline_view_key.set(MeshPipelineViewLayoutKey::DEPTH_PREPASS, depth_prepass);
        mesh_pipeline_view_key.set(
            MeshPipelineViewLayoutKey::MOTION_VECTOR_PREPASS,
            motion_vector_prepass,
        );
        mesh_pipeline_view_key.set(
            MeshPipelineViewLayoutKey::DEFERRED_PREPASS,
            deferred_prepass,
        );
        mesh_pipeline_view_key.set(MeshPipelineViewLayoutKey::ATMOSPHERE, true);

        let id = specializer.specialize(
            &pipeline_cache,
            &layouts,
            RenderSkyPipelineKey {
                msaa_samples: msaa.samples(),
                dual_source_blending: render_device
                    .features()
                    .contains(WgpuFeatures::DUAL_SOURCE_BLENDING),
                mesh_pipeline_view_key,
            },
        );
        commands.entity(entity).insert(RenderSkyPipelineId(id));
    }
}

#[derive(Component)]
pub struct AtmosphereTextures {
    pub transmittance_lut: CachedTexture,
    pub multiscattering_lut: CachedTexture,
    pub sky_view_lut: CachedTexture,
    pub aerial_view_lut: CachedTexture,
}

// Render world representation of an environment map light for the atmosphere
#[derive(Component, ExtractComponent, Clone)]
pub struct AtmosphereEnvironmentMap {
    pub environment_map: Handle<Image>,
    pub size: UVec2,
}

#[derive(Component)]
pub struct AtmosphereProbeTextures {
    pub environment: TextureView,
    pub transmittance_lut: CachedTexture,
    pub multiscattering_lut: CachedTexture,
    pub sky_view_lut: CachedTexture,
}

pub(super) fn prepare_view_textures(
    views: Query<(Entity, &AtmosphereSettings), With<Atmosphere>>,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
    mut commands: Commands,
) {
    for (entity, settings) in &views {
        let transmittance_lut = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("transmittance_lut"),
                size: settings.transmittance_lut_size.to_extents(),
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
                size: settings.multiscattering_lut_size.to_extents(),
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
                size: settings.sky_view_lut_size.to_extents(),
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
                size: settings.aerial_view_lut_size.to_extents(),
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

pub(super) fn prepare_probe_textures(
    view_textures: Query<&AtmosphereTextures, With<Atmosphere>>,
    probes: Query<
        (Entity, &AtmosphereEnvironmentMap),
        (
            With<AtmosphereEnvironmentMap>,
            Without<AtmosphereProbeTextures>,
        ),
    >,
    gpu_images: Res<RenderAssets<GpuImage>>,
    mut commands: Commands,
) {
    for (probe, render_env_map) in &probes {
        let environment = gpu_images.get(&render_env_map.environment_map).unwrap();
        // create a cube view
        let environment_view = environment.texture.create_view(&TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2Array),
            ..Default::default()
        });
        // Get the first view entity's textures to borrow
        if let Some(view_textures) = view_textures.iter().next() {
            commands.entity(probe).insert(AtmosphereProbeTextures {
                environment: environment_view,
                transmittance_lut: view_textures.transmittance_lut.clone(),
                multiscattering_lut: view_textures.multiscattering_lut.clone(),
                sky_view_lut: view_textures.sky_view_lut.clone(),
            });
        }
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
#[repr(C, align(16))]
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
    views: Query<(Entity, &ExtractedView), (With<Atmosphere>, With<Camera3d>)>,
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
        let world_from_view = view.world_from_view.to_matrix();
        let camera_pos = world_from_view.w_axis.truncate();

        // Keep a fixed atmosphere space orientation (Y up)
        let atmo_y = Vec3::Y;
        let camera_z = world_from_view.z_axis.truncate();

        // Project camera's forward onto the horizontal plane
        let forward = (camera_z - camera_z.dot(atmo_y) * atmo_y).normalize();
        let atmo_z = forward;
        let atmo_x = atmo_y.cross(atmo_z).normalize();

        // Create transform with fixed orientation but offset position
        let world_from_atmosphere = Mat4::from_cols(
            atmo_x.extend(0.0),
            atmo_y.extend(0.0),
            atmo_z.extend(0.0),
            camera_pos.extend(1.0),
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

#[derive(Component)]
pub(crate) struct AtmosphereProbeBindGroups {
    pub environment: BindGroup,
}

pub(super) fn prepare_atmosphere_bind_groups(
    views: Query<
        (
            Entity,
            &AtmosphereTextures,
            &ViewDepthTexture,
            &ViewShadowBindings,
            &Msaa,
        ),
        (With<Camera3d>, With<Atmosphere>),
    >,
    probes: Query<
        (Entity, &AtmosphereProbeTextures, &AtmosphereGlobalTransform),
        With<AtmosphereEnvironmentMap>,
    >,
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    layouts: Res<AtmosphereBindGroupLayouts>,
    render_sky_layouts: Res<RenderSkyPipeline>,
    samplers: Res<AtmosphereSamplers>,
    view_uniforms: Res<ViewUniforms>,
    lights_uniforms: Res<LightMeta>,
    atmosphere_transforms: Res<AtmosphereTransforms>,
    atmosphere_uniforms: Res<ComponentUniforms<Atmosphere>>,
    settings_uniforms: Res<ComponentUniforms<AtmosphereSettings>>,
    shadow_samplers: Res<ShadowSamplers>,
    images: Res<RenderAssets<GpuImage>>,
    bluenoise: Res<Bluenoise>,

    mut commands: Commands,
) {
    if views.iter().len() == 0 {
        return;
    }
    let Some(bluenoise_image) = images.get(&bluenoise.texture) else {
        return;
    };

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

    // Get shadow bindings from first view
    let shadow_bindings = views.iter().next().map(|(_, _, _, bindings, _)| bindings);

    // get the gpu image for the bluenoise texture
    let bluenoise_texture_view = bluenoise_image.texture.create_view(&TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..Default::default()
    });

    for (entity, textures, view_depth_texture, shadow_bindings, msaa) in &views {
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
                (14, &shadow_bindings.directional_light_depth_texture_view),
                (15, &shadow_samplers.directional_light_comparison_sampler),
                (16, &bluenoise_texture_view),
                (17, &samplers.blue_noise),
            )),
        );

        let aerial_view_lut = render_device.create_bind_group(
            "aerial_view_lut_bind_group",
            &layouts.aerial_view_lut,
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
                (13, &textures.aerial_view_lut.default_view),
                (14, &shadow_bindings.directional_light_depth_texture_view),
                (15, &shadow_samplers.directional_light_comparison_sampler),
                (16, &bluenoise_texture_view),
                (17, &samplers.blue_noise),
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
                (7, &textures.multiscattering_lut.default_view),
                (8, &samplers.multiscattering_lut),
                (9, &textures.sky_view_lut.default_view),
                (10, &samplers.sky_view_lut),
                (11, &textures.aerial_view_lut.default_view),
                (12, &samplers.aerial_view_lut),
                (13, view_depth_texture.view()),
                (14, &shadow_bindings.directional_light_depth_texture_view),
                (15, &shadow_samplers.directional_light_comparison_sampler),
                (16, &bluenoise_texture_view),
                (17, &samplers.blue_noise),
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

    for (entity, textures, transform) in &probes {
        // Skip if no shadow bindings are available
        let Some(shadow_bindings) = shadow_bindings else {
            continue;
        };

        let transform_matrix = transform.0.to_matrix();
        let mut probe_transform_data = UniformBuffer::from(transform_matrix);
        probe_transform_data.write_buffer(&render_device, &queue);

        let environment = render_device.create_bind_group(
            "environment_bind_group",
            &layouts.environment,
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
                (9, &textures.sky_view_lut.default_view),
                (10, &samplers.sky_view_lut),
                (13, &textures.environment),
                (14, &shadow_bindings.directional_light_depth_texture_view),
                (15, &shadow_samplers.directional_light_comparison_sampler),
                (16, &bluenoise_texture_view),
                (17, &samplers.blue_noise),
                (18, &probe_transform_data),
            )),
        );

        commands
            .entity(entity)
            .insert(AtmosphereProbeBindGroups { environment });
    }
}

#[derive(ShaderType)]
#[repr(C)]
pub(crate) struct AtmosphereData {
    pub atmosphere: Atmosphere,
    pub settings: AtmosphereSettings,
}

#[derive(Resource)]
pub struct AtmosphereBuffer {
    pub(crate) buffer: StorageBuffer<AtmosphereData>,
}

impl FromWorld for AtmosphereBuffer {
    fn from_world(world: &mut World) -> Self {
        let data = world
            .query_filtered::<(&Atmosphere, &AtmosphereSettings), With<Camera3d>>()
            .iter(world)
            .next()
            .map_or_else(
                || AtmosphereData {
                    atmosphere: Atmosphere::default(),
                    settings: AtmosphereSettings::default(),
                },
                |(atmosphere, settings)| AtmosphereData {
                    atmosphere: atmosphere.clone(),
                    settings: settings.clone(),
                },
            );

        Self {
            buffer: StorageBuffer::from(data),
        }
    }
}

pub(crate) fn prepare_atmosphere_buffer(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    atmosphere_entity: Query<(&Atmosphere, &AtmosphereSettings), With<Camera3d>>,
    mut atmosphere_buffer: ResMut<AtmosphereBuffer>,
) {
    let Ok((atmosphere, settings)) = atmosphere_entity.single() else {
        return;
    };

    atmosphere_buffer.buffer.set(AtmosphereData {
        atmosphere: atmosphere.clone(),
        settings: settings.clone(),
    });
    atmosphere_buffer.buffer.write_buffer(&device, &queue);
}

pub fn prepare_atmosphere_probe_components(
    probes: Query<(Entity, &AtmosphereEnvironmentMapLight), (Without<AtmosphereEnvironmentMap>,)>,
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
) {
    for (entity, env_map_light) in &probes {
        // Create a cubemap image in the main world that we can reference
        let mut environment_image = Image::new_fill(
            Extent3d {
                width: env_map_light.size.x,
                height: env_map_light.size.y,
                depth_or_array_layers: 6,
            },
            TextureDimension::D2,
            &[0; 8],
            TextureFormat::Rgba16Float,
            RenderAssetUsages::all(),
        );

        environment_image.texture_view_descriptor = Some(TextureViewDescriptor {
            dimension: Some(TextureViewDimension::Cube),
            ..Default::default()
        });

        environment_image.texture_descriptor.usage = TextureUsages::TEXTURE_BINDING
            | TextureUsages::STORAGE_BINDING
            | TextureUsages::COPY_SRC;

        // Add the image to assets to get a handle
        let environment_handle = images.add(environment_image);

        commands.entity(entity).insert(AtmosphereEnvironmentMap {
            environment_map: environment_handle.clone(),
            size: env_map_light.size,
        });

        commands
            .entity(entity)
            .insert(GeneratedEnvironmentMapLight {
                environment_map: environment_handle,
                intensity: env_map_light.intensity,
                rotation: env_map_light.rotation,
                affects_lightmapped_mesh_diffuse: env_map_light.affects_lightmapped_mesh_diffuse,
            });
    }
}
