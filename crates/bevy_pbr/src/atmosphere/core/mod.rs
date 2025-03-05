use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, Assets};
use bevy_color::ColorToComponents;
use bevy_core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::With,
    removal_detection::RemovedComponents,
    resource::Resource,
    schedule::IntoSystemConfigs,
    system::{Commands, Query, Res, ResMut},
    world::{FromWorld, World},
};
use bevy_math::{UVec2, Vec3};
use bevy_reflect::Reflect;
use bevy_render::{
    graph::CameraDriverLabel,
    render_graph::RenderGraphApp,
    render_resource::{
        binding_types::{sampler, storage_buffer, texture_2d, texture_storage_2d, uniform_buffer},
        BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
        CachedComputePipelineId, ComputePipelineDescriptor, DynamicStorageBuffer,
        DynamicUniformBuffer, Extent3d, FilterMode, PipelineCache, Sampler, SamplerBindingType,
        SamplerDescriptor, Shader, ShaderStages, ShaderType, StorageTextureAccess,
        TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
    },
    renderer::RenderDevice,
    sync_world::RenderEntity,
    texture::{CachedTexture, TextureCache},
    Extract, ExtractSchedule, Render, RenderApp, RenderSet,
};

use super::{Atmosphere, Planet, ScatteringProfile};

mod node;

mod shaders {
    use bevy_asset::{weak_handle, Handle};
    use bevy_render::render_resource::Shader;

    pub const TRANSMITTANCE_LUT: Handle<Shader> =
        weak_handle!("a4187282-8cb1-42d3-889c-cbbfb6044183");
    pub const MULTISCATTERING_LUT: Handle<Shader> =
        weak_handle!("bde3a71a-73e9-49fe-a379-a81940c67a1e");
}

pub struct CoreAtmospherePlugin;

impl Plugin for CoreAtmospherePlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            shaders::TRANSMITTANCE_LUT,
            "transmittance_lut.wgsl",
            Shader::from_wgsl
        );

        load_internal_asset!(
            app,
            shaders::MULTISCATTERING_LUT,
            "multiscattering_lut.wgsl",
            Shader::from_wgsl
        );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<Layout>()
            .init_resource::<Pipelines>()
            .add_systems(ExtractSchedule, extract_atmospheres)
            .add_systems(
                Render,
                (
                    prepare_luts.in_set(RenderSet::PrepareAssets),
                    prepare_bind_groups.in_set(RenderSet::PrepareBindGroups),
                ),
            )
            .add_render_graph_node::<node::LutsNode>(Core3d, node::LutsLabel)
            .add_render_graph_edges(Core3d, (node::LutsLabel, CameraDriverLabel));
    }
}

#[derive(Clone, Reflect, Component, ShaderType)]
pub struct Settings {
    /// The size of the transmittance LUT
    pub transmittance_lut_size: UVec2,

    /// The size of the multiscattering LUT
    pub multiscattering_lut_size: UVec2,

    /// The number of points to sample along each ray when
    /// computing the transmittance LUT
    pub transmittance_lut_samples: u32,

    /// The number of rays to sample when computing each
    /// pixel of the multiscattering LUT
    pub multiscattering_lut_dirs: u32,

    /// The number of points to sample along each ray when
    /// computing the multiscattering LUT.
    pub multiscattering_lut_samples: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            transmittance_lut_size: UVec2::new(256, 128),
            multiscattering_lut_size: UVec2::new(32, 32),
            transmittance_lut_samples: 40,
            multiscattering_lut_dirs: 64,
            multiscattering_lut_samples: 20,
        }
    }
}

#[derive(ShaderType)]
struct GpuPlanet {
    ground_albedo: Vec3,
    lower_radius: f32,
    lower_radius_sq: f32,
    upper_radius: f32,
    upper_radius_sq: f32,
    space_altitude: f32,
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

#[derive(ShaderType, Component)]
struct GpuAtmosphere {
    profile: ScatteringProfile,
    planet: GpuPlanet,
}

#[derive(Resource)]
pub struct Uniforms {
    atmospheres: DynamicStorageBuffer<GpuAtmosphere>,
    settings: DynamicUniformBuffer<Settings>,
}

#[derive(Resource)]
pub struct UniformIndex(pub u32);

pub fn extract_atmospheres(
    atmospheres: Extract<Query<(RenderEntity, &Atmosphere, &Settings, &Planet)>>,
    scattering_profiles: Extract<Assets<ScatteringProfile>>,
    mut uniforms: ResMut<Uniforms>,
    mut commands: Commands,
) {
    uniforms.atmospheres.clear();
    uniforms.settings.clear();

    for (render_entity, atmosphere, settings, planet) in &atmospheres {
        let profile = scattering_profiles.get(atmosphere.0.id());
        let gpu_atmosphere = GpuAtmosphere {
            profile,
            planet: planet.clone().into(),
        };

        //since we're extracting both at once, we can assume their indices are the same
        let atmospheres_index = uniforms.atmospheres.push(gpu_atmosphere);
        let settings_index = uniforms.settings.push(settings);
        debug_assert_eq(atmospheres_index, settings_index);

        commands
            .entity(render_entity)
            .insert((settings.clone(), UniformIndex(atmospheres_index)));
    }
}

#[derive(Resource)]
pub struct Layout {
    pub transmittance_lut: BindGroupLayout,
    pub multiscattering_lut: BindGroupLayout,
    pub sampler: Sampler,
}

impl FromWorld for Layout {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let transmittance_lut = render_device.create_bind_group_layout(
            "transmittance_lut_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (0, storage_buffer::<GpuAtmosphere>(true)),
                    (4, uniform_buffer::<Settings>(true)),
                    (
                        // transmittance lut storage texture
                        10,
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
                    (4, uniform_buffer::<Settings>(true)),
                    (6, texture_2d(TextureSampleType::Float { filterable: true })), // transmittance lut
                    (
                        // multiscattering lut storage texture
                        10,
                        texture_storage_2d(
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
            sampler,
        }
    }
}

#[derive(Resource)]
pub struct Pipelines {
    pub transmittance_lut: CachedComputePipelineId,
    pub multiscattering_lut: CachedComputePipelineId,
}

impl FromWorld for Pipelines {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let layout = world.resource::<Layout>();

        let transmittance_lut = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("transmittance_lut_pipeline".into()),
            layout: vec![layout.transmittance_lut.clone()],
            push_constant_ranges: vec![],
            shader: shaders::TRANSMITTANCE_LUT,
            shader_defs: vec![],
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        });

        let multiscattering_lut =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("multi_scattering_lut_pipeline".into()),
                layout: vec![layout.multiscattering_lut.clone()],
                push_constant_ranges: vec![],
                shader: shaders::MULTISCATTERING_LUT,
                shader_defs: vec![],
                entry_point: "main".into(),
                zero_initialize_workgroup_memory: false,
            });

        Self {
            transmittance_lut,
            multiscattering_lut,
        }
    }
}

#[derive(Component)]
pub struct Luts {
    pub transmittance_lut: CachedTexture,
    pub multiscattering_lut: CachedTexture,
}

fn prepare_luts(
    atmospheres: Query<(Entity, &Settings), (With<Planet>, With<Atmosphere>)>,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
    mut commands: Commands,
) {
    for (entity, settings) in &atmospheres {
        let transmittance_lut = texture_cache.get(
            &render_device,
            TextureDescriptor {
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
            },
        );

        let multiscattering_lut = texture_cache.get(
            &render_device,
            TextureDescriptor {
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
            },
        );

        commands.entity(entity).insert(Luts {
            transmittance_lut,
            multiscattering_lut,
        });
    }
}

#[derive(Component)]
pub struct BindGroups {
    pub transmittance_lut: BindGroup,
    pub multiscattering_lut: BindGroup,
}

fn prepare_bind_groups(
    atmospheres: Query<(Entity, &Luts, &GpuAtmosphere)>,
    render_device: Res<RenderDevice>,
    layout: Res<Layout>,
    mut commands: Commands,
) {
    for (entity, core_luts, gpu_atmosphere) in &atmospheres {
        let transmittance_lut = render_device.create_bind_group(
            "transmittance_lut_bind_group",
            &layout.transmittance_lut,
            &BindGroupEntries::with_indices((
                (0, todo!()), //TODO: MAKE ATMOSPHERE STORAGE BUFFER
                (4, todo!()), //TODO: MAKE CORE SETTINGS UNIFORM BUFFER
                (10, &core_luts.transmittance_lut.default_view),
            )),
        );

        let multiscattering_lut = render_device.create_bind_group(
            "multiscattering_lut_bind_group",
            &layout.multiscattering_lut,
            &BindGroupEntries::with_indices((
                (0, todo!()), //TODO: MAKE ATMOSPHERE STORAGE BUFFER
                (1, &layout.sampler),
                (4, todo!()), //TODO: MAKE CORE SETTINGS UNIFORM BUFFER
                (6, &core_luts.transmittance_lut.default_view),
                (10, &core_luts.multiscattering_lut.default_view),
            )),
        );

        commands.entity(entity).insert(BindGroups {
            transmittance_lut,
            multiscattering_lut,
        });
    }
}
