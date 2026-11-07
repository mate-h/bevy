//! Like [`EnvironmentMapLight`], but filtered in realtime from a cubemap.
//!
//! An environment map needs to be processed to be able to support uses beyond a simple skybox,
//! such as reflections, and ambient light contribution.
//! This process is called filtering, and can either be done ahead of time (prefiltering), or
//! in realtime, although at a reduced quality. Prefiltering is preferred, but not always possible:
//! sometimes you only gain access to an environment map at runtime, for whatever reason.
//! Typically this is from realtime reflection probes, but can also be from other sources.
//!
//! In any case, Bevy supports both modes of filtering.
//! This module provides realtime filtering via [`bevy_light::GeneratedEnvironmentMapLight`].
//! For prefiltered environment maps, see [`bevy_light::EnvironmentMapLight`].
//! These components are intended to be added to a camera.
use bevy_app::{App, Plugin, Update};
use bevy_asset::{embedded_asset, load_embedded_asset, AssetServer, Assets, RenderAssetUsages};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::Without,
    resource::Resource,
    schedule::IntoScheduleConfigs,
    system::{Commands, Query, Res, ResMut},
};
use bevy_image::Image;
use bevy_math::{Quat, Vec4};
use bevy_render::{
    diagnostic::RecordDiagnostics,
    render_asset::RenderAssets,
    render_resource::{
        binding_types::*, AddressMode, BindGroup, BindGroupEntries, BindGroupLayoutDescriptor,
        BindGroupLayoutEntries, Buffer, BufferDescriptor, BufferInitDescriptor, BufferUsages,
        CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor, DownlevelFlags,
        Extent3d, FilterMode, MipmapFilterMode, PipelineCache, Sampler, SamplerBindingType,
        SamplerDescriptor, ShaderStages, ShaderType, StorageTextureAccess, Texture, TextureAspect,
        TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
        TextureView, TextureViewDescriptor, TextureViewDimension, UniformBuffer,
    },
    renderer::{RenderAdapter, RenderContext, RenderDevice, RenderQueue},
    sync_component::{SyncComponent, SyncComponentPlugin},
    sync_world::RenderEntity,
    texture::{CachedTexture, GpuImage, TextureCache},
    Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
};

// Implementation: generate diffuse and specular cubemaps required by PBR
// from a given high-res cubemap by
//
// 1. Copying the base mip (level 0) of the source cubemap into an intermediate
//    storage texture.
// 2. Generating mipmaps with a quadratic B-spline filter.
// 3. Convolving the mip chain twice:
//    * a [Lambertian convolution] for the 32 × 32 diffuse cubemap, with a
//      projection onto [spherical harmonics]
//    * a [GGX convolution] for each mip of the specular cubemap, with the fitted
//      filters of [Manson and Sloan].
//
// [Lambertian convolution]: https://bruop.github.io/ibl/#:~:text=Lambertian%20Diffuse%20Component
// [spherical harmonics]: https://cseweb.ucsd.edu/~ravir/papers/envmap/envmap.pdf
// [GGX convolution]: https://gpuopen.com/download/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf
// [Manson and Sloan]: https://doi.org/10.1111/cgf.12822

use bevy_light::{EnvironmentMapLight, GeneratedEnvironmentMapLight};
use bevy_diagnostic::FrameCount;
use tracing::info;

use crate::environment_map::EnvironmentMapExposures;

/// Manson–Sloan filter coefficients for sources with 128² faces, fitted with
/// [EsotericaFilterFitter] (`FilterFitter --fit --curve esoterica --write-binary`,
/// the `const_8` cubemap table without its 144-byte header). Row `i` filters for
/// perceptual roughness `i / 6`.
///
/// [EsotericaFilterFitter]: https://github.com/bazhenovc/EsotericaFilterFitter
const FILTER_TABLE_128: &[u8] = include_bytes!("manson_sloan_128.bin");
/// The same as [`FILTER_TABLE_128`], fitted for 256² faces (`--base 256`).
const FILTER_TABLE_256: &[u8] = include_bytes!("manson_sloan_256.bin");

/// The number of roughness levels in the filter tables, and the maximum number
/// of mips in the generated specular map.
const FILTER_TABLE_LEVELS: u32 = 7;

/// Stores the bind group layouts for the environment map generation pipelines
#[derive(Resource)]
pub struct GeneratorBindGroupLayouts {
    pub downsample: BindGroupLayoutDescriptor,
    pub radiance: BindGroupLayoutDescriptor,
    pub sh_projection: BindGroupLayoutDescriptor,
    pub irradiance: BindGroupLayoutDescriptor,
    pub copy: BindGroupLayoutDescriptor,
}

/// Samplers for the environment map generation pipelines
#[derive(Resource)]
pub struct GeneratorSamplers {
    pub linear: Sampler,
}

/// Pipelines for the environment map generation pipelines
#[derive(Resource)]
pub struct GeneratorPipelines {
    pub downsample: CachedComputePipelineId,
    pub copy: CachedComputePipelineId,
    pub radiance: CachedComputePipelineId,
    pub sh_projection: CachedComputePipelineId,
    pub irradiance: CachedComputePipelineId,
}

/// The Manson–Sloan filter tables, fitted for sources with 128² and 256² faces.
#[derive(Resource)]
pub struct GeneratorFilterTables {
    pub table_128: Buffer,
    pub table_256: Buffer,
}

pub struct EnvironmentMapGenerationPlugin;

impl Plugin for EnvironmentMapGenerationPlugin {
    fn build(&self, _: &mut App) {}
    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            let adapter = render_app.world().resource::<RenderAdapter>();
            let device = render_app.world().resource::<RenderDevice>();

            let limit_support = device.limits().max_compute_workgroup_storage_size != 0
                && device.limits().max_compute_workgroup_size_x != 0;

            let downlevel_support = adapter
                .get_downlevel_capabilities()
                .flags
                .contains(DownlevelFlags::COMPUTE_SHADERS);

            if !limit_support || !downlevel_support {
                info!("Disabling EnvironmentMapGenerationPlugin because compute is not supported on this platform. This is safe to ignore if you are not using EnvironmentMapGenerationPlugin.");
                return;
            }
        } else {
            return;
        }

        embedded_asset!(app, "environment_filter.wesl");
        embedded_asset!(app, "environment_downsample.wesl");
        embedded_asset!(app, "copy.wesl");

        app.add_plugins(SyncComponentPlugin::<GeneratedEnvironmentMapLight, Self>::default())
            .add_systems(Update, generate_environment_map_light);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(
                ExtractSchedule,
                extract_generated_environment_map_entities.after(generate_environment_map_light),
            )
            .add_systems(
                Render,
                (
                    prepare_generated_environment_map_bind_groups
                        .in_set(RenderSystems::PrepareBindGroups),
                    prepare_generated_environment_map_intermediate_textures
                        .in_set(RenderSystems::PrepareResources),
                    (downsampling_system, filtering_system)
                        .chain()
                        .after(RenderSystems::PrepareBindGroups)
                        .before(RenderSystems::Render),
                ),
            )
            .add_systems(
                RenderStartup,
                initialize_generated_environment_map_resources,
            );
    }
}

/// Initializes all render-world resources used by the environment-map generator once on
/// [`bevy_render::RenderStartup`].
pub fn initialize_generated_environment_map_resources(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    asset_server: Res<AssetServer>,
) {
    let filter_table = |label, contents| {
        render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: BufferUsages::STORAGE,
        })
    };
    commands.insert_resource(GeneratorFilterTables {
        table_128: filter_table("manson_sloan_filter_table_128", FILTER_TABLE_128),
        table_256: filter_table("manson_sloan_filter_table_256", FILTER_TABLE_256),
    });

    // Bind group layouts
    let downsample = BindGroupLayoutDescriptor::new(
        "environment_downsample_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                // Source mip
                texture_2d_array(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                // Destination mip
                texture_storage_2d_array(
                    TextureFormat::Rgba16Float,
                    StorageTextureAccess::WriteOnly,
                ),
            ),
        ),
    );

    let radiance = BindGroupLayoutDescriptor::new(
        "radiance_bind_group_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                (1, sampler(SamplerBindingType::Filtering)), // Source sampler
                // Output specular map
                (
                    2,
                    texture_storage_2d_array(
                        TextureFormat::Rgba16Float,
                        StorageTextureAccess::WriteOnly,
                    ),
                ),
                (3, uniform_buffer::<FilteringConstants>(false)), // Uniforms
                (4, storage_buffer_read_only_sized(false, None)), // Filter table
                // Source environment cubemap, sampled across face edges
                (
                    5,
                    texture_cube(TextureSampleType::Float { filterable: true }),
                ),
            ),
        ),
    );

    let sh_projection = BindGroupLayoutDescriptor::new(
        "sh_projection_bind_group_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                // Source environment cubemap
                (
                    0,
                    texture_2d_array(TextureSampleType::Float { filterable: true }),
                ),
                // Output spherical harmonics
                (7, storage_buffer::<EnvironmentMapSh>(false)),
            ),
        ),
    );

    let irradiance = BindGroupLayoutDescriptor::new(
        "irradiance_bind_group_layout",
        &BindGroupLayoutEntries::with_indices(
            ShaderStages::COMPUTE,
            (
                // Source environment cubemap
                (
                    0,
                    texture_2d_array(TextureSampleType::Float { filterable: true }),
                ),
                // Output irradiance map
                (
                    2,
                    texture_storage_2d_array(
                        TextureFormat::Rgba16Float,
                        StorageTextureAccess::WriteOnly,
                    ),
                ),
                (3, uniform_buffer::<FilteringConstants>(false)), // Uniforms (face offset)
            ),
        ),
    );

    let copy = BindGroupLayoutDescriptor::new(
        "copy_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                // Source cubemap
                texture_2d_array(TextureSampleType::Float { filterable: true }),
                // Destination mip0
                texture_storage_2d_array(
                    TextureFormat::Rgba16Float,
                    StorageTextureAccess::WriteOnly,
                ),
            ),
        ),
    );

    let layouts = GeneratorBindGroupLayouts {
        downsample,
        radiance,
        sh_projection,
        irradiance,
        copy,
    };

    // Samplers
    let linear = render_device.create_sampler(&SamplerDescriptor {
        label: Some("generator_linear_sampler"),
        address_mode_u: AddressMode::ClampToEdge,
        address_mode_v: AddressMode::ClampToEdge,
        address_mode_w: AddressMode::ClampToEdge,
        mag_filter: FilterMode::Linear,
        min_filter: FilterMode::Linear,
        mipmap_filter: MipmapFilterMode::Linear,
        ..Default::default()
    });

    let samplers = GeneratorSamplers { linear };

    // Pipelines
    let env_filter_shader = load_embedded_asset!(asset_server.as_ref(), "environment_filter.wesl");
    let downsample_shader =
        load_embedded_asset!(asset_server.as_ref(), "environment_downsample.wesl");
    let copy_shader = load_embedded_asset!(asset_server.as_ref(), "copy.wesl");

    let downsample = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("environment_downsample_pipeline".into()),
        layout: vec![layouts.downsample.clone()],
        immediate_size: 0,
        shader: downsample_shader,
        shader_defs: vec![],
        entry_point: Some("downsample".into()),
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });

    // Radiance map for specular environment maps
    let radiance = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("radiance_pipeline".into()),
        layout: vec![layouts.radiance.clone()],
        immediate_size: 0,
        shader: env_filter_shader.clone(),
        shader_defs: vec![],
        entry_point: Some("generate_radiance_map".into()),
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });

    // Spherical harmonic projection of the diffuse lighting, once per probe
    let sh_projection = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("sh_projection_pipeline".into()),
        layout: vec![layouts.sh_projection.clone()],
        immediate_size: 0,
        shader: env_filter_shader.clone(),
        shader_defs: vec![],
        entry_point: Some("project_irradiance_sh".into()),
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });

    // Irradiance map for diffuse environment maps
    let irradiance = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("irradiance_pipeline".into()),
        layout: vec![layouts.irradiance.clone()],
        immediate_size: 0,
        shader: env_filter_shader,
        shader_defs: vec![],
        entry_point: Some("generate_irradiance_map".into()),
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });

    // Copy pipeline handles format conversion and populates mip0 when formats differ
    let copy_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("copy_pipeline".into()),
        layout: vec![layouts.copy.clone()],
        immediate_size: 0,
        shader: copy_shader,
        shader_defs: vec![],
        entry_point: Some("copy".into()),
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });

    let pipelines = GeneratorPipelines {
        downsample,
        radiance,
        sh_projection,
        irradiance,
        copy: copy_pipeline,
    };

    // Insert all resources into the render world
    commands.insert_resource(layouts);
    commands.insert_resource(samplers);
    commands.insert_resource(pipelines);
}

pub fn extract_generated_environment_map_entities(
    query: Extract<
        Query<(
            RenderEntity,
            &GeneratedEnvironmentMapLight,
            &EnvironmentMapLight,
        )>,
    >,
    mut commands: Commands,
    render_images: Res<RenderAssets<GpuImage>>,
    mut exposures: ResMut<EnvironmentMapExposures>,
) {
    for (entity, filtered_env_map, env_map_light) in query.iter() {
        let Some(env_map) = render_images.get(&filtered_env_map.environment_map) else {
            continue;
        };

        let diffuse_map = render_images.get(&env_map_light.diffuse_map);
        let specular_map = render_images.get(&env_map_light.specular_map);

        // continue if the diffuse map is not found
        if diffuse_map.is_none() || specular_map.is_none() {
            continue;
        }

        let diffuse_map = diffuse_map.unwrap();
        let specular_map = specular_map.unwrap();

        // Filtering runs ahead of the render graph that writes this frame's environment maps, so the
        // filtered maps hold the exposure their source was last written with.
        let exposure = exposures.get(filtered_env_map.environment_map.id());
        exposures.insert(env_map_light.specular_map.id(), exposure);

        let render_filtered_env_map = RenderEnvironmentMap {
            environment_map: env_map.clone(),
            diffuse_map: diffuse_map.clone(),
            specular_map: specular_map.clone(),
            intensity: filtered_env_map.intensity,
            rotation: filtered_env_map.rotation,
            affects_lightmapped_mesh_diffuse: filtered_env_map.affects_lightmapped_mesh_diffuse,
        };
        commands
            .get_entity(entity)
            .expect("Entity not synced to render world")
            .insert(render_filtered_env_map);
    }
}

// A render-world specific version of FilteredEnvironmentMapLight that uses CachedTexture
#[derive(Component, Clone)]
pub struct RenderEnvironmentMap {
    pub environment_map: GpuImage,
    pub diffuse_map: GpuImage,
    pub specular_map: GpuImage,
    pub intensity: f32,
    pub rotation: Quat,
    pub affects_lightmapped_mesh_diffuse: bool,
}

#[derive(Component)]
pub struct IntermediateTextures {
    pub environment_map: CachedTexture,
}

/// The number of excess lights an [`EnvironmentMapSh`] can hold
/// (`ENVIRONMENT_SH_MAX_EXCESS` in `environment_sh.wesl`).
pub const ENVIRONMENT_SH_MAX_EXCESS: usize = 64;

/// The contents of an [`EnvironmentMapShBuffer`]: the diffuse lighting of a
/// generated environment map as 9 RGB spherical harmonic coefficients (L2) plus
/// up to [`ENVIRONMENT_SH_MAX_EXCESS`] point lights for the excess of texels too
/// bright to project without ringing, such as the sun.
///
/// Mirrors `EnvironmentMapSh` in `environment_sh.wesl`, which documents the
/// layout (2208 bytes; std430 and std140 agree, so it binds as a storage or a
/// uniform buffer). It is only ever written by the GPU; this type sizes the
/// buffer and the bind group layout entry.
#[derive(Clone, Copy, ShaderType)]
pub struct EnvironmentMapSh {
    /// 9 × (R, G, B, 0), convolved with the clamped cosine and divided by π.
    pub coefficients: [Vec4; 9],
    /// The number of excess lights in use.
    pub excess_count: u32,
    /// 64 × (x, y, z, 0), unit directions toward each excess light.
    pub excess_directions: [Vec4; ENVIRONMENT_SH_MAX_EXCESS],
    /// 64 × (R, G, B, 0), radiance × solid angle of each excess light.
    pub excess_radiances: [Vec4; ENVIRONMENT_SH_MAX_EXCESS],
}

/// The GPU buffer holding a probe's [`EnvironmentMapSh`], on the render entity of
/// each [`GeneratedEnvironmentMapLight`]. Written once per frame by the
/// `project_irradiance_sh` pass before the irradiance pass reads it.
#[derive(Component, Clone)]
pub struct EnvironmentMapShBuffer {
    pub buffer: Buffer,
}

/// Returns the total number of mip levels for the provided square texture size.
/// `size` must be a power of two greater than zero. For example, `size = 512` → `9`.
#[inline]
fn compute_mip_count(size: u32) -> u32 {
    debug_assert!(size.is_power_of_two());
    32 - size.leading_zeros()
}

/// Returns the number of mip levels of the specular map generated from a source
/// of the given size: one per row of the filter table, so that mip `i` stores
/// perceptual roughness `i / 6`.
#[inline]
fn compute_specular_mip_count(size: u32) -> u32 {
    compute_mip_count(size).min(FILTER_TABLE_LEVELS)
}

/// Prepares textures needed for downsampling
pub fn prepare_generated_environment_map_intermediate_textures(
    light_probes: Query<(
        Entity,
        &RenderEnvironmentMap,
        Option<&EnvironmentMapShBuffer>,
    )>,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
    mut commands: Commands,
) {
    for (entity, env_map_light, sh_buffer) in &light_probes {
        // The spherical harmonics buffer persists across frames
        if sh_buffer.is_none() {
            let buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("environment_map_sh"),
                size: EnvironmentMapSh::min_size().get(),
                usage: BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            commands
                .entity(entity)
                .insert(EnvironmentMapShBuffer { buffer });
        }

        let base_size = env_map_light.environment_map.texture_descriptor.size.width;
        let mip_level_count = compute_mip_count(base_size);

        let environment_map = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("intermediate_environment_map"),
                size: Extent3d {
                    width: base_size,
                    height: base_size,
                    depth_or_array_layers: 6, // Cubemap faces
                },
                mip_level_count,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::TEXTURE_BINDING
                    | TextureUsages::STORAGE_BINDING
                    | TextureUsages::COPY_DST,
                view_formats: &[],
            },
        );

        commands
            .entity(entity)
            .insert(IntermediateTextures { environment_map });
    }
}

/// Constants for filtering
#[derive(Clone, Copy, ShaderType)]
pub struct FilteringConstants {
    perceptual_roughness: f32,
    /// `log2` of the source face size over the face size the filter table is
    /// fitted for, added to the mip level of each tap.
    lod_bias: f32,
    /// The first cubemap face this frame's dispatches filter ([`EnvironmentMapFilterSlicing`]).
    face_offset: u32,
}

/// Spreads the filtering of a generated environment map over frames: each frame filters
/// `faces_per_frame` of the six faces, cycling through them, instead of all six.
///
/// For a source cubemap that is itself rendered a few faces per frame and changes slowly, such as
/// the atmosphere's. Render world only. `faces_per_frame` must divide 6.
#[derive(Component, Clone, Copy)]
pub struct EnvironmentMapFilterSlicing {
    pub faces_per_frame: u32,
}

impl EnvironmentMapFilterSlicing {
    fn faces(slicing: Option<&Self>) -> u32 {
        slicing.map_or(6, |slicing| slicing.faces_per_frame)
    }
}

/// Stores bind groups for the environment map generation pipelines
#[derive(Component)]
pub struct GeneratorBindGroups {
    pub downsample: Vec<BindGroup>, // One per mip level after the first
    pub radiance: Vec<BindGroup>,   // One per mip level
    pub sh_projection: BindGroup,
    pub irradiance: BindGroup,
    pub copy: BindGroup,
}

/// Prepares bind groups for environment map generation pipelines
pub fn prepare_generated_environment_map_bind_groups(
    light_probes: Query<(
        Entity,
        &IntermediateTextures,
        &RenderEnvironmentMap,
        &EnvironmentMapShBuffer,
        Option<&EnvironmentMapFilterSlicing>,
    )>,
    frame_count: Res<FrameCount>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    queue: Res<RenderQueue>,
    layouts: Res<GeneratorBindGroupLayouts>,
    samplers: Res<GeneratorSamplers>,
    filter_tables: Res<GeneratorFilterTables>,
    mut commands: Commands,
) {
    for (entity, textures, env_map_light, sh_buffer, slicing) in &light_probes {
        let face_offset = (frame_count.0 * EnvironmentMapFilterSlicing::faces(slicing)) % 6;

        // Determine mip chain based on input size
        let base_size = env_map_light.environment_map.texture_descriptor.size.width;
        let mip_count = compute_mip_count(base_size);

        // Create downsampling bind groups, one per destination mip
        let downsample_bind_groups = (1..mip_count)
            .map(|mip| {
                let source = textures
                    .environment_map
                    .texture
                    .create_view(&TextureViewDescriptor {
                        dimension: Some(TextureViewDimension::D2Array),
                        base_mip_level: mip - 1,
                        mip_level_count: Some(1),
                        ..Default::default()
                    });
                let destination =
                    create_storage_view(&textures.environment_map.texture, mip, &render_device);
                render_device.create_bind_group(
                    Some(format!("environment_downsample_bind_group_mip_{mip}").as_str()),
                    &pipeline_cache.get_bind_group_layout(&layouts.downsample),
                    &BindGroupEntries::sequential((&source, &samplers.linear, &destination)),
                )
            })
            .collect();

        // Use the table fitted for the nearest face size at or below the source's,
        // and shift its taps to the source's mip chain
        let (filter_table, table_size) = if base_size >= 256 {
            (&filter_tables.table_256, 256u32)
        } else {
            (&filter_tables.table_128, 128u32)
        };
        let lod_bias = base_size.ilog2() as f32 - table_size.ilog2() as f32;

        // A cube view of the source mip chain, so filtering crosses face edges
        let environment_map_cube =
            textures
                .environment_map
                .texture
                .create_view(&TextureViewDescriptor {
                    dimension: Some(TextureViewDimension::Cube),
                    ..Default::default()
                });

        // Create radiance map bind groups for each mip level
        let specular_mip_count = compute_specular_mip_count(base_size);
        let mut radiance_bind_groups = Vec::with_capacity(specular_mip_count as usize);

        for mip in 0..specular_mip_count {
            let radiance_constants = FilteringConstants {
                perceptual_roughness: mip as f32 / (specular_mip_count - 1).max(1) as f32,
                lod_bias,
                face_offset,
            };

            let mut radiance_constants_buffer = UniformBuffer::from(radiance_constants);
            radiance_constants_buffer.write_buffer(&render_device, &queue);

            let mip_storage_view =
                create_storage_view(&env_map_light.specular_map.texture, mip, &render_device);
            let bind_group = render_device.create_bind_group(
                Some(format!("radiance_bind_group_mip_{mip}").as_str()),
                &pipeline_cache.get_bind_group_layout(&layouts.radiance),
                &BindGroupEntries::with_indices((
                    (1, &samplers.linear),
                    (2, &mip_storage_view),
                    (3, &radiance_constants_buffer),
                    (4, filter_table.as_entire_binding()),
                    (5, &environment_map_cube),
                )),
            );

            radiance_bind_groups.push(bind_group);
        }

        // Create the spherical harmonic projection bind group
        let sh_projection_bind_group = render_device.create_bind_group(
            "sh_projection_bind_group",
            &pipeline_cache.get_bind_group_layout(&layouts.sh_projection),
            &BindGroupEntries::with_indices((
                (0, &textures.environment_map.default_view),
                (7, sh_buffer.buffer.as_entire_binding()),
            )),
        );

        // Create irradiance bind group; only `face_offset` is read there
        let irradiance_constants = FilteringConstants {
            perceptual_roughness: 1.0,
            lod_bias,
            face_offset,
        };

        let mut irradiance_constants_buffer = UniformBuffer::from(irradiance_constants);
        irradiance_constants_buffer.write_buffer(&render_device, &queue);

        // create a 2d array view
        let irradiance_map =
            env_map_light
                .diffuse_map
                .texture
                .create_view(&TextureViewDescriptor {
                    dimension: Some(TextureViewDimension::D2Array),
                    ..Default::default()
                });

        let irradiance_bind_group = render_device.create_bind_group(
            "irradiance_bind_group",
            &pipeline_cache.get_bind_group_layout(&layouts.irradiance),
            &BindGroupEntries::with_indices((
                (0, &textures.environment_map.default_view),
                (2, &irradiance_map),
                (3, &irradiance_constants_buffer),
            )),
        );

        // Create copy bind group (source env map → destination mip0)
        let src_view = env_map_light
            .environment_map
            .texture
            .create_view(&TextureViewDescriptor {
                dimension: Some(TextureViewDimension::D2Array),
                ..Default::default()
            });

        let dst_view = create_storage_view(&textures.environment_map.texture, 0, &render_device);

        let copy_bind_group = render_device.create_bind_group(
            "copy_bind_group",
            &pipeline_cache.get_bind_group_layout(&layouts.copy),
            &BindGroupEntries::with_indices(((0, &src_view), (1, &dst_view))),
        );

        commands.entity(entity).insert(GeneratorBindGroups {
            downsample: downsample_bind_groups,
            radiance: radiance_bind_groups,
            sh_projection: sh_projection_bind_group,
            irradiance: irradiance_bind_group,
            copy: copy_bind_group,
        });
    }
}

/// Helper function to create a storage texture view for a specific mip level
fn create_storage_view(texture: &Texture, mip: u32, _render_device: &RenderDevice) -> TextureView {
    texture.create_view(&TextureViewDescriptor {
        label: Some(format!("storage_view_mip_{mip}").as_str()),
        format: Some(texture.format()),
        dimension: Some(TextureViewDimension::D2Array),
        aspect: TextureAspect::All,
        base_mip_level: mip,
        mip_level_count: Some(1),
        base_array_layer: 0,
        array_layer_count: Some(texture.depth_or_array_layers()),
        usage: Some(TextureUsages::STORAGE_BINDING),
    })
}

pub fn downsampling_system(
    query: Query<(&GeneratorBindGroups, &RenderEnvironmentMap)>,
    pipeline_cache: Res<PipelineCache>,
    pipelines: Option<Res<GeneratorPipelines>>,
    mut ctx: RenderContext,
) {
    let Some(pipelines) = pipelines else {
        return;
    };

    let Some(downsample_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.downsample)
    else {
        return;
    };

    let Some(copy_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.copy) else {
        return;
    };

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();

    for (bind_groups, env_map_light) in &query {
        // Copy base mip using compute shader with pre-built bind group
        {
            let mut compute_pass =
                ctx.command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("lightprobe_copy"),
                        timestamp_writes: None,
                    });

            let pass_span = diagnostics.pass_span(&mut compute_pass, "lightprobe_copy");

            compute_pass.set_pipeline(copy_pipeline);
            compute_pass.set_bind_group(0, &bind_groups.copy, &[]);

            let tex_size = env_map_light.environment_map.texture_descriptor.size;
            let wg_x = tex_size.width.div_ceil(8);
            let wg_y = tex_size.height.div_ceil(8);
            compute_pass.dispatch_workgroups(wg_x, wg_y, 6);

            pass_span.end(&mut compute_pass);
        }

        // Downsample each mip into the next
        {
            let mut compute_pass =
                ctx.command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("lightprobe_downsampling"),
                        timestamp_writes: None,
                    });

            let pass_span = diagnostics.pass_span(&mut compute_pass, "lightprobe_downsampling");

            compute_pass.set_pipeline(downsample_pipeline);

            let base_size = env_map_light.environment_map.texture_descriptor.size.width;
            for (mip, bind_group) in (1..).zip(&bind_groups.downsample) {
                compute_pass.set_bind_group(0, bind_group, &[]);
                let workgroup_count = (base_size >> mip).div_ceil(8);
                compute_pass.dispatch_workgroups(workgroup_count, workgroup_count, 6);
            }

            pass_span.end(&mut compute_pass);
        }
    }
}

pub fn filtering_system(
    query: Query<(
        &GeneratorBindGroups,
        &RenderEnvironmentMap,
        Option<&EnvironmentMapFilterSlicing>,
    )>,
    pipeline_cache: Res<PipelineCache>,
    pipelines: Option<Res<GeneratorPipelines>>,
    mut ctx: RenderContext,
) {
    let Some(pipelines) = pipelines else {
        return;
    };

    let Some(radiance_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.radiance) else {
        return;
    };
    let Some(sh_projection_pipeline) =
        pipeline_cache.get_compute_pipeline(pipelines.sh_projection)
    else {
        return;
    };
    let Some(irradiance_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.irradiance)
    else {
        return;
    };

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();

    for (bind_groups, env_map_light, slicing) in &query {
        let faces = EnvironmentMapFilterSlicing::faces(slicing);

        // Radiance convolution pass
        {
            let mut compute_pass =
                ctx.command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("lightprobe_radiance_map"),
                        timestamp_writes: None,
                    });

            let pass_span = diagnostics.pass_span(&mut compute_pass, "lightprobe_radiance_map");

            compute_pass.set_pipeline(radiance_pipeline);

            let base_size = env_map_light.specular_map.texture_descriptor.size.width;

            // Process each mip at different roughness levels
            for (mip, bind_group) in bind_groups.radiance.iter().enumerate() {
                compute_pass.set_bind_group(0, bind_group, &[]);

                // Calculate dispatch size based on mip level
                let mip_size = base_size >> mip;
                let workgroup_count = mip_size.div_ceil(8);

                compute_pass.dispatch_workgroups(workgroup_count, workgroup_count, faces);
            }

            pass_span.end(&mut compute_pass);
        }

        // Spherical harmonic projection pass: one workgroup per probe, every
        // frame the irradiance pass runs
        {
            let mut compute_pass =
                ctx.command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("lightprobe_sh_projection"),
                        timestamp_writes: None,
                    });

            let sh_span = diagnostics.pass_span(&mut compute_pass, "lightprobe_sh_projection");

            compute_pass.set_pipeline(sh_projection_pipeline);
            compute_pass.set_bind_group(0, &bind_groups.sh_projection, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);

            sh_span.end(&mut compute_pass);
        }

        // Irradiance convolution pass
        // Generate the diffuse environment map
        {
            let mut compute_pass =
                ctx.command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("lightprobe_irradiance_map"),
                        timestamp_writes: None,
                    });

            let irr_span = diagnostics.pass_span(&mut compute_pass, "lightprobe_irradiance_map");

            compute_pass.set_pipeline(irradiance_pipeline);
            compute_pass.set_bind_group(0, &bind_groups.irradiance, &[]);

            // 32×32 texture processed with 8×8 workgroups
            compute_pass.dispatch_workgroups(4, 4, faces);

            irr_span.end(&mut compute_pass);
        }
    }
}

/// System that generates an `EnvironmentMapLight` component based on the `GeneratedEnvironmentMapLight` component
pub fn generate_environment_map_light(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    query: Query<(Entity, &GeneratedEnvironmentMapLight), Without<EnvironmentMapLight>>,
) {
    for (entity, filtered_env_map) in &query {
        // Validate and fetch the source cubemap so we can size our targets correctly
        let Some(src_image) = images.get(&filtered_env_map.environment_map) else {
            // Texture not ready yet – try again next frame
            continue;
        };

        let base_size = src_image.texture_descriptor.size.width;

        // Sanity checks – square, power-of-two, ≤ 8192
        if src_image.texture_descriptor.size.height != base_size
            || !base_size.is_power_of_two()
            || base_size > 8192
        {
            panic!(
                "GeneratedEnvironmentMapLight source cubemap must be square power-of-two ≤ 8192, got {}×{}",
                base_size, src_image.texture_descriptor.size.height
            );
        }

        let mip_count = compute_specular_mip_count(base_size);

        // Create a placeholder for the irradiance map
        let mut diffuse = Image::new_fill(
            Extent3d {
                width: 32,
                height: 32,
                depth_or_array_layers: 6,
            },
            TextureDimension::D2,
            &[0; 8],
            TextureFormat::Rgba16Float,
            RenderAssetUsages::all(),
        );

        diffuse.texture_descriptor.usage =
            TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING;

        diffuse.texture_view_descriptor = Some(TextureViewDescriptor {
            dimension: Some(TextureViewDimension::Cube),
            ..Default::default()
        });

        let diffuse_handle = images.add(diffuse);

        // Create a placeholder for the specular map. It matches the input cubemap resolution.
        let mut specular = Image::new_fill(
            Extent3d {
                width: base_size,
                height: base_size,
                depth_or_array_layers: 6,
            },
            TextureDimension::D2,
            &[0; 8],
            TextureFormat::Rgba16Float,
            RenderAssetUsages::all(),
        );

        // Set up for mipmaps
        specular.texture_descriptor.usage =
            TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING;
        specular.texture_descriptor.mip_level_count = mip_count;

        // When setting mip_level_count, we need to allocate appropriate data size
        // For GPU-generated mipmaps, we can set data to None since the GPU will generate the data
        specular.data = None;

        specular.texture_view_descriptor = Some(TextureViewDescriptor {
            dimension: Some(TextureViewDimension::Cube),
            mip_level_count: Some(mip_count),
            ..Default::default()
        });

        let specular_handle = images.add(specular);

        // Add the EnvironmentMapLight component with the placeholder handles
        commands.entity(entity).insert(EnvironmentMapLight {
            diffuse_map: diffuse_handle,
            specular_map: specular_handle,
            intensity: filtered_env_map.intensity,
            rotation: filtered_env_map.rotation,
            affects_lightmapped_mesh_diffuse: filtered_env_map.affects_lightmapped_mesh_diffuse,
        });
    }
}

impl SyncComponent<RenderApp, EnvironmentMapGenerationPlugin> for GeneratedEnvironmentMapLight {
    type Target = RenderEnvironmentMap;
}
