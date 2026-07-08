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
use bevy_math::{Quat, UVec2};
use bevy_render::{
    diagnostic::RecordDiagnostics,
    render_asset::RenderAssets,
    render_resource::{
        binding_types::*, AddressMode, BindGroup, BindGroupEntries, BindGroupLayoutDescriptor,
        BindGroupLayoutEntries, BindingResource, Buffer, BufferBinding, BufferInitDescriptor,
        BufferUsages, CachedComputePipelineId, ComputePassDescriptor, ComputePipelineDescriptor,
        DownlevelFlags, Extent3d, FilterMode, MipmapFilterMode, Origin3d, PipelineCache, Sampler,
        SamplerBindingType, SamplerDescriptor, ShaderStages, ShaderType, StorageTextureAccess,
        TexelCopyTextureInfo, Texture, TextureAspect, TextureDescriptor, TextureDimension,
        TextureFormat, TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor,
        TextureViewDimension, UniformBuffer,
    },
    renderer::{RenderAdapter, RenderContext, RenderDevice, RenderQueue},
    settings::WgpuFeatures,
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
// 2. Generating mipmaps with quadratic B-spline + Jacobian weighting (Manson & Sloan EGSR 2015 §4).
// 3. Convolving the mip chain twice:
//    * a [Lambertian convolution] for the 32 × 32 diffuse cubemap
//    * a [Manson–Sloan gather] for the specular cubemap (constant 8×3 trilinear taps + table).
//
// [Lambertian convolution]: https://bruop.github.io/ibl/#:~:text=Lambertian%20Diffuse%20Component
// [Manson–Sloan gather]: https://diglib.eg.org/handle/10.2312/egsr.20151131

use bevy_light::{
    EnvironmentMapLight, GeneratedEnvironmentMapLight, SpecularEnvironmentIntegration,
};
use bevy_shader::ShaderDefVal;
use tracing::info;

use crate::Bluenoise;

use core::num::NonZero;

use super::manson_sloan;

/// Stores the bind group layouts for the environment map generation pipelines
#[derive(Resource)]
pub struct GeneratorBindGroupLayouts {
    pub quadratic_downsample: BindGroupLayoutDescriptor,
    pub radiance: BindGroupLayoutDescriptor,
    pub irradiance: BindGroupLayoutDescriptor,
    pub copy: BindGroupLayoutDescriptor,
    pub temporal_blend: BindGroupLayoutDescriptor,
}

/// Samplers for the environment map generation pipelines
#[derive(Resource)]
pub struct GeneratorSamplers {
    pub linear: Sampler,
}

/// Pipelines for the environment map generation pipelines
#[derive(Resource)]
pub struct GeneratorPipelines {
    pub quadratic_downsample: CachedComputePipelineId,
    pub copy: CachedComputePipelineId,
    pub radiance: CachedComputePipelineId,
    pub irradiance: CachedComputePipelineId,
    pub temporal_blend: CachedComputePipelineId,
}

/// Uniforms for Manson–Sloan pass-1 cubemap downsample (4-tap Jacobian-weighted corners per
/// `downsample_cubemap.txt`; one dispatch per mip level).
#[derive(Clone, Copy, ShaderType)]
#[repr(C)]
pub struct QuadraticDownsampleUniforms {
    pub src_mip_level: u32,
    pub dst_mip_level: u32,
    pub dst_size: u32,
    pub _pad: u32,
}

/// Uniforms for temporal blending of filtered specular cubemaps.
#[derive(Clone, Copy, ShaderType)]
#[repr(C)]
pub struct TemporalBlendUniforms {
    pub alpha: f32,
    pub _pad: [f32; 3],
}

/// Constants for filtering.
#[derive(Clone, Copy, ShaderType)]
#[repr(C)]
pub struct FilteringConstants {
    /// Output specular mip index (for debugging / future use).
    mip_level: f32,
    sample_count: u32,
    /// Perceptual roughness in \[0, 1\] for this mip — matches
    /// `environment_map.wgsl`: `radiance_level = perceptual_roughness * (num_levels - 1)`.
    /// Not GGX α (no Filament-style clamp-then-square); that mapping is for BRDF shading, not env mip LOD.
    perceptual_roughness: f32,
    /// `log2(cubemap_face_size / 128)` — baked Manson–Sloan LODs assume 128² faces in SA_texel.
    lod_resolution_bias: f32,
    /// Width/height of the storage view this dispatch writes (one cubemap face).
    ///
    /// `textureDimensions(output_texture)` is not reliable for single-mip storage views on some
    /// backends; we pass the extent explicitly.
    output_size: UVec2,
    noise_size_bits: UVec2,
}

/// GPU buffer storing Manson–Sloan polynomial coefficients (`manson_sloan::FILTER_TABLE_SIZE` bytes).
#[derive(Resource, Clone)]
pub struct MansonSloanFilterTableBuffer(pub Buffer);

pub struct EnvironmentMapGenerationPlugin;

impl Plugin for EnvironmentMapGenerationPlugin {
    fn build(&self, _: &mut App) {}
    fn finish(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            let adapter = render_app.world().resource::<RenderAdapter>();
            let device = render_app.world().resource::<RenderDevice>();

            // Cubemap SPD requires at least 6 storage textures
            let limit_support = device.limits().max_storage_textures_per_shader_stage >= 6
                && device.limits().max_compute_workgroup_storage_size != 0
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

        embedded_asset!(app, "environment_filter.wgsl");
        embedded_asset!(app, "quadratic_b_spline_downsample.wgsl");
        embedded_asset!(app, "temporal_blend_environment_map.wgsl");
        embedded_asset!(app, "copy.wgsl");

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
    _render_adapter: Res<RenderAdapter>,
    pipeline_cache: Res<PipelineCache>,
    asset_server: Res<AssetServer>,
) {
    let table_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("manson_sloan_poly_table".into()),
        contents: manson_sloan::FILTER_TABLE_BYTES,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    commands.insert_resource(MansonSloanFilterTableBuffer(table_buffer));

    let table_min_binding = NonZero::new(manson_sloan::FILTER_TABLE_SIZE as u64)
        .expect("filter table size must be non-zero");

    let quadratic_downsample = BindGroupLayoutDescriptor::new(
        "quadratic_downsample_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d_array(TextureSampleType::Float { filterable: true }),
                texture_storage_2d_array(
                    TextureFormat::Rgba16Float,
                    StorageTextureAccess::WriteOnly,
                ),
                uniform_buffer::<QuadraticDownsampleUniforms>(false),
            ),
        ),
    );

    let radiance = BindGroupLayoutDescriptor::new(
        "radiance_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d_array(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                texture_storage_2d_array(
                    TextureFormat::Rgba16Float,
                    StorageTextureAccess::WriteOnly,
                ),
                uniform_buffer::<FilteringConstants>(false),
                texture_2d_array(TextureSampleType::Float { filterable: true }),
                storage_buffer_read_only_sized(false, Some(table_min_binding)),
            ),
        ),
    );

    let irradiance = BindGroupLayoutDescriptor::new(
        "irradiance_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d_array(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                texture_storage_2d_array(
                    TextureFormat::Rgba16Float,
                    StorageTextureAccess::WriteOnly,
                ),
                uniform_buffer::<FilteringConstants>(false),
                texture_2d_array(TextureSampleType::Float { filterable: true }),
                storage_buffer_read_only_sized(false, Some(table_min_binding)),
            ),
        ),
    );

    let temporal_blend = BindGroupLayoutDescriptor::new(
        "temporal_blend_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d_array(TextureSampleType::Float { filterable: true }),
                texture_2d_array(TextureSampleType::Float { filterable: true }),
                texture_storage_2d_array(
                    TextureFormat::Rgba16Float,
                    StorageTextureAccess::WriteOnly,
                ),
                uniform_buffer::<TemporalBlendUniforms>(false),
            ),
        ),
    );

    let copy = BindGroupLayoutDescriptor::new(
        "copy_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_2d_array(TextureSampleType::Float { filterable: true }),
                texture_storage_2d_array(
                    TextureFormat::Rgba16Float,
                    StorageTextureAccess::WriteOnly,
                ),
            ),
        ),
    );

    let layouts = GeneratorBindGroupLayouts {
        quadratic_downsample,
        radiance,
        irradiance,
        copy,
        temporal_blend,
    };

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

    let features = render_device.features();
    let mut shader_defs = vec![];
    if features.contains(WgpuFeatures::SUBGROUP) {
        shader_defs.push(ShaderDefVal::Int("SUBGROUP_SUPPORT".into(), 1));
    }
    shader_defs.push(ShaderDefVal::Bool("ARRAY_TEXTURE".into(), true));
    #[cfg(feature = "bluenoise_texture")]
    {
        shader_defs.push(ShaderDefVal::Int("HAS_BLUE_NOISE".into(), 1));
    }

    let env_filter_shader = load_embedded_asset!(asset_server.as_ref(), "environment_filter.wgsl");
    let copy_shader = load_embedded_asset!(asset_server.as_ref(), "copy.wgsl");
    let quadratic_shader =
        load_embedded_asset!(asset_server.as_ref(), "quadratic_b_spline_downsample.wgsl");
    let temporal_shader =
        load_embedded_asset!(asset_server.as_ref(), "temporal_blend_environment_map.wgsl");

    let quadratic_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("quadratic_downsample_pipeline".into()),
        layout: vec![layouts.quadratic_downsample.clone()],
        immediate_size: 0,
        shader: quadratic_shader,
        shader_defs: shader_defs.clone(),
        entry_point: Some("downsample_quadratic_mip".into()),
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });

    let radiance_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("radiance_pipeline".into()),
        layout: vec![layouts.radiance.clone()],
        immediate_size: 0,
        shader: env_filter_shader.clone(),
        shader_defs: shader_defs.clone(),
        entry_point: Some("generate_radiance_map".into()),
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });

    let irradiance_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("irradiance_pipeline".into()),
        layout: vec![layouts.irradiance.clone()],
        immediate_size: 0,
        shader: env_filter_shader,
        shader_defs: shader_defs.clone(),
        entry_point: Some("generate_irradiance_map".into()),
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });

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

    let temporal_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("temporal_blend_specular_pipeline".into()),
        layout: vec![layouts.temporal_blend.clone()],
        immediate_size: 0,
        shader: temporal_shader,
        shader_defs: shader_defs.clone(),
        entry_point: Some("temporal_blend_specular".into()),
        zero_initialize_workgroup_memory: false,
        constants: vec![],
    });

    let pipelines = GeneratorPipelines {
        quadratic_downsample: quadratic_pipeline,
        radiance: radiance_pipeline,
        irradiance: irradiance_pipeline,
        copy: copy_pipeline,
        temporal_blend: temporal_pipeline,
    };

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

        let render_filtered_env_map = RenderEnvironmentMap {
            environment_map: env_map.clone(),
            diffuse_map: diffuse_map.clone(),
            specular_map: specular_map.clone(),
            intensity: filtered_env_map.intensity,
            rotation: filtered_env_map.rotation,
            affects_lightmapped_mesh_diffuse: filtered_env_map.affects_lightmapped_mesh_diffuse,
            temporal_blend: filtered_env_map.temporal_blend,
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
    /// `0` = off; otherwise blend factor toward this frame's filter (`out = lerp(history, filtered, clamp(alpha,0,1))`).
    pub temporal_blend: f32,
}

#[derive(Component)]
pub struct IntermediateTextures {
    pub environment_map: CachedTexture,
    pub specular_scratch: Option<CachedTexture>,
    pub specular_history: Option<CachedTexture>,
}

/// Returns the total number of mip levels for the provided square texture size.
/// `size` must be a power of two greater than zero. For example, `size = 512` → `9`.
#[inline]
fn compute_mip_count(size: u32) -> u32 {
    debug_assert!(size.is_power_of_two());
    32 - size.leading_zeros()
}

/// Prepares textures needed for single pass downsampling
pub fn prepare_generated_environment_map_intermediate_textures(
    light_probes: Query<(Entity, &RenderEnvironmentMap)>,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
    mut commands: Commands,
) {
    for (entity, env_map_light) in &light_probes {
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

        let spec_desc = &env_map_light.specular_map.texture_descriptor;
        let temporal = env_map_light.temporal_blend > 0.0;
        let specular_scratch = temporal.then(|| {
            texture_cache.get(
                &render_device,
                TextureDescriptor {
                    label: Some("specular_filter_scratch"),
                    size: spec_desc.size,
                    mip_level_count: spec_desc.mip_level_count,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba16Float,
                    usage: TextureUsages::TEXTURE_BINDING
                        | TextureUsages::STORAGE_BINDING
                        | TextureUsages::COPY_DST,
                    view_formats: &[],
                },
            )
        });
        let specular_history = temporal.then(|| {
            texture_cache.get(
                &render_device,
                TextureDescriptor {
                    label: Some("specular_filter_history"),
                    size: spec_desc.size,
                    mip_level_count: spec_desc.mip_level_count,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: TextureFormat::Rgba16Float,
                    usage: TextureUsages::TEXTURE_BINDING
                        | TextureUsages::COPY_SRC
                        | TextureUsages::COPY_DST,
                    view_formats: &[],
                },
            )
        });

        commands.entity(entity).insert(IntermediateTextures {
            environment_map,
            specular_scratch,
            specular_history,
        });
    }
}

/// Stores bind groups for the environment map generation pipelines
#[derive(Component)]
pub struct GeneratorBindGroups {
    pub quadratic: Vec<BindGroup>,
    pub radiance: Vec<BindGroup>,
    pub irradiance: BindGroup,
    pub copy: BindGroup,
    pub temporal: Option<Vec<BindGroup>>,
}

/// Prepares bind groups for environment map generation pipelines
pub fn prepare_generated_environment_map_bind_groups(
    light_probes: Query<(Entity, &IntermediateTextures, &RenderEnvironmentMap)>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    queue: Res<RenderQueue>,
    layouts: Res<GeneratorBindGroupLayouts>,
    samplers: Res<GeneratorSamplers>,
    render_images: Res<RenderAssets<GpuImage>>,
    bluenoise: Res<Bluenoise>,
    manson_sloan_table: Res<MansonSloanFilterTableBuffer>,
    mut commands: Commands,
) {
    let Some(stbn_texture) = render_images.get(&bluenoise.texture) else {
        return;
    };

    assert!(stbn_texture.texture_descriptor.size.width.is_power_of_two());
    assert!(stbn_texture
        .texture_descriptor
        .size
        .height
        .is_power_of_two());
    let noise_size_bits = UVec2::new(
        stbn_texture.texture_descriptor.size.width.trailing_zeros(),
        stbn_texture.texture_descriptor.size.height.trailing_zeros(),
    );

    let table_binding = BindingResource::Buffer(BufferBinding {
        buffer: &manson_sloan_table.0,
        offset: 0,
        size: None,
    });

    for (entity, textures, env_map_light) in &light_probes {
        let base_size = env_map_light.environment_map.texture_descriptor.size.width;
        let mip_count = compute_mip_count(base_size);

        let mut quadratic_bind_groups = Vec::with_capacity(mip_count.saturating_sub(1) as usize);
        for dst_mip in 1..mip_count {
            let src_mip = dst_mip - 1;
            let dst_size = base_size >> dst_mip;
            let q_const = QuadraticDownsampleUniforms {
                src_mip_level: src_mip,
                dst_mip_level: dst_mip,
                dst_size,
                _pad: 0,
            };
            let mut q_buf = UniformBuffer::from(q_const);
            q_buf.write_buffer(&render_device, &queue);

            let src_view = textures
                .environment_map
                .texture
                .create_view(&TextureViewDescriptor {
                    dimension: Some(TextureViewDimension::D2Array),
                    base_mip_level: src_mip,
                    mip_level_count: Some(1),
                    ..Default::default()
                });
            let dst_view =
                create_storage_view(&textures.environment_map.texture, dst_mip, &render_device);

            let bg = render_device.create_bind_group(
                Some(format!("quadratic_downsample_mip_{dst_mip}").as_str()),
                &pipeline_cache.get_bind_group_layout(&layouts.quadratic_downsample),
                &BindGroupEntries::sequential((&src_view, &dst_view, &q_buf)),
            );
            quadratic_bind_groups.push(bg);
        }

        let stbn_texture_view = stbn_texture
            .texture
            .clone()
            .create_view(&TextureViewDescriptor {
                dimension: Some(TextureViewDimension::D2Array),
                ..Default::default()
            });

        let num_mips = mip_count as usize;
        let mut radiance_bind_groups = Vec::with_capacity(num_mips);

        let use_temporal = env_map_light.temporal_blend > 0.0;
        let spec_tex: &Texture = if use_temporal {
            &textures
                .specular_scratch
                .as_ref()
                .expect("scratch texture must exist when temporal_blend > 0")
                .texture
        } else {
            &env_map_light.specular_map.texture
        };

        let lod_resolution_bias = (base_size as f32 / 128.0).log2();
        for mip in 0..num_mips {
            let perceptual_roughness = mip as f32 / (num_mips - 1).max(1) as f32;
            let face_extent = base_size >> mip;
            let radiance_constants = FilteringConstants {
                mip_level: mip as f32,
                sample_count: 0,
                perceptual_roughness,
                lod_resolution_bias,
                output_size: UVec2::splat(face_extent),
                noise_size_bits,
            };

            let mut radiance_constants_buffer = UniformBuffer::from(radiance_constants);
            radiance_constants_buffer.write_buffer(&render_device, &queue);

            let mip_storage_view = create_storage_view(spec_tex, mip as u32, &render_device);

            let bind_group = render_device.create_bind_group(
                Some(format!("radiance_bind_group_mip_{mip}").as_str()),
                &pipeline_cache.get_bind_group_layout(&layouts.radiance),
                &BindGroupEntries::with_indices((
                    (0u32, &textures.environment_map.default_view),
                    (1u32, &samplers.linear),
                    (2u32, &mip_storage_view),
                    (3u32, &radiance_constants_buffer),
                    (4u32, &stbn_texture_view),
                    (5u32, table_binding.clone()),
                )),
            );

            radiance_bind_groups.push(bind_group);
        }

        let irradiance_constants = FilteringConstants {
            mip_level: 0.0,
            sample_count: 1024,
            perceptual_roughness: 1.0,
            lod_resolution_bias: 0.0,
            output_size: UVec2::new(32, 32),
            noise_size_bits,
        };

        let mut irradiance_constants_buffer = UniformBuffer::from(irradiance_constants);
        irradiance_constants_buffer.write_buffer(&render_device, &queue);

        let irradiance_map =
            env_map_light
                .diffuse_map
                .texture
                .create_view(&TextureViewDescriptor {
                    dimension: Some(TextureViewDimension::D2Array),
                    ..Default::default()
                });

        let irradiance_bind_group = render_device.create_bind_group(
            Some("irradiance_bind_group"),
            &pipeline_cache.get_bind_group_layout(&layouts.irradiance),
            &BindGroupEntries::with_indices((
                (0u32, &textures.environment_map.default_view),
                (1u32, &samplers.linear),
                (2u32, &irradiance_map),
                (3u32, &irradiance_constants_buffer),
                (4u32, &stbn_texture_view),
                (5u32, table_binding.clone()),
            )),
        );

        let src_view = env_map_light
            .environment_map
            .texture
            .create_view(&TextureViewDescriptor {
                dimension: Some(TextureViewDimension::D2Array),
                ..Default::default()
            });

        let dst_view = create_storage_view(&textures.environment_map.texture, 0, &render_device);

        let copy_bind_group = render_device.create_bind_group(
            Some("copy_bind_group"),
            &pipeline_cache.get_bind_group_layout(&layouts.copy),
            &BindGroupEntries::with_indices(((0, &src_view), (1, &dst_view))),
        );

        let temporal_bind_groups = if use_temporal {
            let scratch = textures
                .specular_scratch
                .as_ref()
                .expect("scratch texture must exist when temporal_blend > 0");
            let history = textures
                .specular_history
                .as_ref()
                .expect("history texture must exist when temporal_blend > 0");
            let alpha = env_map_light.temporal_blend.clamp(0.0, 1.0);
            let t_const = TemporalBlendUniforms {
                alpha,
                _pad: [0.0; 3],
            };
            let mut t_buf = UniformBuffer::from(t_const);
            t_buf.write_buffer(&render_device, &queue);

            let mut t_groups = Vec::with_capacity(num_mips);
            for mip in 0..mip_count {
                let scratch_view = scratch.texture.create_view(&TextureViewDescriptor {
                    dimension: Some(TextureViewDimension::D2Array),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    aspect: TextureAspect::All,
                    ..Default::default()
                });
                let history_view = history.texture.create_view(&TextureViewDescriptor {
                    dimension: Some(TextureViewDimension::D2Array),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    aspect: TextureAspect::All,
                    ..Default::default()
                });
                let out_view =
                    create_storage_view(&env_map_light.specular_map.texture, mip, &render_device);
                let bg = render_device.create_bind_group(
                    Some(format!("temporal_blend_mip_{mip}").as_str()),
                    &pipeline_cache.get_bind_group_layout(&layouts.temporal_blend),
                    &BindGroupEntries::sequential((
                        &scratch_view,
                        &history_view,
                        &out_view,
                        &t_buf,
                    )),
                );
                t_groups.push(bg);
            }
            Some(t_groups)
        } else {
            None
        };

        commands.entity(entity).insert(GeneratorBindGroups {
            quadratic: quadratic_bind_groups,
            radiance: radiance_bind_groups,
            irradiance: irradiance_bind_group,
            copy: copy_bind_group,
            temporal: temporal_bind_groups,
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

    let Some(copy_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.copy) else {
        return;
    };
    let Some(quadratic_pipeline) =
        pipeline_cache.get_compute_pipeline(pipelines.quadratic_downsample)
    else {
        return;
    };

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();

    for (bind_groups, env_map_light) in &query {
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

        // One compute pass for the full mip chain: a pass per mip forces wgpu/backend barriers
        // and shows up as separate labeled passes (and internal prepasses) in tools.
        if !bind_groups.quadratic.is_empty() {
            let mut compute_pass =
                ctx.command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("lightprobe_quadratic_downsample"),
                        timestamp_writes: None,
                    });

            let pass_span =
                diagnostics.pass_span(&mut compute_pass, "lightprobe_quadratic_downsample");

            compute_pass.set_pipeline(quadratic_pipeline);

            let base = env_map_light.environment_map.texture_descriptor.size.width;
            for (dst_mip_minus_one, bg) in bind_groups.quadratic.iter().enumerate() {
                let dst_mip = dst_mip_minus_one + 1;
                compute_pass.set_bind_group(0, bg, &[]);
                let dst_size = base >> dst_mip;
                let wg = dst_size.div_ceil(8);
                compute_pass.dispatch_workgroups(wg, wg, 6);
            }

            pass_span.end(&mut compute_pass);
        }
    }
}

pub fn filtering_system(
    query: Query<(
        &GeneratorBindGroups,
        &IntermediateTextures,
        &RenderEnvironmentMap,
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
    let Some(irradiance_pipeline) = pipeline_cache.get_compute_pipeline(pipelines.irradiance)
    else {
        return;
    };

    let diagnostics = ctx.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let encoder = ctx.command_encoder();

    for (bind_groups, intermediate, env_map_light) in &query {
        let base_size = env_map_light.specular_map.texture_descriptor.size.width;

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("lightprobe_radiance_map"),
                timestamp_writes: None,
            });

            let pass_span = diagnostics.pass_span(&mut compute_pass, "lightprobe_radiance_map");

            compute_pass.set_pipeline(radiance_pipeline);

            for (mip, bind_group) in bind_groups.radiance.iter().enumerate() {
                compute_pass.set_bind_group(0, bind_group, &[]);

                let mip_size = base_size >> mip;
                let workgroup_count = mip_size.div_ceil(8);

                compute_pass.dispatch_workgroups(workgroup_count, workgroup_count, 6);
            }

            pass_span.end(&mut compute_pass);
        }

        if let (Some(temporal_binds), Some(history_tex)) =
            (&bind_groups.temporal, &intermediate.specular_history)
        {
            if let Some(temporal_pipeline) =
                pipeline_cache.get_compute_pipeline(pipelines.temporal_blend)
            {
                {
                    let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("lightprobe_temporal_blend_specular"),
                        timestamp_writes: None,
                    });
                    let pass_span = diagnostics.pass_span(&mut compute_pass, "lightprobe_temporal");
                    compute_pass.set_pipeline(temporal_pipeline);
                    for (mip, temporal_bg) in temporal_binds.iter().enumerate() {
                        compute_pass.set_bind_group(0, temporal_bg, &[]);
                        let mip_size = base_size >> mip;
                        let wg = mip_size.div_ceil(8);
                        compute_pass.dispatch_workgroups(wg, wg, 6);
                    }
                    pass_span.end(&mut compute_pass);
                }

                let mip_count = compute_mip_count(base_size);
                for mip in 0..mip_count {
                    let dim = base_size >> mip;
                    encoder.copy_texture_to_texture(
                        TexelCopyTextureInfo {
                            texture: &env_map_light.specular_map.texture,
                            mip_level: mip,
                            origin: Origin3d::ZERO,
                            aspect: TextureAspect::All,
                        },
                        TexelCopyTextureInfo {
                            texture: &history_tex.texture,
                            mip_level: mip,
                            origin: Origin3d::ZERO,
                            aspect: TextureAspect::All,
                        },
                        Extent3d {
                            width: dim,
                            height: dim,
                            depth_or_array_layers: 6,
                        },
                    );
                }
            }
        }

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("lightprobe_irradiance_map"),
                timestamp_writes: None,
            });

            let irr_span = diagnostics.pass_span(&mut compute_pass, "lightprobe_irradiance_map");

            compute_pass.set_pipeline(irradiance_pipeline);
            compute_pass.set_bind_group(0, &bind_groups.irradiance, &[]);

            compute_pass.dispatch_workgroups(4, 4, 6);

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

        let mip_count = compute_mip_count(base_size);

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

        // Set up for mipmaps (+ copy when temporal filtering writes scratch then blends)
        let mut spec_usage = TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING;
        if filtered_env_map.temporal_blend > 0.0 {
            spec_usage |= TextureUsages::COPY_SRC | TextureUsages::COPY_DST;
        }
        specular.texture_descriptor.usage = spec_usage;
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
            specular_environment_integration: SpecularEnvironmentIntegration::MansonSloan,
        });
    }
}

impl SyncComponent<RenderApp, EnvironmentMapGenerationPlugin> for GeneratedEnvironmentMapLight {
    type Target = RenderEnvironmentMap;
}
