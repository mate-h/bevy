//! 3D FBM Noise Generation for Atmospheric Effects
//!
//! This module generates a 3D texture filled with Fractional Brownian Motion (FBM) noise,
//! which can be used for atmospheric effects like clouds, fog, and volumetric effects.

use bevy_asset::load_embedded_asset;
use bevy_ecs::{
    resource::Resource,
    system::{Res, ResMut},
    world::{FromWorld, World},
};
use bevy_math::UVec2;
use bevy_render::{
    render_resource::{binding_types::*, *},
    renderer::{RenderDevice, RenderQueue},
    texture::CachedTexture,
};
use bevy_utils::default;

/// Parameters for controlling FBM noise generation
#[derive(Clone, Copy, ShaderType, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct FbmNoiseParams {
    /// Number of octaves for the FBM noise (more octaves = more detail)
    pub octaves: u32,
    /// Base frequency of the noise
    pub frequency: f32,
    /// Base amplitude of the noise
    pub amplitude: f32,
    /// Frequency multiplier for each octave (typically 2.0)
    pub lacunarity: f32,
    /// Amplitude multiplier for each octave (typically 0.5)
    pub persistence: f32,
}

impl Default for FbmNoiseParams {
    fn default() -> Self {
        Self {
            octaves: 4,
            frequency: 1.0,
            amplitude: 1.0,
            lacunarity: 2.0,
            persistence: 0.5,
        }
    }
}

/// Size of the 2D noise textures
#[derive(Clone, Copy)]
pub struct NoiseTextureSize {
    /// Cloud macro map (coverage/type/detail) resolution.
    pub size: UVec2,
    /// Curl noise texture resolution (HZD uses 128x128).
    pub curl_size: UVec2,
}

impl Default for NoiseTextureSize {
    fn default() -> Self {
        Self {
            // Higher resolution reduces visible repetition in cloud coverage
            // and carries more octaves of detail. Still cheap: generated once
            // at startup.
            size: UVec2::new(512, 512),
            curl_size: UVec2::new(128, 128),
        }
    }
}

/// Resource containing the bind group layout for the FBM noise pass
#[derive(Resource)]
pub struct FbmNoiseBindGroupLayout {
    pub layout: BindGroupLayout,
    pub descriptor: BindGroupLayoutDescriptor,
}

impl FromWorld for FbmNoiseBindGroupLayout {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let descriptor = BindGroupLayoutDescriptor::new(
            "fbm_noise_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (
                        // 2D noise texture storage
                        13,
                        texture_storage_2d(
                            // Packed cloud noise (coverage + type controls + detail).
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                    (
                        // FBM parameters uniform buffer
                        14,
                        uniform_buffer::<FbmNoiseParams>(false),
                    ),
                ),
            ),
        );

        let layout =
            render_device.create_bind_group_layout(descriptor.label.as_ref(), &descriptor.entries);

        Self { layout, descriptor }
    }
}

/// Resource containing the compute pipelines for 2D cloud noise generation
#[derive(Resource)]
pub struct FbmNoisePipeline {
    pub pipeline: CachedComputePipelineId,
    pub curl_pipeline: CachedComputePipelineId,
}

impl FromWorld for FbmNoisePipeline {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let layout = world.resource::<FbmNoiseBindGroupLayout>();

        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("fbm_noise_2d_pipeline".into()),
            layout: vec![layout.descriptor.clone()],
            shader: load_embedded_asset!(world, "fbm_noise_3d.wgsl"),
            entry_point: Some("main".into()),
            ..default()
        });

        let curl_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("curl_noise_2d_pipeline".into()),
            layout: vec![layout.descriptor.clone()],
            shader: load_embedded_asset!(world, "fbm_noise_3d.wgsl"),
            entry_point: Some("curl_main".into()),
            ..default()
        });

        Self {
            pipeline,
            curl_pipeline,
        }
    }
}

/// Resource containing the generated 2D noise texture
#[derive(Resource)]
pub struct FbmNoiseTexture {
    pub texture: CachedTexture,
    pub size: UVec2,
}

/// Resource containing the generated 2D curl noise texture (signed RGB).
#[derive(Resource)]
pub struct CurlNoiseTexture {
    pub texture: CachedTexture,
    pub size: UVec2,
}

/// Resource containing the bind groups for the noise generation passes
#[derive(Resource)]
pub struct FbmNoiseBindGroup {
    pub bind_group: BindGroup,
    pub curl_bind_group: BindGroup,
}

/// Resource containing the uniform buffer for FBM parameters
#[derive(Resource)]
pub struct FbmNoiseParamsBuffer {
    pub buffer: Buffer,
}

/// System to initialize the FBM noise texture.
///
/// Allocates GPU memory directly instead of the render [`TextureCache`](bevy_render::texture::TextureCache)
/// so this `RenderStartup` system does not need to run after `init_gpu_resource::<TextureCache>`.
pub fn init_fbm_noise_texture(
    render_device: Res<RenderDevice>,
    mut commands: bevy_ecs::system::Commands,
) {
    let size = NoiseTextureSize::default();

    let make_texture = |label: &'static str, size: UVec2| {
        let texture_descriptor = TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };

        let texture = render_device.create_texture(&texture_descriptor);
        let default_view = texture.create_view(&TextureViewDescriptor::default());
        CachedTexture {
            texture,
            default_view,
        }
    };

    commands.insert_resource(FbmNoiseTexture {
        // Packed cloud noise (coverage + type controls + detail).
        texture: make_texture("fbm_noise_2d_texture", size.size),
        size: size.size,
    });
    commands.insert_resource(CurlNoiseTexture {
        // Signed curl noise (RGB) for detail distortion; rgba16float keeps the sign.
        texture: make_texture("curl_noise_2d_texture", size.curl_size),
        size: size.curl_size,
    });
}

/// System to prepare the FBM noise bind groups
pub fn prepare_fbm_noise_bind_group(
    mut commands: bevy_ecs::system::Commands,
    render_device: Res<RenderDevice>,
    layout: Res<FbmNoiseBindGroupLayout>,
    texture: Res<FbmNoiseTexture>,
    curl_texture: Res<CurlNoiseTexture>,
    params_buffer: Res<FbmNoiseParamsBuffer>,
) {
    let bind_group = render_device.create_bind_group(
        "fbm_noise_bind_group",
        &layout.layout,
        &BindGroupEntries::with_indices((
            (13, &texture.texture.default_view),
            (14, params_buffer.buffer.as_entire_binding()),
        )),
    );

    let curl_bind_group = render_device.create_bind_group(
        "curl_noise_bind_group",
        &layout.layout,
        &BindGroupEntries::with_indices((
            (13, &curl_texture.texture.default_view),
            (14, params_buffer.buffer.as_entire_binding()),
        )),
    );

    commands.insert_resource(FbmNoiseBindGroup {
        bind_group,
        curl_bind_group,
    });
}

/// System to initialize the FBM noise parameters buffer
pub fn init_fbm_noise_params_buffer(
    mut commands: bevy_ecs::system::Commands,
    render_device: Res<RenderDevice>,
) {
    let params = FbmNoiseParams::default();
    let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("fbm_noise_params_buffer"),
        contents: bytemuck::cast_slice(&[params]),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    commands.insert_resource(FbmNoiseParamsBuffer { buffer });
}

/// Returns the dispatch workgroup counts for the 2D noise texture based on its size
pub fn get_noise_dispatch_size(texture_size: UVec2) -> UVec2 {
    const WORKGROUP_SIZE: u32 = 8;
    UVec2::new(
        texture_size.x.div_ceil(WORKGROUP_SIZE),
        texture_size.y.div_ceil(WORKGROUP_SIZE),
    )
}

/// Resource to track if the 2D noise textures have been generated
#[derive(Resource, Default)]
pub struct NoiseGenerated {
    pub map: bool,
    pub curl: bool,
}

/// System to generate the 2D cloud noise textures (each runs once)
pub fn generate_fbm_noise_once(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<FbmNoisePipeline>,
    bind_group: Option<Res<FbmNoiseBindGroup>>,
    texture: Res<FbmNoiseTexture>,
    curl_texture: Res<CurlNoiseTexture>,
    mut noise_generated: ResMut<NoiseGenerated>,
) {
    if noise_generated.map && noise_generated.curl {
        return;
    }

    let Some(bind_group) = bind_group else {
        return;
    };

    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("fbm_noise_generation"),
    });
    let mut any_dispatched = false;

    if !noise_generated.map
        && let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline)
    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("fbm_noise_2d_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group.bind_group, &[]);

        let dispatch_size = get_noise_dispatch_size(texture.size);
        compute_pass.dispatch_workgroups(dispatch_size.x, dispatch_size.y, 1);

        noise_generated.map = true;
        any_dispatched = true;
    }

    if !noise_generated.curl
        && let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.curl_pipeline)
    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("curl_noise_2d_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group.curl_bind_group, &[]);

        let dispatch_size = get_noise_dispatch_size(curl_texture.size);
        compute_pass.dispatch_workgroups(dispatch_size.x, dispatch_size.y, 1);

        noise_generated.curl = true;
        any_dispatched = true;
    }

    if any_dispatched {
        let command_buffer = encoder.finish();
        render_queue.submit([command_buffer]);
    }
}
