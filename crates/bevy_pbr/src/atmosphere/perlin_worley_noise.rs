//! Tileable 3D Perlin–Worley Noise Generation for Volumetric Clouds
//!
//! Generates two 3D textures (Schneider, "The Real-Time Volumetric Cloudscapes of
//! Horizon Zero Dawn", SIGGRAPH 2015):
//!
//! Shape noise (128^3):
//! - R: Perlin–Worley (remapped billowy Perlin by low-frequency Worley FBM)
//! - G: Worley FBM (base frequency)
//! - B: Worley FBM (2x frequency)
//! - A: Worley FBM (4x frequency)
//!
//! Detail/erosion noise (32^3):
//! - R/G/B: Worley FBM octave groups at increasing frequency

use bevy_asset::load_embedded_asset;
use bevy_ecs::{
    resource::Resource,
    system::{Res, ResMut},
    world::{FromWorld, World},
};
use bevy_math::UVec3;
use bevy_render::{
    render_resource::{binding_types::*, *},
    renderer::{RenderDevice, RenderQueue},
    texture::CachedTexture,
};
use bevy_utils::default;

/// Parameters for controlling Perlin–Worley noise generation.
#[derive(Clone, Copy, ShaderType, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct PerlinWorleyNoiseParams {
    /// Base frequency / tile period (must be > 0).
    pub base_frequency: u32,
    /// Number of octaves for Perlin FBM.
    pub perlin_octaves: u32,
    /// Z animation offset (in texture space).
    pub z_offset: f32,
    /// Padding.
    pub _pad0: f32,
}

impl Default for PerlinWorleyNoiseParams {
    fn default() -> Self {
        Self {
            base_frequency: 4,
            perlin_octaves: 7,
            z_offset: 0.0,
            _pad0: 0.0,
        }
    }
}

/// Size of the 3D noise textures.
#[derive(Clone, Copy)]
pub struct PerlinWorleyTextureSize {
    /// Shape noise resolution (HZD uses 128^3; 8MB at RGBA8).
    pub size: UVec3,
    /// Detail/erosion noise resolution (HZD uses 32^3).
    pub detail_size: UVec3,
}

impl Default for PerlinWorleyTextureSize {
    fn default() -> Self {
        Self {
            size: UVec3::new(128, 128, 128),
            detail_size: UVec3::new(32, 32, 32),
        }
    }
}

#[derive(Resource)]
pub struct PerlinWorleyNoiseBindGroupLayout {
    pub layout: BindGroupLayout,
    pub descriptor: BindGroupLayoutDescriptor,
}

impl FromWorld for PerlinWorleyNoiseBindGroupLayout {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let descriptor = BindGroupLayoutDescriptor::new(
            "perlin_worley_noise_bind_group_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::COMPUTE,
                (
                    (
                        // 3D noise texture storage
                        13,
                        texture_storage_3d(
                            TextureFormat::Rgba8Unorm,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                    (
                        // params uniform buffer
                        14,
                        uniform_buffer::<PerlinWorleyNoiseParams>(false),
                    ),
                ),
            ),
        );

        let layout =
            render_device.create_bind_group_layout(descriptor.label.as_ref(), &descriptor.entries);

        Self { layout, descriptor }
    }
}

#[derive(Resource)]
pub struct PerlinWorleyNoisePipeline {
    pub pipeline: CachedComputePipelineId,
    pub detail_pipeline: CachedComputePipelineId,
}

impl FromWorld for PerlinWorleyNoisePipeline {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let layout = world.resource::<PerlinWorleyNoiseBindGroupLayout>();

        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("perlin_worley_noise_3d_pipeline".into()),
            layout: vec![layout.descriptor.clone()],
            shader: load_embedded_asset!(world, "perlin_worley_noise_3d.wgsl"),
            entry_point: Some("main".into()),
            ..default()
        });

        let detail_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cloud_detail_noise_3d_pipeline".into()),
            layout: vec![layout.descriptor.clone()],
            shader: load_embedded_asset!(world, "perlin_worley_noise_3d.wgsl"),
            entry_point: Some("detail_main".into()),
            ..default()
        });

        Self {
            pipeline,
            detail_pipeline,
        }
    }
}

/// The generated 3D Perlin–Worley shape texture.
#[derive(Resource)]
pub struct PerlinWorleyNoiseTexture {
    pub texture: CachedTexture,
    pub size: UVec3,
}

/// The generated 3D Worley detail/erosion texture.
#[derive(Resource)]
pub struct CloudDetailNoiseTexture {
    pub texture: CachedTexture,
    pub size: UVec3,
}

#[derive(Resource)]
pub struct PerlinWorleyNoiseBindGroup {
    pub bind_group: BindGroup,
    pub detail_bind_group: BindGroup,
}

#[derive(Resource)]
pub struct PerlinWorleyNoiseParamsBuffer {
    pub buffer: Buffer,
    pub detail_buffer: Buffer,
}

/// Tracks if the 3D noise textures have been generated.
#[derive(Resource, Default)]
pub struct PerlinWorleyNoiseGenerated {
    pub shape: bool,
    pub detail: bool,
}

/// Allocates GPU memory directly for the same reason as `init_fbm_noise_texture` in `fbm_noise`.
pub fn init_perlin_worley_noise_texture(
    render_device: Res<RenderDevice>,
    mut commands: bevy_ecs::system::Commands,
) {
    let size = PerlinWorleyTextureSize::default();

    let make_texture = |label: &'static str, size: UVec3| {
        let texture_descriptor = TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: size.z,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D3,
            format: TextureFormat::Rgba8Unorm,
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

    commands.insert_resource(PerlinWorleyNoiseTexture {
        texture: make_texture("perlin_worley_noise_3d_texture", size.size),
        size: size.size,
    });
    commands.insert_resource(CloudDetailNoiseTexture {
        texture: make_texture("cloud_detail_noise_3d_texture", size.detail_size),
        size: size.detail_size,
    });
}

pub fn init_perlin_worley_noise_params_buffer(
    mut commands: bevy_ecs::system::Commands,
    render_device: Res<RenderDevice>,
) {
    let params = PerlinWorleyNoiseParams::default();
    let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("perlin_worley_noise_params_buffer"),
        contents: bytemuck::cast_slice(&[params]),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    // Lower base frequency for the detail texture: it is much smaller (32^3) and
    // tiles at a higher world frequency, so the Worley cells need fewer texels.
    let detail_params = PerlinWorleyNoiseParams {
        base_frequency: 2,
        ..PerlinWorleyNoiseParams::default()
    };
    let detail_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("cloud_detail_noise_params_buffer"),
        contents: bytemuck::cast_slice(&[detail_params]),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    commands.insert_resource(PerlinWorleyNoiseParamsBuffer {
        buffer,
        detail_buffer,
    });
}

pub fn prepare_perlin_worley_noise_bind_group(
    mut commands: bevy_ecs::system::Commands,
    render_device: Res<RenderDevice>,
    layout: Res<PerlinWorleyNoiseBindGroupLayout>,
    texture: Res<PerlinWorleyNoiseTexture>,
    detail_texture: Res<CloudDetailNoiseTexture>,
    params_buffer: Res<PerlinWorleyNoiseParamsBuffer>,
) {
    let bind_group = render_device.create_bind_group(
        "perlin_worley_noise_bind_group",
        &layout.layout,
        &BindGroupEntries::with_indices((
            (13, &texture.texture.default_view),
            (14, params_buffer.buffer.as_entire_binding()),
        )),
    );

    let detail_bind_group = render_device.create_bind_group(
        "cloud_detail_noise_bind_group",
        &layout.layout,
        &BindGroupEntries::with_indices((
            (13, &detail_texture.texture.default_view),
            (14, params_buffer.detail_buffer.as_entire_binding()),
        )),
    );

    commands.insert_resource(PerlinWorleyNoiseBindGroup {
        bind_group,
        detail_bind_group,
    });
}

pub fn get_3d_dispatch_size(texture_size: UVec3) -> UVec3 {
    const WORKGROUP_SIZE: u32 = 4;
    UVec3::new(
        texture_size.x.div_ceil(WORKGROUP_SIZE),
        texture_size.y.div_ceil(WORKGROUP_SIZE),
        texture_size.z.div_ceil(WORKGROUP_SIZE),
    )
}

/// Generate the Perlin–Worley shape and Worley detail 3D textures once
/// (each as soon as its pipeline is ready).
pub fn generate_perlin_worley_noise_once(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<PerlinWorleyNoisePipeline>,
    bind_group: Option<Res<PerlinWorleyNoiseBindGroup>>,
    texture: Res<PerlinWorleyNoiseTexture>,
    detail_texture: Res<CloudDetailNoiseTexture>,
    mut generated: ResMut<PerlinWorleyNoiseGenerated>,
) {
    if generated.shape && generated.detail {
        return;
    }

    let Some(bind_group) = bind_group else {
        return;
    };

    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("perlin_worley_noise_generation"),
    });
    let mut any_dispatched = false;

    if !generated.shape
        && let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline)
    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("perlin_worley_noise_3d_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group.bind_group, &[]);

        let dispatch_size = get_3d_dispatch_size(texture.size);
        compute_pass.dispatch_workgroups(dispatch_size.x, dispatch_size.y, dispatch_size.z);

        generated.shape = true;
        any_dispatched = true;
    }

    if !generated.detail
        && let Some(compute_pipeline) =
            pipeline_cache.get_compute_pipeline(pipeline.detail_pipeline)
    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloud_detail_noise_3d_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group.detail_bind_group, &[]);

        let dispatch_size = get_3d_dispatch_size(detail_texture.size);
        compute_pass.dispatch_workgroups(dispatch_size.x, dispatch_size.y, dispatch_size.z);

        generated.detail = true;
        any_dispatched = true;
    }

    if any_dispatched {
        let command_buffer = encoder.finish();
        render_queue.submit([command_buffer]);
    }
}
