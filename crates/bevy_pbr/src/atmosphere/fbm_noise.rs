//! 3D FBM Noise Generation for Atmospheric Effects
//!
//! This module generates a 3D texture filled with Fractional Brownian Motion (FBM) noise,
//! which can be used for atmospheric effects like clouds, fog, and volumetric effects.

use bevy_asset::{load_embedded_asset, AssetId, Assets};
use bevy_image::Image;
use bevy_ecs::{
    query::With,
    resource::Resource,
    system::{Local, Query, Res, ResMut},
    world::{FromWorld, World},
};
use bevy_math::{UVec2, UVec3};
use bevy_render::{
    render_resource::{binding_types::*, *},
    renderer::{RenderAdapter, RenderDevice, RenderQueue},
    settings::WgpuFeatures,
    texture::CachedTexture,
    Extract,
};
use bevy_utils::default;
use tracing::warn;

use super::{
    cloud_profile::{
        f32_to_f16, physical_profile_texels, CloudProfiles, CLOUD_PROFILE_HEIGHT_RES,
        CLOUD_PROFILE_TYPE_RES,
    },
    perlin_worley_noise::{PerlinWorleyNoiseGenerated, PerlinWorleyNoiseTexture},
    VolumetricCloudLayer,
};

/// Storage format of the cirrus generating field: R16Float is a storage format only with the
/// adapter-specific format features, not in WebGPU's guaranteed set; elsewhere the field falls
/// back to Rgba16Float (a core storage format), which the generator declares through
/// `CIRRUS_STORAGE_RGBA16`. R16Float is kept where possible (2.8 MB with mips vs 11 MB).
#[derive(Resource, Clone, Copy)]
pub struct CloudNoiseFormats {
    pub cirrus: TextureFormat,
}

impl CloudNoiseFormats {
    pub fn cirrus_shader_defs(&self) -> Vec<bevy_shader::ShaderDefVal> {
        if self.cirrus == TextureFormat::Rgba16Float {
            vec!["CIRRUS_STORAGE_RGBA16".into()]
        } else {
            Vec::new()
        }
    }
}

impl FromWorld for CloudNoiseFormats {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let adapter = world.resource::<RenderAdapter>();
        let adapter_formats = device
            .features()
            .contains(WgpuFeatures::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES);
        let storage = |format: TextureFormat| {
            if adapter_formats
                && adapter
                    .get_texture_format_features(format)
                    .allowed_usages
                    .contains(TextureUsages::STORAGE_BINDING)
            {
                format
            } else {
                TextureFormat::Rgba16Float
            }
        };
        Self {
            cirrus: storage(TextureFormat::R16Float),
        }
    }
}

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
    /// Cloud life-cycle phase, `evolution_time / lifetime + 0.5` ([`crate::VolumetricCloudLayer`]).
    pub life_t: f32,
    /// 1 when the life cycle is on (`lifetime > 0`).
    pub life_enabled: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

impl Default for FbmNoiseParams {
    fn default() -> Self {
        Self {
            octaves: 4,
            frequency: 1.0,
            amplitude: 1.0,
            lacunarity: 2.0,
            persistence: 0.5,
            life_t: 0.5,
            life_enabled: 0,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

/// Size of the 2D noise texture
#[derive(Clone, Copy)]
pub struct NoiseTextureSize {
    pub size: UVec2,
}

impl Default for NoiseTextureSize {
    fn default() -> Self {
        Self {
            // Procedural weather map (design.md §1.3 / §3.2): 512² RGBA.
            // R coverage, G type, B density scale, A spare.
            size: UVec2::new(512, 512),
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
                            // Weather map (coverage, type, density scale, spare).
                            TextureFormat::Rgba16Float,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                    (
                        // FBM parameters uniform buffer
                        14,
                        uniform_buffer::<FbmNoiseParams>(false),
                    ),
                    (
                        // Cirrus generating field (`cirrus` entry point).
                        15,
                        texture_storage_2d(
                            world.resource::<CloudNoiseFormats>().cirrus,
                            StorageTextureAccess::WriteOnly,
                        ),
                    ),
                ),
            ),
        );

        let layout =
            render_device.create_bind_group_layout(descriptor.label.as_ref(), &descriptor.entries);

        Self { layout, descriptor }
    }
}

/// Resource containing the compute pipeline for FBM noise generation
#[derive(Resource)]
pub struct FbmNoisePipeline {
    pub pipeline: CachedComputePipelineId,
    pub cirrus_pipeline: CachedComputePipelineId,
}

impl FromWorld for FbmNoisePipeline {
    fn from_world(world: &mut World) -> Self {
        let pipeline_cache = world.resource::<PipelineCache>();
        let layout = world.resource::<FbmNoiseBindGroupLayout>();

        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("fbm_noise_2d_pipeline".into()),
            layout: vec![layout.descriptor.clone()],
            shader: load_embedded_asset!(world, "fbm_noise_3d.wesl"),
            entry_point: Some("main".into()),
            ..default()
        });
        let cirrus_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("cirrus_field_pipeline".into()),
            layout: vec![layout.descriptor.clone()],
            shader: load_embedded_asset!(world, "fbm_noise_3d.wesl"),
            entry_point: Some("cirrus".into()),
            shader_defs: world.resource::<CloudNoiseFormats>().cirrus_shader_defs(),
            ..default()
        });

        Self { pipeline, cirrus_pipeline }
    }
}

/// Resource containing the generated 2D noise texture
#[derive(Resource)]
pub struct FbmNoiseTexture {
    pub texture: CachedTexture,
    pub size: UVec2,
    /// Cirrus generating field (R16F, unit Gaussian, -5/3 spectrum; tile = cirrus outer scale),
    /// with a mip chain; each mip is generated as the band-limited field (its resolvable octaves).
    pub cirrus: CachedTexture,
    /// Single-mip views of [`FbmNoiseTexture::cirrus`] for generation.
    pub cirrus_mips: Vec<TextureView>,
    /// Vertical profile lookups (Rgba16Float, height x type; `cloud_profile.rs`).
    pub profile: CachedTexture,
}

/// Size of the cirrus generating-field texture.
pub const CIRRUS_TEXTURE_SIZE: u32 = 1024;

/// Resource containing the bind group for the noise generation pass
#[derive(Resource)]
pub struct FbmNoiseBindGroup {
    pub bind_group: BindGroup,
    /// One per cirrus mip (binding 15 = that mip).
    pub cirrus_bind_groups: Vec<BindGroup>,
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
    render_queue: Res<RenderQueue>,
    formats: Res<CloudNoiseFormats>,
    mut commands: bevy_ecs::system::Commands,
) {
    let size = NoiseTextureSize::default();

    let texture_descriptor = TextureDescriptor {
        label: Some("cloud_weather_map_texture"),
        size: Extent3d {
            width: size.size.x,
            height: size.size.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        // Weather map (coverage, type, density scale, spare).
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    };

    let texture = render_device.create_texture(&texture_descriptor);
    let default_view = texture.create_view(&TextureViewDescriptor::default());

    let cirrus_texture = render_device.create_texture(&TextureDescriptor {
        label: Some("cirrus_generating_field_texture"),
        size: Extent3d {
            width: CIRRUS_TEXTURE_SIZE,
            height: CIRRUS_TEXTURE_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: CIRRUS_TEXTURE_SIZE.ilog2() + 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: formats.cirrus,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let cirrus_view = cirrus_texture.create_view(&TextureViewDescriptor::default());
    let cirrus_mips = (0..CIRRUS_TEXTURE_SIZE.ilog2() + 1)
        .map(|mip| {
            cirrus_texture.create_view(&TextureViewDescriptor {
                base_mip_level: mip,
                mip_level_count: Some(1),
                ..TextureViewDescriptor::default()
            })
        })
        .collect();

    let profile_texture = render_device.create_texture(&TextureDescriptor {
        label: Some("cloud_profile_lut_texture"),
        size: CLOUD_PROFILE_TEXTURE_EXTENT,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_DST
            | TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    write_cloud_profile(&render_queue, &profile_texture, physical_profile_texels());
    // Noise quantile rows: identity (threshold 1 - profile, as Nubis) until measured.
    let identity: Vec<u8> = (0..CLOUD_NOISE_QUANTILE_ROWS * CLOUD_PROFILE_HEIGHT_RES as u32)
        .map(|i| i as usize % CLOUD_PROFILE_HEIGHT_RES)
        .flat_map(|i| {
            let q = i as f32 / (CLOUD_PROFILE_HEIGHT_RES - 1) as f32;
            [q, 0.0, 0.0, 1.0].map(f32_to_f16)
        })
        .flat_map(u16::to_le_bytes)
        .collect();
    render_queue.write_texture(
        TexelCopyTextureInfo {
            texture: &profile_texture,
            mip_level: 0,
            origin: Origin3d {
                x: 0,
                y: CLOUD_PROFILE_TYPE_RES as u32,
                z: 0,
            },
            aspect: TextureAspect::All,
        },
        &identity,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(8 * CLOUD_PROFILE_HEIGHT_RES as u32),
            rows_per_image: None,
        },
        Extent3d {
            width: CLOUD_PROFILE_HEIGHT_RES as u32,
            height: CLOUD_NOISE_QUANTILE_ROWS,
            depth_or_array_layers: 1,
        },
    );
    let profile_view = profile_texture.create_view(&TextureViewDescriptor::default());

    commands.insert_resource(FbmNoiseTexture {
        texture: CachedTexture {
            texture,
            default_view,
        },
        size: size.size,
        cirrus: CachedTexture {
            texture: cirrus_texture,
            default_view: cirrus_view,
        },
        cirrus_mips,
        profile: CachedTexture {
            texture: profile_texture,
            default_view: profile_view,
        },
    });
}

/// The profile rows (one per cloud type).
const CLOUD_PROFILE_EXTENT: Extent3d = Extent3d {
    width: CLOUD_PROFILE_HEIGHT_RES as u32,
    height: CLOUD_PROFILE_TYPE_RES as u32,
    depth_or_array_layers: 1,
};
/// Noise quantile rows after the profile rows: one per shape-noise mip (128^3 -> 8 levels; must
/// match `clouds.wesl` and `cloud_noise_quantiles.wesl`).
const CLOUD_NOISE_QUANTILE_ROWS: u32 = 8;
/// The profile rows plus the noise quantile rows.
const CLOUD_PROFILE_TEXTURE_EXTENT: Extent3d = Extent3d {
    width: CLOUD_PROFILE_HEIGHT_RES as u32,
    height: CLOUD_PROFILE_TYPE_RES as u32 + CLOUD_NOISE_QUANTILE_ROWS,
    depth_or_array_layers: 1,
};

fn write_cloud_profile(queue: &RenderQueue, texture: &Texture, rgba16f: &[u8]) {
    queue.write_texture(
        texture.as_image_copy(),
        rgba16f,
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(8 * CLOUD_PROFILE_HEIGHT_RES as u32),
            rows_per_image: None,
        },
        CLOUD_PROFILE_EXTENT,
    );
}

/// Profile lookup texels (Rgba16Float) chosen in the extract schedule, uploaded by
/// [`upload_cloud_profiles`].
#[derive(Resource, Default)]
pub struct PendingCloudProfiles(pub(crate) Option<Vec<u8>>);

/// What the lookup holds.
#[derive(PartialEq)]
pub(crate) enum ProfileSource {
    Physical,
    Image(AssetId<Image>),
    Nubis(AssetId<Image>),
}

/// Resamples a Nubis-style vertical profile gradient (height up, type right, any format) into the
/// lookup rows: u = height in the layer (0 bottom .. 1 top), R = the image value, G = B = A = 1.
fn nubis_lookup_texels(image: &Image) -> Option<Vec<u8>> {
    let (w, h) = (image.width(), image.height());
    if w == 0 || h == 0 {
        return None;
    }
    let mut out = Vec::with_capacity(8 * CLOUD_PROFILE_HEIGHT_RES * CLOUD_PROFILE_TYPE_RES);
    for j in 0..CLOUD_PROFILE_TYPE_RES {
        let x = ((j as f32 + 0.5) / CLOUD_PROFILE_TYPE_RES as f32 * w as f32) as u32;
        for i in 0..CLOUD_PROFILE_HEIGHT_RES {
            // Texel i at height i / (N - 1); the image's top row is the layer top.
            let fy = (1.0 - i as f32 / (CLOUD_PROFILE_HEIGHT_RES - 1) as f32) * (h - 1) as f32;
            let (y0, t) = (fy.floor() as u32, fy.fract());
            let y1 = (y0 + 1).min(h - 1);
            let value = |y| image.get_color_at(x.min(w - 1), y).ok().map(|c| c.to_linear().red);
            let v = value(y0)? * (1.0 - t) + value(y1)? * t;
            out.extend([v.clamp(0.0, 1.0), 1.0, 1.0, 1.0].map(f32_to_f16).iter().flat_map(|h| h.to_le_bytes()));
        }
    }
    Some(out)
}

/// Picks the lookups of the first cloud layer's [`CloudProfiles`] and queues an upload when they
/// change: a loaded image of the lookup format, else the physical set (the analytic envelope is
/// evaluated in the shader and leaves the lookups unused). One set for all atmospheres, like the
/// weather map.
pub(crate) fn extract_cloud_profiles(
    layers: Extract<Query<&CloudProfiles, With<VolumetricCloudLayer>>>,
    images: Extract<Res<Assets<Image>>>,
    mut last: Local<Option<ProfileSource>>,
    mut pending: ResMut<PendingCloudProfiles>,
) {
    let Some(profiles) = layers.iter().next() else {
        return;
    };
    let expected = (
        TextureFormat::Rgba16Float,
        CLOUD_PROFILE_EXTENT.width,
        CLOUD_PROFILE_EXTENT.height,
    );
    if let CloudProfiles::NubisLookup(handle) = profiles {
        let source = ProfileSource::Nubis(handle.id());
        if last.as_ref() == Some(&source) {
            return;
        }
        if let Some(texels) = images.get(handle).and_then(nubis_lookup_texels) {
            pending.0 = Some(texels);
            *last = Some(source);
        }
        return;
    }
    let image = match profiles {
        CloudProfiles::Image(handle) => images.get(handle).map(|image| (handle.id(), image)),
        _ => None,
    };
    let usable = image.filter(|(_, image)| {
        let d = &image.texture_descriptor;
        (d.format, d.size.width, d.size.height) == expected && image.data.is_some()
    });
    let source = usable.map_or(ProfileSource::Physical, |(id, _)| ProfileSource::Image(id));
    if last.as_ref() == Some(&source) {
        return;
    }
    if let (Some((_, image)), None) = (image, usable) {
        let d = &image.texture_descriptor;
        warn!(
            "CloudProfiles image is {:?} {}x{}, expected Rgba16Float {}x{}: using the physical profiles",
            d.format, d.size.width, d.size.height, expected.1, expected.2
        );
    }
    pending.0 = Some(match usable {
        Some((_, image)) => image.data.clone().unwrap_or_default(),
        None => physical_profile_texels().to_vec(),
    });
    *last = Some(source);
}

pub fn upload_cloud_profiles(
    mut pending: ResMut<PendingCloudProfiles>,
    texture: Option<Res<FbmNoiseTexture>>,
    queue: Res<RenderQueue>,
) {
    if let (Some(texture), Some(texels)) = (texture, pending.0.take()) {
        write_cloud_profile(&queue, &texture.profile.texture, &texels);
    }
}

/// Measures the shape-noise distribution of each noise mip once the 3D noise exists and writes
/// its quantiles into the last rows of the profile texture (`cloud_noise_quantiles.wesl`), so the erosion threshold
/// at a dimensional profile p is exceeded with probability p (the profile is the cloud fraction).
#[derive(Resource)]
pub struct CloudNoiseQuantilePipeline {
    layout: BindGroupLayoutDescriptor,
    histogram: CachedComputePipelineId,
    quantiles: CachedComputePipelineId,
}

impl FromWorld for CloudNoiseQuantilePipeline {
    fn from_world(world: &mut World) -> Self {
        let layout = BindGroupLayoutDescriptor::new(
            "cloud_noise_quantiles_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    texture_3d(TextureSampleType::Float { filterable: false }),
                    storage_buffer_sized(false, None),
                    texture_storage_2d(TextureFormat::Rgba16Float, StorageTextureAccess::WriteOnly),
                ),
            ),
        );
        let shader = load_embedded_asset!(world, "cloud_noise_quantiles.wesl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = |entry: &'static str| {
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(format!("cloud_noise_quantiles_{entry}").into()),
                layout: vec![layout.clone()],
                shader: shader.clone(),
                entry_point: Some(entry.into()),
                ..default()
            })
        };
        Self {
            histogram: pipeline("histogram"),
            quantiles: pipeline("quantiles"),
            layout,
        }
    }
}

/// Runs [`CloudNoiseQuantilePipeline`] once, after the shape noise is generated.
pub fn compute_cloud_noise_quantiles(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<CloudNoiseQuantilePipeline>,
    profile: Option<Res<FbmNoiseTexture>>,
    noise: Option<Res<PerlinWorleyNoiseTexture>>,
    generated: Res<PerlinWorleyNoiseGenerated>,
    mut done: Local<bool>,
) {
    let (Some(profile), Some(noise)) = (profile, noise) else {
        return;
    };
    if *done || generated.0 < noise.mip_level_count {
        return;
    }
    let (Some(histogram), Some(quantiles)) = (
        pipeline_cache.get_compute_pipeline(pipeline.histogram),
        pipeline_cache.get_compute_pipeline(pipeline.quantiles),
    ) else {
        return;
    };
    // 256 bins per mip, zero-initialized at creation.
    let bins = render_device.create_buffer(&BufferDescriptor {
        label: Some("cloud_noise_histogram"),
        size: 256 * 4 * CLOUD_NOISE_QUANTILE_ROWS as u64,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let bind_group = render_device.create_bind_group(
        "cloud_noise_quantiles_bind_group",
        &pipeline_cache.get_bind_group_layout(&pipeline.layout),
        &BindGroupEntries::sequential((
            &noise.texture.default_view,
            bins.as_entire_binding(),
            &profile.profile.default_view,
        )),
    );
    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("cloud_noise_quantiles"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cloud_noise_quantiles"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_pipeline(histogram);
        let n = (noise.size + UVec3::splat(3)) / 4;
        pass.dispatch_workgroups(n.x, n.y, n.z);
        pass.set_pipeline(quantiles);
        pass.dispatch_workgroups(1, 1, 1);
    }
    render_queue.submit([encoder.finish()]);
    *done = true;
}

/// System to prepare the FBM noise bind group
pub fn prepare_fbm_noise_bind_group(
    mut commands: bevy_ecs::system::Commands,
    render_device: Res<RenderDevice>,
    layout: Res<FbmNoiseBindGroupLayout>,
    texture: Res<FbmNoiseTexture>,
    params_buffer: Res<FbmNoiseParamsBuffer>,
) {
    let bind_group = render_device.create_bind_group(
        "fbm_noise_bind_group",
        &layout.layout,
        &BindGroupEntries::with_indices((
            (13, &texture.texture.default_view),
            (14, params_buffer.buffer.as_entire_binding()),
            (15, &texture.cirrus_mips[0]),
        )),
    );
    let cirrus_bind_groups = texture
        .cirrus_mips
        .iter()
        .map(|view| {
            render_device.create_bind_group(
                "cirrus_field_bind_group",
                &layout.layout,
                &BindGroupEntries::with_indices((
                    (13, &texture.texture.default_view),
                    (14, params_buffer.buffer.as_entire_binding()),
                    (15, view),
                )),
            )
        })
        .collect();

    commands.insert_resource(FbmNoiseBindGroup {
        bind_group,
        cirrus_bind_groups,
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

/// Resource to track if noise has been generated
#[derive(Resource, Default)]
pub struct NoiseGenerated(pub bool);

/// System to generate the weather map: once, and again whenever the cloud life-cycle clock
/// ([`crate::VolumetricCloudLayer::evolution_time`]) advances (the map holds the blended generations).
pub fn generate_fbm_noise_once(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    pipeline: Res<FbmNoisePipeline>,
    bind_group: Option<Res<FbmNoiseBindGroup>>,
    texture: Res<FbmNoiseTexture>,
    params_buffer: Option<Res<FbmNoiseParamsBuffer>>,
    layers: Query<&crate::GpuCloudLayers>,
    mut last_life_t: Local<Option<f32>>,
    mut cirrus_done: Local<bool>,
    mut noise_generated: ResMut<NoiseGenerated>,
) {
    let life = layers
        .iter()
        .next()
        .filter(|layer| layer.lifetime > 0.0)
        .map(|layer| layer.evolution_time / layer.lifetime + 0.5);
    // Re-bake only when the phase moved by > 1/2000 of a cycle (a weight change far below
    // visibility; at 1x with a 20 min life time that is every ~0.6 s).
    let unchanged = match (life, *last_life_t) {
        (Some(t), Some(last)) => (t - last).abs() < 5e-4,
        (None, _) => true,
        _ => false,
    };
    if noise_generated.0 && unchanged && *cirrus_done {
        return;
    }
    if let Some(params_buffer) = params_buffer.as_ref() {
        let params = FbmNoiseParams {
            life_t: life.unwrap_or(0.5),
            life_enabled: life.is_some() as u32,
            ..FbmNoiseParams::default()
        };
        render_queue.write_buffer(&params_buffer.buffer, 0, bytemuck::cast_slice(&[params]));
    }

    let Some(bind_group) = bind_group else {
        return;
    };

    let Some(compute_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.pipeline) else {
        return;
    };

    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("fbm_noise_generation"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("fbm_noise_2d_pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group.bind_group, &[]);

        let dispatch_size = get_noise_dispatch_size(texture.size);
        compute_pass.dispatch_workgroups(dispatch_size.x, dispatch_size.y, 1);

        // The cirrus generating field does not evolve: generate it once.
        if !*cirrus_done {
            if let Some(cirrus_pipeline) = pipeline_cache.get_compute_pipeline(pipeline.cirrus_pipeline) {
                compute_pass.set_pipeline(cirrus_pipeline);
                for (mip, mip_bind_group) in bind_group.cirrus_bind_groups.iter().enumerate() {
                    compute_pass.set_bind_group(0, mip_bind_group, &[]);
                    let n = get_noise_dispatch_size(UVec2::splat((CIRRUS_TEXTURE_SIZE >> mip).max(1)));
                    compute_pass.dispatch_workgroups(n.x, n.y, 1);
                }
                *cirrus_done = true;
            }
        }
    }

    let command_buffer = encoder.finish();
    render_queue.submit([command_buffer]);

    noise_generated.0 = true;
    *last_life_t = life;
}
