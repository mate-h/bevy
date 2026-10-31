//! Main-world side of [`EnvironmentMapSourceExtraction`].
//!
//! The generator writes the extracted sources of a [`GeneratedEnvironmentMapLight`]
//! into a small storage buffer (see `environment_sources.wesl`). That buffer is read
//! back every frame, and the sources drive [`DirectionalLight`]s with an angular
//! size, matched to the previous frame's lights by direction and smoothed. With
//! [`EnvironmentMapSourceOutput::Shading`] there is no readback: the render world
//! copies the buffer into the view's `environment_map_sources` binding instead.

use bevy_asset::{Assets, Handle, RenderAssetUsages};
use bevy_color::{Color, ColorToComponents, LinearRgba};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    observer::On,
    query::{With, Without},
    resource::Resource,
    system::{Commands, Query, ResMut},
};
use bevy_light::{
    DirectionalLight, EnvironmentMapSourceExtraction, EnvironmentMapSourceLight,
    EnvironmentMapSourceOutput, GeneratedEnvironmentMapLight,
};
use bevy_math::Vec3;
use bevy_render::{
    gpu_readback::{Readback, ReadbackComplete},
    render_resource::{Buffer, BufferDescriptor, BufferUsages},
    renderer::RenderDevice,
    storage::ShaderBuffer,
};
use bevy_transform::components::Transform;

/// `Sources` in `environment_sources.wesl`: a 16-byte header and 4 sources of 32 bytes.
pub const SOURCES_BYTES: u64 = 16 + 4 * 32;

/// Zeroed sources for every binding slot, bound to views without sources of their
/// own (`environment_map_sources` in `mesh_view_bindings.wesl`).
#[derive(Resource)]
pub struct EnvironmentMapSourcesFallback(pub Buffer);

impl EnvironmentMapSourcesFallback {
    pub fn new(render_device: &RenderDevice) -> Self {
        Self(render_device.create_buffer(&BufferDescriptor {
            label: Some("environment_map_sources_fallback"),
            size: super::MAX_VIEW_LIGHT_PROBES as u64 * SOURCES_BYTES,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        }))
    }
}

/// A view's sources, one [`SOURCES_BYTES`] slot per environment map binding index.
#[derive(Component)]
pub struct ViewEnvironmentMapSources(pub Buffer);

/// Two lights whose directions are closer than this are the same source.
const MATCH_COS: f32 = 0.94; // ~20°

/// The GPU buffer the generator writes extracted sources into.
#[derive(Component)]
pub struct EnvironmentMapSourceBuffer {
    pub buffer: Handle<ShaderBuffer>,
    readback: Entity,
}

/// The latest sources read back from the GPU.
#[derive(Component, Default)]
pub struct EnvironmentMapSourceState {
    pub sources: Option<Vec<ExtractedSource>>,
}

/// One extracted source, in environment map space and radiance units.
#[derive(Clone, Copy, Debug)]
pub struct ExtractedSource {
    pub direction: Vec3,
    pub angular_size: f32,
    pub illuminance: Vec3,
    pub isotropy: f32,
}

#[derive(Component)]
struct SourceReadbackOwner(Entity);

pub fn prepare_environment_map_source_extraction(
    mut commands: Commands,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    new: Query<(Entity, &EnvironmentMapSourceExtraction), Without<EnvironmentMapSourceBuffer>>,
) {
    for (entity, settings) in &new {
        let mut buffer = ShaderBuffer::with_size(SOURCES_BYTES, RenderAssetUsages::RENDER_WORLD);
        buffer.label = "environment_map_sources".into();
        buffer.buffer_usage |= BufferUsages::COPY_SRC;
        let buffer = buffers.add(buffer);
        // Shading reads the buffer on the GPU, so only lights need a readback.
        let mut readback = commands.spawn(SourceReadbackOwner(entity));
        if settings.output == EnvironmentMapSourceOutput::DirectionalLights {
            readback
                .insert(Readback::buffer(buffer.clone()))
                .observe(store_source_readback);
        }
        let readback = readback.id();
        commands.entity(entity).insert((
            EnvironmentMapSourceBuffer { buffer, readback },
            EnvironmentMapSourceState::default(),
        ));
    }
}

fn store_source_readback(
    event: On<ReadbackComplete>,
    owners: Query<&SourceReadbackOwner>,
    mut states: Query<&mut EnvironmentMapSourceState>,
) {
    let Ok(owner) = owners.get(event.entity) else {
        return;
    };
    let Ok(mut state) = states.get_mut(owner.0) else {
        return;
    };
    if event.data.len() < SOURCES_BYTES as usize {
        return;
    }
    let word = |index: usize| {
        let bytes: [u8; 4] = event.data[index * 4..index * 4 + 4].try_into().unwrap();
        bytes
    };
    let float = |index: usize| f32::from_le_bytes(word(index));
    let count = u32::from_le_bytes(word(0)).min(4) as usize;
    let sources = (0..count)
        .map(|i| {
            let base = 4 + i * 8;
            ExtractedSource {
                direction: Vec3::new(float(base), float(base + 1), float(base + 2)),
                angular_size: float(base + 3),
                illuminance: Vec3::new(float(base + 4), float(base + 5), float(base + 6)),
                isotropy: float(base + 7),
            }
        })
        .filter(|source| source.direction.is_finite() && source.illuminance.is_finite())
        .collect();
    state.sources = Some(sources);
}

fn luminance(rgb: Vec3) -> f32 {
    rgb.dot(Vec3::new(0.2126, 0.7152, 0.0722))
}

/// The light a source should become, in world space and lux.
struct TargetLight {
    direction: Vec3,
    angular_size: f32,
    illuminance: f32,
    color: Vec3,
}

pub fn update_environment_map_source_lights(
    mut commands: Commands,
    owners: Query<(
        Entity,
        &EnvironmentMapSourceExtraction,
        &GeneratedEnvironmentMapLight,
        &EnvironmentMapSourceState,
    )>,
    mut lights: Query<(
        Entity,
        &EnvironmentMapSourceLight,
        &mut DirectionalLight,
        &mut Transform,
    )>,
) {
    for (owner, settings, environment_map, state) in &owners {
        let Some(sources) = &state.sources else {
            continue;
        };
        let targets: Vec<TargetLight> = sources
            .iter()
            .filter_map(|source| {
                let illuminance = luminance(source.illuminance);
                (illuminance > 0.0).then(|| TargetLight {
                    direction: (environment_map.rotation * source.direction).normalize(),
                    angular_size: source.angular_size,
                    illuminance: illuminance * environment_map.intensity,
                    color: source.illuminance / illuminance,
                })
            })
            .collect();
        let brightest = targets
            .iter()
            .map(|target| target.illuminance)
            .fold(0.0, f32::max);

        let keep = settings.smoothing.clamp(0.0, 0.999);
        let mut matched = vec![false; targets.len()];
        for (entity, light_owner, mut light, mut transform) in &mut lights {
            if light_owner.owner != owner {
                continue;
            }
            let current = transform.back().as_vec3();
            let best = targets
                .iter()
                .enumerate()
                .filter(|(i, target)| !matched[*i] && target.direction.dot(current) > MATCH_COS)
                .max_by(|a, b| {
                    a.1.direction
                        .dot(current)
                        .total_cmp(&b.1.direction.dot(current))
                });
            match best {
                Some((i, target)) => {
                    matched[i] = true;
                    let direction = current.lerp(target.direction, 1.0 - keep).normalize();
                    *transform = Transform::default().looking_to(-direction, Vec3::Y);
                    light.illuminance =
                        light.illuminance * keep + target.illuminance * (1.0 - keep);
                    light.angular_size =
                        light.angular_size * keep + target.angular_size * (1.0 - keep);
                    let color = LinearRgba::from(light.color).to_vec3();
                    let color = color.lerp(target.color, 1.0 - keep);
                    light.color = Color::linear_rgb(color.x, color.y, color.z);
                    light.shadow_maps_enabled =
                        settings.shadows_enabled && target.illuminance >= brightest;
                }
                None => {
                    // The source is gone: fade out, then remove.
                    light.illuminance *= keep;
                    if light.illuminance <= 1e-4 * brightest.max(1.0) {
                        commands.entity(entity).despawn();
                    }
                }
            }
        }
        for (target, _) in targets
            .iter()
            .zip(&matched)
            .filter(|(_, matched)| !**matched)
        {
            commands.spawn((
                DirectionalLight {
                    // Fade in from zero, like any other smoothed change.
                    illuminance: target.illuminance * (1.0 - keep),
                    color: Color::linear_rgb(target.color.x, target.color.y, target.color.z),
                    angular_size: target.angular_size,
                    shadow_maps_enabled: settings.shadows_enabled
                        && target.illuminance >= brightest,
                    ..Default::default()
                },
                Transform::default().looking_to(-target.direction, Vec3::Y),
                EnvironmentMapSourceLight { owner },
            ));
        }
    }
}

/// Removes the lights and buffers of entities that no longer extract sources.
pub fn cleanup_environment_map_source_lights(
    mut commands: Commands,
    lights: Query<(Entity, &EnvironmentMapSourceLight)>,
    owners: Query<(), With<EnvironmentMapSourceExtraction>>,
    stopped: Query<(Entity, &EnvironmentMapSourceBuffer), Without<EnvironmentMapSourceExtraction>>,
) {
    for (entity, light) in &lights {
        if owners.get(light.owner).is_err() {
            commands.entity(entity).despawn();
        }
    }
    for (entity, buffer) in &stopped {
        commands.entity(buffer.readback).despawn();
        commands
            .entity(entity)
            .remove::<(EnvironmentMapSourceBuffer, EnvironmentMapSourceState)>();
    }
}
