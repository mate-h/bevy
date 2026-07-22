//! Small planet atmosphere casting an umbra on a second body.
//!
//! Controls:
//! - Space: pause / resume receiver motion
//! - 1 / 2: lookup-texture / raymarched atmosphere
//! - Up / Down: sun angular size ±0.1 rad
//! - RMB + WASD/QE: FreeCamera (Shift run, scroll speed)

use bevy::{
    camera::Exposure,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    core_pipeline::tonemapping::Tonemapping,
    light::{atmosphere::ScatteringMedium, light_consts::lux, Atmosphere, SunDisk},
    pbr::{AtmosphereMode, AtmosphereSettings},
    post_process::bloom::Bloom,
    prelude::*,
};

const OCCLUDER_INNER: f32 = 2_000.0;
const OCCLUDER_OUTER: f32 = 4_000.0;
const OCCLUDER_POS: Vec3 = Vec3::ZERO;

const RECEIVER_RADIUS: f32 = 5_000.0;
const RECEIVER_DIST_NEAR: f32 = 8_000.0;
const RECEIVER_DIST_FAR: f32 = 22_000.0;
const RECEIVER_MOTION_SPEED: f32 = 1.0;

const SUN_ANGULAR_SIZE_INITIAL: f32 = 0.5;
const SUN_ANGULAR_SIZE_STEP: f32 = 0.1;

#[derive(Component)]
struct UmbraReceiver;

#[derive(Resource)]
struct OrbitState {
    phase: f32,
    paused: bool,
}

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(GlobalAmbientLight::NONE)
        .insert_resource(OrbitState {
            phase: 0.0,
            paused: false,
        })
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Bevy Examples - Atmosphere Umbra".into(),
                    ..default()
                }),
                ..default()
            }),
            FreeCameraPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, (orbit_receiver, controls))
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut mediums: ResMut<Assets<ScatteringMedium>>,
) {
    let medium = mediums.add(
        ScatteringMedium::earth(256, 256)
            .with_density_multiplier(100.0)
            .with_label("umbra_test_earth"),
    );

    // Non-default GlobalTransform avoids Atmosphere's on_add hook relocating the planet.
    commands.spawn((
        Atmosphere {
            inner_radius: OCCLUDER_INNER,
            outer_radius: OCCLUDER_OUTER,
            ground_albedo: Vec3::splat(0.25),
            medium,
        },
        Transform::from_translation(OCCLUDER_POS),
        GlobalTransform::from_translation(OCCLUDER_POS + Vec3::X * 0.001),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(OCCLUDER_INNER))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.15, 0.35, 0.75),
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform::from_translation(OCCLUDER_POS),
    ));

    commands.spawn((
        UmbraReceiver,
        Mesh3d(meshes.add(Sphere::new(RECEIVER_RADIUS))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.85, 0.75, 0.55),
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform::from_translation(receiver_position(0.0)),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: lux::RAW_SUNLIGHT,
            ..default()
        },
        SunDisk {
            angular_size: SUN_ANGULAR_SIZE_INITIAL,
            intensity: 1.0,
        },
        Transform::default().looking_to(-Vec3::X, Vec3::Y),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(8_000.0, 12_000.0, 28_000.0).looking_at(OCCLUDER_POS, Vec3::Y),
        AtmosphereSettings {
            rendering_method: AtmosphereMode::LookupTexture,
            aerial_view_lut_max_distance: 1.0e5,
            ..default()
        },
        Exposure { ev100: 11.0 },
        Tonemapping::AcesFitted,
        Bloom::NATURAL,
        FreeCamera {
            walk_speed: 2_000.0,
            run_speed: 8_000.0,
            ..default()
        },
    ));
}

fn receiver_position(phase: f32) -> Vec3 {
    let t = 0.5 + 0.5 * phase.sin();
    let distance = RECEIVER_DIST_NEAR.lerp(RECEIVER_DIST_FAR, t);
    Vec3::new(-distance, 0.0, 0.0)
}

fn orbit_receiver(
    time: Res<Time>,
    mut state: ResMut<OrbitState>,
    mut receiver: Query<&mut Transform, With<UmbraReceiver>>,
) {
    if !state.paused {
        state.phase += time.delta_secs() * RECEIVER_MOTION_SPEED;
    }
    for mut transform in &mut receiver {
        transform.translation = receiver_position(state.phase);
    }
}

fn controls(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<OrbitState>,
    mut camera: Query<&mut AtmosphereSettings, With<Camera3d>>,
    mut sun_disks: Query<&mut SunDisk>,
) {
    if keyboard.just_pressed(KeyCode::Space) {
        state.paused = !state.paused;
    }

    if keyboard.just_pressed(KeyCode::Digit1) {
        for mut settings in &mut camera {
            settings.rendering_method = AtmosphereMode::LookupTexture;
        }
    }
    if keyboard.just_pressed(KeyCode::Digit2) {
        for mut settings in &mut camera {
            settings.rendering_method = AtmosphereMode::Raymarched;
        }
    }

    let sun_delta = if keyboard.just_pressed(KeyCode::ArrowUp) {
        SUN_ANGULAR_SIZE_STEP
    } else if keyboard.just_pressed(KeyCode::ArrowDown) {
        -SUN_ANGULAR_SIZE_STEP
    } else {
        0.0
    };
    if sun_delta != 0.0 {
        for mut sun in &mut sun_disks {
            sun.angular_size = (sun.angular_size + sun_delta).max(0.0);
        }
    }
}
