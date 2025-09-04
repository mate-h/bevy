//! This example showcases pbr atmospheric scattering
#[path = "../helpers/camera_controller.rs"]
mod camera_controller;

use camera_controller::{CameraController, CameraControllerPlugin};
use std::f32::consts::PI;

use bevy::{
    camera::Exposure,
    core_pipeline::{bloom::Bloom, tonemapping::Tonemapping},
    input::keyboard::KeyCode,
    light::{light_consts::lux, AtmosphereEnvironmentMapLight, CascadeShadowConfigBuilder},
    pbr::{Atmosphere, AtmosphereMode, AtmosphereSettings, EarthAtmosphere, Falloff, PhaseFunction, ScatteringMedium, ScatteringTerm},
    prelude::*,
};

#[derive(Resource)]
struct GameState {
    paused: bool,
}

// impl default for GameState
impl Default for GameState {
    fn default() -> Self {
        Self { paused: true }
    }
}


fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(GameState::default())
        .add_plugins((DefaultPlugins, CameraControllerPlugin))
        .add_systems(
            Startup,
            (setup_camera_fog, setup_terrain_scene, print_controls),
        )
        .add_systems(Update, (dynamic_scene, atmosphere_controls))
        .run();
}

fn print_controls() {
    println!("Atmosphere Example Controls:");
    println!("    1          - Switch to default rendering method");
    println!("    2          - Switch to raymarched rendering method");
    println!("    Enter      - Pause/Resume sun motion");
    println!("    Up/Down    - Increase/Decrease exposure");
}

fn atmosphere_controls(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut atmosphere_settings: Query<&mut AtmosphereSettings>,
    mut game_state: ResMut<GameState>,
    mut camera_exposure: Query<&mut Exposure, With<Camera3d>>,
    time: Res<Time>,
) {
    if keyboard_input.just_pressed(KeyCode::Digit1) {
        for mut settings in &mut atmosphere_settings {
            settings.rendering_method = AtmosphereMode::LookupTexture;
            println!("Switched to default rendering method");
        }
    }

    if keyboard_input.just_pressed(KeyCode::Digit2) {
        for mut settings in &mut atmosphere_settings {
            settings.rendering_method = AtmosphereMode::Raymarched;
            println!("Switched to raymarched rendering method");
        }
    }
    if keyboard_input.just_pressed(KeyCode::Enter) {
        game_state.paused = !game_state.paused;
    }

    if keyboard_input.pressed(KeyCode::ArrowUp) {
        for mut exposure in &mut camera_exposure {
            exposure.ev100 -= time.delta_secs() * 2.0;
        }
    }

    if keyboard_input.pressed(KeyCode::ArrowDown) {
        for mut exposure in &mut camera_exposure {
            exposure.ev100 += time.delta_secs() * 2.0;
        }
    }
}

fn setup_camera_fog(mut commands: Commands, mut resources: ResMut<Assets<ScatteringMedium>>) {

    let custom_medium_input = ScatteringMedium::new(256, 256, [
        ScatteringTerm {
            absorption: Vec3::ZERO,
            scattering: Vec3::new(5.802e-6, 13.558e-6, 33.100e-6),
            falloff: Falloff::Exponential { scale: 12.5 },
            phase: PhaseFunction::Rayleigh,
        },
        ScatteringTerm {
            absorption: Vec3::splat(3.996e-6),
            scattering: Vec3::splat(0.444e-6),
            falloff: Falloff::Exponential { scale: 83.5 },
            phase: PhaseFunction::Mie { bias: 0.8 },
        },
        ScatteringTerm {
            absorption: Vec3::new(0.650e-6, 1.881e-6, 0.085e-6),
            scattering: Vec3::ZERO,
            falloff: Falloff::Tent {
                center: 0.75,
                width: 0.3,
            },
            phase: PhaseFunction::Isotropic,
        },
    ],).with_label("custom_atmosphere");

    let custom_medium = resources.add(custom_medium_input);

    let custom_atmosphere = Atmosphere {
        bottom_radius: 6_360_000.0,
        top_radius: 6_460_000.0,
        ground_albedo: Vec3::splat(0.3),
        medium: custom_medium,
    };

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-1.2, 0.15, 0.0).looking_at(Vec3::Y * 0.1, Vec3::Y),
        // get the default `Atmosphere` component
        custom_atmosphere,
        // The scene is in units of 10km, so we need to scale up the
        // aerial view lut distance and set the scene scale accordingly.
        // Most usages of this feature will not need to adjust this.
        AtmosphereSettings {
            aerial_view_lut_max_distance: 3.2e5,
            scene_units_to_m: 1e+4,
            ..Default::default()
        },
        // The directional light illuminance used in this scene
        // (the one recommended for use with this feature) is
        // quite bright, so raising the exposure compensation helps
        // bring the scene to a nicer brightness range.
        Exposure::SUNLIGHT,
        // Tonemapper chosen just because it looked good with the scene, any
        // tonemapper would be fine :)
        Tonemapping::AcesFitted,
        // Bloom gives the sun a much more natural look.
        Bloom::NATURAL,
        // Enables the atmosphere to drive reflections and ambient lighting (IBL) for this view
        AtmosphereEnvironmentMapLight::default(),
        CameraController::default(),
    ));
}

#[derive(Component)]
struct Terrain;

fn setup_terrain_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    // Configure a properly scaled cascade shadow map for this scene (defaults are too large, mesh units are in km)
    let cascade_shadow_config = CascadeShadowConfigBuilder {
        first_cascade_far_bound: 0.3,
        maximum_distance: 15.0,
        ..default()
    }
    .build();

    // Sun
    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
            // lux::RAW_SUNLIGHT is recommended for use with this feature, since
            // other values approximate sunlight *post-scattering* in various
            // conditions. RAW_SUNLIGHT in comparison is the illuminance of the
            // sun unfiltered by the atmosphere, so it is the proper input for
            // sunlight to be filtered by the atmosphere.
            illuminance: lux::RAW_SUNLIGHT,
            ..default()
        },
        Transform::from_xyz(1.0, 0.4, 0.0).looking_at(Vec3::ZERO, Vec3::Y),
        cascade_shadow_config,
    ));

    let sphere_mesh = meshes.add(Mesh::from(Sphere { radius: 1.0 }));

    // light probe spheres
    commands.spawn((
        Mesh3d(sphere_mesh.clone()),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            metallic: 1.0,
            perceptual_roughness: 0.0,
            ..default()
        })),
        Transform::from_xyz(-0.3, 0.1, -0.1).with_scale(Vec3::splat(0.05)),
    ));

    commands.spawn((
        Mesh3d(sphere_mesh.clone()),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            metallic: 0.0,
            perceptual_roughness: 1.0,
            ..default()
        })),
        Transform::from_xyz(-0.3, 0.1, 0.1).with_scale(Vec3::splat(0.05)),
    ));

    // Terrain
    commands.spawn((
        Terrain,
        SceneRoot(
            asset_server.load(GltfAssetLabel::Scene(0).from_asset("models/terrain/terrain.glb")),
        ),
        Transform::from_xyz(-1.0, 0.0, -0.5)
            .with_scale(Vec3::splat(0.5))
            .with_rotation(Quat::from_rotation_y(PI / 2.0)),
    ));
}

fn dynamic_scene(
    mut suns: Query<&mut Transform, With<DirectionalLight>>,
    time: Res<Time>,
    sun_motion_state: Res<GameState>,
) {
    // Only rotate the sun if motion is not paused
    if !sun_motion_state.paused {
        suns.iter_mut()
            .for_each(|mut tf| tf.rotate_x(-time.delta_secs() * PI / 10.0));
    }
}
