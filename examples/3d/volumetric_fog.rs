//! Demonstrates volumetric fog and lighting (light shafts or god rays).
//! Note: On Wasm, this example only runs on WebGPU
//!
//! Adds the free camera plugin so you can fly around (WASD, QE, mouse after grab; right-click or M).

use bevy::{
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    color::palettes::css::RED,
    core_pipeline::tonemapping::Tonemapping,
    light::Skybox,
    light::{FogVolume, VolumetricFog, VolumetricLight},
    post_process::bloom::Bloom,
    prelude::*,
};

const DIRECTIONAL_LIGHT_MOVEMENT_SPEED: f32 = 0.02;

/// Center of the `VolumetricFogExample` interior — matches the camera look-at.
const ROOM_CENTER: Vec3 = Vec3::new(-1.5, 1.7, 3.5);

/// Default camera position (must stay in sync with `setup`).
const DEFAULT_CAMERA_POSITION: Vec3 = Vec3::new(-1.7, 1.5, 4.5);

/// How far past [`ROOM_CENTER`] along the view axis (deeper into the room, away from the camera) the stress grid is centered.
const STRESS_GRID_DEPTH_PAST_ROOM_CENTER: f32 = 2.75;

/// Grid of extra volumetric point lights to stress light clustering (native / WebGPU).
///
/// Total count is `STRESS_LIGHT_GRID_DIMS.x * STRESS_LIGHT_GRID_DIMS.y * STRESS_LIGHT_GRID_DIMS.z`.
/// These use `shadow_maps_enabled: false` so cost is dominated by clustering and volumetrics, not shadow maps.
const STRESS_LIGHT_GRID_DIMS: UVec3 = UVec3::new(4, 2, 4);
const STRESS_LIGHT_SPACING: f32 = 0.42;
const STRESS_LIGHT_INTENSITY: f32 = 550.0;
const STRESS_LIGHT_RANGE: f32 = 16.0;

/// Marks the original point/spot lights toggled with P/L (not the stress-test grid).
#[derive(Component)]
struct MainDemoLight;

/// The current settings that the user has chosen.
#[derive(Resource)]
struct AppSettings {
    /// Whether volumetric spot light is on.
    volumetric_spotlight: bool,
    /// Whether volumetric point light is on.
    volumetric_pointlight: bool,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            volumetric_spotlight: true,
            volumetric_pointlight: true,
        }
    }
}

// Define a struct to store parameters for the point light's movement.
#[derive(Component)]
struct MoveBackAndForthHorizontally {
    min_x: f32,
    max_x: f32,
    speed: f32,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FreeCameraPlugin)
        .insert_resource(ClearColor(Color::Srgba(Srgba {
            red: 0.02,
            green: 0.02,
            blue: 0.02,
            alpha: 1.0,
        })))
        .insert_resource(GlobalAmbientLight::NONE)
        .init_resource::<AppSettings>()
        .add_systems(Startup, setup)
        .add_systems(Update, tweak_scene)
        .add_systems(Update, (move_directional_light, move_point_light))
        .add_systems(Update, adjust_app_settings)
        .run();
}

/// Initializes the scene.
fn setup(mut commands: Commands, asset_server: Res<AssetServer>, app_settings: Res<AppSettings>) {
    // Spawn the glTF scene.
    commands.spawn(WorldAssetRoot(asset_server.load(
        GltfAssetLabel::Scene(0).from_asset("models/VolumetricFogExample/VolumetricFogExample.glb"),
    )));

    // Spawn the camera.
    commands
        .spawn((
            Camera3d::default(),
            Transform::from_translation(DEFAULT_CAMERA_POSITION).looking_at(ROOM_CENTER, Vec3::Y),
            Tonemapping::TonyMcMapface,
            Bloom::default(),
            FreeCamera {
                walk_speed: 2.5,
                run_speed: 7.0,
                ..default()
            },
        ))
        .insert(Skybox {
            image: asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2"),
            brightness: 1000.0,
            ..default()
        })
        .insert(VolumetricFog {
            // This value is explicitly set to 0 since we have no environment map light
            ambient_intensity: 0.0,
            ..default()
        });

    // Add the point light
    commands.spawn((
        Transform::from_xyz(-0.4, 1.9, 1.0),
        PointLight {
            shadow_maps_enabled: true,
            range: 150.0,
            color: RED.into(),
            intensity: 10_000.0,
            ..default()
        },
        VolumetricLight,
        MoveBackAndForthHorizontally {
            min_x: -1.93,
            max_x: -0.4,
            speed: -0.2,
        },
        MainDemoLight,
    ));

    // Add the spot light
    commands.spawn((
        Transform::from_xyz(-1.8, 3.9, -2.7).looking_at(Vec3::ZERO, Vec3::Y),
        SpotLight {
            intensity: 50_000.0, // lumens
            color: Color::WHITE,
            shadow_maps_enabled: true,
            inner_angle: 0.76,
            outer_angle: 0.94,
            ..default()
        },
        VolumetricLight,
        MainDemoLight,
    ));

    spawn_cluster_stress_lights(&mut commands);

    // Add the fog volume.
    commands.spawn((
        FogVolume::default(),
        Transform::from_scale(Vec3::splat(35.0)),
    ));

    // Add the help text.
    commands.spawn((
        create_text(&app_settings),
        Node {
            position_type: PositionType::Absolute,
            top: px(12),
            left: px(12),
            ..default()
        },
    ));
}

fn create_text(app_settings: &AppSettings) -> Text {
    let stress_count = STRESS_LIGHT_GRID_DIMS.x * STRESS_LIGHT_GRID_DIMS.y * STRESS_LIGHT_GRID_DIMS.z;
    format!(
        "{}\n{}\n{}\n{}",
        "Arrow keys: rotate the directional light (WASD/QE: free camera)",
        if app_settings.volumetric_pointlight {
            "Press P to turn volumetric main point light off"
        } else {
            "Press P to turn volumetric main point light on"
        },
        if app_settings.volumetric_spotlight {
            "Press L to turn volumetric main spot light off"
        } else {
            "Press L to turn volumetric main spot light on"
        },
        format!(
            "Cluster stress: {stress_count} extra volumetric point lights (no shadows)"
        ),
    )
    .into()
}

/// Spreads many [`PointLight`]s through the room so clustered forward + volumetric fog do real work.
fn spawn_cluster_stress_lights(commands: &mut Commands) {
    let gx = STRESS_LIGHT_GRID_DIMS.x as f32;
    let gy = STRESS_LIGHT_GRID_DIMS.y as f32;
    let gz = STRESS_LIGHT_GRID_DIMS.z as f32;
    let s = STRESS_LIGHT_SPACING;
    // Back of the room along the default shot: past the look-at point, still in front of the camera.
    let view_into_room = (ROOM_CENTER - DEFAULT_CAMERA_POSITION).normalize();
    let origin = ROOM_CENTER + view_into_room * STRESS_GRID_DEPTH_PAST_ROOM_CENTER;

    let mut idx = 0u32;
    for iz in 0..STRESS_LIGHT_GRID_DIMS.z {
        for iy in 0..STRESS_LIGHT_GRID_DIMS.y {
            for ix in 0..STRESS_LIGHT_GRID_DIMS.x {
                let fx = ix as f32;
                let fy = iy as f32;
                let fz = iz as f32;
                let pos = origin
                    + Vec3::new(
                        (fx - (gx - 1.0) * 0.5) * s,
                        (fy - (gy - 1.0) * 0.5) * s,
                        (fz - (gz - 1.0) * 0.5) * s,
                    );

                // Golden-angle hue steps so neighboring lights differ in color.
                let hue = (idx as f32 * 137.508) % 360.0;
                idx += 1;

                commands.spawn((
                    Transform::from_translation(pos),
                    PointLight {
                        shadow_maps_enabled: true,
                        range: STRESS_LIGHT_RANGE,
                        color: Color::hsl(hue, 0.75, 0.55),
                        intensity: STRESS_LIGHT_INTENSITY,
                        ..default()
                    },
                    VolumetricLight,
                ));
            }
        }
    }
}

/// A system that makes directional lights in the glTF scene into volumetric
/// lights with shadows.
fn tweak_scene(
    mut commands: Commands,
    mut lights: Query<(Entity, &mut DirectionalLight), Changed<DirectionalLight>>,
) {
    for (light, mut directional_light) in lights.iter_mut() {
        // Shadows are needed for volumetric lights to work.
        directional_light.shadow_maps_enabled = true;
        commands.entity(light).insert(VolumetricLight);
    }
}

/// Processes user requests to move the directional light.
fn move_directional_light(
    input: Res<ButtonInput<KeyCode>>,
    mut directional_lights: Query<&mut Transform, With<DirectionalLight>>,
) {
    let mut delta_theta = Vec2::ZERO;
    if input.pressed(KeyCode::ArrowUp) {
        delta_theta.y += DIRECTIONAL_LIGHT_MOVEMENT_SPEED;
    }
    if input.pressed(KeyCode::ArrowDown) {
        delta_theta.y -= DIRECTIONAL_LIGHT_MOVEMENT_SPEED;
    }
    if input.pressed(KeyCode::ArrowLeft) {
        delta_theta.x += DIRECTIONAL_LIGHT_MOVEMENT_SPEED;
    }
    if input.pressed(KeyCode::ArrowRight) {
        delta_theta.x -= DIRECTIONAL_LIGHT_MOVEMENT_SPEED;
    }

    if delta_theta == Vec2::ZERO {
        return;
    }

    let delta_quat = Quat::from_euler(EulerRot::XZY, delta_theta.y, 0.0, delta_theta.x);
    for mut transform in directional_lights.iter_mut() {
        transform.rotate(delta_quat);
    }
}

// Toggle point light movement between left and right.
fn move_point_light(
    timer: Res<Time>,
    mut objects: Query<(&mut Transform, &mut MoveBackAndForthHorizontally)>,
) {
    for (mut transform, mut move_data) in objects.iter_mut() {
        let mut translation = transform.translation;
        let mut need_toggle = false;
        translation.x += move_data.speed * timer.delta_secs();
        if translation.x > move_data.max_x {
            translation.x = move_data.max_x;
            need_toggle = true;
        } else if translation.x < move_data.min_x {
            translation.x = move_data.min_x;
            need_toggle = true;
        }
        if need_toggle {
            move_data.speed = -move_data.speed;
        }
        transform.translation = translation;
    }
}

// Adjusts app settings per user input.
fn adjust_app_settings(
    mut commands: Commands,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut app_settings: ResMut<AppSettings>,
    mut point_lights: Query<Entity, (With<PointLight>, With<MainDemoLight>)>,
    mut spot_lights: Query<Entity, (With<SpotLight>, With<MainDemoLight>)>,
    mut text: Query<&mut Text>,
) {
    // If there are no changes, we're going to bail for efficiency. Record that
    // here.
    let mut any_changes = false;

    // If the user pressed P, toggle volumetric state of the point light.
    if keyboard_input.just_pressed(KeyCode::KeyP) {
        app_settings.volumetric_pointlight = !app_settings.volumetric_pointlight;
        any_changes = true;
    }
    // If the user pressed L, toggle volumetric state of the spot light.
    if keyboard_input.just_pressed(KeyCode::KeyL) {
        app_settings.volumetric_spotlight = !app_settings.volumetric_spotlight;
        any_changes = true;
    }

    // If there were no changes, bail out.
    if !any_changes {
        return;
    }

    // Update volumetric settings.
    for point_light in point_lights.iter_mut() {
        if app_settings.volumetric_pointlight {
            commands.entity(point_light).insert(VolumetricLight);
        } else {
            commands.entity(point_light).remove::<VolumetricLight>();
        }
    }
    for spot_light in spot_lights.iter_mut() {
        if app_settings.volumetric_spotlight {
            commands.entity(spot_light).insert(VolumetricLight);
        } else {
            commands.entity(spot_light).remove::<VolumetricLight>();
        }
    }

    // Update the help text.
    for mut text in text.iter_mut() {
        *text = create_text(&app_settings);
    }
}
