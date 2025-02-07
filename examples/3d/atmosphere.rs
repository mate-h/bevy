//! This example showcases pbr atmospheric scattering

use std::f32::consts::PI;

use bevy::{
    core_pipeline::{bloom::Bloom, tonemapping::Tonemapping},
    pbr::{light_consts::lux, Atmosphere, AtmosphereSettings, CascadeShadowConfigBuilder},
    prelude::*,
    render::camera::Exposure,
};

/// The current settings that the user can toggle.
#[derive(Resource)]
struct AtmosphereUserSettings {
    /// Current atmosphere type
    atmosphere_type: AtmosphereType,
    /// Current view position (ground or space)
    view_type: ViewType,
}

#[derive(Default)]
enum AtmosphereType {
    #[default]
    Earth,
    Mars,
}

#[derive(Default)]
enum ViewType {
    #[default]
    Ground,
    Space,
}

impl Default for AtmosphereUserSettings {
    fn default() -> Self {
        Self {
            atmosphere_type: AtmosphereType::Earth,
            view_type: ViewType::Ground,
        }
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .init_resource::<AtmosphereUserSettings>()
        .add_systems(Startup, (setup_camera_fog, setup_terrain_scene, setup_ui))
        .add_systems(
            Update,
            (dynamic_scene, adjust_atmosphere_settings, update_ui_text),
        )
        .run();
}

fn setup_camera_fog(mut commands: Commands, settings: ResMut<AtmosphereUserSettings>) {
    let (position, target) = match settings.view_type {
        ViewType::Ground => (Vec3::new(-1.2, 0.15, 0.0), Vec3::Y * 0.1),
        ViewType::Space => (Vec3::new(-1.2, 7.0, 0.0), Vec3::Y * 6.8),
    };

    let atmosphere = match settings.atmosphere_type {
        AtmosphereType::Earth => Atmosphere::EARTH,
        AtmosphereType::Mars => Atmosphere::MARS,
    };

    commands.spawn((
        Camera3d::default(),
        // HDR is required for atmospheric scattering to be properly applied to the scene
        Camera {
            hdr: true,
            ..default()
        },
        Transform::from_translation(position).looking_at(target, Vec3::Y),
        atmosphere,
        // The scene is in units of 10km, so we need to scale up the
        // aerial view lut distance and set the scene scale accordingly.
        // Most usages of this feature will not need to adjust this.
        AtmosphereSettings {
            aerial_view_lut_max_distance: 3.2e5,
            scene_units_to_m: 1e+4,
            ..Default::default()
        },
        // The directional light illuminance  used in this scene
        // (the one recommended for use with this feature) is
        // quite bright, so raising the exposure compensation helps
        // bring the scene to a nicer brightness range.
        Exposure::SUNLIGHT,
        // Tonemapper chosen just because it looked good with the scene, any
        // tonemapper would be fine :)
        Tonemapping::AcesFitted,
        // Bloom gives the sun a much more natural look.
        Bloom::NATURAL,
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
        maximum_distance: 3.0,
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
        Transform::from_xyz(1.0, -0.4, 0.0).looking_at(Vec3::ZERO, Vec3::Y),
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

fn setup_ui(mut commands: Commands, settings: Res<AtmosphereUserSettings>) {
    commands.spawn((
        create_text(&settings),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            left: Val::Px(12.0),
            ..default()
        },
    ));
}

fn create_text(settings: &AtmosphereUserSettings) -> Text {
    format!(
        "{}\n{}",
        if matches!(settings.atmosphere_type, AtmosphereType::Earth) {
            "Press M to switch to Mars atmosphere"
        } else {
            "Press M to switch to Earth atmosphere"
        },
        if matches!(settings.view_type, ViewType::Ground) {
            "Press V to switch to Space view"
        } else {
            "Press V to switch to Ground view"
        },
    )
    .into()
}

fn adjust_atmosphere_settings(
    mut commands: Commands,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut settings: ResMut<AtmosphereUserSettings>,
    camera: Query<Entity, With<Camera3d>>,
) {
    let mut rebuild_camera = false;

    if keyboard_input.just_pressed(KeyCode::KeyM) {
        settings.atmosphere_type = match settings.atmosphere_type {
            AtmosphereType::Earth => AtmosphereType::Mars,
            AtmosphereType::Mars => AtmosphereType::Earth,
        };
        rebuild_camera = true;
    }

    if keyboard_input.just_pressed(KeyCode::KeyV) {
        settings.view_type = match settings.view_type {
            ViewType::Ground => ViewType::Space,
            ViewType::Space => ViewType::Ground,
        };
        rebuild_camera = true;
    }

    if rebuild_camera {
        // Remove existing camera
        if let Ok(camera_entity) = camera.get_single() {
            commands.entity(camera_entity).despawn();
        }
        // Setup new camera with current settings
        setup_camera_fog(commands, settings);
    }
}

fn dynamic_scene(mut suns: Query<&mut Transform, With<DirectionalLight>>, time: Res<Time>) {
    suns.iter_mut()
        .for_each(|mut tf| tf.rotate_x(-time.delta_secs() * PI / 10.0));
}

fn update_ui_text(mut text: Query<&mut Text>, settings: Res<AtmosphereUserSettings>) {
    for mut text in text.iter_mut() {
        *text = create_text(&settings);
    }
}
