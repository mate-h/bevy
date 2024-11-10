//! This example showcases pbr atmospheric scattering
//!
//! ## Controls
//!
//! | Key Binding        | Action                                 |
//! |:-------------------|:---------------------------------------|
//! | `Spacebar`         | Toggle Atmospheric Fog                 |
//! | `S`                | Toggle Directional Light Fog Influence |

use std::f32::consts::PI;

use bevy::{
    core_pipeline::{auto_exposure::AutoExposure, core_3d::Camera3dDepthTextureUsage},
    pbr::{Atmosphere, CascadeShadowConfigBuilder},
    prelude::*,
};
use bevy_internal::core_pipeline::tonemapping::Tonemapping;
use bevy_render::render_resource::TextureUsages;
use light_consts::lux;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(
            Startup,
            (setup_camera_fog, setup_terrain_scene, setup_instructions),
        )
        .add_systems(Update, rotate_sun)
        .run();
}

fn setup_camera_fog(mut commands: Commands) {
    commands.spawn((
        Camera3d {
            depth_texture_usages: Camera3dDepthTextureUsage::from(
                TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            ),
            ..Default::default()
        },
        Camera {
            hdr: true,
            ..default()
        },
        Msaa::Off,
        Tonemapping::AcesFitted,
        Transform::from_xyz(-1.2, 0.15, 0.0).looking_at(Vec3::Y * 0.1, Vec3::Y),
        Atmosphere::EARTH,
        AutoExposure {
            range: -500.0..=500.0,
            ..Default::default()
        },
    ));
}

fn setup_terrain_scene(mut commands: Commands, asset_server: Res<AssetServer>) {
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
            color: Color::srgb(0.98, 0.95, 0.82),
            shadows_enabled: true,
            illuminance: 10.0,
            ..default()
        },
        Transform::from_xyz(1.0, -1.0, 0.0).looking_at(Vec3::ZERO, Vec3::Y),
        cascade_shadow_config,
    ));

    // Terrain
    commands.spawn(SceneRoot(asset_server.load(
        GltfAssetLabel::Scene(0).from_asset("models/terrain/Mountains.gltf"),
    )));
}

//TODO: update this
fn setup_instructions(mut commands: Commands) {
    // commands.spawn(
    //     TextBundle::from_section(
    //         "Press Spacebar to Toggle Atmospheric Fog.\nPress S to Toggle Directional Light Fog Influence.",
    //         TextStyle::default(),
    //     )
    //     .with_style(Style {
    //         position_type: PositionType::Absolute,
    //         bottom: Val::Px(12.0),
    //         left: Val::Px(12.0),
    //         ..default()
    //     }),
    // );
}

// fn toggle_system(keycode: Res<ButtonInput<KeyCode>>, mut fog: Query<&mut FogSettings>) {
//     let mut fog_settings = fog.single_mut();
//
//     if keycode.just_pressed(KeyCode::Space) {
//         let a = fog_settings.color.alpha();
//         fog_settings.color.set_alpha(1.0 - a);
//     }
//
//     if keycode.just_pressed(KeyCode::KeyS) {
//         let a = fog_settings.directional_light_color.alpha();
//         fog_settings.directional_light_color.set_alpha(0.5 - a);
//     }
// }

fn rotate_sun(mut sun: Query<&mut Transform, With<DirectionalLight>>, time: Res<Time>) {
    sun.single_mut().rotate_z(time.delta_secs() * PI / 30.0);
}