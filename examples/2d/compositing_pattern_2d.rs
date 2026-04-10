//! Reference layout for visually comparing 2D alpha compositing against an offline composite.
//!
//! See also: <https://github.com/bevyengine/bevy/pull/23049>

use bevy::{
    camera::CompositingSpace,
    core_pipeline::tonemapping::{DebandDither, Tonemapping},
    post_process::bloom::Bloom,
    prelude::*,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(ClearColor(Color::BLACK))
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    asset_server: Res<AssetServer>,
) {
    commands.spawn((
        Camera2d,
        Camera {
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        CompositingSpace::Srgb,
        Tonemapping::TonyMcMapface,
        Bloom::default(),
        DebandDither::Enabled,
    ));

    let display_size = Vec2::splat(280.0);
    let gap = 40.0;
    let half_span = display_size.x * 0.5 + gap * 0.5;

    let composited = asset_server.load("textures/pattern-test-composited.png");
    commands.spawn((
        Sprite {
            image: composited,
            custom_size: Some(display_size),
            ..default()
        },
        Transform::from_xyz(-half_span, 0.0, 0.0),
    ));

    let pattern = asset_server.load("textures/pattern-test.png");
    commands.spawn((
        Sprite {
            image: pattern,
            custom_size: Some(display_size),
            ..default()
        },
        Transform::from_xyz(half_span, 0.0, 0.0),
    ));

    // Bloom probe, this shows missing bloom because the main pass is not using the 
    // correct texture format when using sRGB compositing.
    commands.spawn((
        Mesh2d(meshes.add(Circle::new(28.0))),
        MeshMaterial2d(materials.add(Color::srgb(12.0, 6.0, 2.0))),
        Transform::from_xyz(0.0, -210.0, 1.0),
    ));
}
