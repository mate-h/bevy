//! Demonstrates dynamic volumetric cloud rendering with Bevy's atmospheric scattering system.
//!
//! This example showcases:
//! - Procedural volumetric cloud generation using noise functions
//! - Dynamic cloud movement and animation through time-based displacement
//! - Integration with the atmospheric scattering system for realistic lighting
//! - Cloud density and coverage controls
//! - Light scattering and shadowing within cloud volumes
//!
//! The clouds are rendered using ray-marching techniques in a fragment shader,
//! with cloud properties controlled through the Clouds component.
//! Time-based animation is provided through the globals uniform binding.

use bevy::{
    core_pipeline::tonemapping::Tonemapping,
    pbr::{
        Atmosphere, AtmosphereSettings, CascadeShadowConfigBuilder, Clouds, DirectionalLight,
        DirectionalLightShadowMap,
    },
    prelude::*,
    render::camera::Exposure,
};

fn main() {
    App::new()
        .insert_resource(DirectionalLightShadowMap { size: 4096 })
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, (rotate_sun, update_clouds))
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Setup atmosphere camera
    commands.spawn((
        Camera3d::default(),
        Camera {
            hdr: true,
            ..default()
        },
        Transform::from_xyz(0.0, 1.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
        // Add the atmosphere components
        Atmosphere::default(),
        AtmosphereSettings::default(),
        Exposure::SUNLIGHT,
        Tonemapping::AcesFitted,
        // Add the clouds component with enhanced settings
        Clouds {
            coverage: 0.4,
            altitude: 2000.0,
            thickness: 4000.0,
            density: 0.06,
            shape_scale: Vec3::new(0.00005, 0.00010, 0.00005), // Smaller scale for larger cloud formations
            detail_scale: Vec3::new(0.001, 0.001, 0.001),      // More detail
            detail_strength: 0.3,                              // Stronger detail erosion
            wind_speed: Vec2::new(8.0, 5.0),                   // Faster wind for more movement
            brightness: 1.5,                                   // Brighter clouds
            ray_march_steps: 64, // Double the samples for better quality
            light_samples: 8,    // More light samples for better lighting
            ..default()
        },
    ));

    // Ground plane
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.5, 0.3),
            ..default()
        })),
        Transform::from_scale(Vec3::splat(100.0)),
    ));

    // Some objects to cast shadows and show the atmosphere effect
    for i in -2..=2 {
        for j in -2..=2 {
            if i == 0 && j == 0 {
                // Place a larger object in the center
                commands.spawn((
                    Mesh3d(meshes.add(Cuboid::default().mesh())),
                    MeshMaterial3d(materials.add(StandardMaterial {
                        base_color: Color::srgb(0.8, 0.7, 0.6),
                        ..default()
                    })),
                    Transform::from_xyz(0.0, 0.75, 0.0).with_scale(Vec3::splat(1.5)),
                ));
            } else {
                // Scatter smaller objects
                let distance = (i * i + j * j) as f32 * 0.2 + 1.0;
                let height = ((i - j) as f32).sin() * 0.5 + 1.0;
                let color_r = ((i as f32) / 4.0) + 0.5;
                let color_g = ((j as f32) / 4.0) + 0.5;

                commands.spawn((
                    Mesh3d(meshes.add(Cuboid::default().mesh())),
                    MeshMaterial3d(materials.add(StandardMaterial {
                        base_color: Color::srgb(color_r, color_g, 0.5),
                        ..default()
                    })),
                    Transform::from_xyz(i as f32 * distance, height * 0.2, j as f32 * distance)
                        .with_scale(Vec3::splat(0.4)),
                ));
            }
        }
    }

    // Configure shadows
    let cascade_shadow_config = CascadeShadowConfigBuilder {
        first_cascade_far_bound: 5.0,
        maximum_distance: 30.0,
        ..default()
    }
    .build();

    // Directional light (sun)
    commands.spawn((
        DirectionalLight {
            illuminance: 100000.0,
            shadows_enabled: true,
            ..default()
        },
        cascade_shadow_config,
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_4)),
    ));
}

// Rotate the sun around to show time of day changes
fn rotate_sun(time: Res<Time>, mut query: Query<&mut Transform, With<DirectionalLight>>) {
    for mut transform in &mut query {
        // Rotate slowly around the Y axis
        let angle = time.elapsed_secs() * 0.05;
        let y = angle.sin();
        transform.rotation = Quat::from_rotation_x(-1.0 + y * 0.5) * Quat::from_rotation_y(angle);
    }
}

// Update cloud parameters over time for a more dynamic look
fn update_clouds(time: Res<Time>, mut clouds_query: Query<&mut Clouds>) {
    let elapsed = time.elapsed_secs();

    for mut clouds in &mut clouds_query {
        // Vary cloud coverage over time
        clouds.coverage = (elapsed.sin() * 0.2 + 0.5).clamp(0.1, 0.9);

        // Subtly adjust detail strength for variation
        clouds.detail_strength = (elapsed * 0.1).sin() * 0.1 + 0.2;
    }
}
