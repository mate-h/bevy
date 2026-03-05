use bevy::{
    camera::Exposure,
    camera_controller::free_camera::{FreeCamera, FreeCameraPlugin},
    core_pipeline::tonemapping::Tonemapping,
    light::{
        Atmosphere, AtmosphereEnvironmentMapLight, atmosphere::{Falloff, PhaseFunction, ScatteringMedium, ScatteringTerm}, light_consts::lux
    },
    pbr::{AtmosphereMode, AtmosphereSettings},
    prelude::*,
};

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(GlobalAmbientLight::NONE)
        .add_plugins((DefaultPlugins, FreeCameraPlugin))
        .add_systems(Startup, (setup_camera, setup_scene))
        .run();
}

fn setup_camera(
    mut commands: Commands,
    mut scattering_mediums: ResMut<Assets<ScatteringMedium>>,
) {
    // Scattering medium: Mie scattering for water droplets (fog).
    let height_fog_medium = scattering_mediums.add(
        ScatteringMedium::new(
            128,
            128,
            [ScatteringTerm {
                absorption: Vec3::splat(4e-4),
                scattering: Vec3::splat(0.04),
                falloff: Falloff::Exponential { scale: 0.1 / 10.0 },
                phase: PhaseFunction::Mie { asymmetry: 0.8 },
            }],
        )
        .with_label("height_fog"),
    );

    let atmosphere = Atmosphere {
        bottom_radius: 6_360_000.0,
        top_radius: 6_370_000.0,
        ground_albedo: Vec3::splat(0.3),
        medium: height_fog_medium,
    };

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 2.0, 8.0).looking_at(Vec3::new(0.0, 0.5, 0.0), Vec3::Y),
        atmosphere,
        AtmosphereSettings {
            scene_units_to_m: 1000.0,
            rendering_method: AtmosphereMode::Raymarched,
            ..default()
        },
        Exposure { ev100: 12.0 },
        Tonemapping::AcesFitted,
        FreeCamera::default(),
        AtmosphereEnvironmentMapLight::default(),
    ));
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Sun
    commands.spawn((
        DirectionalLight {
            illuminance: lux::RAW_SUNLIGHT,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(1.0, 1.0, 0.5).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Ground plane
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(1.0)))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.2, 0.22, 0.2),
            ..default()
        })),
        Transform::from_scale(Vec3::splat(30.0)),
    ));

    for (mesh, color, positions) in [
        (meshes.add(Cuboid::new(1.0, 1.0, 1.0)), Color::srgb(0.6, 0.5, 0.4), [(-2.0, -2.0), (2.0, -1.0), (-1.0, 2.0)]),
        (meshes.add(Sphere::new(0.5)), Color::srgb(0.5, 0.6, 0.7), [(0.0, 0.0), (3.0, 2.0), (-3.0, 1.0)]),
        (meshes.add(Cylinder::new(0.5, 1.0)), Color::srgb(0.6, 0.55, 0.5), [(1.0, -3.0), (-2.0, 1.0), (0.0, -1.0)]),
    ] {
        let mat = materials.add(StandardMaterial { base_color: color, ..default() });
        for (x, z) in positions {
            commands.spawn((Mesh3d(mesh.clone()), MeshMaterial3d(mat.clone()), Transform::from_xyz(x, 0.5, z)));
        }
    }
}
