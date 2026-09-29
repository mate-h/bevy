//! This example shows how to place reflection probes in the scene.
//!
//! Press Space to toggle reflection modes:
//!
//! 1. A prefiltered [`EnvironmentMapLight`] (Pisa)
//! 2. A runtime-generated [`GeneratedEnvironmentMapLight`] (Manson–Sloan)
//!
//! // 3. A reflection probe with the cubes scene (disabled for A/B comparison)
//!
//! The generated path uses a **128²** source cubemap (box-downsampled from Pisa's sharp
//! mip 0) so Manson–Sloan filtering runs at the table's native resolution
//! (`lod_resolution_bias = 0`) without inheriting prefiltered specular blur.
//!
//! Press Enter to pause or resume rotation.
//!
//! Reflection probes don't work on WebGL 2 or WebGPU.

use bevy::{
    asset::RenderAssetUsages,
    camera::{Exposure, Hdr},
    core_pipeline::tonemapping::Tonemapping,
    image::TextureFormatPixelInfo,
    light::{ParallaxCorrection, Skybox},
    pbr::generate::generate_environment_map_light,
    prelude::*,
    render::render_resource::{
        Extent3d, TextureDimension, TextureUsages, TextureViewDescriptor, TextureViewDimension,
    },
};

use std::{
    f32::consts::PI,
    fmt::{Display, Formatter, Result as FmtResult},
};

static STOP_ROTATION_HELP_TEXT: &str = "Press Enter to stop rotation";
static START_ROTATION_HELP_TEXT: &str = "Press Enter to start rotation";

static REFLECTION_MODE_HELP_TEXT: &str = "Press Space to switch reflection mode";

const ENV_MAP_INTENSITY: f32 = 5000.0;
/// Face size the Manson–Sloan coefficient table was fitted for.
const GENERATED_ENV_MAP_FACE_SIZE: u32 = 128;

// The mode the application is in.
#[derive(Resource)]
struct AppStatus {
    // Which environment maps the user has requested to display.
    reflection_mode: ReflectionMode,
    // Whether the user has requested the scene to rotate.
    rotating: bool,
    // The current roughness of the central sphere
    sphere_roughness: f32,
}

// Which environment maps the user has requested to display.
#[derive(Clone, Copy, PartialEq)]
enum ReflectionMode {
    // Only a world environment map is shown.
    EnvironmentMap = 0,
    // Both a world environment map and a reflection probe are present. The
    // reflection probe is shown in the sphere.
    ReflectionProbe = 1,
    // A generated environment map is shown.
    GeneratedEnvironmentMap = 2,
}

// The various reflection maps.
#[derive(Resource)]
struct Cubemaps {
    // The blurry diffuse cubemap that reflects the world, but not the cubes.
    diffuse_environment_map: Handle<Image>,

    // The specular cubemap mip chain that reflects the world, but not the cubes.
    specular_environment_map: Handle<Image>,

    // 128² specular source for GPU Manson–Sloan filtering (native table resolution).
    specular_environment_map_128: Handle<Image>,

    // The specular cubemap mip chain that reflects both the world and the cubes.
    specular_reflection_probe: Handle<Image>,
}

fn main() {
    // Create the app.
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(GlobalAmbientLight::NONE)
        .init_resource::<AppStatus>()
        .init_resource::<Cubemaps>()
        .add_systems(Startup, setup)
        .add_systems(PreUpdate, add_environment_map_to_camera)
        .add_systems(
            Update,
            change_reflection_type.before(generate_environment_map_light),
        )
        .add_systems(Update, toggle_rotation)
        .add_systems(Update, change_sphere_roughness)
        .add_systems(
            Update,
            rotate_camera
                .after(toggle_rotation)
                .after(change_reflection_type),
        )
        .add_systems(Update, update_text.after(rotate_camera))
        .add_systems(Update, setup_environment_map_usage)
        .run();
}

// Spawns all the scene objects.
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    _asset_server: Res<AssetServer>,
    app_status: Res<AppStatus>,
    _cubemaps: Res<Cubemaps>,
) {
    spawn_camera(&mut commands);
    spawn_sphere(&mut commands, &mut meshes, &mut materials, &app_status);
    // Reflection probe mode disabled — A/B prefiltered vs generated only.
    // spawn_reflection_probe(&mut commands, &cubemaps);
    // spawn_scene(&mut commands, &asset_server);
    spawn_text(&mut commands, &app_status);
}

// Spawns the cubes, light, and camera.
#[allow(dead_code)]
fn spawn_scene(commands: &mut Commands, asset_server: &AssetServer) {
    commands.spawn((
        WorldAssetRoot(
            asset_server.load(GltfAssetLabel::Scene(0).from_asset("models/cubes/Cubes.glb")),
        ),
        CubesScene,
    ));
}

// Spawns the camera.
fn spawn_camera(commands: &mut Commands) {
    commands.spawn((
        Camera3d::default(),
        Hdr,
        Exposure { ev100: 11.0 },
        Tonemapping::AcesFitted,
        Transform::from_xyz(-3.883, 0.325, 2.781).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

// Creates the sphere mesh and spawns it.
fn spawn_sphere(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    app_status: &AppStatus,
) {
    // Create a sphere mesh.
    let sphere_mesh = meshes.add(Sphere::new(1.0).mesh().ico(7).unwrap());

    // Create a sphere.
    commands.spawn((
        Mesh3d(sphere_mesh.clone()),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Srgba::hex("#ffffff").unwrap().into(),
            metallic: 1.0,
            perceptual_roughness: app_status.sphere_roughness,
            ..StandardMaterial::default()
        })),
        SphereMaterial,
    ));
}

// Spawns the reflection probe.
#[allow(dead_code)]
fn spawn_reflection_probe(commands: &mut Commands, cubemaps: &Cubemaps) {
    commands.spawn((
        LightProbe::default(),
        EnvironmentMapLight {
            diffuse_map: cubemaps.diffuse_environment_map.clone(),
            specular_map: cubemaps.specular_reflection_probe.clone(),
            intensity: ENV_MAP_INTENSITY,
            ..default()
        },
        // 2.0 because the sphere's radius is 1.0 and we want to fully enclose it.
        Transform::from_scale(Vec3::splat(2.0)),
        // Disable parallax correction because the reflected scene is quite
        // distant.
        ParallaxCorrection::None,
    ));
}

// Spawns the help text.
fn spawn_text(commands: &mut Commands, app_status: &AppStatus) {
    // Create the text.
    commands.spawn((
        app_status.create_text(),
        Node {
            position_type: PositionType::Absolute,
            bottom: px(12),
            left: px(12),
            ..default()
        },
    ));
}

// Adds a world environment map to the camera. This separate system is needed because the camera is
// managed by the scene spawner, as it's part of the glTF file with the cubes, so we have to add
// the environment map after the fact.
fn add_environment_map_to_camera(
    mut commands: Commands,
    query: Query<Entity, Added<Camera3d>>,
    cubemaps: Res<Cubemaps>,
) {
    for camera_entity in query.iter() {
        commands
            .entity(camera_entity)
            .insert(create_camera_environment_map_light(&cubemaps))
            .insert(Skybox {
                image: Some(cubemaps.specular_environment_map.clone()),
                brightness: ENV_MAP_INTENSITY,
                ..default()
            });
    }
}

// A system that handles switching between different reflection modes.
fn change_reflection_type(
    mut commands: Commands,
    light_probe_query: Query<Entity, With<LightProbe>>,
    cubes_scene_query: Query<Entity, With<CubesScene>>,
    camera_query: Query<Entity, With<Camera3d>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut app_status: ResMut<AppStatus>,
    cubemaps: Res<Cubemaps>,
    _asset_server: Res<AssetServer>,
) {
    // Only do anything if space was pressed.
    if !keyboard.just_pressed(KeyCode::Space) {
        return;
    }

    // Toggle prefiltered ↔ generated (reflection probe mode commented out).
    app_status.reflection_mode = match app_status.reflection_mode {
        ReflectionMode::EnvironmentMap => ReflectionMode::GeneratedEnvironmentMap,
        ReflectionMode::GeneratedEnvironmentMap | ReflectionMode::ReflectionProbe => {
            ReflectionMode::EnvironmentMap
        }
    };

    // Remove light probes
    for light_probe in light_probe_query.iter() {
        commands.entity(light_probe).despawn();
    }
    // Remove existing cube scenes
    for scene_entity in cubes_scene_query.iter() {
        commands.entity(scene_entity).despawn();
    }
    // match app_status.reflection_mode {
    //     ReflectionMode::EnvironmentMap | ReflectionMode::GeneratedEnvironmentMap => {}
    //     ReflectionMode::ReflectionProbe => {
    //         spawn_reflection_probe(&mut commands, &cubemaps);
    //         spawn_scene(&mut commands, &asset_server);
    //     }
    // }

    // Update the environment-map components on the camera entity/entities
    for camera in camera_query.iter() {
        // Remove any existing environment-map components
        commands
            .entity(camera)
            .remove::<(EnvironmentMapLight, GeneratedEnvironmentMapLight)>();

        match app_status.reflection_mode {
            ReflectionMode::EnvironmentMap /* | ReflectionMode::ReflectionProbe */ => {
                commands
                    .entity(camera)
                    .insert(create_camera_environment_map_light(&cubemaps));
            }

            // GPU-filtered environment map generated at runtime (128² MS-native source).
            ReflectionMode::GeneratedEnvironmentMap => {
                commands
                    .entity(camera)
                    .insert(GeneratedEnvironmentMapLight {
                        environment_map: cubemaps.specular_environment_map_128.clone(),
                        intensity: ENV_MAP_INTENSITY,
                        ..default()
                    });
            }

            ReflectionMode::ReflectionProbe => {
                // Disabled — fall through to prefiltered.
                commands
                    .entity(camera)
                    .insert(create_camera_environment_map_light(&cubemaps));
            }
        }
    }
}

// A system that handles enabling and disabling rotation.
fn toggle_rotation(keyboard: Res<ButtonInput<KeyCode>>, mut app_status: ResMut<AppStatus>) {
    if keyboard.just_pressed(KeyCode::Enter) {
        app_status.rotating = !app_status.rotating;
    }
}

// A system that updates the help text.
fn update_text(mut text_query: Query<&mut Text>, app_status: Res<AppStatus>) {
    for mut text in text_query.iter_mut() {
        *text = app_status.create_text();
    }
}

impl TryFrom<u32> for ReflectionMode {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(ReflectionMode::EnvironmentMap),
            1 => Ok(ReflectionMode::ReflectionProbe),
            2 => Ok(ReflectionMode::GeneratedEnvironmentMap),
            _ => Err(()),
        }
    }
}

impl Display for ReflectionMode {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        let text = match *self {
            ReflectionMode::EnvironmentMap => "Environment map",
            ReflectionMode::ReflectionProbe => "Reflection probe",
            ReflectionMode::GeneratedEnvironmentMap => "Generated environment map",
        };
        formatter.write_str(text)
    }
}

impl AppStatus {
    // Constructs the help text at the bottom of the screen based on the
    // application status.
    fn create_text(&self) -> Text {
        let rotation_help_text = if self.rotating {
            STOP_ROTATION_HELP_TEXT
        } else {
            START_ROTATION_HELP_TEXT
        };

        format!(
            "{}\n{}\nRoughness: {:.2}\n{}\nUp/Down arrows to change roughness",
            self.reflection_mode,
            rotation_help_text,
            self.sphere_roughness,
            REFLECTION_MODE_HELP_TEXT
        )
        .into()
    }
}

// Creates the world environment map light, used as a fallback if no reflection
// probe is applicable to a mesh.
fn create_camera_environment_map_light(cubemaps: &Cubemaps) -> EnvironmentMapLight {
    EnvironmentMapLight {
        diffuse_map: cubemaps.diffuse_environment_map.clone(),
        specular_map: cubemaps.specular_environment_map.clone(),
        intensity: ENV_MAP_INTENSITY,
        ..default()
    }
}

// Rotates the camera a bit every frame.
fn rotate_camera(
    time: Res<Time>,
    mut camera_query: Query<&mut Transform, With<Camera3d>>,
    app_status: Res<AppStatus>,
) {
    if !app_status.rotating {
        return;
    }

    for mut transform in camera_query.iter_mut() {
        transform.translation = Vec2::from_angle(time.delta_secs() * PI / 5.0)
            .rotate(transform.translation.xz())
            .extend(transform.translation.y)
            .xzy();
        transform.look_at(Vec3::ZERO, Vec3::Y);
    }
}

// Loads the cubemaps from the assets directory.
impl FromWorld for Cubemaps {
    fn from_world(world: &mut World) -> Self {
        Cubemaps {
            diffuse_environment_map: world
                .load_asset("environment_maps/debug_filter_test_2k_diffuse.ktx2"),
            specular_environment_map: world
                .load_asset("environment_maps/debug_filter_test_2k_specular.ktx2"),
            // Filled once the 512² specular map is loaded (see `setup_environment_map_usage`).
            specular_environment_map_128: Handle::default(),
            specular_reflection_probe: world
                .load_asset("environment_maps/cubes_reflection_probe_specular_rgb9e5_zstd.ktx2"),
        }
    }
}

fn setup_environment_map_usage(mut cubemaps: ResMut<Cubemaps>, mut images: ResMut<Assets<Image>>) {
    let Some(image) = images.get(&cubemaps.specular_environment_map) else {
        return;
    };

    if !image
        .texture_descriptor
        .usage
        .contains(TextureUsages::COPY_SRC)
    {
        let mut image = images.get_mut(&cubemaps.specular_environment_map).unwrap();
        image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    }

    // Build a sharp 128² MS source by box-downsampling Pisa mip 0 (not extracting
    // prefiltered mip 2, which already encodes mid roughness).
    if cubemaps.specular_environment_map_128 == Handle::default() {
        let image = images.get(&cubemaps.specular_environment_map).unwrap();
        if let Some(image_128) = downsample_cubemap_to_face_size(image, GENERATED_ENV_MAP_FACE_SIZE)
        {
            cubemaps.specular_environment_map_128 = images.add(image_128);
            info!(
                "Prepared sharp {}² cubemap for GeneratedEnvironmentMapLight (MS native resolution)",
                GENERATED_ENV_MAP_FACE_SIZE
            );
        }
    }
}

/// Box-downsamples cubemap mip 0 to `face_size` (MipMajor rgb9e5).
fn downsample_cubemap_to_face_size(image: &Image, face_size: u32) -> Option<Image> {
    let descriptor = &image.texture_descriptor;
    let data = image.data.as_ref()?;
    let faces = descriptor.size.depth_or_array_layers;
    if faces != 6 || descriptor.dimension != TextureDimension::D2 {
        warn!(
            "Expected a 6-face cubemap, got {} layers / {:?}",
            faces, descriptor.dimension
        );
        return None;
    }
    if descriptor.format != bevy::render::render_resource::TextureFormat::Rgb9e5Ufloat {
        warn!(
            "Sharp downsample currently supports Rgb9e5Ufloat only, got {:?}",
            descriptor.format
        );
        return None;
    }

    let src_w = descriptor.size.width;
    let src_h = descriptor.size.height;
    if src_w % face_size != 0 || src_h % face_size != 0 {
        warn!("Cannot evenly downsample {src_w}x{src_h} → {face_size}²");
        return None;
    }

    let pixel_size = descriptor.format.pixel_size().ok()?;
    let scale_x = src_w / face_size;
    let scale_y = src_h / face_size;
    let src_face_stride = (src_w as usize) * (src_h as usize) * pixel_size;
    let mut out = vec![0u8; (face_size as usize) * (face_size as usize) * pixel_size * 6];

    for face in 0..faces as usize {
        let src_face = &data[face * src_face_stride..(face + 1) * src_face_stride];
        let dst_face_offset = face * (face_size as usize) * (face_size as usize) * pixel_size;

        for dy in 0..face_size {
            for dx in 0..face_size {
                let mut sum = [0.0f32; 3];
                let mut count = 0.0f32;
                for sy in 0..scale_y {
                    for sx in 0..scale_x {
                        let x = dx * scale_x + sx;
                        let y = dy * scale_y + sy;
                        let off = ((y as usize) * (src_w as usize) + (x as usize)) * pixel_size;
                        let packed = u32::from_le_bytes(src_face[off..off + 4].try_into().ok()?);
                        let rgb = unpack_rgb9e5(packed);
                        sum[0] += rgb[0];
                        sum[1] += rgb[1];
                        sum[2] += rgb[2];
                        count += 1.0;
                    }
                }
                let inv = 1.0 / count;
                let packed = pack_rgb9e5([sum[0] * inv, sum[1] * inv, sum[2] * inv]);
                let dst_off =
                    dst_face_offset + ((dy as usize) * (face_size as usize) + (dx as usize)) * 4;
                out[dst_off..dst_off + 4].copy_from_slice(&packed.to_le_bytes());
            }
        }
    }

    let mut extracted = Image::new(
        Extent3d {
            width: face_size,
            height: face_size,
            depth_or_array_layers: faces,
        },
        descriptor.dimension,
        out,
        descriptor.format,
        RenderAssetUsages::all(),
    );
    extracted.texture_descriptor.mip_level_count = 1;
    extracted.texture_descriptor.usage |= TextureUsages::COPY_SRC | TextureUsages::TEXTURE_BINDING;
    extracted.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });
    extracted.sampler = image.sampler.clone();
    Some(extracted)
}

fn unpack_rgb9e5(v: u32) -> [f32; 3] {
    const EXP_BIAS: i32 = 15;
    const MANTISSA_BITS: i32 = 9;
    let exponent = ((v >> 27) & 0x1f) as i32 - EXP_BIAS - MANTISSA_BITS;
    let scale = 2f32.powi(exponent);
    [
        ((v) & 0x1ff) as f32 * scale,
        ((v >> 9) & 0x1ff) as f32 * scale,
        ((v >> 18) & 0x1ff) as f32 * scale,
    ]
}

fn pack_rgb9e5(rgb_in: [f32; 3]) -> u32 {
    const EXP_BIAS: i32 = 15;
    const MANTISSA_BITS: i32 = 9;
    const MANTISSA_VALUES: i32 = 512;
    const MAX_RGB9E5: f32 = 65408.0;

    let rgb = [
        rgb_in[0].clamp(0.0, MAX_RGB9E5),
        rgb_in[1].clamp(0.0, MAX_RGB9E5),
        rgb_in[2].clamp(0.0, MAX_RGB9E5),
    ];
    let maxrgb = rgb[0].max(rgb[1]).max(rgb[2]);
    if maxrgb < f32::MIN_POSITIVE {
        return 0;
    }

    let mut exp_shared =
        ((-EXP_BIAS - 1).max(floor_log2_f32(maxrgb))) + 1 + EXP_BIAS;
    let mut denom = 2f32.powi(exp_shared - EXP_BIAS - MANTISSA_BITS);
    let maxm = (maxrgb / denom + 0.5).floor() as i32;
    if maxm == MANTISSA_VALUES {
        denom *= 2.0;
        exp_shared += 1;
    }

    let n0 = ((rgb[0] / denom + 0.5).floor() as u32).min(511);
    let n1 = ((rgb[1] / denom + 0.5).floor() as u32).min(511);
    let n2 = ((rgb[2] / denom + 0.5).floor() as u32).min(511);
    ((exp_shared as u32) << 27) | (n2 << 18) | (n1 << 9) | n0
}

fn floor_log2_f32(x: f32) -> i32 {
    let bits = x.to_bits();
    let biased = ((bits & 0x7f80_0000) >> 23) as i32;
    biased - 127
}

impl Default for AppStatus {
    fn default() -> Self {
        Self {
            reflection_mode: ReflectionMode::EnvironmentMap,
            rotating: false,
            sphere_roughness: 0.2,
        }
    }
}

#[derive(Component)]
struct SphereMaterial;

#[derive(Component)]
struct CubesScene;

// A system that changes the sphere's roughness with up/down arrow keys
fn change_sphere_roughness(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut app_status: ResMut<AppStatus>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    sphere_query: Query<&MeshMaterial3d<StandardMaterial>, With<SphereMaterial>>,
) {
    let roughness_delta = if keyboard.pressed(KeyCode::ArrowUp) {
        0.01 // Decrease roughness
    } else if keyboard.pressed(KeyCode::ArrowDown) {
        -0.01 // Increase roughness
    } else {
        0.0 // No change
    };

    if roughness_delta != 0.0 {
        // Update the app status
        app_status.sphere_roughness =
            (app_status.sphere_roughness + roughness_delta).clamp(0.0, 1.0);

        // Update the sphere material
        for material_handle in sphere_query.iter() {
            if let Some(mut material) = materials.get_mut(&material_handle.0) {
                material.perceptual_roughness = app_status.sphere_roughness;
            }
        }
    }
}
