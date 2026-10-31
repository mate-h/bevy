//! This example shows how to place reflection probes in the scene.
//!
//! Press Space to toggle reflection modes:
//!
//! 1. A prefiltered [`EnvironmentMapLight`] (Pisa)
//! 2. A runtime-generated [`GeneratedEnvironmentMapLight`] (Manson–Sloan)
//! 3. Solari pathtracer reference (Monte Carlo env-map reflections, mip 0 only)
//!
//! // Reflection probe + cubes scene temporarily disabled for A/B comparison
//!
//! The generated path uses a **128²** source cubemap (box-downsampled from Pisa's sharp
//! mip 0) so Manson–Sloan filtering runs at the table's native resolution
//! (`lod_resolution_bias = 0`) without inheriting prefiltered specular blur.
//!
//! Press Enter to pause or resume rotation.
//! In pathtracer mode, samples accumulate automatically while the camera is still.
//! Accumulation resets when the camera moves or roughness changes.
//!
//! Press M to measure pathtracer error against the prefiltered environment map.
//! The score is linear light on sphere pixels, with tonemapping off. The process
//! exits when the measurement finishes. Red in the overlay is where the
//! environment map is brighter than the pathtracer. Bake the cubemaps with
//! `blender-envmap --no-tonemap`: the
//! default bake Reinhard-compresses the specular chain and only inverts that
//! when `--white-point` is set.
//!
//! `IBL_FILTER=fis` enables experimental 64-sample shade-time filtered importance
//! sampling using a box-filtered radiance pyramid built from the same mip 0.
//! `IBL_FIS_SAMPLES` overrides the default 64 samples (1–4096).
//! `IBL_PROFILE=1` uses a fixed 640² background window without measurement
//! readbacks. `python tools/ibl_profile.py --output /tmp/ibl-gpu.json` records
//! warmed Metal GPU pass timings and deletes its temporary xctrace bundles.
//! `IBL_MEASURE=1` starts measurement automatically after loading. Run the seven
//! stress cases with `bash tools/ibl_stress.sh` (after building this example).
//! `IBL_REFERENCE_SPP` requests an exact GPU sample count (default 512, 1–16777216),
//! and `IBL_REFERENCE_SEED` selects the u32 RNG seed (default 0). The dedicated
//! RGBA32Float readback carries RGB and actual spp together. This establishes
//! capture provenance; reference convergence remains unqualified.
//! `IBL_REFERENCE_TIMEOUT_SECS` defaults to 3600 for long convergence runs.
//! `IBL_CAPTURE_DIR` optionally saves immutable RGB f32 captures and geometry masks
//! under a unique run directory, with a manifest written after all payloads.
//!
//! Reflection probes don't work on WebGL 2 or WebGPU.
//! The pathtracer mode requires a GPU with ray-query support (`bevy_solari`).
//!
//! Note: Solari refuses to build its scene bind group with zero directional/emissive
//! lights, even when an environment map is present. A zero-illuminance directional
//! light is spawned so the pathtracer can run in an otherwise env-map-only scene.

#[path = "reflection_probes/metrics.rs"]
mod metrics;
#[path = "reflection_probes/reference_capture.rs"]
mod reference_capture;
use reference_capture::{reference_rgb, validate_reference_capture, ReferenceConfig};

use bevy::{
    asset::RenderAssetUsages,
    camera::{CameraMainTextureUsages, Exposure, Hdr, RenderTarget},
    core_pipeline::tonemapping::{DebandDither, Tonemapping},
    image::TextureFormatPixelInfo,
    light::{ParallaxCorrection, Skybox},
    mesh::Indices,
    pbr::generate::generate_environment_map_light,
    prelude::*,
    render::{
        gpu_readback::{Readback, ReadbackComplete},
        render_resource::{
            Extent3d, TextureDataOrder, TextureDimension, TextureFormat, TextureUsages,
            TextureViewDescriptor, TextureViewDimension,
        },
    },
    solari::{
        pathtracer::{Pathtracer, PathtracingPlugin},
        scene::{RaytracingMesh3d, RaytracingScenePlugin},
    },
};

use std::{
    f32::consts::PI,
    fmt::{Display, Formatter, Result as FmtResult},
};

static STOP_ROTATION_HELP_TEXT: &str = "Press Enter to stop rotation";
static START_ROTATION_HELP_TEXT: &str = "Press Enter to start rotation";

static REFLECTION_MODE_HELP_TEXT: &str =
    "Press Space to switch reflection mode\nPress M to measure pathtracer vs environment map\nAfter M, red on the sphere is where the environment map is brighter";

/// Longest side of the measurement target. The other side follows the window aspect
/// so the error image covers the same view as the sphere on screen.
const MEASURE_MAX_SIDE: u32 = 640;
/// Readbacks to wait after enabling the measure camera before storing the environment-map image.
const MEASURE_IBL_FRAMES: u64 = 8;

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
    // Kept for when the cubes scene is re-enabled.
    #[allow(dead_code)]
    ReflectionProbe = 1,
    // A generated environment map is shown.
    GeneratedEnvironmentMap = 2,
    // Solari pathtracer reference for environment-map reflections.
    Pathtracer = 3,
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

    // Mip 0 only, at the source face size. `IBL_FILTER=ms` feeds this to the runtime filter.
    specular_mip0: Handle<Image>,
    specular_radiance_pyramid: Handle<Image>,

    // The specular cubemap mip chain that reflects both the world and the cubes.
    specular_reflection_probe: Handle<Image>,
}

// Keep the experiment local to this harness until quality/cost is established.
fn configure_fis_shader(mut shaders: ResMut<Assets<Shader>>, mut configured: Local<bool>) {
    if *configured || std::env::var("IBL_FILTER").ok().as_deref() != Some("fis") {
        return;
    }
    for (_, shader) in shaders.iter_mut() {
        if shader.path.ends_with("light_probe/environment_map.wesl") {
            if let bevy::shader::Source::Wesl(source) = &mut shader.source
                && source.contains("const SHADE_TIME_FIS: bool = false;")
            {
                let samples = std::env::var("IBL_FIS_SAMPLES")
                    .ok()
                    .map(|s| {
                        s.parse::<u32>()
                            .expect("IBL_FIS_SAMPLES must be an integer")
                    })
                    .unwrap_or(64);
                assert!(
                    (1..=4096).contains(&samples),
                    "IBL_FIS_SAMPLES must be 1..=4096"
                );
                info!("Enabled shade-time FIS ({samples} samples)");
                *configured = true;
                *source = source
                    .replace(
                        "const SHADE_TIME_FIS: bool = false;",
                        "const SHADE_TIME_FIS: bool = true;",
                    )
                    .replace(
                        "const FIS_SAMPLES: u32 = 64u;",
                        &format!("const FIS_SAMPLES: u32 = {samples}u;"),
                    )
                    .into();
            }
        }
    }
}

fn main() {
    // Create the app.
    // Use RaytracingScenePlugin + PathtracingPlugin only — full SolariPlugins also
    // enables SolariLightingPlugin, which forces deferred opaque rendering and leaves
    // non-pathtracer modes black (no SolariLighting / DeferredPrepass on the camera).
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    focused: false,
                    resolution: if std::env::var_os("IBL_PROFILE").is_some() {
                        bevy::window::WindowResolution::new(640, 640)
                            .with_scale_factor_override(1.0)
                    } else {
                        default()
                    },
                    ..default()
                }),
                ..default()
            }),
            RaytracingScenePlugin,
            PathtracingPlugin,
        ))
        .insert_resource(bevy::winit::WinitSettings::continuous())
        .insert_resource(GlobalAmbientLight::NONE)
        .init_resource::<AppStatus>()
        .init_resource::<Cubemaps>()
        .init_resource::<MeasureReadbacks>()
        .init_resource::<PathtracerErrorMeasure>()
        .add_systems(Startup, setup)
        .add_systems(Update, configure_fis_shader)
        .add_systems(PreUpdate, add_environment_map_to_camera)
        .add_systems(
            Update,
            change_reflection_type
                .before(generate_environment_map_light)
                .after(clear_pathtracer_reset),
        )
        .add_systems(Update, toggle_rotation)
        // Clear `Pathtracer::reset` at the start of the frame so a one-shot reset
        // (roughness / mode switch) only clears accumulation for a single Extract.
        // Camera motion is handled separately in Solari's extract via GlobalTransform.
        .add_systems(Update, clear_pathtracer_reset)
        .add_systems(
            Update,
            change_sphere_roughness.after(clear_pathtracer_reset),
        )
        .add_systems(
            Update,
            rotate_camera
                .after(toggle_rotation)
                .after(change_reflection_type),
        )
        .add_systems(
            Update,
            drive_pathtracer_error_measure
                .after(clear_pathtracer_reset)
                .before(update_text),
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
    mut images: ResMut<Assets<Image>>,
    _asset_server: Res<AssetServer>,
    app_status: Res<AppStatus>,
    _cubemaps: Res<Cubemaps>,
) {
    spawn_camera(&mut commands);
    spawn_measure_camera(&mut commands, &mut images);
    spawn_error_overlay(&mut commands, &mut images);
    spawn_sphere(&mut commands, &mut meshes, &mut materials, &app_status);
    // Solari pathtracer needs ≥1 directional/emissive light to build its bind group.
    // Illuminance 0 contributes no energy — env-map reflections remain the only lighting.
    commands.spawn((
        DirectionalLight {
            illuminance: 0.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::default(),
    ));
    // Reflection probe mode disabled — A/B prefiltered vs generated vs pathtracer.
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
    // `IBL_VIEW=back` looks away from the disc, the view where the narrow-source skirt shows up.
    let translation = if std::env::var("IBL_VIEW").ok().as_deref() == Some("back") {
        Vec3::new(2.737, 0.325, 3.916)
    } else {
        Vec3::new(-3.883, 0.325, 2.781)
    };
    commands.spawn((
        Camera3d::default(),
        Hdr,
        Exposure { ev100: 11.0 },
        Tonemapping::AcesFitted,
        Transform::from_translation(translation).looking_at(Vec3::ZERO, Vec3::Y),
        // Required for Solari pathtracer writes into the main texture.
        CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
        Msaa::Off,
    ));
}

/// Hidden HDR camera. Press M renders the environment map, then the pathtracer, and
/// reads both back so the error is in linear light rather than the window's tonemap.
fn spawn_measure_camera(commands: &mut Commands, images: &mut Assets<Image>) {
    let mut image = Image::new_target_texture(
        MEASURE_MAX_SIDE,
        MEASURE_MAX_SIDE,
        TextureFormat::Rgba16Float,
        None,
    );
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let image = images.add(image);
    commands.insert_resource(MeasureTarget {
        image: image.clone(),
        width: MEASURE_MAX_SIDE,
        height: MEASURE_MAX_SIDE,
    });

    let mut camera = Camera::default();
    camera.is_active = false;
    camera.order = -1;

    // `Camera3d` already inserts `Camera`. Replace it afterwards so this view stays
    // inactive until a measurement starts.
    let camera_entity = commands
        .spawn((
            Camera3d::default(),
            DebandDither::Disabled,
            Hdr,
            Tonemapping::None,
            Exposure { ev100: 11.0 },
            CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
            Msaa::Off,
            RenderTarget::Image(image.clone().into()),
            MeasureCamera,
        ))
        .id();
    commands.entity(camera_entity).insert(camera);
    if std::env::var_os("IBL_PROFILE").is_none() {
        commands
            .spawn(Readback::texture(image))
            .observe(store_measure_readback);
    }
}

// Creates the sphere mesh and spawns it.
fn spawn_sphere(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    app_status: &AppStatus,
) {
    // Create a sphere mesh compatible with Solari raytracing (UVs, tangents, U32 indices).
    let mut sphere = Sphere::new(1.0).mesh().ico(7).unwrap();
    sphere.generate_tangents().unwrap();
    if let Some(indices) = sphere.indices_mut()
        && let Indices::U16(_) = indices
    {
        *indices = Indices::U32(indices.iter().map(|i| i as u32).collect());
    }
    let sphere_mesh = meshes.add(sphere);

    // Create a sphere.
    commands.spawn((
        Mesh3d(sphere_mesh.clone()),
        RaytracingMesh3d(sphere_mesh),
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

/// Full-window overlay. Red is environment-map light the pathtracer does not have.
#[derive(Component)]
struct MeasureErrorPanel;

/// HDR target the measurement camera renders into. Sized to the window aspect on each run.
#[derive(Resource)]
struct MeasureTarget {
    image: Handle<Image>,
    width: u32,
    height: u32,
}

fn measure_resolution(window_width: u32, window_height: u32) -> UVec2 {
    let window_width = window_width.max(1);
    let window_height = window_height.max(1);
    let long_side = window_width.max(window_height);
    let scale = (MEASURE_MAX_SIDE as f32 / long_side as f32).min(1.0);
    UVec2::new(
        ((window_width as f32) * scale).round().max(1.0) as u32,
        ((window_height as f32) * scale).round().max(1.0) as u32,
    )
}

fn spawn_error_overlay(commands: &mut Commands, images: &mut Assets<Image>) {
    let image = images.add(Image::new(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        vec![0, 0, 0, 0],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    ));
    commands.spawn((
        ImageNode::new(image).with_mode(NodeImageMode::Stretch),
        Node {
            position_type: PositionType::Absolute,
            top: px(0),
            left: px(0),
            width: percent(100.0),
            height: percent(100.0),
            display: bevy::ui::Display::None,
            ..default()
        },
        MeasureErrorPanel,
    ));
}

// Spawns the help text.
fn spawn_text(commands: &mut Commands, app_status: &AppStatus) {
    // Create the text.
    commands.spawn((
        Text::new(app_status.create_text()),
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
    query: Query<Entity, (Added<Camera3d>, Without<MeasureCamera>)>,
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
    camera_query: Query<Entity, (With<Camera3d>, Without<MeasureCamera>)>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut app_status: ResMut<AppStatus>,
    cubemaps: Res<Cubemaps>,
    measure: Res<PathtracerErrorMeasure>,
    _asset_server: Res<AssetServer>,
) {
    // Only do anything if space was pressed.
    if !keyboard.just_pressed(KeyCode::Space) || !measure.is_idle() {
        return;
    }

    let leaving_pathtracer = app_status.reflection_mode == ReflectionMode::Pathtracer;

    // Cycle: prefiltered → generated → pathtracer → …
    app_status.reflection_mode = match app_status.reflection_mode {
        ReflectionMode::EnvironmentMap => ReflectionMode::GeneratedEnvironmentMap,
        ReflectionMode::GeneratedEnvironmentMap => ReflectionMode::Pathtracer,
        ReflectionMode::Pathtracer | ReflectionMode::ReflectionProbe => {
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
    //     ReflectionMode::Pathtracer => {}
    // }

    // Keep Mesh3d always — pathtracer overwrites the main pass anyway. Removing it
    // and re-adding it after pathtracer left IBL modes permanently black.

    // Update the environment-map components on the camera entity/entities
    for camera in camera_query.iter() {
        // Do not use `remove_with_requires::<Pathtracer>()`: that also strips `Hdr`
        // (required by Pathtracer), which disables tonemapping and blacks out IBL.
        commands.entity(camera).remove::<(
            EnvironmentMapLight,
            GeneratedEnvironmentMapLight,
            Pathtracer,
        )>();

        // Re-assert camera components pathtracer mode must not leave behind.
        if leaving_pathtracer {
            commands.entity(camera).insert((
                Hdr,
                CameraMainTextureUsages::default().with(TextureUsages::STORAGE_BINDING),
                Msaa::Off,
                Skybox {
                    image: Some(cubemaps.specular_environment_map.clone()),
                    brightness: ENV_MAP_INTENSITY,
                    ..default()
                },
            ));
        }

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

            // Pathtracer samples the sharp specular mip 0 via EnvironmentMapLight.
            ReflectionMode::Pathtracer => {
                commands
                    .entity(camera)
                    .insert(create_camera_environment_map_light(&cubemaps))
                    .insert(Pathtracer {
                        reset: true,
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
fn update_text(
    mut text_query: Query<&mut Text>,
    app_status: Res<AppStatus>,
    measure: Res<PathtracerErrorMeasure>,
) {
    for mut text in text_query.iter_mut() {
        *text = format!("{}\n{}", app_status.create_text(), measure.report).into();
    }
}

impl TryFrom<u32> for ReflectionMode {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(ReflectionMode::EnvironmentMap),
            1 => Ok(ReflectionMode::ReflectionProbe),
            2 => Ok(ReflectionMode::GeneratedEnvironmentMap),
            3 => Ok(ReflectionMode::Pathtracer),
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
            ReflectionMode::Pathtracer => "Solari pathtracer (reference)",
        };
        formatter.write_str(text)
    }
}

impl AppStatus {
    // Constructs the help text at the bottom of the screen based on the
    // application status.
    fn create_text(&self) -> String {
        let rotation_help_text = if self.rotating {
            STOP_ROTATION_HELP_TEXT
        } else {
            START_ROTATION_HELP_TEXT
        };

        let pathtracer_hint = if self.reflection_mode == ReflectionMode::Pathtracer {
            "\nPathtracer: accumulating (resets on camera move / roughness change)"
        } else {
            ""
        };

        format!(
            "{}\n{}\nRoughness: {:.2}\n{}\nUp/Down arrows to change roughness{}",
            self.reflection_mode,
            rotation_help_text,
            self.sphere_roughness,
            REFLECTION_MODE_HELP_TEXT,
            pathtracer_hint
        )
    }
}

// Creates the world environment map light, used as a fallback if no reflection
// probe is applicable to a mesh.
fn create_camera_environment_map_light(cubemaps: &Cubemaps) -> EnvironmentMapLight {
    EnvironmentMapLight {
        diffuse_map: cubemaps.diffuse_environment_map.clone(),
        specular_map: if std::env::var("IBL_FILTER").ok().as_deref() == Some("fis")
            && cubemaps.specular_radiance_pyramid != Handle::default()
        {
            cubemaps.specular_radiance_pyramid.clone()
        } else {
            cubemaps.specular_environment_map.clone()
        },
        intensity: ENV_MAP_INTENSITY,
        ..default()
    }
}

// Rotates the camera a bit every frame.
fn rotate_camera(
    time: Res<Time>,
    mut camera_query: Query<&mut Transform, (With<Camera3d>, Without<MeasureCamera>)>,
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

/// Map stem under `environment_maps/`. Unset keeps the 12° linear disc.
///
/// The stress set, all Karis-filtered at 256² with the same light direction:
/// `stress_furnace` (constant, the case split-sum is exact for), `stress_wide`
/// (60° cap, a source that covers a large part of the hemisphere), `stress_narrow`
/// (12° disc, the narrow-highlight case).
fn ibl_stress_stem() -> String {
    std::env::var("IBL_STRESS").unwrap_or_else(|_| "debug_filter_test_2k".to_string())
}

/// `IBL_FILTER=ms` measures the runtime Manson–Sloan probe. Unset measures the Karis chain.
fn ibl_filter_is_manson_sloan() -> bool {
    std::env::var("IBL_FILTER").ok().as_deref() == Some("ms")
}

// Loads the cubemaps from the assets directory.
impl FromWorld for Cubemaps {
    fn from_world(world: &mut World) -> Self {
        let stem = ibl_stress_stem();
        Cubemaps {
            diffuse_environment_map: world
                .load_asset(format!("environment_maps/{stem}_diffuse.ktx2")),
            specular_environment_map: world
                .load_asset(format!("environment_maps/{stem}_specular.ktx2")),
            // Filled once the 512² specular map is loaded (see `setup_environment_map_usage`).
            specular_environment_map_128: Handle::default(),
            specular_mip0: Handle::default(),
            specular_radiance_pyramid: Handle::default(),
            specular_reflection_probe: world
                .load_asset("environment_maps/cubes_reflection_probe_specular_rgb9e5_zstd.ktx2"),
        }
    }
}

fn setup_environment_map_usage(
    mut cubemaps: ResMut<Cubemaps>,
    mut images: ResMut<Assets<Image>>,
    mut maps: Query<&mut EnvironmentMapLight>,
) {
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

    // Mip 0 only, full face size, so the runtime filter does not convolve a Karis mip.
    if cubemaps.specular_mip0 == Handle::default() {
        let image = images.get(&cubemaps.specular_environment_map).unwrap();
        let face = image.texture_descriptor.size.width;
        if let Some(sharp) = downsample_cubemap_to_face_size(image, face) {
            cubemaps.specular_mip0 = images.add(sharp);
            info!("Prepared MS source {face}² from specular mip 0");
        }
    }

    if std::env::var("IBL_FILTER").ok().as_deref() == Some("fis")
        && cubemaps.specular_radiance_pyramid == Handle::default()
    {
        let source = images.get(&cubemaps.specular_environment_map).unwrap();
        let face = source.texture_descriptor.size.width;
        let mut pyramid = downsample_cubemap_to_face_size(source, face).unwrap();
        let mut data = pyramid.data.take().unwrap();
        let levels = face.ilog2() + 1;
        for level in 1..levels {
            let mip = downsample_cubemap_to_face_size(source, face >> level).unwrap();
            data.extend_from_slice(mip.data.as_ref().unwrap());
        }
        pyramid.data = Some(data);
        pyramid.data_order = TextureDataOrder::MipMajor;
        pyramid.texture_descriptor.mip_level_count = levels;
        cubemaps.specular_radiance_pyramid = images.add(pyramid);
        for mut map in &mut maps {
            if map.specular_map == cubemaps.specular_environment_map {
                map.specular_map = cubemaps.specular_radiance_pyramid.clone();
            }
        }
        info!("Prepared FIS radiance pyramid {face}², {levels} levels");
    }

    // Build a sharp 128² MS source by box-downsampling mip 0 (not extracting
    // a prefiltered mip, which already encodes mid roughness).
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
    if descriptor.format != TextureFormat::Rgb9e5Ufloat {
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

    let mut exp_shared = ((-EXP_BIAS - 1).max(floor_log2_f32(maxrgb))) + 1 + EXP_BIAS;
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
            sphere_roughness: std::env::var("IBL_ROUGHNESS")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0.2),
        }
    }
}

#[derive(Component)]
struct SphereMaterial;

#[derive(Component)]
struct CubesScene;

/// `Pathtracer::reset` must be true for only one frame. Solari's extract copies it into the
/// render world; if it stays true, accumulation never converges.
fn clear_pathtracer_reset(mut pathtracer: Query<&mut Pathtracer>) {
    for mut pathtracer in &mut pathtracer {
        pathtracer.reset = false;
    }
}

// A system that changes the sphere's roughness with up/down arrow keys
fn change_sphere_roughness(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut app_status: ResMut<AppStatus>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    sphere_query: Query<&MeshMaterial3d<StandardMaterial>, With<SphereMaterial>>,
    mut pathtracers: Query<&mut Pathtracer>,
    measure: Res<PathtracerErrorMeasure>,
) {
    if !measure.is_idle() {
        return;
    }

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

        // One-shot reset — `clear_pathtracer_reset` clears this next frame.
        for mut pathtracer in &mut pathtracers {
            pathtracer.reset = true;
        }
    }
}

#[derive(Component)]
struct MeasureCamera;

#[derive(Resource, Default)]
struct MeasureReadbacks {
    latest: Option<Vec<u8>>,
    generation: u64,
}

#[derive(Clone, Copy)]
enum MeasurePhase {
    Idle,
    /// Environment-map image is stored once `generation` reaches `ready_at`.
    CaptureEnvironmentMap {
        ready_at: u64,
    },
    /// Wait for the dedicated accumulation readback with an exact GPU sample count.
    CapturePathtracer,
}

#[derive(Resource)]
struct PathtracerErrorMeasure {
    phase: MeasurePhase,
    environment_map: Option<Vec<u8>>,
    report: String,
    started_at: f32,
    last_generation: u64,
    run: u64,
    reference_readback: Option<Entity>,
    reference_capture: Option<Vec<u8>>,
    config: ReferenceConfig,
}

impl Default for PathtracerErrorMeasure {
    fn default() -> Self {
        Self {
            phase: MeasurePhase::Idle,
            environment_map: None,
            report: "Press M to measure pathtracer error against the environment map.".into(),
            started_at: 0.0,
            last_generation: 0,
            run: 0,
            reference_readback: None,
            reference_capture: None,
            config: ReferenceConfig::default(),
        }
    }
}

impl PathtracerErrorMeasure {
    fn is_idle(&self) -> bool {
        matches!(self.phase, MeasurePhase::Idle)
    }
}

fn store_measure_readback(event: On<ReadbackComplete>, mut frames: ResMut<MeasureReadbacks>) {
    frames.latest = Some(event.data.clone());
    frames.generation = frames.generation.wrapping_add(1);
}

fn drive_pathtracer_error_measure(
    mut commands: Commands,
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    cubemaps: Res<Cubemaps>,
    mut measure: ResMut<PathtracerErrorMeasure>,
    readbacks: Res<MeasureReadbacks>,
    mut app_status: ResMut<AppStatus>,
    window_cameras: Query<
        (&GlobalTransform, &Projection),
        (With<Camera3d>, Without<MeasureCamera>),
    >,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    mut measure_cameras: Query<
        (
            Entity,
            &mut Camera,
            &mut Transform,
            &GlobalTransform,
            &mut Projection,
        ),
        With<MeasureCamera>,
    >,
    mut images: ResMut<Assets<Image>>,
    mut measure_target: ResMut<MeasureTarget>,
    mut error_panels: Query<(&ImageNode, &mut Node), With<MeasureErrorPanel>>,
) {
    if measure.is_idle() {
        if keyboard.just_pressed(KeyCode::KeyM)
            || (std::env::var_os("IBL_MEASURE").is_some() && time.elapsed_secs() > 5.0)
        {
            if let Ok((_, mut node)) = error_panels.single_mut() {
                node.display = bevy::ui::Display::None;
            }
            start_pathtracer_error_measure(
                &mut commands,
                &mut measure,
                &readbacks,
                &mut app_status,
                &window_cameras,
                &windows,
                &mut measure_cameras,
                &mut images,
                &mut measure_target,
                time.elapsed_secs(),
                &cubemaps,
            );
        }
        return;
    }

    if time.elapsed_secs() - measure.started_at > measure.config.timeout_secs {
        finish_measure(
            &mut commands,
            &mut measure_cameras,
            &mut measure,
            "Measurement timed out.".into(),
            false,
        );
        return;
    }

    let latest = if matches!(measure.phase, MeasurePhase::CaptureEnvironmentMap { .. }) {
        if readbacks.generation == measure.last_generation {
            return;
        }
        let Some(latest) = readbacks.latest.clone() else {
            return;
        };
        if latest.len() != row_stride(measure_target.width) * measure_target.height as usize {
            return;
        }
        measure.last_generation = readbacks.generation;
        latest
    } else {
        let Some(latest) = measure.reference_capture.take() else {
            return;
        };
        latest
    };

    match measure.phase {
        MeasurePhase::Idle => {}
        MeasurePhase::CaptureEnvironmentMap { ready_at } => {
            if readbacks.generation < ready_at {
                return;
            }
            measure.environment_map = Some(latest);
            let mut output = Image::new_target_texture(
                measure_target.width,
                measure_target.height,
                TextureFormat::Rgba32Float,
                None,
            );
            output.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::COPY_SRC;
            let output = images.add(output);
            let run = measure.run;
            let readback = commands
                .spawn(Readback::texture(output.clone()))
                .observe(
                    move |event: On<ReadbackComplete>,
                          mut measure: ResMut<PathtracerErrorMeasure>| {
                        if reference_capture::accepts_snapshot(
                            measure.run,
                            run,
                            matches!(measure.phase, MeasurePhase::CapturePathtracer),
                        ) {
                            measure.reference_capture = Some(event.data.clone());
                        }
                    },
                )
                .id();
            measure.reference_readback = Some(readback);
            for (entity, _, _, _, _) in &mut measure_cameras {
                commands.entity(entity).insert(Pathtracer {
                    reset: true,
                    seed: measure.config.seed,
                    max_samples: measure.config.spp,
                    accumulation_output: Some(output.clone()),
                });
            }
            measure.phase = MeasurePhase::CapturePathtracer;
            measure.report = format!(
                "Accumulating {} exact GPU samples (seed {})…",
                measure.config.spp, measure.config.seed
            );
        }
        MeasurePhase::CapturePathtracer => {
            let Some(reference) = measure.environment_map.clone() else {
                return;
            };
            match validate_reference_capture(
                &latest,
                measure_target.width,
                measure_target.height,
                measure.config.spp,
            ) {
                Ok(false) => return,
                Ok(true) => {}
                Err(reason) => {
                    finish_measure(
                        &mut commands,
                        &mut measure_cameras,
                        &mut measure,
                        format!("Reference rejected: {reason}"),
                        false,
                    );
                    return;
                }
            }
            let samples = u64::from(measure.config.spp);
            let report = match measure_cameras.single() {
                Ok((_, _, _, global_transform, projection)) => {
                    if let Ok((panel, mut node)) = error_panels.single_mut() {
                        if let Some(mut image) = images.get_mut(&panel.image) {
                            let rgba = error_overlay_rgba(
                                &reference,
                                &latest,
                                global_transform,
                                projection,
                                measure_target.width,
                                measure_target.height,
                            );
                            image.resize(Extent3d {
                                width: measure_target.width,
                                height: measure_target.height,
                                depth_or_array_layers: 1,
                            });
                            image.data = Some(rgba);
                        }
                        node.display = bevy::ui::Display::Flex;
                    }
                    sphere_error_report(
                        &reference,
                        &latest,
                        global_transform,
                        projection,
                        app_status.sphere_roughness,
                        samples,
                        measure.config.seed,
                        measure_target.width,
                        measure_target.height,
                    )
                }
                Err(_) => Err("Measurement camera is missing.".into()),
            };
            let (report, success) = match report {
                Ok(report) => (report, true),
                Err(reason) => (reason, false),
            };
            info!("{report}");
            finish_measure(
                &mut commands,
                &mut measure_cameras,
                &mut measure,
                report,
                success,
            );
        }
    }
}

fn start_pathtracer_error_measure(
    commands: &mut Commands,
    measure: &mut PathtracerErrorMeasure,
    readbacks: &MeasureReadbacks,
    app_status: &mut AppStatus,
    window_cameras: &Query<
        (&GlobalTransform, &Projection),
        (With<Camera3d>, Without<MeasureCamera>),
    >,
    windows: &Query<&Window, With<bevy::window::PrimaryWindow>>,
    measure_cameras: &mut Query<
        (
            Entity,
            &mut Camera,
            &mut Transform,
            &GlobalTransform,
            &mut Projection,
        ),
        With<MeasureCamera>,
    >,
    images: &mut Assets<Image>,
    measure_target: &mut MeasureTarget,
    started_at: f32,
    cubemaps: &Cubemaps,
) {
    let Some((window_transform, window_projection)) = window_cameras.iter().next() else {
        measure.report = "No view camera to measure from.".into();
        return;
    };
    let Ok(window) = windows.single() else {
        measure.report = "No window to measure from.".into();
        return;
    };
    let Ok((entity, mut camera, mut transform, _, mut projection)) = measure_cameras.single_mut()
    else {
        measure.report = "Measurement camera is missing.".into();
        return;
    };

    let size = measure_resolution(window.physical_width(), window.physical_height());
    measure_target.width = size.x;
    measure_target.height = size.y;
    if let Some(mut image) = images.get_mut(&measure_target.image) {
        image.resize(Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        });
    }
    // Match the window framing. A square target puts the sphere in a different place.
    *projection = window_projection.clone();
    if let Projection::Perspective(perspective) = &mut *projection {
        perspective.aspect_ratio = size.x as f32 / size.y as f32;
    }

    app_status.rotating = false;
    *transform = window_transform.compute_transform();
    camera.is_active = true;
    if ibl_filter_is_manson_sloan() {
        // The generator inserts `EnvironmentMapLight` from this sharp mip 0.
        // The pathtracer then samples that chain's mip 0, the same capture.
        commands
            .entity(entity)
            .remove::<(Pathtracer, EnvironmentMapLight)>()
            .insert((
                GeneratedEnvironmentMapLight {
                    environment_map: cubemaps.specular_mip0.clone(),
                    intensity: ENV_MAP_INTENSITY,
                    ..default()
                },
                Skybox {
                    image: Some(cubemaps.specular_mip0.clone()),
                    brightness: ENV_MAP_INTENSITY,
                    ..default()
                },
            ));
    } else {
        commands.entity(entity).remove::<Pathtracer>().insert((
            create_camera_environment_map_light(cubemaps),
            Skybox {
                image: Some(cubemaps.specular_environment_map.clone()),
                brightness: ENV_MAP_INTENSITY,
                ..default()
            },
        ));
    }

    let config = match ReferenceConfig::from_env() {
        Ok(config) => config,
        Err(reason) => {
            measure.report = reason;
            camera.is_active = false;
            commands.write_message(AppExit::error());
            return;
        }
    };
    measure.config = config;
    measure.run = measure.run.wrapping_add(1);
    measure.reference_capture = None;
    measure.environment_map = None;
    measure.started_at = started_at;
    measure.last_generation = readbacks.generation;
    // The runtime filter needs a few frames to replace the placeholder cubemap.
    let ibl_frames = if ibl_filter_is_manson_sloan() {
        MEASURE_IBL_FRAMES * 4
    } else {
        MEASURE_IBL_FRAMES
    };
    measure.phase = MeasurePhase::CaptureEnvironmentMap {
        ready_at: readbacks.generation + ibl_frames,
    };
    info!(
        "IBL stress map {} filter {} roughness {:.2} view {}",
        ibl_stress_stem(),
        std::env::var("IBL_FILTER").unwrap_or_else(|_| "karis".into()),
        app_status.sphere_roughness,
        std::env::var("IBL_VIEW").unwrap_or_else(|_| "front".to_string()),
    );
    measure.report = format!(
        "Measuring roughness {:.2}: capturing the environment map…",
        app_status.sphere_roughness
    );
}

fn finish_measure(
    commands: &mut Commands,
    measure_cameras: &mut Query<
        (
            Entity,
            &mut Camera,
            &mut Transform,
            &GlobalTransform,
            &mut Projection,
        ),
        With<MeasureCamera>,
    >,
    measure: &mut PathtracerErrorMeasure,
    report: String,
    success: bool,
) {
    for (entity, mut camera, _, _, _) in measure_cameras {
        camera.is_active = false;
        commands.entity(entity).remove::<Pathtracer>();
    }
    if let Some(entity) = measure.reference_readback.take() {
        commands.entity(entity).despawn();
    }
    measure.reference_capture = None;
    measure.phase = MeasurePhase::Idle;
    measure.environment_map = None;
    measure.report = report;
    commands.write_message(if success {
        AppExit::Success
    } else {
        AppExit::error()
    });
}

fn sphere_error_report(
    candidate_capture: &[u8],
    fixed_reference_capture: &[u8],
    transform: &GlobalTransform,
    projection: &Projection,
    roughness: f32,
    samples: u64,
    seed: u32,
    width: u32,
    height: u32,
) -> Result<String, String> {
    let world_from_clip = transform.to_matrix() * projection.get_clip_from_view().inverse();
    let origin = transform.translation();
    let stride = row_stride(width);
    let mut candidate = Vec::new();
    let mut fixed_reference = Vec::new();
    let mut geometry_mask = Vec::new();
    for y in 0..height {
        for x in 0..width {
            // Geometry never depends on either image, including decode validity.
            let uv = Vec2::new(
                (x as f32 + 0.5) / width as f32,
                (y as f32 + 0.5) / height as f32,
            );
            let ndc = uv * 2.0 - 1.0;
            let target = world_from_clip * Vec4::new(ndc.x, -ndc.y, 1.0, 1.0);
            let direction = (target.xyz() / target.w - origin).normalize_or_zero();
            geometry_mask.push(direction != Vec3::ZERO && ray_hits_unit_sphere(origin, direction));
            let (Some(c), Some(r)) = (
                pixel_rgb(candidate_capture, stride, x, y),
                reference_rgb(fixed_reference_capture, width, x, y).map(Vec3::from),
            ) else {
                return Err("Metric v2 rejected capture: missing or nonfinite RGB.".into());
            };
            candidate.push(c.to_array().map(f64::from));
            fixed_reference.push(r.to_array().map(f64::from));
        }
    }
    // Retain the full geometry for scene-image/silhouette comparisons and also
    // report a one-pixel 8-neighbor erosion for shading-integral comparisons.
    let mut eroded = vec![false; geometry_mask.len()];
    if width >= 3 && height >= 3 {
        for y in 1..height as usize - 1 {
            for x in 1..width as usize - 1 {
                eroded[y * width as usize + x] = (y - 1..=y + 1)
                    .all(|yy| (x - 1..=x + 1).all(|xx| geometry_mask[yy * width as usize + xx]));
            }
        }
    }
    let score = |mask: &[bool]| {
        metrics::score(
            &candidate,
            &fixed_reference,
            mask,
            width as usize,
            height as usize,
        )
    };
    let report = match score(&geometry_mask) {
        Ok(report) => report,
        Err(reason) => return Err(format!("Metric v2 rejected capture: {reason}.")),
    };
    let shading = match score(&eroded) {
        Ok(report) => report,
        Err(reason) => return Err(format!("Metric v2 rejected capture: {reason}.")),
    };
    if let Err(reason) = export_capture(
        &candidate,
        &fixed_reference,
        &geometry_mask,
        width,
        height,
        samples,
        seed,
        roughness,
        transform,
        projection,
    ) {
        return Err(format!("Capture export failed: {reason}"));
    }
    // Keep JSON on separate log lines so stress tooling can archive metric-v2
    // artifacts without interpreting rounded UI text as machine measurements.
    info!(
        "IBL_METRIC_V2 {}",
        report.json().replace(
            "\"actual_spp\":null,\"rng_seed\":null",
            &format!("\"actual_spp\":{samples},\"rng_seed\":{seed}")
        )
    );
    info!(
        "IBL_METRIC_V2_ERODED {}",
        shading.json_with_geometry_role(true).replace(
            "\"actual_spp\":null,\"rng_seed\":null",
            &format!("\"actual_spp\":{samples},\"rng_seed\":{seed}")
        )
    );
    let full = &report.unpooled[0];
    let lit = &report.unpooled[3];
    Ok(format!(
        "Metric v2 (candidate minus fixed reference; positive bias means excess)\n\
         Environment map vs fixed pathtracer reference, roughness {roughness:.2}, {samples} actual GPU spp, seed {seed}, {} sphere pixels ({} lit)\n\
         lit relRMSE {}   4×4 lit relRMSE {}   luminance bias {}   max |Δ| {}   peak {:.3}\n\
         core bias {}   skirt bias {}\n\
         dark leakage mean/P {}   p99/P {}   integrated positive excess {}\n\
         Masks and scales use only the fixed reference. Origin-anchored 4×4 tiles use geometry-population weights. null means undefined; floor flags/reasons and full/eroded results are in metric-v2 logs. Reference remains unqualified; linear exposed light, tonemap off.",
        full.population, lit.population, lit.relative_rmse.display(),
        report.pooled_4x4[3].relative_rmse.display(), full.signed_bias.display(),
        full.maximum_error.display(), report.peak, report.unpooled[1].signed_bias.display(),
        report.unpooled[2].signed_bias.display(), report.unpooled[4].leakage_mean.display(),
        report.unpooled[4].leakage_p99.display(), report.integrated_positive_excess.display(),
    ))
}

/// Transparent except where the environment map is brighter than the pathtracer.
/// Drawn stretched over the window, so it sits on the same sphere.
fn error_overlay_rgba(
    reference: &[u8],
    pathtracer: &[u8],
    transform: &GlobalTransform,
    projection: &Projection,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let world_from_view = transform.to_matrix();
    let world_from_clip = world_from_view * projection.get_clip_from_view().inverse();
    let origin = transform.translation();
    let stride = row_stride(width);
    let width_f = width as f32;
    let height_f = height as f32;
    let mut max_excess = 0.0f32;
    let mut excesses = vec![0.0f32; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let Some((reference, pathtracer)) = sphere_pixel(
                reference,
                pathtracer,
                stride,
                x,
                y,
                width_f,
                height_f,
                origin,
                world_from_clip,
            ) else {
                continue;
            };
            let excess = (luminance(reference) - luminance(pathtracer)).max(0.0);
            max_excess = max_excess.max(excess);
            excesses[(y * width + x) as usize] = excess;
        }
    }

    let excess_scale = max_excess.max(1.0e-4);
    let mut rgba = vec![0u8; (width * height * 4) as usize];
    for (i, excess) in excesses.into_iter().enumerate() {
        let red = (excess / excess_scale).sqrt().clamp(0.0, 1.0);
        if red < 0.04 {
            continue;
        }
        let offset = i * 4;
        rgba[offset] = 255;
        rgba[offset + 3] = (red * 220.0) as u8;
    }
    rgba
}

fn sphere_pixel(
    reference: &[u8],
    pathtracer: &[u8],
    stride: usize,
    x: u32,
    y: u32,
    width: f32,
    height: f32,
    origin: Vec3,
    world_from_clip: Mat4,
) -> Option<(Vec3, Vec3)> {
    let uv = Vec2::new((x as f32 + 0.5) / width, (y as f32 + 0.5) / height);
    let ndc = uv * 2.0 - 1.0;
    let target = world_from_clip * Vec4::new(ndc.x, -ndc.y, 1.0, 1.0);
    let direction = (target.xyz() / target.w - origin).normalize_or_zero();
    if direction == Vec3::ZERO || !ray_hits_unit_sphere(origin, direction) {
        return None;
    }
    Some((
        pixel_rgb(reference, stride, x, y)?,
        Vec3::from(reference_rgb(pathtracer, width as u32, x, y)?),
    ))
}

fn ray_hits_unit_sphere(origin: Vec3, direction: Vec3) -> bool {
    let b = origin.dot(direction);
    let c = origin.length_squared() - 1.0;
    let discriminant = b * b - c;
    if discriminant < 0.0 {
        return false;
    }
    let root = discriminant.sqrt();
    -b - root > 0.0 || -b + root > 0.0
}

fn luminance(rgb: Vec3) -> f32 {
    rgb.dot(Vec3::new(0.2126, 0.7152, 0.0722))
}

fn row_stride(width: u32) -> usize {
    let tight = width as usize * 8;
    tight.div_ceil(256) * 256
}

fn pixel_rgb(buffer: &[u8], stride: usize, x: u32, y: u32) -> Option<Vec3> {
    let offset = y as usize * stride + x as usize * 8;
    let bytes = buffer.get(offset..offset + 6)?;
    let rgb = Vec3::new(
        f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])),
        f16_to_f32(u16::from_le_bytes([bytes[2], bytes[3]])),
        f16_to_f32(u16::from_le_bytes([bytes[4], bytes[5]])),
    );
    rgb.is_finite().then_some(rgb)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    let out = match exp {
        0 if mant == 0 => sign << 31,
        0 => {
            let mut mantissa = mant as u32;
            let mut exponent = -14;
            while mantissa & 0x400 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            mantissa &= 0x3ff;
            let exponent = (exponent + 127) as u32;
            (sign << 31) | (exponent << 23) | (mantissa << 13)
        }
        31 => (sign << 31) | 0x7f80_0000 | ((mant as u32) << 13),
        _ => {
            let exponent = (exp as i32 - 15 + 127) as u32;
            (sign << 31) | (exponent << 23) | ((mant as u32) << 13)
        }
    };
    f32::from_bits(out)
}

/// Files are created in an exclusive run directory; completed runs are never overwritten.
fn export_capture(
    candidate: &[[f64; 3]],
    reference: &[[f64; 3]],
    geometry: &[bool],
    width: u32,
    height: u32,
    actual_spp: u64,
    seed: u32,
    roughness: f32,
    transform: &GlobalTransform,
    projection: &Projection,
) -> Result<(), String> {
    let Some(root) = std::env::var_os("IBL_CAPTURE_DIR") else {
        return Ok(());
    };
    let root = std::path::PathBuf::from(root);
    std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let directory = root.join(format!("run-{stamp}-{}", std::process::id()));
    std::fs::create_dir(&directory).map_err(|e| e.to_string())?;
    let write = |name: &str, bytes: &[u8]| -> Result<(), String> {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join(name))
            .map_err(|e| e.to_string())?;
        file.write_all(bytes).map_err(|e| e.to_string())
    };
    let dense = |pixels: &[[f64; 3]]| {
        pixels
            .iter()
            .flat_map(|p| p.iter().flat_map(|v| (*v as f32).to_le_bytes()))
            .collect::<Vec<_>>()
    };
    write("candidate.f32", &dense(candidate))?;
    write("reference.f32", &dense(reference))?;
    write(
        "geometry.mask",
        &geometry.iter().map(|&v| u8::from(v)).collect::<Vec<_>>(),
    )?;
    let hash_bytes = |bytes: &[u8]| {
        let value = bytes.iter().fold(0xcbf29ce484222325_u64, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
        });
        format!("fnv1a64:{value:016x}")
    };
    let hash_file = |path: &std::path::Path| -> Result<String, String> {
        use std::io::Read;
        let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let mut value = 0xcbf29ce484222325_u64;
        let mut buffer = [0u8; 65536];
        loop {
            let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
            if count == 0 {
                break;
            }
            for byte in &buffer[..count] {
                value = (value ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
            }
        }
        Ok(format!("fnv1a64:{value:016x}"))
    };
    // Match Bevy's default file asset reader base-path precedence.
    let asset_base = std::env::var_os("BEVY_ASSET_ROOT")
        .or_else(|| std::env::var_os("CARGO_MANIFEST_DIR"))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::env::current_exe()
                .unwrap_or_default()
                .parent()
                .unwrap_or(std::path::Path::new("."))
                .to_path_buf()
        });
    let source_path = asset_base.join("assets").join(format!(
        "environment_maps/{}_specular.ktx2",
        ibl_stress_stem()
    ));
    let source_hash = hash_file(&source_path);
    let build_hash = std::env::current_exe()
        .map_err(|e| e.to_string())
        .and_then(|path| hash_file(&path));
    // These bytes are embedded in this executable, so working-tree edits after
    // compilation cannot masquerade as the build's shader/model provenance.
    let model_hash = hash_bytes(include_bytes!(
        "../../crates/bevy_solari/src/scene/brdf.wesl"
    ));
    let shader_hash = hash_bytes(include_bytes!(
        "../../crates/bevy_solari/src/pathtracer/pathtracer.wesl"
    ));
    let dfg_hash = hash_bytes(include_bytes!(
        "../../crates/bevy_pbr/src/environment_map/dfg.ktx2"
    ));
    let manifest = serde_json::json!({
        "schema_version": 1, "width": width, "height": height,
        "format": "rgb32float-le", "encoding": "rgb32float-le", "row_stride_bytes": width * 12,
        "candidate_file": "candidate.f32", "reference_file": "reference.f32", "geometry_file": "geometry.mask",
        "geometry_encoding": "u8_0_or_1", "reference_qualified": false,
        "actual_spp": actual_spp, "seed": seed,
        "gpu_capture_format": "rgba32float-le", "gpu_capture_row_stride_bytes": reference_capture::stride(width),
        "identity": {
            "environment": ibl_stress_stem(), "view": std::env::var("IBL_VIEW").unwrap_or_else(|_| "front".into()),
            "roughness": roughness, "exposure_ev100": 11.0,
            "material": {"metallic":1.0,"base_color_linear":[1.0,1.0,1.0],"perceptual_roughness":roughness},
            "filter": std::env::var("IBL_FILTER").unwrap_or_else(|_| "karis".into()),
            "model": "solari_environment_compensated_brdf",
            "camera_world_from_view": transform.to_matrix().to_cols_array(),
            "camera_clip_from_view": projection.get_clip_from_view().to_cols_array(),
            "source_hash": source_hash.as_ref().ok(), "model_hash": model_hash,
            "build_hash": build_hash.as_ref().ok(), "shader_hash": shader_hash, "dfg_hash": dfg_hash,
            "hash_algorithm": "fnv1a64_noncryptographic",
            "source_hash_unavailable_reason": source_hash.as_ref().err(),
            "build_hash_unavailable_reason": build_hash.as_ref().err()
        }
    });
    // Write the manifest last, so consumers discover only complete captures.
    write(
        "manifest.json",
        &serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?,
    )?;
    info!(
        "IBL_CAPTURE_MANIFEST {}",
        directory.join("manifest.json").display()
    );
    Ok(())
}
