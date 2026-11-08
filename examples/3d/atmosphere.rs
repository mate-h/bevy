//! This example showcases pbr atmospheric scattering
//!
//! Environment variables (all optional; used by the cloud validation scripts):
//! - Scene: `ATMOSPHERE_VIEW=hero|sun|up|under|backlit`, `ATMOSPHERE_CAM`, `ATMOSPHERE_SUN_ELEV_DEG`,
//!   `ATMOSPHERE_SUNSET`, `ATMOSPHERE_TERRAIN`, `ATMOSPHERE_NO_CLOUDS`, `ATMOSPHERE_NO_CIRRUS`,
//!   `ATMOSPHERE_NO_CUMULUS` (cirrus only), `ATMOSPHERE_WIND=slow|opposing`, `ATMOSPHERE_NO_TAA`,
//!   `ATMOSPHERE_MESH_PLANE`, `ATMOSPHERE_MSAA=2|4|8`, `ATMOSPHERE_WINDOW=w,h`,
//!   `ATMOSPHERE_CLOUD_QUALITY=low|medium|high|cinematic`, `ATMOSPHERE_CLOUD_LIGHT_TAPS=<n>`, `ATMOSPHERE_SECOND_THIN=1` (cirrostratus layer),
//!   `ATMOSPHERE_INFLUENCE=hole|coast` (a world-anchored clear region; a maritime/continental coast), `ATMOSPHERE_CLOUD_STEP_SAMPLES=<n>`,
//!   `ATMOSPHERE_CLOUD_PROFILE=nubis|adiabatic|nubis_envelope`,
//!   `ATMOSPHERE_CLOUD_LWC=<g/m3>`,
//!   `ATMOSPHERE_CLOUD_DENSITY=<x>`,
//!   `ATMOSPHERE_VOXEL_CLOUD=parkouring|stormbird|none` (a placed Nubis voxel cloud, default
//!   parkouring; convert the pack with `tools/nvdf_to_ktx2.py` into `assets/volumes/nubis/`),
//!   `ATMOSPHERE_VOXEL_ONLY=1` (no sky cumulus or cirrus: the voxel cloud alone),
//!   `ATMOSPHERE_VOXEL_BOUNDS=0` (start with its box hidden, B toggles),
//!   `ATMOSPHERE_LUT_MODE`.
//! - Near-field froxel volumetrics (`VolumetricsSettings`, Phase 2): `ATMOSPHERE_FROXELS=1`
//!   enables the grid (8 px tiles x 64 slices to 128 m; the haze march then covers only the far
//!   field, and the camera's `VolumetricFog` per-pixel pass is skipped: the grid consumes the
//!   `FogVolume`s), `ATMOSPHERE_FROXEL_DISTANCE=<m>` its range, `ATMOSPHERE_FROXEL_FOG=<1/m>` a
//!   height fog extinction at the ground (`ATMOSPHERE_FROXEL_FOG_FALLOFF=<1/m>`, default 0.01),
//!   `ATMOSPHERE_FROXEL_AMBIENT=<scale>` the sky ambient's scale (default 1; the grid turns on the
//!   camera's atmosphere probe, whose harmonics are the ambient).
//!   `ATMOSPHERE_FROXEL_TEST_LIGHT=1|<lm>` spawns a shadow-casting `VolumetricLight` point light
//!   (1e9 lm, or the given flux), a `FogVolume` box (sigma_t 0.02/m, a ~200 m visibility fog;
//!   `ATMOSPHERE_FROXEL_TEST_FOG=<1/m>`) and occluders in front of the hero camera so local-light
//!   shafts and shadowed fog are visible (works without the grid too, via the per-pixel
//!   `VolumetricFog` pass); with the grid it also spawns two alpha-blended panes at 30 and 100 m
//!   (red, blue) that take the froxel aerial perspective in the transparent pass
//!   (`ATMOSPHERE_FROXEL_TRANSPARENT=0` turns that apply off).
//! - Time: `ATMOSPHERE_CAM_STEP`, `ATMOSPHERE_FREEZE`, `ATMOSPHERE_FIXED_DT`,
//!   `ATMOSPHERE_CLOUD_TIMESCALE=<x>` (cloud clock multiplier; `T` toggles a 60x time-lapse),
//!   `ATMOSPHERE_SUN_SWEEP=<deg/frame>,<first>,<last>` (turns the sun by a fixed angle per frame),
//!   `ATMOSPHERE_WIND_SCALE=<x>` (wind profile multiplier, 0..2; `-` / `=`).
//! - Settings panel (Tab): tabs of sliders for the wind, cumulus, cirrus, camera and the debug
//!   view; `ATMOSPHERE_SETTINGS=wind|clouds|cirrus|scene|froxels|debug` starts with it open on that tab,
//!   `ATMOSPHERE_DEBUG_VIEW=1` on the LUT / buffer previews.
//! - Cloud shadows: `ATMOSPHERE_SHADOW_INTERLEAVE`, `ATMOSPHERE_SHADOW_SAMPLES`,
//!   `ATMOSPHERE_SHADOW_TEMPORAL_ALPHA`.
//! - Capture / runs: `ATMOSPHERE_BACKGROUND`, `ATMOSPHERE_SCREENSHOT`, `ATMOSPHERE_SCREENSHOT_SECS`,
//!   `ATMOSPHERE_SCREENSHOT_FRAMES`, `ATMOSPHERE_SCREENSHOT_FROM`, `ATMOSPHERE_EXIT_SECS`, `ATMOSPHERE_EXIT_FRAME`,
//!   `ATMOSPHERE_DIAGNOSTICS`, `ATMOSPHERE_CLEAR`.
//! - Read by `bevy_pbr`: `ATMOSPHERE_DBG` (comma-separated `ATM_DBG_*` shader defs, see
//!   `atmosphere/resources.rs`), `ATMOSPHERE_GPU_TIMING` (per-pass CSV), `CLOUD_SKY_SH_BOUNDARY=1`
//!   (the cloud field's diffusion solve with the sky and ground light at its boundary, off by default).
#[cfg(feature = "free_camera")]
use bevy::camera_controller::free_camera::{FreeCamera, FreeCameraPlugin, FreeCameraState};
use std::f32::consts::PI;

use bevy::{
    anti_alias::taa::TemporalAntiAliasing,
    camera::Exposure,
    core_pipeline::tonemapping::Tonemapping,
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    ecs::template::OptionTemplate,
    feathers::{
        controls::{FeathersSlider, FeathersTab, FeathersTabList},
        dark_theme::create_dark_theme,
        display::{caption, label_small},
        theme::{ThemeBackgroundColor, ThemeBorderColor, UiTheme},
        tokens, FeathersPlugins,
    },
    input::keyboard::KeyCode,
    light::{
        atmosphere::{Aerosol, ScatteringMedium},
        light_consts::lux,
        Atmosphere, AtmosphereEnvironmentMapLight, FogVolume, SunDisk, VolumetricFog,
        VolumetricLight,
    },
    pbr::{
        AtmosphereMode, AtmosphereSettings, AtmosphereWind, CloudTimeScale, CloudWeather,
        CloudBlend, CloudInfluence, CloudInfluenceShape, CloudInfluenceValues, CloudLayerOf,
        CloudProfiles, ThinCloudLayer, VolumetricCloudLayer, VoxelCloud,
        AtmosphereDebugView, AtmosphereDebugViewPlugin,
        DefaultOpaqueRendererMethod,
        VolumetricCloudSettings, VolumetricsSettings,
    },
    post_process::bloom::Bloom,
    prelude::*,
    render::{
        diagnostic::RenderDiagnosticsPlugin,
        view::screenshot::{save_to_disk, Screenshot},
    },
    ui_widgets::{tablist_self_update, SelectedTab, SliderPrecision, SliderValue, ValueChange},
};

#[derive(Resource)]
struct GameState {
    paused: bool,
    planet_scale_view: bool,
}

impl Default for GameState {
    fn default() -> Self {
        Self {
            paused: true,
            planet_scale_view: false,
        }
    }
}

#[derive(Resource)]
struct AtmospherePresets {
    earth: Handle<ScatteringMedium>,
    mars: Handle<ScatteringMedium>,
}

fn main() {
    let mut app = App::new();
    // Continuous mode waits for the OS redraw callback, which macOS stops delivering for occluded
    // windows (the app then freezes in the background). In background test runs drive updates
    // from the event loop instead (reactive with ~0 wait => ControlFlow::Poll).
    let background = std::env::var("ATMOSPHERE_BACKGROUND").is_ok();
    let update_mode = if background {
        bevy::winit::UpdateMode::reactive(core::time::Duration::from_micros(100))
    } else {
        bevy::winit::UpdateMode::Continuous
    };
    app.insert_resource(bevy::winit::WinitSettings {
        focused_mode: update_mode,
        unfocused_mode: update_mode,
    })
    .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .insert_resource(ClearColor(
            // Debug: `ATMOSPHERE_CLEAR=r,g,b` (linear) to test background leakage through the
            // sky's dual-source transmittance blend.
            std::env::var("ATMOSPHERE_CLEAR")
                .ok()
                .and_then(|v| {
                    let c: Vec<f32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                    (c.len() == 3).then(|| Color::linear_rgb(c[0], c[1], c[2]))
                })
                .unwrap_or(Color::BLACK),
        ))
        .insert_resource(GameState::default())
        .insert_resource(GlobalAmbientLight::NONE)
        .add_plugins((
            // `ATMOSPHERE_BACKGROUND=1` (used by the test scripts): start unfocused so macOS does
            // not activate the app / steal focus; rendering stays continuous when unfocused.
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    focused: !background,
                    visible: !background,
                    // No vsync in background runs: an occluded window's drawables are not
                    // display-synced, so acquisition doesn't stall.
                    present_mode: if background {
                        bevy::window::PresentMode::AutoNoVsync
                    } else {
                        bevy::window::PresentMode::AutoVsync
                    },
                    // Fixed 1280x720 physical pixels so dumps match across displays (retina = 2x);
                    // `ATMOSPHERE_WINDOW=w,h` for another size.
                    resolution: {
                        let (w, h) = window_size();
                        bevy::window::WindowResolution::new(w, h).with_scale_factor_override(1.0)
                    },
                    ..default()
                }),
                ..default()
            }),
            #[cfg(feature = "free_camera")]
            FreeCameraPlugin,
        ))
        .add_plugins((AtmosphereDebugViewPlugin, FeathersPlugins))
        .insert_resource(UiTheme(create_dark_theme()))
        .init_resource::<WindControl>()
        .init_resource::<PointerOnUi>()
        // After picking has found what the pointer is over, before the camera controller
        // (`RunFixedMainLoop`) and `sun_drag` read the mouse.
        .add_systems(PreUpdate, pointer_on_ui.after(bevy::picking::PickingSystems::Last))
        .add_systems(
            Startup,
            (
                setup_camera_fog,
                setup_terrain_scene,
                setup_settings_panel.after(setup_camera_fog),
                spawn_voxel_cloud,
                print_controls,
            ),
        )
        .add_systems(
            Update,
            (
                dynamic_scene,
                sun_drag,
                sun_sweep,
                atmosphere_controls,
                settings_panel_controls,
                populate_debug_page,
                debug_overlay_scale,
                (wind_speed_controls, apply_wind).chain(),
                sync_setting_sliders,
                cycle_cloud_profiles,
                visibility_toggle.after(atmosphere_controls),
                camera_pose_title,
                draw_voxel_cloud_bounds,
                report_voxel_cloud_data,
            ),
        );

    // Optional benchmarking hooks (used for profiling the atmosphere/cloud passes):
    // - `ATMOSPHERE_DIAGNOSTICS=1` logs frame time and per-pass render timings once per second.
    // - `ATMOSPHERE_SCREENSHOT=<path.png>` saves a screenshot after `ATMOSPHERE_SCREENSHOT_SECS` (default 8 s).
    // - `ATMOSPHERE_EXIT_SECS=<secs>` exits the app after the given number of seconds.
    // - `ATMOSPHERE_EXIT_FRAME=<n>` exits after n frames (e.g. just past the last dump frame).
    let bench = BenchSettings::from_env();
    if bench.diagnostics {
        app.add_plugins((
            FrameTimeDiagnosticsPlugin::default(),
            RenderDiagnosticsPlugin,
            LogDiagnosticsPlugin::default(),
        ));
    }
    // `ATMOSPHERE_FREEZE=1`: pause virtual time (sun motion, wind/noise offsets, animations) so
    // realtime and reference renders see identical clouds.
    if std::env::var("ATMOSPHERE_FREEZE").is_ok() {
        app.add_systems(Startup, |mut time: ResMut<Time<Virtual>>| time.pause());
    }
    if let Some(scale) = std::env::var("ATMOSPHERE_CLOUD_TIMESCALE").ok().and_then(|v| v.parse().ok()) {
        app.insert_resource(CloudTimeScale(scale));
    }
    // `ATMOSPHERE_FIXED_DT=1`: advance time by exactly 1/60 s per frame, so animated captures at
    // a given frame are reproducible regardless of GPU speed.
    if std::env::var("ATMOSPHERE_FIXED_DT").is_ok() {
        app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
            core::time::Duration::from_secs_f64(1.0 / 60.0),
        ));
    }
    // `ATMOSPHERE_CAM_STEP=x,y,z`: move the camera by this many metres every frame (independent
    // of time, so frozen runs still move); used to test shadow-map / field recentering.
    if let Some(step) = std::env::var("ATMOSPHERE_CAM_STEP").ok().and_then(|v| {
        let c: Vec<f32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        (c.len() == 3).then(|| Vec3::new(c[0], c[1], c[2]))
    }) {
        app.add_systems(Update, move |mut cams: Query<&mut Transform, With<Camera3d>>| {
            for mut t in &mut cams {
                t.translation += step;
            }
        });
    }
    // `ATMOSPHERE_CLOUD_LIFETIME_OFF_SECS=<t>`: at t (real time), turn the cloud life cycle off
    // (static weather); used to test that the weather map re-bakes.
    if let Some(secs) = env_f32("ATMOSPHERE_CLOUD_LIFETIME_OFF_SECS") {
        app.add_systems(
            Update,
            move |time: Res<Time<Real>>, mut layers: Query<&mut VolumetricCloudLayer>| {
                if time.elapsed_secs() >= secs {
                    for mut layer in &mut layers {
                        if layer.lifetime != 0.0 {
                            layer.lifetime = 0.0;
                        }
                    }
                }
            },
        );
    }
    // `ATMOSPHERE_AEROSOL_EDIT_SECS=<t>`: at t, switch the Earth medium's aerosol to
    // `Aerosol::URBAN` by editing the asset in place (same handle); used to test that the
    // atmosphere LUTs follow in-place medium edits.
    if let Some(secs) = env_f32("ATMOSPHERE_AEROSOL_EDIT_SECS") {
        app.add_systems(
            Update,
            // Real time: `ATMOSPHERE_FREEZE` pauses virtual time.
            move |time: Res<Time<Real>>,
                  presets: Res<AtmospherePresets>,
                  mut media: ResMut<Assets<ScatteringMedium>>,
                  mut done: Local<bool>| {
                if !*done && time.elapsed_secs() >= secs {
                    *done = true;
                    if let Some(mut medium) = media.get_mut(&presets.earth) {
                        *medium = ScatteringMedium::earth_with_aerosol(256, 256, Aerosol::URBAN);
                    }
                }
            },
        );
    }
    if bench.screenshot.is_some()
        || bench.exit_secs.is_some()
        || bench.exit_frame.is_some()
        || bench.no_clouds
    {
        app.insert_resource(bench).add_systems(Update, bench_hooks);
    }
    app.run();
}

#[derive(Resource, Default)]
struct BenchSettings {
    diagnostics: bool,
    screenshot: Option<String>,
    screenshot_secs: f32,
    exit_secs: Option<f32>,
    exit_frame: Option<u32>,
    no_clouds: bool,
    sunset: bool,
    screenshot_taken: bool,
    startup_frames: Option<u32>,
    frames_from: u32,
    frame: u32,
}

impl BenchSettings {
    fn from_env() -> Self {
        let var = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
        Self {
            diagnostics: var("ATMOSPHERE_DIAGNOSTICS").is_some(),
            screenshot: var("ATMOSPHERE_SCREENSHOT"),
            screenshot_secs: var("ATMOSPHERE_SCREENSHOT_SECS")
                .and_then(|v| v.parse().ok())
                .unwrap_or(8.0),
            exit_secs: var("ATMOSPHERE_EXIT_SECS").and_then(|v| v.parse().ok()),
            exit_frame: var("ATMOSPHERE_EXIT_FRAME").and_then(|v| v.parse().ok()),
            no_clouds: var("ATMOSPHERE_NO_CLOUDS").is_some(),
            sunset: var("ATMOSPHERE_SUNSET").is_some(),
            screenshot_taken: false,
            startup_frames: var("ATMOSPHERE_SCREENSHOT_FRAMES").and_then(|v| v.parse().ok()),
            frames_from: var("ATMOSPHERE_SCREENSHOT_FROM")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            frame: 0,
        }
    }
}

fn bench_hooks(
    mut commands: Commands,
    time: Res<Time<Real>>,
    mut bench: ResMut<BenchSettings>,
    mut exit: MessageWriter<AppExit>,
    cloud_layers: Query<Entity, With<VolumetricCloudLayer>>,
    offscreen: Option<Res<OffscreenTarget>>,
    second: Option<Res<SecondCameraTarget>>,
) {
    let offscreen = OffscreenTarget(offscreen.and_then(|o| o.0.clone()));
    let _sunset = bench.sunset; // sun tilt applied at spawn via ATMOSPHERE_SUNSET
    if bench.no_clouds {
        for entity in &cloud_layers {
            commands
                .entity(entity)
                .remove::<(VolumetricCloudLayer, ThinCloudLayer)>();
        }
    }
    let elapsed = time.elapsed_secs();
    if !bench.screenshot_taken
        && elapsed >= bench.screenshot_secs
        && let Some(path) = bench.screenshot.clone()
    {
        bench.screenshot_taken = true;
        commands
            .spawn(offscreen.0.clone().map_or_else(Screenshot::primary_window, Screenshot::image))
            .observe(save_to_disk(path));
        if let (Some(second), Ok(path)) = (second.as_ref(), std::env::var("ATMOSPHERE_SECOND_SCREENSHOT")) {
            commands
                .spawn(Screenshot::image(second.0.clone()))
                .observe(save_to_disk(path));
        }
    }
    // `ATMOSPHERE_SCREENSHOT_FRAMES=<n>` + `ATMOSPHERE_SCREENSHOT=<dir/prefix>`: also capture each
    // of the first n frames (startup convergence debugging) to `<prefix>_fNN.png`;
    // `ATMOSPHERE_SCREENSHOT_FROM=<f>` starts the n frames at frame f (temporal noise).
    if let (Some(n), Some(prefix)) = (bench.startup_frames, bench.screenshot.clone())
        && bench.frame >= bench.frames_from
        && bench.frame < bench.frames_from + n
    {
        commands
            .spawn(offscreen.0.clone().map_or_else(Screenshot::primary_window, Screenshot::image))
            .observe(save_to_disk(format!("{prefix}_f{:02}.png", bench.frame)));
    }
    bench.frame += 1;
    if bench.exit_secs.is_some_and(|secs| elapsed >= secs)
        || bench.exit_frame.is_some_and(|f| bench.frame >= f)
    {
        exit.write(AppExit::Success);
    }
}

fn print_controls() {
    println!("Atmosphere Example Controls:");
    println!("    1          - Switch to lookup texture rendering method");
    println!("    2          - Switch to raymarched rendering method");
    println!("    3          - Switch to Earth atmosphere");
    println!("    4          - Switch to Mars atmosphere");
    println!("    5          - Switch between normal and planet scale (space) view");
    println!("    C          - Toggle cloud layer");
    println!("    Left/Right - Decrease/Increase cloud coverage");
    println!("    P          - Cycle cloud profiles (physical, Nubis, adiabatic, Nubis envelope)");
    println!("    Enter      - Pause/Resume sun motion");
    println!("    T          - Toggle 60x cloud time-lapse (wind, rising thermals, cloud life cycle)");
    println!("    Right drag - Slide the time of day (sun position)");
    println!("    Up/Down    - Increase/Decrease exposure");
    println!("    - / =      - Decrease/Increase the wind speed (whole profile: drift and lean), 0x .. 2x");
    println!("    Tab        - Toggle the settings panel (wind, clouds, cirrus, scene, debug view)");
    println!("    [ / ]      - Debug view: move the slice of the 3D textures down/up");
    println!("    B          - Show/hide the voxel cloud's bounding box");
}

/// The settings panel's pages, one per tab.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
enum Page {
    #[default]
    Wind,
    Clouds,
    Cirrus,
    Scene,
    Froxels,
    Debug,
}

/// The settings panel's root node (Tab shows / hides it).
#[derive(Component, Clone, Copy, Default)]
struct SettingsPanel;

/// The settings panel's tab strip.
#[derive(Component, Clone, Copy, Default)]
struct SettingsTabList;

/// A tab of the settings panel and the page it shows.
#[derive(Component, Clone, Copy, Default)]
struct SettingsTab(Page);

/// A page of the settings panel, shown while its tab is selected.
#[derive(Component, Clone, Copy, Default)]
struct SettingsPage(Page);

/// Whether the settings panel is open, and on which page.
#[derive(Resource, Default)]
struct SettingsUi {
    open: bool,
    page: Page,
}

/// A value the settings panel edits. Read from and written to the scene, so the sliders follow
/// the keyboard shortcuts too.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
enum Setting {
    #[default]
    WindSpeed,
    WindShear,
    WindHeading,
    TimeScale,
    Coverage,
    Density,
    Detail,
    CloudSize,
    Updraft,
    Lifetime,
    CirrusOpticalDepth,
    CirrusCoverage,
    CirrusFallSpeed,
    CirrusTop,
    Exposure,
    MarchSteps,
    LightTaps,
    FroxelEnabled,
    FroxelDistance,
    FroxelUniformity,
    FroxelFogDensity,
    FroxelFogFalloff,
    DebugSlice,
}

/// The [`Setting`] a slider edits.
#[derive(Component, Clone, Copy, Default)]
struct SettingSlider(Setting);

/// The wind as edited by the settings panel and - / =: the atmosphere's wind profile as spawned
/// (`base`), scaled, with its shear (the change of the wind above the surface) scaled again, and
/// turned to a heading.
#[derive(Resource)]
struct WindControl {
    base: Option<AtmosphereWind>,
    /// Multiplier on the whole profile (drift and lean), 0..2.
    speed: f32,
    /// Extra multiplier on the shear only (lean, fall streaks), 0..2.
    shear: f32,
    /// Heading of the surface wind, degrees from +X toward +Z.
    heading_deg: f32,
}

impl Default for WindControl {
    fn default() -> Self {
        Self {
            base: None,
            // `ATMOSPHERE_WIND_SCALE=<x>` sets the speed at startup.
            speed: env_f32("ATMOSPHERE_WIND_SCALE").map_or(1.0, |s| s.clamp(0.0, 2.0)),
            shear: 1.0,
            heading_deg: 0.0,
        }
    }
}

fn heading_deg(velocity: Vec3) -> f32 {
    velocity.z.atan2(velocity.x).to_degrees().rem_euclid(360.0)
}

/// Applies [`WindControl`] to the atmosphere's wind profile. Up to 2x speed and shear: with
/// stronger shear the lean changes across air masses faster than the field can follow
/// (`VolumetricCloudLayer::wind_shear`).
fn apply_wind(mut control: ResMut<WindControl>, mut winds: Query<&mut AtmosphereWind>) {
    if control.base.is_none() {
        let Some(wind) = winds.iter().next() else {
            return;
        };
        control.heading_deg = wind.levels.first().map_or(0.0, |&(_, v)| heading_deg(v));
        control.base = Some(wind.clone());
    } else if !control.is_changed() {
        return;
    }
    let Some(base) = control.base.as_ref() else {
        return;
    };
    let surface = base.levels.first().map_or(Vec3::ZERO, |&(_, v)| v);
    let turn = Quat::from_rotation_y(-(control.heading_deg - heading_deg(surface)).to_radians());
    let levels: Vec<(f32, Vec3)> = base
        .levels
        .iter()
        .map(|&(z, v)| (z, turn * (surface + (v - surface) * control.shear) * control.speed))
        .collect();
    for mut wind in &mut winds {
        wind.levels = levels.clone();
    }
}

/// - / = scale the whole wind profile: the clouds' drift and, through the shear, their lean.
fn wind_speed_controls(keyboard_input: Res<ButtonInput<KeyCode>>, mut control: ResMut<WindControl>) {
    let speed = if keyboard_input.just_pressed(KeyCode::Equal) {
        (control.speed + 0.25).min(2.0)
    } else if keyboard_input.just_pressed(KeyCode::Minus) {
        (control.speed - 0.25).max(0.0)
    } else {
        return;
    };
    control.speed = speed;
    let (bottom, top) = bevy::pbr::physical_profile_bounds();
    let shear = control
        .base
        .as_ref()
        .map_or(0.0, |b| b.mean_shear(bottom, top).length())
        * speed
        * control.shear;
    println!("Wind x{speed:.2} (cloud-layer shear {:.1} m/s per km)", shear * 1e3);
}

/// Edge of the debug grid at UI scale 1 (logical px): 5 x 4 tiles with labels, below the tabs.
const DEBUG_GRID: Vec2 = Vec2::new(5.0 * (DEBUG_TILE + 4.0) + 24.0, 4.0 * (DEBUG_TILE + 26.0) + 110.0);

/// The settings panel at the display's scale: the window is pinned to scale factor 1 (dumps
/// match across displays), so on a HiDPI display the panel would be half size. The UI scale is
/// the display's own scale factor (2 on HiDPI), at most what keeps the debug grid within 60 % of
/// the window's width and its height.
fn debug_overlay_scale(windows: Query<&Window, With<bevy::window::PrimaryWindow>>, mut ui_scale: ResMut<UiScale>) {
    let Ok(window) = windows.single() else {
        return;
    };
    let fit = (0.6 * window.physical_width() as f32 / DEBUG_GRID.x)
        .min(0.98 * window.physical_height() as f32 / DEBUG_GRID.y);
    let scale = window.resolution.base_scale_factor().min(fit).max(0.5);
    if (ui_scale.0 - scale).abs() > 1e-3 {
        ui_scale.0 = scale;
    }
}

/// Edge of a debug preview in the overlay (logical px).
const DEBUG_TILE: f32 = 136.0;

/// Width of the settings pages other than the debug grid (logical px).
const SETTINGS_WIDTH: f32 = 300.0;

fn settings_tab(text: &'static str, page: Page) -> impl Scene {
    bsn! {
        @FeathersTab {
            @caption: bsn! { @caption(text) }
        }
        SettingsTab(page)
    }
}

fn settings_page(page: Page) -> impl Scene {
    bsn! {
        Node {
            display: Display::None,
            flex_direction: FlexDirection::Column,
            row_gap: px(4),
            padding: px(8),
            min_width: px(SETTINGS_WIDTH),
            border: UiRect {
                left: px(1),
                right: px(1),
                bottom: px(1),
            },
        }
        SettingsPage(page)
        ThemeBackgroundColor(tokens::PANE_BODY_BG)
        ThemeBorderColor(tokens::TAB_STRIP_BG)
    }
}

/// A labelled slider editing `setting` over `min..=max`, shown with `precision` decimals.
fn setting_row(label: &'static str, setting: Setting, min: f32, max: f32, precision: i32) -> impl Scene {
    bsn! {
        Node {
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            row_gap: px(2),
        }
        Children [
            @label_small(label)
            --
            @FeathersSlider {
                @min: min,
                @max: max,
            }
            SliderPrecision(precision)
            SettingSlider(setting)
            on(apply_setting)
        ]
    }
}

/// The settings panel (Tab): tabbed pages of sliders for the wind, the cumulus and cirrus layers,
/// the camera, and the debug view (`AtmosphereDebugView`, a grid of live previews of the
/// atmosphere's lookup tables, cloud buffers, cloud field, weather and noise).
fn settings_panel() -> impl Scene {
    bsn! {
        Node {
            position_type: PositionType::Absolute,
            top: px(8),
            right: px(8),
            display: Display::Flex,
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Stretch,
        }
        SettingsPanel
        Children [
            @FeathersTabList {
                @selected: OptionTemplate::Some(#wind_tab),
            }
            SettingsTabList
            on(tablist_self_update)
            Children [
                #wind_tab
                @settings_tab("Wind", Page::Wind)
                --
                @settings_tab("Clouds", Page::Clouds)
                --
                @settings_tab("Cirrus", Page::Cirrus)
                --
                @settings_tab("Scene", Page::Scene)
                --
                @settings_tab("Froxels", Page::Froxels)
                --
                @settings_tab("Debug", Page::Debug)
            ]
            --
            @settings_page(Page::Wind)
            Children [
                @setting_row("Wind speed (x profile, - / =)", Setting::WindSpeed, 0.0, 2.0, 2)
                --
                @setting_row("Wind shear (x, cloud lean)", Setting::WindShear, 0.0, 2.0, 2)
                --
                @setting_row("Wind heading (deg)", Setting::WindHeading, 0.0, 360.0, 0)
                --
                @setting_row("Cloud time scale (T: 60x)", Setting::TimeScale, 0.0, 120.0, 1)
                --
                @setting_row("Updraft speed (m/s)", Setting::Updraft, 0.0, 8.0, 1)
                --
                @setting_row("Cloud life time (min, 0 = static)", Setting::Lifetime, 0.0, 60.0, 1)
            ]
            --
            @settings_page(Page::Clouds)
            Children [
                @setting_row("Coverage (Left / Right)", Setting::Coverage, 0.0, 3.0, 2)
                --
                @setting_row("Density", Setting::Density, 0.0, 3.0, 2)
                --
                @setting_row("Detail noise strength", Setting::Detail, 0.0, 1.0, 2)
                --
                @setting_row("Cloud size (shape noise period, km)", Setting::CloudSize, 1.0, 20.0, 1)
            ]
            --
            @settings_page(Page::Cirrus)
            Children [
                @setting_row("Optical depth", Setting::CirrusOpticalDepth, 0.0, 3.0, 2)
                --
                @setting_row("Coverage", Setting::CirrusCoverage, 0.0, 1.0, 2)
                --
                @setting_row("Ice fall speed (m/s)", Setting::CirrusFallSpeed, 0.0, 3.0, 2)
                --
                @setting_row("Top altitude (km)", Setting::CirrusTop, 5.0, 13.0, 1)
            ]
            --
            @settings_page(Page::Scene)
            Children [
                @setting_row("Exposure (EV100, Up / Down)", Setting::Exposure, 6.0, 18.0, 1)
                --
                @setting_row("Cloud march steps", Setting::MarchSteps, 16.0, 256.0, 0)
                --
                @setting_row("Cloud light taps", Setting::LightTaps, 4.0, 24.0, 0)
            ]
            --
            @settings_page(Page::Froxels)
            Children [
                @setting_row("Enabled (0 / 1; needs ATMOSPHERE_FROXELS=1)", Setting::FroxelEnabled, 0.0, 1.0, 0)
                --
                @setting_row("Grid range d_max (m)", Setting::FroxelDistance, 16.0, 512.0, 0)
                --
                @setting_row("Slice distribution uniformity", Setting::FroxelUniformity, 0.0, 1.0, 2)
                --
                @setting_row("Height fog extinction at ground (1e-3/m)", Setting::FroxelFogDensity, 0.0, 50.0, 1)
                --
                @setting_row("Height fog falloff (1e-3/m)", Setting::FroxelFogFalloff, 0.0, 100.0, 1)
            ]
            --
            @settings_page(Page::Debug)
            Children [
                @setting_row("3D texture slice ([ / ])", Setting::DebugSlice, 0.0, 0.999, 2)
            ]
        ]
    }
}

/// Spawns the settings panel, hidden until Tab; `ATMOSPHERE_SETTINGS=wind|clouds|cirrus|scene|froxels|debug`
/// starts with it open on that page, `ATMOSPHERE_DEBUG_VIEW=1` on the debug view.
fn setup_settings_panel(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    cameras: Query<(Entity, &Camera), With<Camera3d>>,
) {
    let view = AtmosphereDebugView::all(&mut images);
    let page = match std::env::var("ATMOSPHERE_SETTINGS").as_deref() {
        _ if std::env::var("ATMOSPHERE_DEBUG_VIEW").is_ok() => Some(Page::Debug),
        Ok("wind") => Some(Page::Wind),
        Ok("clouds") => Some(Page::Clouds),
        Ok("cirrus") => Some(Page::Cirrus),
        Ok("scene") => Some(Page::Scene),
        Ok("froxels") => Some(Page::Froxels),
        Ok("debug") => Some(Page::Debug),
        _ => None,
    };
    commands.insert_resource(SettingsUi {
        open: page.is_some(),
        page: page.unwrap_or_default(),
    });
    let mut panel = commands.spawn_scene(settings_panel());
    panel.insert(if page.is_some() { Visibility::Inherited } else { Visibility::Hidden });
    // On the main camera, which renders offscreen in background runs.
    if let Some((camera, _)) = cameras.iter().min_by_key(|(_, c)| c.order) {
        panel.insert(UiTargetCamera(camera));
    }
    commands.insert_resource(view);
}

/// Fills the debug page with the previews of the debug view, once it is spawned.
fn populate_debug_page(
    mut commands: Commands,
    pages: Query<(Entity, &SettingsPage), Added<SettingsPage>>,
    view: Res<AtmosphereDebugView>,
) {
    for (page, _) in pages.iter().filter(|(_, p)| p.0 == Page::Debug) {
        commands.entity(page).with_children(|parent| {
            parent
                .spawn(Node {
                    width: px(5.0 * (DEBUG_TILE + 4.0)),
                    flex_wrap: FlexWrap::Wrap,
                    column_gap: px(4),
                    row_gap: px(2),
                    ..default()
                })
                .with_children(|grid| {
                    for panel in &view.panels {
                        grid.spawn(Node {
                            flex_direction: FlexDirection::Column,
                            width: px(DEBUG_TILE),
                            ..default()
                        })
                        .with_children(|tile| {
                            tile.spawn((
                                ImageNode::new(panel.image.clone()),
                                Node {
                                    width: px(DEBUG_TILE),
                                    height: px(DEBUG_TILE),
                                    ..default()
                                },
                            ));
                            tile.spawn((
                                Text::new(panel.label.clone()),
                                TextFont {
                                    font_size: FontSize::Px(10.0),
                                    ..default()
                                },
                                TextColor(Color::srgb(0.85, 0.85, 0.85)),
                            ));
                        });
                    }
                });
        });
    }
}

/// Tab opens / closes the settings panel; the selected tab picks the page. The engine draws the
/// debug previews only while the debug page is shown. [ and ] move the slice of every 3D texture
/// (cloud field and aerial perspective: height / distance).
fn settings_panel_controls(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut ui: ResMut<SettingsUi>,
    mut view: ResMut<AtmosphereDebugView>,
    mut panels: Query<&mut Visibility, With<SettingsPanel>>,
    lists: Query<(Entity, &SelectedTab), With<SettingsTabList>>,
    tabs: Query<(Entity, &SettingsTab)>,
    mut pages: Query<(&SettingsPage, &mut Node)>,
    mut commands: Commands,
    mut initialized: Local<bool>,
) {
    // Start on the page `SettingsUi` was set up with (the tab list starts on the wind tab).
    if !*initialized {
        let Some(tab) = tabs.iter().find(|(_, t)| t.0 == ui.page).map(|(e, _)| e) else {
            return;
        };
        for (list, _) in &lists {
            commands.entity(list).insert(SelectedTab(Some(tab)));
        }
        *initialized = true;
    } else {
        for (_, selected) in &lists {
            if let Some(page) = selected.0.and_then(|e| tabs.get(e).ok()).map(|(_, t)| t.0)
                && page != ui.page
            {
                ui.page = page;
            }
        }
    }
    if keyboard_input.just_pressed(KeyCode::Tab) {
        ui.open = !ui.open;
    }
    if ui.is_changed() {
        for mut visibility in &mut panels {
            *visibility = if ui.open { Visibility::Inherited } else { Visibility::Hidden };
        }
        for (page, mut node) in &mut pages {
            node.display = if page.0 == ui.page { Display::Flex } else { Display::None };
        }
    }
    let enabled = ui.open && ui.page == Page::Debug;
    if view.enabled != enabled {
        view.enabled = enabled;
    }
    let step = if keyboard_input.just_pressed(KeyCode::BracketRight) {
        1.0 / 32.0
    } else if keyboard_input.just_pressed(KeyCode::BracketLeft) {
        -1.0 / 32.0
    } else {
        return;
    };
    for panel in &mut view.panels {
        panel.slice = (panel.slice + step).clamp(0.0, 1.0 - 1e-3);
    }
}

/// A slider moved: write its value to the scene.
fn apply_setting(change: On<ValueChange<f32>>, sliders: Query<&SettingSlider>, mut commands: Commands) {
    let Ok(&SettingSlider(setting)) = sliders.get(change.source) else {
        return;
    };
    let value = change.value;
    commands.entity(change.source).insert(SliderValue(value));
    commands.queue(move |world: &mut World| write_setting(world, setting, value));
}

/// Keeps the sliders on the scene's values (the keyboard shortcuts change them too).
fn sync_setting_sliders(world: &mut World) {
    if !world.get_resource::<SettingsUi>().is_some_and(|ui| ui.open) {
        return;
    }
    let sliders: Vec<(Entity, Setting, f32, i32)> = world
        .query::<(Entity, &SettingSlider, &SliderValue, &SliderPrecision)>()
        .iter(world)
        .map(|(entity, slider, value, precision)| (entity, slider.0, value.0, precision.0))
        .collect();
    for (entity, setting, shown, precision) in sliders {
        // Rounded to the slider's precision: the slider prints values with more decimals in full.
        let factor = 10f32.powi(precision);
        if let Some(value) = read_setting(world, setting).map(|v| (v * factor).round() / factor)
            && (value - shown).abs() > 1e-4 * (1.0 + value.abs())
        {
            world.entity_mut(entity).insert(SliderValue(value));
        }
    }
}

/// The planet's radius: cloud layer altitudes are radii from the planet center.
fn ground_radius(world: &mut World) -> f32 {
    world
        .query::<&Atmosphere>()
        .iter(world)
        .next()
        .map_or(6_360_000.0, |a| a.inner_radius)
}

fn read_setting(world: &mut World, setting: Setting) -> Option<f32> {
    fn first<C: Component, R>(world: &mut World, f: impl Fn(&C) -> R) -> Option<R> {
        world.query::<&C>().iter(world).next().map(f)
    }
    fn cirrus<R>(world: &mut World, f: impl Fn(&ThinCloudLayer) -> R) -> Option<R> {
        world
            .query_filtered::<&ThinCloudLayer, With<Atmosphere>>()
            .iter(world)
            .next()
            .map(f)
    }
    let wind = world.resource::<WindControl>();
    Some(match setting {
        Setting::WindSpeed => wind.speed,
        Setting::WindShear => wind.shear,
        Setting::WindHeading => wind.heading_deg,
        Setting::TimeScale => world.resource::<CloudTimeScale>().0,
        Setting::Coverage => first(world, |c: &VolumetricCloudLayer| c.coverage)?,
        Setting::Density => first(world, |c: &VolumetricCloudLayer| c.cloud_density)?,
        Setting::Detail => first(world, |c: &VolumetricCloudLayer| c.detail_strength)?,
        Setting::CloudSize => first(world, |c: &VolumetricCloudLayer| c.shape_noise_scale * 1e-3)?,
        Setting::Updraft => first(world, |c: &VolumetricCloudLayer| c.updraft_speed)?,
        Setting::Lifetime => first(world, |c: &VolumetricCloudLayer| c.lifetime / 60.0)?,
        Setting::CirrusOpticalDepth => cirrus(world, |c| c.optical_depth)?,
        Setting::CirrusCoverage => cirrus(world, |c| c.coverage)?,
        Setting::CirrusFallSpeed => cirrus(world, |c| c.fall_speed)?,
        Setting::CirrusTop => {
            let ground = ground_radius(world);
            cirrus(world, |c| (c.top - ground) * 1e-3)?
        }
        Setting::Exposure => first(world, |e: &Exposure| e.ev100)?,
        Setting::MarchSteps => first(world, |s: &VolumetricCloudSettings| s.max_steps as f32)?,
        Setting::LightTaps => first(world, |s: &VolumetricCloudSettings| s.light_taps as f32)?,
        Setting::FroxelEnabled => first(world, |s: &VolumetricsSettings| s.enabled as u32 as f32)?,
        Setting::FroxelDistance => first(world, |s: &VolumetricsSettings| s.max_distance)?,
        Setting::FroxelUniformity => first(world, |s: &VolumetricsSettings| s.depth_uniformity)?,
        Setting::FroxelFogDensity => first(world, |s: &VolumetricsSettings| s.fog_density * 1e3)?,
        Setting::FroxelFogFalloff => first(world, |s: &VolumetricsSettings| s.fog_height_falloff * 1e3)?,
        Setting::DebugSlice => world.resource::<AtmosphereDebugView>().panels.first()?.slice,
    })
}

fn write_setting(world: &mut World, setting: Setting, value: f32) {
    fn each<C: Component<Mutability = bevy::ecs::component::Mutable>>(world: &mut World, f: impl Fn(&mut C)) {
        for mut c in world.query::<&mut C>().iter_mut(world) {
            f(&mut c);
        }
    }
    fn cirrus(world: &mut World, f: impl Fn(&mut ThinCloudLayer)) {
        for mut c in world
            .query_filtered::<&mut ThinCloudLayer, With<Atmosphere>>()
            .iter_mut(world)
        {
            f(&mut c);
        }
    }
    match setting {
        Setting::WindSpeed => world.resource_mut::<WindControl>().speed = value,
        Setting::WindShear => world.resource_mut::<WindControl>().shear = value,
        Setting::WindHeading => world.resource_mut::<WindControl>().heading_deg = value,
        Setting::TimeScale => world.resource_mut::<CloudTimeScale>().0 = value,
        Setting::Coverage => each(world, |c: &mut VolumetricCloudLayer| c.coverage = value),
        Setting::Density => each(world, |c: &mut VolumetricCloudLayer| c.cloud_density = value),
        Setting::Detail => each(world, |c: &mut VolumetricCloudLayer| c.detail_strength = value),
        Setting::CloudSize => each(world, |c: &mut VolumetricCloudLayer| c.shape_noise_scale = value * 1e3),
        Setting::Updraft => each(world, |c: &mut VolumetricCloudLayer| c.updraft_speed = value),
        Setting::Lifetime => each(world, |c: &mut VolumetricCloudLayer| c.lifetime = value * 60.0),
        Setting::CirrusOpticalDepth => cirrus(world, |c| c.optical_depth = value),
        Setting::CirrusCoverage => cirrus(world, |c| c.coverage = value),
        Setting::CirrusFallSpeed => cirrus(world, |c| c.fall_speed = value),
        Setting::CirrusTop => {
            let ground = ground_radius(world);
            cirrus(world, |c| c.top = ground + value * 1e3);
        }
        Setting::Exposure => each(world, |e: &mut Exposure| e.ev100 = value),
        Setting::MarchSteps => each(world, |s: &mut VolumetricCloudSettings| s.max_steps = value.round() as u32),
        Setting::LightTaps => each(world, |s: &mut VolumetricCloudSettings| s.light_taps = value.round() as u32),
        Setting::FroxelEnabled => each(world, |s: &mut VolumetricsSettings| s.enabled = value >= 0.5),
        Setting::FroxelDistance => each(world, |s: &mut VolumetricsSettings| s.max_distance = value),
        Setting::FroxelUniformity => each(world, |s: &mut VolumetricsSettings| s.depth_uniformity = value),
        Setting::FroxelFogDensity => each(world, |s: &mut VolumetricsSettings| s.fog_density = value * 1e-3),
        Setting::FroxelFogFalloff => each(world, |s: &mut VolumetricsSettings| s.fog_height_falloff = value * 1e-3),
        Setting::DebugSlice => {
            for panel in &mut world.resource_mut::<AtmosphereDebugView>().panels {
                panel.slice = value;
            }
        }
    }
}

fn atmosphere_controls(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut planet_atmosphere: Query<(&mut Atmosphere, &mut GlobalTransform)>,
    mut camera_settings: Query<&mut AtmosphereSettings, With<Camera3d>>,
    mut sun_disks: Query<&mut SunDisk, With<DirectionalLight>>,
    mut atmosphere_entities: Query<(Entity, Option<&mut VolumetricCloudLayer>), With<Atmosphere>>,
    mut commands: Commands,
    atmosphere_presets: Res<AtmospherePresets>,
    mut game_state: ResMut<GameState>,
    mut camera_exposure: Query<&mut Exposure, With<Camera3d>>,
    mut cloud_time_scale: ResMut<CloudTimeScale>,
    time: Res<Time>,
) {
    if keyboard_input.just_pressed(KeyCode::Digit3) {
        for (mut atmosphere, mut transform) in &mut planet_atmosphere {
            *atmosphere = Atmosphere::earth(atmosphere_presets.earth.clone());
            *transform = GlobalTransform::from_translation(-Vec3::Y * atmosphere.inner_radius);
        }
        for mut sun_disk in &mut sun_disks {
            sun_disk.angular_size = SunDisk::EARTH.angular_size;
        }
        println!("Switched to Earth atmosphere");
    }

    if keyboard_input.just_pressed(KeyCode::Digit4) {
        for (mut atmosphere, mut transform) in &mut planet_atmosphere {
            *atmosphere = Atmosphere::mars(atmosphere_presets.mars.clone());
            *transform = GlobalTransform::from_translation(-Vec3::Y * atmosphere.inner_radius);
        }
        for mut sun_disk in &mut sun_disks {
            sun_disk.angular_size = SunDisk::MARS.angular_size;
        }
        println!("Switched to Mars atmosphere");
    }

    if keyboard_input.just_pressed(KeyCode::Digit1) {
        for mut settings in &mut camera_settings {
            settings.rendering_method = AtmosphereMode::LookupTexture;
            println!("Switched to lookup texture rendering method");
        }
    }

    if keyboard_input.just_pressed(KeyCode::Digit2) {
        for mut settings in &mut camera_settings {
            settings.rendering_method = AtmosphereMode::Raymarched;
            println!("Switched to raymarched rendering method");
        }
    }

    if keyboard_input.just_pressed(KeyCode::Digit5) {
        game_state.planet_scale_view = !game_state.planet_scale_view;
        if game_state.planet_scale_view {
            for mut settings in &mut camera_settings {
                settings.aerial_view_lut_size = UVec3::new(160, 88, 64);
                settings.aerial_view_lut_max_distance = 6.2e4;
            }
            println!("Switched to planet scale view");
        } else {
            for mut settings in &mut camera_settings {
                settings.aerial_view_lut_size = UVec3::new(32, 32, 32);
                settings.aerial_view_lut_max_distance = 3.2e4;
            }
            println!("Switched to normal view");
        }
    }

    if keyboard_input.just_pressed(KeyCode::KeyC) {
        let has_clouds = atmosphere_entities
            .iter()
            .any(|(_, cloud_layer)| cloud_layer.is_some());
        let entities: Vec<Entity> = atmosphere_entities
            .iter()
            .map(|(entity, _)| entity)
            .collect();
        if has_clouds {
            for entity in entities {
                commands.entity(entity).remove::<VolumetricCloudLayer>();
            }
            println!("Cloud layer disabled");
        } else {
            for entity in entities {
                commands.entity(entity).insert(VolumetricCloudLayer::default());
            }
            println!("Cloud layer enabled");
        }
    }

    if keyboard_input.pressed(KeyCode::ArrowLeft) {
        for (_, cloud_layer) in atmosphere_entities.iter_mut() {
            if let Some(mut cloud_layer) = cloud_layer {
                cloud_layer.coverage = (cloud_layer.coverage - time.delta_secs()).max(0.0);
                println!("Cloud coverage scale: {:.2}", cloud_layer.coverage);
            }
        }
    }

    if keyboard_input.pressed(KeyCode::ArrowRight) {
        for (_, cloud_layer) in atmosphere_entities.iter_mut() {
            if let Some(mut cloud_layer) = cloud_layer {
                cloud_layer.coverage = (cloud_layer.coverage + time.delta_secs()).min(5.0);
                println!("Cloud coverage scale: {:.2}", cloud_layer.coverage);
            }
        }
    }

    if keyboard_input.just_pressed(KeyCode::Enter) {
        game_state.paused = !game_state.paused;
    }

    if keyboard_input.just_pressed(KeyCode::KeyT) {
        cloud_time_scale.0 = if cloud_time_scale.0 > 1.0 { 1.0 } else { 60.0 };
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

/// Cycles the vertical cloud profiles of the cloud layer: the baked physical set, Nubis's
/// reconstructed lookup, and the two analytic envelopes.
fn cycle_cloud_profiles(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut profiles: Query<&mut CloudProfiles, With<VolumetricCloudLayer>>,
) {
    if !keyboard_input.just_pressed(KeyCode::KeyP) {
        return;
    }
    for mut profile in &mut profiles {
        let (next, name) = match *profile {
            CloudProfiles::Physical | CloudProfiles::Image(_) => (CloudProfiles::Nubis, "Nubis"),
            CloudProfiles::Nubis => (CloudProfiles::ADIABATIC, "adiabatic envelope"),
            CloudProfiles::Envelope { bottom, .. } if bottom == 1.0 => {
                (CloudProfiles::NUBIS_ENVELOPE, "Nubis envelope")
            }
            CloudProfiles::Envelope { .. } => (CloudProfiles::Physical, "physical"),
        };
        *profile = next;
        println!("Cloud profiles: {name}");
    }
}

fn visibility_toggle(
    game_state: Res<GameState>,
    mut visibility_query: Query<&mut Visibility, With<FogVolume>>,
) {
    let visibility = if game_state.planet_scale_view {
        Visibility::Hidden
    } else {
        Visibility::Inherited
    };
    for mut v in &mut visibility_query {
        *v = visibility;
    }
}

fn setup_camera_fog(
    mut commands: Commands,
    mut scattering_mediums: ResMut<Assets<ScatteringMedium>>,
    mut images: ResMut<Assets<Image>>,
    asset_server: Res<AssetServer>,
) {
    // `ATMOSPHERE_BACKGROUND=1`: render the camera into an offscreen image of the window's size
    // instead of the window. macOS stops redraws/drawables for occluded windows, so a
    // window-backed run freezes when it is behind other windows; the window itself stays hidden
    // and unfocused.
    let (target_w, target_h) = window_size();
    let offscreen = std::env::var("ATMOSPHERE_BACKGROUND").is_ok().then(|| {
        images.add(Image::new_target_texture(
            target_w,
            target_h,
            bevy::render::render_resource::TextureFormat::Rgba8Unorm,
            Some(bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb),
        ))
    });
    commands.insert_resource(OffscreenTarget(offscreen.clone()));
    // The default aerosol (Bruneton 2017, τ≈0.005) is very clear air: its clear-sky anti-sun
    // L/Lz is 1.5–2.3× steeper than CIE S 011/E:2003 type 12 and Poly Haven HDRs.
    // `ATMOSPHERE_AEROSOL=urban` starts with `Aerosol::URBAN`.
    let aerosol = match std::env::var("ATMOSPHERE_AEROSOL").as_deref() {
        Ok("urban") => Aerosol::URBAN,
        _ => Aerosol::CONTINENTAL_CLEAN,
    };
    let earth_medium =
        scattering_mediums.add(ScatteringMedium::earth_with_aerosol(256, 256, aerosol));
    let mars_phase = asset_server.load("textures/mars_mie_phase.ktx2");
    let mars_medium = scattering_mediums.add(ScatteringMedium::mars(256, 256, mars_phase));

    commands.insert_resource(AtmospherePresets {
        earth: earth_medium.clone(),
        mars: mars_medium.clone(),
    });

    // Spawn earth atmosphere
    commands.spawn((
        // Ground albedo 0.28 to match the cloud ground bounce (CLOUD_GROUND_ALBEDO) and the reference.
        Atmosphere {
            ground_albedo: Vec3::splat(0.28),
            ..Atmosphere::earth(earth_medium)
        },
        // Volumetric cloud layer. Scattering/absorption come from droplet microphysics
        // (`VolumetricCloudLayer::with_liquid_water`): peak cumulus liquid water content 1 g/m^3 at density 1,
        // effective radius 10 um.
        VolumetricCloudLayer {
            cloud_layer_start: 6_361_500.0, // cloud base 1.5 km above the surface
            cloud_layer_end: 6_364_000.0,   // tops 4 km (2.5 km thick layer)
            // Used for enabling/disabling, actual density comes from noise (normalized [0, 1]);
            // `ATMOSPHERE_NO_CUMULUS` leaves only the cirrus (its history moves with its own wind).
            // `ATMOSPHERE_CLOUD_DENSITY=<x>` scales the optical depth (debug).
            cloud_density: if std::env::var("ATMOSPHERE_NO_CUMULUS").is_ok() {
                0.0
            } else {
                std::env::var("ATMOSPHERE_CLOUD_DENSITY").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0)
            },
            // Larger scale = larger cloud features (more “big cumulus”, less “small puffs”)
            noise_scale: 32_000.0, // weather tile (shape noise: shape_noise_scale, 6.4 km)
            noise_offset: Vec3::ZERO,
            // Detail noise tile; `ATMOSPHERE_CLOUD_DETAIL_SCALE=<m>` overrides (validation).
            detail_noise_scale: std::env::var("ATMOSPHERE_CLOUD_DETAIL_SCALE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(600.0),
            // `ATMOSPHERE_CLOUD_LIFETIME=<s>` sets the cloud life time (0 = static weather).
            lifetime: env_f32("ATMOSPHERE_CLOUD_LIFETIME").unwrap_or(1200.0),
            // `ATMOSPHERE_CLOUD_EVOLUTION=<s>` starts the cloud clock (life-cycle phase) there.
            evolution_time: env_f32("ATMOSPHERE_CLOUD_EVOLUTION").unwrap_or(0.0),
            // `ATMOSPHERE_CLOUD_COVERAGE=<x>` sets the starting coverage scale (Left/Right change it);
            // `ATMOSPHERE_VOXEL_ONLY` clears the sky's cumulus (the layer stays: the voxel cloud
            // takes its optical properties).
            coverage: if voxel_only() {
                0.0
            } else {
                std::env::var("ATMOSPHERE_CLOUD_COVERAGE")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1.0)
            },
            // `ATMOSPHERE_CLOUD_DETAIL=<x>` overrides the detail-erosion strength (validation).
            detail_strength: std::env::var("ATMOSPHERE_CLOUD_DETAIL")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.0),
            ..default()
        }
        .with_liquid_water(
            std::env::var("ATMOSPHERE_CLOUD_LWC").ok().and_then(|v| v.parse::<f32>().ok()).map_or(VolumetricCloudLayer::CUMULUS_LIQUID_WATER, |g| g * 1e-3),
            VolumetricCloudLayer::CUMULUS_EFFECTIVE_RADIUS,
        ),
        // Vertical cloud profiles: physically based by default (baked parcel ensemble);
        // `ATMOSPHERE_CLOUD_PROFILE=nubis` selects Nubis's vertical profile, `adiabatic` or
        // `nubis_envelope` an analytic envelope.
        match std::env::var("ATMOSPHERE_CLOUD_PROFILE").as_deref() {
            Ok("nubis") => CloudProfiles::Nubis,
            Ok("adiabatic") => CloudProfiles::ADIABATIC,
            Ok("nubis_envelope") => CloudProfiles::NUBIS_ENVELOPE,
            _ => CloudProfiles::Physical,
        },
        CloudWeather::default(),
        // High-altitude cirrus (thin ice cloud at 9 km); `ATMOSPHERE_NO_CIRRUS` removes it.
        ThinCloudLayer {
            optical_depth: if std::env::var("ATMOSPHERE_NO_CIRRUS").is_ok() || voxel_only() { 0.0 } else { 0.3 },
            ..ThinCloudLayer::cirrus()
        },
        // One wind profile for both layers (each takes its wind and shear at its altitude).
        // `ATMOSPHERE_WIND=slow`: 2 m/s everywhere (sub-texel shadow motion per frame);
        // `opposing`: cumulus and cirrus winds in opposite directions (per-layer reprojection).
        match std::env::var("ATMOSPHERE_WIND").as_deref() {
            Ok("slow") => AtmosphereWind::constant(Vec3::X * 2.0),
            Ok("opposing") => AtmosphereWind {
                levels: vec![(0.0, Vec3::X * 10.0), (4000.0, Vec3::X * 15.0), (8000.0, Vec3::X * -30.0)],
            },
            _ => AtmosphereWind::fair_weather(),
        },
    ))
    // Further layers are entities related to the atmosphere (`CloudLayerOf`):
    // `ATMOSPHERE_SECOND_THIN=1` adds a cirrostratus veil at 11 km above the cirrus.
    .with_related_entities::<CloudLayerOf>(|layers| {
        if std::env::var("ATMOSPHERE_SECOND_THIN").is_ok() && !voxel_only() {
            layers.spawn(ThinCloudLayer {
                top: 6_360_000.0 + 11_500.0,
                ..ThinCloudLayer::cirrostratus()
            });
        }
    });

    // World-anchored weather edits (`CloudInfluence`): `ATMOSPHERE_INFLUENCE=hole` clears a hole
    // 6 km across, 6 km ahead of the default camera (the weather drifts, the hole stays).
    // `ATMOSPHERE_INFLUENCE=coast`: a coastline 8 km ahead, maritime air (few, large droplets:
    // N_d 80 cm^-3, more translucent clouds) before it, continental (N_d 800) beyond.
    if std::env::var("ATMOSPHERE_INFLUENCE").as_deref() == Ok("coast") {
        for (x, droplet_number) in [(-42_000.0, 80.0), (58_000.0, 800.0)] {
            commands.spawn((
                Transform::from_xyz(x, 0.0, 0.0),
                CloudInfluence {
                    shape: CloudInfluenceShape::Rectangle(Vec2::splat(50_000.0)),
                    feather: 1_500.0,
                    values: CloudInfluenceValues { droplet_number: Some(droplet_number), ..default() },
                    ..default()
                },
            ));
        }
    }
    if std::env::var("ATMOSPHERE_INFLUENCE").as_deref() == Ok("hole") {
        commands.spawn((
            Transform::from_xyz(6_000.0, 0.0, 0.0),
            CloudInfluence {
                shape: CloudInfluenceShape::Ellipse(Vec2::splat(3_000.0)),
                feather: 800.0,
                blend: CloudBlend::Subtract,
                values: CloudInfluenceValues { coverage: Some(1.0), ..default() },
                ..default()
            },
        ));
    }

    let mut camera = commands.spawn_empty();
    if let Some(image) = offscreen {
        camera.insert(bevy::camera::RenderTarget::Image(image.into()));
    }
    camera.insert((
        Camera3d::default(),
        // `ATMOSPHERE_VIEW=up|wide` picks an alternative camera for cloud screenshots.
        // `ATMOSPHERE_CAM=x,y,z,tx,ty,tz` overrides the camera position and look-at target.
        match std::env::var("ATMOSPHERE_CAM")
            .ok()
            .map(|v| v.split(',').filter_map(|x| x.trim().parse::<f32>().ok()).collect::<Vec<_>>())
            .filter(|v| v.len() == 6)
        {
            Some(v) => Transform::from_xyz(v[0], v[1], v[2])
                .looking_at(Vec3::new(v[3], v[4], v[5]), Vec3::Y),
            None => match std::env::var("ATMOSPHERE_VIEW").as_deref() {
            Ok("up") => Transform::from_xyz(-2.8, 0.045, 0.0)
                .looking_at(Vec3::new(-1.8, 1.2, 0.3), Vec3::Y),
            // Toward the sun (+X), tilted up; use with ATMOSPHERE_SUNSET=1 for a low sun.
            Ok("sun") => Transform::from_xyz(-2.8, 0.045, 0.0)
                .looking_at(Vec3::new(0.0, 0.5, 0.0), Vec3::Y),
            // Sunset undersides: steeply up toward the sun azimuth (cloud bases over the camera).
            Ok("under") => Transform::from_xyz(-2.8, 0.045, 0.0)
                .looking_at(Vec3::new(-2.2, 1.6, 0.0), Vec3::Y),
            // Inside the cloud slab (2.6 km; world units are metres) looking along it: near-field / step-budget clipping.
            Ok("inslab") => Transform::from_xyz(-2.8, 2600.0, 0.0)
                .looking_at(Vec3::new(5000.0, 2700.0, 0.0), Vec3::Y),
            // Grazing toward the horizon from the ground: far-distance precision / convergence.
            Ok("horizon") => Transform::from_xyz(-2.8, 0.045, 0.0)
                .looking_at(Vec3::new(5.0, 0.12, 0.0), Vec3::Y),
            // Down at the ground from 1 km: cloud shadow map stability.
            Ok("ground") => Transform::from_xyz(-2.8, 1000.0, 0.0)
                .looking_at(Vec3::new(0.0, 0.0, 1500.0), Vec3::Y),
            // Sun behind a cumulus top (silver lining, crepuscular rays), framed like a photo
            // reference. Aimed at the sun for ATMOSPHERE_SUN_ELEV_DEG=20; the camera position was
            // picked by a grid search for an occluded sun pixel.
            Ok("backlit") => Transform::from_xyz(-6000.0, 2.0, 9000.0).looking_at(
                Vec3::new(-6000.0 + 20f32.to_radians().cos(), 2.0 + 20f32.to_radians().sin(), 9000.0),
                Vec3::Y,
            ),
            Ok("wide") => Transform::from_xyz(-2.8, 0.045, 0.0)
                .looking_at(Vec3::new(-2.8, 0.25, -3.0), Vec3::Y),
            // Default ("hero"): eye height, pitched ~15 deg up so the sky fills most of the frame.
            _ => Transform::from_xyz(-2.8, 1.8, 0.0).looking_at(Vec3::new(0.0, 2.55, 0.0), Vec3::Y),
            },
        },
        // Can be adjusted to change the scene scale and rendering quality
        AtmosphereSettings {
            // Profiling override: `ATMOSPHERE_LUT_MODE=1` starts in `AtmosphereMode::LookupTexture`.
            rendering_method: if std::env::var("ATMOSPHERE_LUT_MODE").is_ok() {
                AtmosphereMode::LookupTexture
            } else {
                AtmosphereMode::Raymarched
            },
            // Profiling override: `ATMOSPHERE_SHADOW_SAMPLES=<n>` density samples per shadow ray.
            cloud_shadow_map_samples: std::env::var("ATMOSPHERE_SHADOW_SAMPLES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(AtmosphereSettings::default().cloud_shadow_map_samples),
            ..default()
        },
        // The directional light illuminance used in this scene
        // (the one recommended for use with this feature) is
        // quite bright, so raising the exposure compensation helps
        // bring the scene to a nicer brightness range.
        Exposure { ev100: 13.0 },
        // Tonemapper chosen just because it looked good with the scene, any
        // tonemapper would be fine :)
        Tonemapping::AcesFitted,
        // Bloom gives the sun a much more natural look.
        Bloom::NATURAL,
        // Left-drag looks around; right-drag is the time-of-day slider (`sun_drag`).
        #[cfg(feature = "free_camera")]
        FreeCamera {
            mouse_key_cursor_grab: MouseButton::Left,
            ..default()
        },
        // Planet-scale flight: 100x the default walk/run speed.
        #[cfg(feature = "free_camera")]
        {
            // `FreeCameraState` has private fields, so no struct-update syntax.
            let mut state = FreeCameraState::default();
            state.speed_multiplier = 100.0;
            state
        },
        VolumetricFog {
            ambient_intensity: 0.0,
            ..default()
        },
        // `ATMOSPHERE_MSAA=2|4|8` (the clouds trace multisampled depth; the sky marches at full
        // resolution instead of the half-res haze prepass).
        match std::env::var("ATMOSPHERE_MSAA").as_deref() {
            Ok("2") => Msaa::Sample2,
            Ok("4") => Msaa::Sample4,
            Ok("8") => Msaa::Sample8,
            _ => Msaa::Off,
        },
        // `VolumetricCloudSettings` owns the cloud quality and the shadow update/temporal
        // settings (it overwrites `AtmosphereSettings` every frame), so the overrides go here:
        // `ATMOSPHERE_CLOUD_QUALITY=low|medium|high|cinematic` picks the preset,
        // `ATMOSPHERE_SHADOW_INTERLEAVE=1` traces the whole shadow map every frame instead of
        // time-slicing it, `ATMOSPHERE_SHADOW_TEMPORAL_ALPHA=<a>` sets the history blend
        // (1 = no history), `ATMOSPHERE_CLOUD_STEP_SAMPLES=<n>` the density samples per long
        // in-cloud view step.
        {
            let preset = match std::env::var("ATMOSPHERE_CLOUD_QUALITY").as_deref() {
                Ok("low") => VolumetricCloudSettings::LOW,
                Ok("high") => VolumetricCloudSettings::HIGH,
                Ok("cinematic") => VolumetricCloudSettings::CINEMATIC,
                _ => VolumetricCloudSettings::MEDIUM,
            };
            VolumetricCloudSettings {
                shadow_update_interleave: std::env::var("ATMOSPHERE_SHADOW_INTERLEAVE")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(preset.shadow_update_interleave),
                shadow_temporal_alpha: std::env::var("ATMOSPHERE_SHADOW_TEMPORAL_ALPHA")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(preset.shadow_temporal_alpha),
                light_taps: std::env::var("ATMOSPHERE_CLOUD_LIGHT_TAPS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(preset.light_taps),
                step_samples: std::env::var("ATMOSPHERE_CLOUD_STEP_SAMPLES")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(preset.step_samples),
                // `ATMOSPHERE_CLOUD_MAX_RENDER=w,h`: the cloud/haze buffer size cap.
                max_render_size: std::env::var("ATMOSPHERE_CLOUD_MAX_RENDER")
                    .ok()
                    .and_then(|v| {
                        let (w, h) = v.split_once(',')?;
                        Some(UVec2::new(w.trim().parse().ok()?, h.trim().parse().ok()?))
                    })
                    .unwrap_or(preset.max_render_size),
                ..preset
            }
        },
    ));
    // `ATMOSPHERE_FROXELS=1`: the near-field froxel grid (`VolumetricsSettings`), with an
    // optional height fog (`ATMOSPHERE_FROXEL_FOG=<1/m>` extinction at the ground,
    // `ATMOSPHERE_FROXEL_FOG_FALLOFF=<1/m>`), range (`ATMOSPHERE_FROXEL_DISTANCE=<m>`), sky
    // ambient scale (`ATMOSPHERE_FROXEL_AMBIENT=<scale>`, 0 = none; the ambient needs the
    // camera's atmosphere probe, turned on below) and the transparent-mesh apply
    // (`ATMOSPHERE_FROXEL_TRANSPARENT=0` turns it off).
    if std::env::var("ATMOSPHERE_FROXELS").is_ok() {
        let env = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<f32>().ok());
        let defaults = VolumetricsSettings::default();
        camera.insert(VolumetricsSettings {
            enabled: true,
            max_distance: env("ATMOSPHERE_FROXEL_DISTANCE").unwrap_or(defaults.max_distance),
            fog_density: env("ATMOSPHERE_FROXEL_FOG").unwrap_or(0.0),
            fog_height_falloff: env("ATMOSPHERE_FROXEL_FOG_FALLOFF").unwrap_or(0.01),
            ambient_scale: env("ATMOSPHERE_FROXEL_AMBIENT").unwrap_or(defaults.ambient_scale),
            apply_to_transparents: env("ATMOSPHERE_FROXEL_TRANSPARENT").is_none_or(|v| v != 0.0),
            ..defaults
        });
    }
    // `ATMOSPHERE_VIEWPORT=x,y,w,h`: render the camera into that physical-pixel sub-rectangle of
    // its target (split-screen / inset cameras).
    if let Some(v) = std::env::var("ATMOSPHERE_VIEWPORT")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse::<u32>().ok()).collect::<Vec<_>>())
        .filter(|v| v.len() == 4)
    {
        camera.insert(Camera {
            viewport: Some(bevy::camera::Viewport {
                physical_position: UVec2::new(v[0], v[1]),
                physical_size: UVec2::new(v[2], v[3]),
                ..default()
            }),
            ..default()
        });
    }
    // `ATMOSPHERE_NO_TAA`: no TAA (and so no jitter), to look at the clouds' and haze's own
    // temporal filters alone.
    if std::env::var("ATMOSPHERE_NO_TAA").is_err() {
        camera.insert(TemporalAntiAliasing::default());
    }
    // `ATMOSPHERE_IBL=1`: the atmosphere drives reflections and ambient lighting (IBL) for this
    // view (off by default while debugging the clouds; on with `ATMOSPHERE_OBJECTS`, and with
    // `ATMOSPHERE_FROXELS`, whose sky ambient is the probe's harmonics).
    let ibl = std::env::var("ATMOSPHERE_IBL").is_ok()
        || std::env::var("ATMOSPHERE_OBJECTS").is_ok()
        || std::env::var("ATMOSPHERE_FROXELS").is_ok();
    if ibl {
        camera.insert(AtmosphereEnvironmentMapLight::default());
    }
    // `ATMOSPHERE_SECOND_CAMERA=x,y,z,tx,ty,tz`: a second camera with the same settings, into an
    // offscreen image of the window's size (equal-sized views), to test per-view cloud
    // state. `ATMOSPHERE_SECOND_SCREENSHOT=<path>` saves its image with the main screenshot.
    let main_camera = camera.id();
    if let Some(v) = std::env::var("ATMOSPHERE_SECOND_CAMERA")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse::<f32>().ok()).collect::<Vec<_>>())
        .filter(|v| v.len() == 6)
    {
        let image = images.add(Image::new_target_texture(
            target_w,
            target_h,
            bevy::render::render_resource::TextureFormat::Rgba8Unorm,
            Some(bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb),
        ));
        commands.insert_resource(SecondCameraTarget(image.clone()));
        commands
            .entity(main_camera)
            .clone_and_spawn_with_opt_out(|builder| {
                // Its own environment map (inserted below), not the main camera's image.
                builder.deny::<(
                    AtmosphereEnvironmentMapLight,
                    GeneratedEnvironmentMapLight,
                    EnvironmentMapLight,
                )>();
                #[cfg(feature = "free_camera")]
                builder.deny::<(FreeCamera, FreeCameraState)>();
            })
            .insert_if(AtmosphereEnvironmentMapLight::default(), || ibl)
            .insert((
                Transform::from_xyz(v[0], v[1], v[2]).looking_at(Vec3::new(v[3], v[4], v[5]), Vec3::Y),
                bevy::camera::RenderTarget::Image(image.into()),
                Camera {
                    order: 1,
                    ..default()
                },
            ));
    }
}

/// The second camera's image (`ATMOSPHERE_SECOND_CAMERA`).
#[derive(Resource)]
struct SecondCameraTarget(Handle<Image>);

#[derive(Component)]
struct Terrain;

fn setup_terrain_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut std_materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    // Sun
    commands.spawn((
        DirectionalLight {
            shadow_maps_enabled: true,
            // lux::RAW_SUNLIGHT is recommended for use with this feature, since
            // other values approximate sunlight *post-scattering* in various
            // conditions. RAW_SUNLIGHT in comparison is the illuminance of the
            // sun unfiltered by the atmosphere, so it is the proper input for
            // sunlight to be filtered by the atmosphere.
            // `ATMOSPHERE_SUN_SCALE=<x>` scales the sun's illuminance.
            illuminance: lux::RAW_SUNLIGHT * env_f32("ATMOSPHERE_SUN_SCALE").unwrap_or(1.0),
            ..default()
        },
        // `ATMOSPHERE_SUN_ELEV_DEG=<deg>` fixes the sun elevation (overrides ATMOSPHERE_SUNSET),
        // `ATMOSPHERE_SUN_AZ_DEG=<deg>` turns it about the vertical from +X toward -Z.
        match env_f32("ATMOSPHERE_SUN_ELEV_DEG") {
            Some(deg) => Transform::from_translation(sun_direction(
                deg,
                env_f32("ATMOSPHERE_SUN_AZ_DEG").unwrap_or(0.0),
            )),
            None => Transform::from_xyz(
                1.0,
                if std::env::var("ATMOSPHERE_SUNSET").is_ok() { 0.06 } else { 0.4 },
                0.0,
            ),
        }
        .looking_at(Vec3::ZERO, Vec3::Y),
        VolumetricLight,
        SunDisk::EARTH,
    ));
    // `ATMOSPHERE_SECOND_SUN=<elev_deg>,<az_deg>`: a second sun (no shadow maps, not volumetric,
    // so it sorts after the first and the cloud shadow map follows the first), to test
    // multiple-light cloud shadows.
    if let Some(v) = std::env::var("ATMOSPHERE_SECOND_SUN")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse::<f32>().ok()).collect::<Vec<_>>())
        .filter(|v| v.len() == 2)
    {
        commands.spawn((
            DirectionalLight {
                illuminance: lux::RAW_SUNLIGHT,
                ..default()
            },
            Transform::from_translation(sun_direction(v[0], v[1])).looking_at(Vec3::ZERO, Vec3::Y),
        ));
    }

    // `ATMOSPHERE_FROXEL_TEST_LIGHT=1`: a local-light / fog-volume test in front of the hero
    // camera (at (-2.8, 1.8, 0) looking +x): a 12 x 4 x 12 m `FogVolume` box over the ground, a
    // shadow-casting point light with `VolumetricLight` above it, and three occluders between
    // the light and the fog so its shadow maps cut shafts into the fog.
    if let Ok(test_light) = std::env::var("ATMOSPHERE_FROXEL_TEST_LIGHT") {
        // Extinction 0.02/m (`ATMOSPHERE_FROXEL_TEST_FOG=<1/m>` overrides): Koschmieder's law
        // (visibility V = 3.912 / sigma_t for a 2% contrast threshold) gives V = 196 m, a fog by
        // the WMO definition (visibility below 1 km; RDR2 slide 31 uses g = 0.9 for low- and
        // high-density fog, Bevy's default g is 0.5). `FogVolume` scales `absorption +
        // scattering` (0.6 by default) by `density_factor`.
        let fog_extinction = env_f32("ATMOSPHERE_FROXEL_TEST_FOG").unwrap_or(0.02);
        let fog = FogVolume::default();
        commands.spawn((
            FogVolume {
                density_factor: fog_extinction / (fog.absorption + fog.scattering),
                ..fog
            },
            Transform::from_xyz(3.0, 2.0, 0.0).with_scale(Vec3::new(12.0, 4.0, 12.0)),
        ));
        // Luminous flux (lm; `ATMOSPHERE_FROXEL_TEST_LIGHT=<lm>` sets it, any other value the
        // default 1e9): against the ~1e4 nit daylight sky at EV100 13, the fog's in-scatter
        // sigma_s x p_HG x E ~ 0.01/m x 0.08/sr x E must reach ~1e4 nit, so E ~ 1e7 lux at the
        // fog's top 3 m below the light: I = E d^2 ~ 1e8 cd, 4 pi I ~ 1e9 lm (about 100 suns;
        // a physical lamp is only visible in fog at night).
        let intensity = test_light.parse::<f32>().unwrap_or(1e9);
        commands.spawn((
            PointLight {
                intensity,
                range: 60.0,
                shadow_maps_enabled: true,
                ..default()
            },
            Transform::from_xyz(3.0, 5.0, 0.0),
            VolumetricLight,
        ));
        let occluder = std_materials.add(StandardMaterial {
            base_color: Color::srgb(0.6, 0.6, 0.6),
            perceptual_roughness: 0.8,
            // The example's default is deferred; the camera has no deferred prepass.
            opaque_render_method: bevy::material::OpaqueRendererMethod::Forward,
            ..default()
        });
        let slab = meshes.add(Cuboid::new(1.6, 0.2, 1.6));
        for (i, position) in [
            Vec3::new(3.0, 3.6, 0.0),
            Vec3::new(1.4, 3.2, -1.6),
            Vec3::new(4.6, 3.0, 1.4),
        ]
        .into_iter()
        .enumerate()
        {
            commands.spawn((
                Mesh3d(slab.clone()),
                MeshMaterial3d(occluder.clone()),
                Transform::from_translation(position)
                    .with_rotation(Quat::from_rotation_y(0.4 * i as f32)),
            ));
        }
        // With `ATMOSPHERE_FROXELS`, two alpha-blended panes facing the hero camera at 30 and
        // 100 m, inside the grid: the froxel aerial perspective of transparent meshes
        // (`VOLUMETRICS_APPLY_MESH`; opaque meshes get theirs from `render_sky`'s composite).
        // Both subtend ~6 degrees; the far one is offset sideways so the near one does not cover it.
        if std::env::var("ATMOSPHERE_FROXELS").is_ok() {
            for (distance, half_size, z, color) in [
                (30.0, 1.5, 0.0, Color::srgba(0.9, 0.2, 0.15, 0.6)),
                (100.0, 5.0, 14.0, Color::srgba(0.15, 0.3, 0.9, 0.6)),
            ] {
                commands.spawn((
                    Mesh3d(meshes.add(Plane3d::new(-Vec3::X, Vec2::splat(half_size)))),
                    MeshMaterial3d(std_materials.add(StandardMaterial {
                        base_color: color,
                        alpha_mode: AlphaMode::Blend,
                        perceptual_roughness: 0.6,
                        ..default()
                    })),
                    Transform::from_xyz(-2.8 + distance, 1.8 + half_size, z),
                ));
            }
        }
    }

    // `ATMOSPHERE_MESH_PLANE=1`: a 40 km grey mesh plane over x < 0, 2 m above the ground, to
    // compare mesh (PBR) cloud shadows with the planet ground's across its edge at x = 0.
    if std::env::var("ATMOSPHERE_MESH_PLANE").is_ok() {
        commands.spawn((
            Mesh3d(meshes.add(Plane3d::new(Vec3::Y, Vec2::splat(20_000.0)))),
            MeshMaterial3d(std_materials.add(StandardMaterial {
                base_color: Color::srgb(0.3, 0.3, 0.3),
                perceptual_roughness: 1.0,
                // The example's default is deferred; the camera has no deferred prepass.
                opaque_render_method: bevy::material::OpaqueRendererMethod::Forward,
                ..default()
            })),
            Transform::from_xyz(-20_000.0, 2.0, 0.0),
        ));
    }

    // `ATMOSPHERE_OBJECTS=1`: meshes to check that clouds sort against geometry and that it is
    // lit by the sky (turns on `ATMOSPHERE_IBL`): props in front of the default camera, towers
    // through the cloud layer (base 1.5 km, tops 4 km) and a 3.5 km mountain 22 km away.
    if std::env::var("ATMOSPHERE_OBJECTS").is_ok() {
        let material = |color: Color, metallic: f32, roughness: f32| StandardMaterial {
            base_color: color,
            metallic,
            perceptual_roughness: roughness,
            // The example's default is deferred; the camera has no deferred prepass.
            opaque_render_method: bevy::material::OpaqueRendererMethod::Forward,
            ..default()
        };
        let props = [
            (
                meshes.add(Cuboid::from_length(1.0)),
                material(Color::srgb(0.8, 0.8, 0.8), 0.0, 0.6),
                Vec3::new(3.0, 0.5, -1.0),
            ),
            (
                meshes.add(Sphere::new(0.6)),
                material(Color::srgb(0.8, 0.3, 0.2), 0.0, 0.4),
                Vec3::new(4.0, 0.6, 1.2),
            ),
            (
                meshes.add(Sphere::new(0.4)),
                material(Color::srgb(0.9, 0.9, 0.9), 1.0, 0.15),
                Vec3::new(2.5, 0.4, 0.4),
            ),
        ];
        let tower = meshes.add(Cylinder::new(150.0, 6000.0));
        let tower_material = std_materials.add(material(Color::srgb(0.5, 0.5, 0.55), 0.0, 0.8));
        for (mesh, material, position) in props {
            commands.spawn((
                Mesh3d(mesh),
                MeshMaterial3d(std_materials.add(material)),
                Transform::from_translation(position),
            ));
        }
        for position in [
            Vec3::new(4_000.0, 3_000.0, -1_500.0),
            Vec3::new(9_000.0, 3_000.0, 2_500.0),
            Vec3::new(15_000.0, 3_000.0, -4_000.0),
        ] {
            commands.spawn((
                Mesh3d(tower.clone()),
                MeshMaterial3d(tower_material.clone()),
                Transform::from_translation(position),
            ));
        }
        commands.spawn((
            Mesh3d(meshes.add(Cone::new(7_000.0, 3_500.0))),
            MeshMaterial3d(std_materials.add(material(Color::srgb(0.35, 0.33, 0.3), 0.0, 1.0))),
            Transform::from_xyz(20_000.0, 1_750.0, 9_000.0),
        ));
    }

    // Terrain: temporarily off by default while tuning clouds + atmosphere (the planet ground
    // sphere still renders); `ATMOSPHERE_TERRAIN=1` brings it back.
    if std::env::var("ATMOSPHERE_TERRAIN").is_err() {
        return;
    }
    commands.spawn((
        Terrain,
        WorldAssetRoot(
            asset_server.load(GltfAssetLabel::Scene(0).from_asset("models/terrain/terrain.glb")),
        ),
        Transform::from_xyz(-1.0, 0.0, -0.5)
            .with_scale(Vec3::splat(0.5))
            .with_rotation(Quat::from_rotation_y(PI / 2.0)),
    ));
}

/// Puts the camera position (whole numbers) and facing direction in the window title.
fn camera_pose_title(
    cameras: Query<&Transform, With<Camera3d>>,
    mut windows: Query<&mut Window, With<bevy::window::PrimaryWindow>>,
) {
    let Ok(transform) = cameras.single() else {
        return;
    };
    let Ok(mut window) = windows.single_mut() else {
        return;
    };
    let p = transform.translation;
    let f = transform.forward();
    let title = format!(
        "{}, {}, {}  facing {:.2}, {:.2}, {:.2}",
        p.x.round() as i32,
        p.y.round() as i32,
        p.z.round() as i32,
        f.x,
        f.y,
        f.z,
    );
    if window.title != title {
        window.title = title;
    }
}

fn env_f32(name: &str) -> Option<f32> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

/// Direction toward a sun at `elev_deg` above the horizon, turned `az_deg` about the vertical
/// from +X toward -Z.
fn sun_direction(elev_deg: f32, az_deg: f32) -> Vec3 {
    let (e, a) = (elev_deg.to_radians(), az_deg.to_radians());
    Vec3::new(e.cos() * a.cos(), e.sin(), -e.cos() * a.sin())
}

fn dynamic_scene(
    mut suns: Query<&mut Transform, With<DirectionalLight>>,
    time: Res<Time>,
    sun_motion_state: Res<GameState>,
) {
    // Only rotate the sun if motion is not paused
    if !sun_motion_state.paused {
        suns.iter_mut()
            .for_each(|mut tf| tf.rotate_x(-time.delta_secs() * PI / 40.0));
    }
}

/// `ATMOSPHERE_SUN_SWEEP=<deg per frame>,<first frame>,<last frame>`: swings the sun along its
/// arc (as right-drag) by a fixed angle per frame, independent of time, so fast lighting changes
/// are reproducible in background runs and dumps.
fn sun_sweep(mut frame: Local<u32>, mut suns: Query<&mut Transform, With<DirectionalLight>>) {
    let Some(v) = std::env::var("ATMOSPHERE_SUN_SWEEP").ok() else {
        return;
    };
    let p: Vec<f32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    let n = *frame;
    *frame += 1;
    if p.len() != 3 || (n as f32) < p[1] || (n as f32) > p[2] {
        return;
    }
    let rotation = Quat::from_rotation_z(-p[0].to_radians());
    for mut tf in &mut suns {
        tf.rotation = rotation * tf.rotation;
    }
}

/// Whether the UI owns the mouse: from a press that starts over the settings panel until every
/// button is released. The camera controller and `sun_drag` ignore the mouse meanwhile, so
/// dragging a slider doesn't turn the camera or the sun. A drag that starts over the scene keeps
/// the camera even when the cursor crosses the panel.
#[derive(Resource, Default)]
struct PointerOnUi(bool);

fn pointer_on_ui(
    mouse: Res<ButtonInput<MouseButton>>,
    hover_map: Res<bevy::picking::hover::HoverMap>,
    nodes: Query<(), With<Node>>,
    mut on_ui: ResMut<PointerOnUi>,
    #[cfg(feature = "free_camera")] mut cameras: Query<&mut FreeCameraState>,
) {
    // Only the first button of a press decides; releasing every button frees the mouse.
    let first_press = mouse.get_just_pressed().next().is_some()
        && mouse.get_pressed().all(|button| mouse.just_pressed(*button));
    if first_press {
        // Hidden UI (the closed panel) is not picked.
        on_ui.0 = hover_map
            .get(&bevy::picking::pointer::PointerId::Mouse)
            .is_some_and(|hits| hits.keys().any(|entity| nodes.contains(*entity)));
    } else if mouse.get_pressed().next().is_none() && on_ui.0 {
        on_ui.0 = false;
    }
    #[cfg(feature = "free_camera")]
    for mut state in &mut cameras {
        if state.enabled == on_ui.0 {
            state.enabled = !on_ui.0;
        }
    }
}

/// Right-drag slides the time of day: horizontal mouse motion swings the sun along its arc
/// (rotation about world Z, the same plane as `ATMOSPHERE_SUN_ELEV_DEG`), 0.15 deg per pixel.
fn sun_drag(
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<bevy::input::mouse::AccumulatedMouseMotion>,
    on_ui: Res<PointerOnUi>,
    mut suns: Query<&mut Transform, With<DirectionalLight>>,
) {
    if on_ui.0 || !buttons.pressed(MouseButton::Right) || motion.delta.x == 0.0 {
        return;
    }
    let rotation = Quat::from_rotation_z(-motion.delta.x * 0.15f32.to_radians());
    for mut tf in &mut suns {
        tf.rotation = rotation * tf.rotation;
    }
}

/// Places a Nubis voxel cloud (`ATMOSPHERE_VOXEL_CLOUD=parkouring|stormbird|none`) 6 km ahead of
/// the default camera, its box from the cumulus layer's base (1.5 km) up 500 m. The textures are
/// not in the repository: `tools/nvdf_to_ktx2.py` converts them from Guerrilla's Nubis Voxel Cloud
/// Pack into `assets/volumes/nubis/`.
fn spawn_voxel_cloud(mut commands: Commands, asset_server: Res<AssetServer>) {
    let name = match std::env::var("ATMOSPHERE_VOXEL_CLOUD").as_deref() {
        Ok("none") => return,
        Ok("stormbird") => "stormbird_cloud",
        _ => "parkouring_cloud",
    };
    let path = format!("volumes/nubis/{name}.ktx2");
    #[cfg(not(target_arch = "wasm32"))]
    if !std::path::Path::new("assets").join(&path).exists() {
        println!("No voxel cloud: assets/{path} is missing (see tools/nvdf_to_ktx2.py)");
        return;
    }
    // The renderer draws voxel clouds within the cumulus layer's altitude range (with the
    // physical profiles, the range of their air masses); the box is at most 500 m tall (the
    // pack's domain), centred 1.75 km up.
    let (bottom, top) = bevy::pbr::physical_profile_bounds();
    println!("Voxel cloud centred at 1750 m; cumulus layer {bottom:.0}..{top:.0} m");
    commands.spawn((
        Name::new(name),
        // Data, not color: load it linear (the image loader assumes sRGB).
        VoxelCloud::nubis_pack(
            asset_server
                .load_builder()
                .with_settings(|settings: &mut bevy::image::ImageLoaderSettings| settings.is_srgb = false)
                .load(path),
        ),
        Transform::from_xyz(6_000.0, 1_750.0, 0.0),
    ));
}

/// `ATMOSPHERE_VOXEL_ONLY`: the voxel cloud alone (the cumulus layer at zero coverage, no cirrus).
fn voxel_only() -> bool {
    std::env::var("ATMOSPHERE_VOXEL_ONLY").is_ok()
}

/// Draws each voxel cloud's box (B toggles it).
fn draw_voxel_cloud_bounds(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut shown: Local<Option<bool>>,
    mut gizmos: Gizmos,
    clouds: Query<(&GlobalTransform, &VoxelCloud)>,
    images: Res<Assets<Image>>,
) {
    // `ATMOSPHERE_VOXEL_BOUNDS=0` starts with the box hidden (captures).
    let shown = shown.get_or_insert_with(|| std::env::var("ATMOSPHERE_VOXEL_BOUNDS").as_deref() != Ok("0"));
    if keyboard_input.just_pressed(KeyCode::KeyB) {
        *shown = !*shown;
    }
    if !*shown {
        return;
    }
    for (transform, cloud) in &clouds {
        if let Some(data) = images.get(&cloud.data) {
            gizmos.cube(cloud.bounds(transform, cloud.size(data)), Color::srgb(1.0, 0.6, 0.1));
        }
    }
}

/// Reports each voxel cloud's texture once it has loaded (size, mips, format).
fn report_voxel_cloud_data(
    clouds: Query<(&Name, &VoxelCloud)>,
    images: Res<Assets<Image>>,
    mut reported: Local<bevy::platform::collections::HashSet<AssetId<Image>>>,
) {
    for (name, cloud) in &clouds {
        if let Some(image) = images.get(&cloud.data)
            && reported.insert(cloud.data.id())
        {
            let d = image.texture_descriptor.size;
            let size = cloud.size(image);
            println!(
                "Voxel cloud {name}: {}x{}x{} {:?} {:?}, {} mips, box {:.0} x {:.0} x {:.0} m",
                d.width,
                d.height,
                d.depth_or_array_layers,
                image.texture_descriptor.dimension,
                image.texture_descriptor.format,
                image.texture_descriptor.mip_level_count,
                size.x,
                size.y,
                size.z
            );
        }
    }
}

/// Offscreen render target used by `ATMOSPHERE_BACKGROUND=1` runs.
#[derive(Resource)]
struct OffscreenTarget(Option<Handle<Image>>);

/// Window (and offscreen target) size in physical pixels: 1280x720 so dumps match across displays,
/// or `ATMOSPHERE_WINDOW=w,h`.
fn window_size() -> (u32, u32) {
    std::env::var("ATMOSPHERE_WINDOW")
        .ok()
        .and_then(|v| {
            let (w, h) = v.split_once(',')?;
            Some((w.trim().parse().ok()?, h.trim().parse().ok()?))
        })
        .unwrap_or((1280, 720))
}
