//! HGIG-style HDR display calibration patterns, with optional OS-sensed
//! calibration.
//!
//! This example renders the three classic calibration patterns (peak
//! luminance, paper white, black level) and lets you adjust the primary
//! window's [`DisplayTarget`] live while looking at them, exactly the flow an
//! in-game "HDR settings" screen would implement. On top of that manual flow it
//! demonstrates Bevy's display-sensing path: a [`DisplayCalibrationPolicy`]
//! opts individual fields into OS-sensed auto-resolution, the renderer merges
//! the user's intent with what the display reports into an
//! [`EffectiveDisplayTarget`], and the live [`WindowDisplayState`] /
//! [`MonitorDisplayCapability`] expose what the platform sees right now.
//!
//! **An HDR display is required to calibrate anything real.** scRGB-linear
//! output is reachable on macOS/iOS (Metal), Windows (Vulkan/DX12), and Linux
//! (Wayland + Vulkan, Mesa 25.1+); PQ (HDR10) output on Vulkan/DX12/Metal when
//! the OS has HDR output enabled (press `T` to cycle the requested transfer).
//! On SDR displays (or unsupported backends) the HDR request is downgraded — PQ
//! falls back to scRGB, then to plain SDR — with a warning in the log; the
//! example still runs, but everything brighter than paper white clips on the
//! SDR fallback.
//!
//! # Manual intent versus sensed calibration
//!
//! [`DisplayTarget`] is the user's *intent* and Bevy never overwrites it. The
//! HGIG-style keys below edit it directly, and a field whose
//! [`DisplayCalibrationPolicy`] is [`AutoField::Keep`] always renders that
//! authored value (top of the precedence ladder). A field set to
//! [`AutoField::Auto`] instead lets the renderer fill it from sensed display
//! information when the platform reports any — the result, and which provenance
//! won per field, lands in [`EffectiveDisplayTarget`], the target the renderer
//! actually encodes for. Press `A` to flip peak / black level / gamut between
//! `Keep` (manual) and `Auto` (OS-sensed) and watch the resolved values and
//! their provenance change in the overlay. Paper white stays manual: it is a
//! viewing-environment preference, not a hardware fact, so the display cannot
//! sense it.
//!
//! The calibration camera uses [`Tonemapping::None`] so the patterns reach
//! the display encoder at their exact authored values: a tone-mapping
//! operator would reshape the patches and the readings would calibrate the
//! operator, not the display. Patch brightness is authored in
//! paper-white-relative units (`1.0` = paper white), so a patch meant to show
//! `N` nits is drawn at `N / paper_white_nits`. The patches read the *resolved*
//! [`EffectiveDisplayTarget`], so when peak is on `Auto` the peak pattern
//! tracks the sensed peak. Press `G` to preview the Gran Turismo 7 operator on
//! the same patterns (this exercises the full HDR tone-mapping path, but the
//! patterns are no longer exact while it is on).

use bevy::{
    camera::{Hdr, ScalingMode},
    core_pipeline::tonemapping::Tonemapping,
    platform::collections::HashSet,
    prelude::*,
    window::{
        AutoField, DisplayCalibrationPolicy, DisplayInfoSource, DisplayProvenance, DisplayTarget,
        DisplayTransfer, EffectiveDisplayTarget, FieldProvenance, Monitor,
        MonitorDisplayCapability, OnMonitor, PrimaryWindow, WindowDisplayState,
        WindowMonitorChanged,
    },
};

/// The `DisplayTarget` requested at startup (and restored with `R`): a common
/// HDR baseline of 200-nit paper white on a 1000-nit display over scRGB.
///
/// The 1000-nit starting candidate exceeds the real peak of most consumer HDR
/// panels (typically 400–800 nits), so the peak-luminance pattern instructs a
/// two-directional procedure: lower the candidate first if the center square
/// is already invisible. When peak is on [`AutoField::Auto`] and the display
/// reports its peak, the sensed value takes over instead.
const INITIAL_HDR_TARGET: DisplayTarget = DisplayTarget::SDR_SRGB
    .with_paper_white(200.0)
    .with_peak(1000.0)
    .with_transfer(DisplayTransfer::ScRgbLinear);

/// The calibration policy requested at startup (and restored with `R`): paper
/// white stays manual (the display cannot sense a viewing preference), while
/// peak, black level, and gamut start manual but can be flipped to OS-sensed
/// with `A`.
const INITIAL_POLICY: DisplayCalibrationPolicy = DisplayCalibrationPolicy {
    paper_white: AutoField::Keep,
    peak_luminance: AutoField::Keep,
    min_luminance: AutoField::Keep,
    gamut: AutoField::Keep,
};

/// Height of the orthographic view volume in world units.
const VIEW_HEIGHT: f32 = 8.0;

/// Absolute luminances (in nits) of the black-level steps, left to right.
const BLACK_STEP_NITS: [f32; 8] = [0.0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0];

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(ClearColor(Color::BLACK))
        .init_resource::<CalibrationPattern>()
        .init_resource::<MonitorChangeNotice>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                select_pattern,
                adjust_display_target,
                toggle_transfer,
                toggle_auto_sensing,
                toggle_gt7_preview,
                watch_monitor_changes,
                update_pattern_visibility,
                update_patch_levels,
                update_ui,
            )
                .chain(),
        )
        .run();
}

/// Which calibration pattern is currently shown.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq, Eq)]
enum CalibrationPattern {
    /// Adjust `peak_luminance_nits` until the center patch merges with the
    /// clipped background.
    #[default]
    PeakLuminance,
    /// Adjust `paper_white_nits` until the reference white card sits at a
    /// comfortable "white UI" brightness.
    PaperWhite,
    /// Find the dimmest near-black step distinguishable from the background
    /// and record it as `min_luminance_nits`.
    BlackLevel,
}

/// Marks the root entity of one calibration pattern; visibility is toggled
/// from [`CalibrationPattern`].
#[derive(Component)]
struct PatternRoot(CalibrationPattern);

/// How bright a calibration patch should be. A system converts this to a
/// paper-white-relative linear color whenever the resolved
/// [`EffectiveDisplayTarget`] changes.
#[derive(Component)]
enum Patch {
    /// An absolute luminance in nits (drawn at `nits / paper_white_nits`).
    Nits(f32),
    /// A fraction of the resolved `peak_luminance_nits`.
    PeakFraction(f32),
    /// A value in paper-white-relative units (`1.0` = paper white).
    PaperWhiteRelative(f32),
}

/// On-screen "window changed monitor" notice with a countdown.
#[derive(Resource, Default)]
struct MonitorChangeNotice {
    seconds_remaining: f32,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    primary_window: Single<Entity, With<PrimaryWindow>>,
) {
    // Request HDR output and the calibration policy. If the display/backend
    // cannot provide HDR the renderer warns once and degrades to plain SDR;
    // this example then runs (and is exercised in CI) on that downgrade path.
    // `DisplayTarget` is the user's intent; `DisplayCalibrationPolicy` says
    // which fields the renderer may auto-resolve from the display.
    commands
        .entity(*primary_window)
        .insert((INITIAL_HDR_TARGET, INITIAL_POLICY));

    commands.spawn((
        Camera3d::default(),
        // An fp16 intermediate, so patch values above paper white (> 1.0)
        // survive until the display-encoding pass.
        Hdr,
        // Calibration patterns must reach the display encoder unmodified; an
        // operator would bend them. Press `G` for a GT7 preview instead.
        Tonemapping::None,
        Projection::from(OrthographicProjection {
            scaling_mode: ScalingMode::FixedVertical {
                viewport_height: VIEW_HEIGHT,
            },
            ..OrthographicProjection::default_3d()
        }),
        Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Patches start black; `update_patch_levels` colors them from the resolved
    // `EffectiveDisplayTarget` on the first frame.
    let patch = |commands: &mut Commands,
                 meshes: &mut Assets<Mesh>,
                 materials: &mut Assets<StandardMaterial>,
                 parent: Entity,
                 size: Vec2,
                 position: Vec3,
                 level: Patch| {
        commands.spawn((
            Mesh3d(meshes.add(Rectangle::from_size(size))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::BLACK,
                unlit: true,
                ..default()
            })),
            Transform::from_translation(position),
            level,
            ChildOf(parent),
        ));
    };

    // Pattern 1: peak luminance. A near-peak checkerboard (10000/9000 nits —
    // both clip to whatever the display can actually do) behind a center
    // patch drawn at the *candidate* peak. The candidate starts at 1000 nits
    // (`INITIAL_HDR_TARGET`), which already clips on dimmer panels, so the
    // procedure is two-directional: if the center square is already invisible,
    // lower the candidate until it appears, then raise it until it just
    // disappears into the clipped background — at that point
    // `peak_luminance_nits` matches the display's real peak (HGIG "MaxTML"
    // pattern). With peak on `Auto`, the sensed peak drives the center patch.
    let peak_root = commands
        .spawn((
            Transform::default(),
            Visibility::Hidden,
            PatternRoot(CalibrationPattern::PeakLuminance),
        ))
        .id();
    let (columns, rows) = (12, 6);
    for column in 0..columns {
        for row in 0..rows {
            let nits = if (column + row) % 2 == 0 {
                10000.0
            } else {
                9000.0
            };
            patch(
                &mut commands,
                &mut meshes,
                &mut materials,
                peak_root,
                Vec2::splat(1.0),
                Vec3::new(
                    column as f32 - (columns - 1) as f32 / 2.0,
                    row as f32 - (rows - 1) as f32 / 2.0,
                    0.0,
                ),
                Patch::Nits(nits),
            );
        }
    }
    patch(
        &mut commands,
        &mut meshes,
        &mut materials,
        peak_root,
        Vec2::splat(2.0),
        Vec3::new(0.0, 0.0, 0.1),
        Patch::PeakFraction(1.0),
    );

    // Pattern 2: paper white. A reference white card at exactly 1.0
    // (paper-white-relative) over a dim surround. The UI text (default white)
    // also renders at paper white. A 203-nit strip (the ITU-R BT.2408
    // reference white for HDR broadcast) is shown for comparison.
    let paper_root = commands
        .spawn((
            Transform::default(),
            Visibility::Hidden,
            PatternRoot(CalibrationPattern::PaperWhite),
        ))
        .id();
    patch(
        &mut commands,
        &mut meshes,
        &mut materials,
        paper_root,
        Vec2::new(40.0, VIEW_HEIGHT),
        Vec3::ZERO,
        Patch::PaperWhiteRelative(0.15),
    );
    patch(
        &mut commands,
        &mut meshes,
        &mut materials,
        paper_root,
        Vec2::new(5.0, 3.0),
        Vec3::new(0.0, 0.4, 0.1),
        Patch::PaperWhiteRelative(1.0),
    );
    patch(
        &mut commands,
        &mut meshes,
        &mut materials,
        paper_root,
        Vec2::new(5.0, 0.8),
        Vec3::new(0.0, -2.0, 0.1),
        Patch::Nits(203.0),
    );

    // Pattern 3: black level. Near-black steps at fixed absolute luminances
    // over a true-black background. Record the dimmest step you can
    // distinguish as `min_luminance_nits`. (On 8-bit SDR the lowest steps
    // quantize to the same code value and are genuinely indistinguishable.)
    let black_root = commands
        .spawn((
            Transform::default(),
            Visibility::Hidden,
            PatternRoot(CalibrationPattern::BlackLevel),
        ))
        .id();
    patch(
        &mut commands,
        &mut meshes,
        &mut materials,
        black_root,
        Vec2::new(40.0, VIEW_HEIGHT),
        Vec3::ZERO,
        Patch::Nits(0.0),
    );
    for (index, nits) in BLACK_STEP_NITS.into_iter().enumerate() {
        patch(
            &mut commands,
            &mut meshes,
            &mut materials,
            black_root,
            Vec2::new(1.2, 2.5),
            Vec3::new((index as f32 - 3.5) * 1.5, 0.0, 0.1),
            Patch::Nits(nits),
        );
    }

    // UI overlay. Note: white UI text is composited at 1.0 in the
    // display-linear buffer, i.e. exactly at paper white.
    commands.spawn((
        Text::default(),
        Node {
            position_type: PositionType::Absolute,
            top: px(12),
            left: px(12),
            ..default()
        },
    ));
}

fn select_pattern(keys: Res<ButtonInput<KeyCode>>, mut pattern: ResMut<CalibrationPattern>) {
    for (key, selected) in [
        (KeyCode::Digit1, CalibrationPattern::PeakLuminance),
        (KeyCode::Digit2, CalibrationPattern::PaperWhite),
        (KeyCode::Digit3, CalibrationPattern::BlackLevel),
    ] {
        if keys.just_pressed(key) {
            *pattern = selected;
        }
    }
}

/// Adjusts the primary window's [`DisplayTarget`] calibration values while
/// keys are held. The renderer reacts live: changing the values re-prepares
/// the display-target uniform (and GT7 parameters when previewing), and a
/// transfer change renegotiates the swapchain.
///
/// These edits always land in the user-authoritative [`DisplayTarget`]. For a
/// field whose policy is [`AutoField::Keep`] that authored value wins outright;
/// for an [`AutoField::Auto`] field the OS-sensed value takes precedence when
/// the display reports one, so the manual edit only shows through (as the
/// `Default`-provenance fallback) while nothing is sensed.
fn adjust_display_target(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut display_target: Single<&mut DisplayTarget, With<PrimaryWindow>>,
    mut policy: Single<&mut DisplayCalibrationPolicy, With<PrimaryWindow>>,
) {
    if keys.just_pressed(KeyCode::KeyR) {
        **display_target = INITIAL_HDR_TARGET;
        **policy = INITIAL_POLICY;
        return;
    }

    let delta_seconds = time.delta_secs();
    let fast = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    let axis = |up: KeyCode, down: KeyCode| {
        (keys.pressed(up) as i8 - keys.pressed(down) as i8) as f32
            * delta_seconds
            * if fast { 4.0 } else { 1.0 }
    };

    let peak = axis(KeyCode::ArrowUp, KeyCode::ArrowDown) * 250.0;
    if peak != 0.0 {
        display_target.peak_luminance_nits =
            (display_target.peak_luminance_nits + peak).clamp(100.0, 10000.0);
    }
    let paper_white = axis(KeyCode::ArrowRight, KeyCode::ArrowLeft) * 50.0;
    if paper_white != 0.0 {
        display_target.paper_white_nits =
            (display_target.paper_white_nits + paper_white).clamp(80.0, 500.0);
    }
    let min_luminance = axis(KeyCode::BracketRight, KeyCode::BracketLeft) * 0.1;
    if min_luminance != 0.0 {
        display_target.min_luminance_nits =
            (display_target.min_luminance_nits + min_luminance).clamp(0.0, 5.0);
    }
}

/// Cycles the requested transfer: sRGB → scRGB-linear → extended-sRGB (encoded,
/// the web HDR path) → PQ (HDR10) → sRGB. The surface is renegotiated the same
/// frame; if the backend cannot provide the matching color space the request
/// degrades to plain SDR with a one-time warning in the log. The gamut is left
/// at its default Rec.709 (the encoder coerces it per transfer); the
/// `tonemapping` example demonstrates the wide-gamut Display-P3 path.
///
/// The transfer is the one calibration field that is *never* auto-resolved:
/// changing it renegotiates the swapchain, so [`DisplayCalibrationPolicy`] has
/// no transfer field and sensing never rewrites it.
fn toggle_transfer(
    keys: Res<ButtonInput<KeyCode>>,
    mut display_target: Single<&mut DisplayTarget, With<PrimaryWindow>>,
) {
    if keys.just_pressed(KeyCode::KeyT) {
        display_target.transfer = match display_target.transfer {
            DisplayTransfer::Srgb => DisplayTransfer::ScRgbLinear,
            DisplayTransfer::ScRgbLinear => DisplayTransfer::ExtendedSrgb,
            DisplayTransfer::ExtendedSrgb => DisplayTransfer::Pq,
            _ => DisplayTransfer::Srgb,
        };
    }
}

/// Flips peak, black level, and gamut between [`AutoField::Keep`] (the manual
/// HGIG values authored above win) and [`AutoField::Auto`] (the renderer
/// resolves them from sensed display information when the platform reports
/// any). Paper white is deliberately omitted: it is a viewing preference the
/// display cannot sense, so it stays manual.
fn toggle_auto_sensing(
    keys: Res<ButtonInput<KeyCode>>,
    mut policy: Single<&mut DisplayCalibrationPolicy, With<PrimaryWindow>>,
) {
    if keys.just_pressed(KeyCode::KeyA) {
        let next = match policy.peak_luminance {
            AutoField::Keep => AutoField::Auto,
            AutoField::Auto => AutoField::Keep,
        };
        policy.peak_luminance = next;
        policy.min_luminance = next;
        policy.gamut = next;
    }
}

/// Toggles a Gran Turismo 7 tone-mapping preview. On an HDR display target
/// the operator automatically runs in its HDR mode, plumbing the resolved
/// peak luminance into the tone curve, so this exercises the full HDR
/// tone-mapping path end to end (add `GranTurismo7Params` to tune it). While
/// the preview is on, the patterns are *not* exact — calibrate with it off.
fn toggle_gt7_preview(
    keys: Res<ButtonInput<KeyCode>>,
    mut tonemapping: Single<&mut Tonemapping, With<Camera3d>>,
) {
    if keys.just_pressed(KeyCode::KeyG) {
        **tonemapping = if **tonemapping == Tonemapping::GranTurismo7 {
            Tonemapping::None
        } else {
            Tonemapping::GranTurismo7
        };
    }
}

/// Listens for the window moving to a different monitor. The new monitor may
/// have a completely different peak luminance and gamut, and `DisplayTarget`
/// is user-authoritative (Bevy never rewrites it), so suggest recalibrating —
/// or rely on `Auto` fields, which the renderer re-resolves against the new
/// monitor's [`MonitorDisplayCapability`] on its own. The first event per
/// window only reports the monitor becoming known at startup, so it is logged
/// but does not raise the notice.
fn watch_monitor_changes(
    mut events: MessageReader<WindowMonitorChanged>,
    monitors: Query<&Monitor>,
    mut notice: ResMut<MonitorChangeNotice>,
    time: Res<Time>,
    mut known_windows: Local<HashSet<Entity>>,
) {
    for event in events.read() {
        let name = event
            .monitor
            .and_then(|monitor| monitors.get(monitor).ok())
            .and_then(|monitor| monitor.name.clone())
            .unwrap_or_else(|| "<unknown>".into());
        if known_windows.insert(event.window) {
            info!("Window {} is on monitor {name}.", event.window);
        } else {
            info!(
                "Window {} moved to monitor {name}; its DisplayTarget calibration may no \
                 longer match - recalibration recommended.",
                event.window
            );
            notice.seconds_remaining = 8.0;
        }
    }
    notice.seconds_remaining = (notice.seconds_remaining - time.delta_secs()).max(0.0);
}

fn update_pattern_visibility(
    pattern: Res<CalibrationPattern>,
    mut roots: Query<(&PatternRoot, &mut Visibility)>,
) {
    if !pattern.is_changed() {
        return;
    }
    for (root, mut visibility) in &mut roots {
        *visibility = if root.0 == *pattern {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
}

/// Recomputes every patch's linear color from the resolved
/// [`EffectiveDisplayTarget`].
///
/// With `Tonemapping::None` the scene-linear value `v` reaches the display
/// encoder unchanged, where `1.0` = paper white; a patch meant to show `N`
/// nits is therefore drawn at `N / paper_white_nits`. Reading the *resolved*
/// target (not the raw `DisplayTarget`) means a sensed peak or paper white
/// shows through the patches when those fields are on `Auto`.
fn update_patch_levels(
    effective: Single<
        &EffectiveDisplayTarget,
        (With<PrimaryWindow>, Changed<EffectiveDisplayTarget>),
    >,
    patches: Query<(&Patch, &MeshMaterial3d<StandardMaterial>)>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let target = effective.target;
    let paper_white = target.sanitized_paper_white_nits();
    for (patch, material) in &patches {
        let value = match patch {
            Patch::Nits(nits) => nits / paper_white,
            Patch::PeakFraction(fraction) => fraction * target.peak_luminance_nits / paper_white,
            Patch::PaperWhiteRelative(value) => *value,
        };
        if let Some(mut material) = materials.get_mut(&material.0) {
            material.base_color = Color::linear_rgb(value, value, value);
        }
    }
}

/// One-letter tag for a field's resolved [`FieldProvenance`], shown beside the
/// effective value so it is clear *why* it has that value.
fn provenance_tag(provenance: FieldProvenance) -> &'static str {
    match provenance {
        FieldProvenance::User => "user",
        FieldProvenance::Hgig => "HGIG",
        FieldProvenance::Policy => "policy",
        FieldProvenance::Os => "OS-sensed",
        FieldProvenance::Default => "default",
    }
}

/// Human-readable name for a [`DisplayInfoSource`].
fn source_name(source: DisplayInfoSource) -> &'static str {
    match source {
        DisplayInfoSource::Unknown => "unknown (nothing sensed)",
        DisplayInfoSource::Os => "OS / windowing system",
        DisplayInfoSource::DerivedFromHeadroom => "derived from EDR headroom",
        DisplayInfoSource::Web => "web coarse capability",
    }
}

/// Renders one optional sensed nits value, or "—" when the platform did not
/// report it.
fn fmt_nits(value: Option<f32>) -> String {
    value.map_or_else(|| "       —".into(), |nits| format!("{nits:7.1}n"))
}

fn update_ui(
    mut text: Single<&mut Text>,
    window: Single<
        (
            &DisplayTarget,
            &DisplayCalibrationPolicy,
            &EffectiveDisplayTarget,
            Option<&WindowDisplayState>,
            Option<&OnMonitor>,
        ),
        With<PrimaryWindow>,
    >,
    capabilities: Query<&MonitorDisplayCapability>,
    tonemapping: Single<&Tonemapping, With<Camera3d>>,
    pattern: Res<CalibrationPattern>,
    notice: Res<MonitorChangeNotice>,
) {
    let (display_target, policy, effective, live, on_monitor) = *window;
    let resolved = &effective.target;
    let provenance: &DisplayProvenance = &effective.provenance;
    let capability = on_monitor.and_then(|on_monitor| capabilities.get(on_monitor.0).ok());

    let auto = |field: AutoField| {
        if field == AutoField::Auto {
            "Auto"
        } else {
            "Keep"
        }
    };

    let mut ui = String::new();

    ui.push_str("HDR calibration (HGIG-style + OS sensing) - requires an HDR display\n\n");

    // The authored intent (what the manual keys edit).
    ui.push_str("DisplayTarget intent (primary window):\n");
    ui.push_str(&format!(
        "  paper white: {:7.1} nits  (Left/Right, Shift = fast)\n",
        display_target.paper_white_nits
    ));
    ui.push_str(&format!(
        "  peak:        {:7.1} nits  (Up/Down, Shift = fast)\n",
        display_target.peak_luminance_nits
    ));
    ui.push_str(&format!(
        "  min:         {:7.3} nits  ([ / ])\n",
        display_target.min_luminance_nits
    ));
    ui.push_str(&format!("  gamut:       {:?}\n", display_target.gamut));
    ui.push_str(&format!(
        "  transfer (requested): {:?}  (T cycles sRGB -> scRGB -> extended-sRGB -> PQ, R resets)\n\n",
        display_target.transfer
    ));

    // Which fields the renderer may auto-resolve.
    ui.push_str(&format!(
        "Calibration policy (A toggles peak/min/gamut Keep<->Auto):\n  \
         paper white {}   peak {}   min {}   gamut {}\n  \
         (paper white is always manual: the display cannot sense a viewing preference)\n\n",
        auto(policy.paper_white),
        auto(policy.peak_luminance),
        auto(policy.min_luminance),
        auto(policy.gamut),
    ));

    // The resolved target the renderer actually encodes for, with provenance.
    ui.push_str("EffectiveDisplayTarget (resolved, what the renderer encodes):\n");
    ui.push_str(&format!(
        "  paper white: {:7.1} nits  [{}]\n",
        resolved.paper_white_nits,
        provenance_tag(provenance.paper_white),
    ));
    ui.push_str(&format!(
        "  peak:        {:7.1} nits  [{}]\n",
        resolved.peak_luminance_nits,
        provenance_tag(provenance.peak_luminance),
    ));
    ui.push_str(&format!(
        "  min:         {:7.3} nits  [{}]\n",
        resolved.min_luminance_nits,
        provenance_tag(provenance.min_luminance),
    ));
    ui.push_str(&format!(
        "  gamut:       {:?}  [{}]\n",
        resolved.gamut,
        provenance_tag(provenance.gamut),
    ));
    ui.push_str(&format!(
        "  transfer:    {:?}  [{}]\n\n",
        resolved.transfer,
        provenance_tag(provenance.transfer),
    ));

    // What the platform reports right now (the inputs the resolver merges).
    ui.push_str("WindowDisplayState (live, sensed):\n");
    match live {
        Some(live) => {
            ui.push_str(&format!(
                "  HDR active: {}   source: {}\n",
                live.hdr_active
                    .map_or_else(|| "unknown".into(), |active| active.to_string()),
                source_name(live.source),
            ));
            ui.push_str(&format!(
                "  SDR white: {}   headroom potential: {}\n",
                fmt_nits(live.sdr_white_nits),
                live.headroom_potential
                    .map_or_else(|| "  —".into(), |h| format!("{h:.2}x")),
            ));
        }
        None => ui.push_str("  not sensed yet (no successful poll)\n"),
    }
    ui.push('\n');

    ui.push_str("MonitorDisplayCapability (static, sensed):\n");
    match capability {
        Some(capability) => {
            ui.push_str(&format!(
                "  peak: {}   full-frame: {}   black: {}\n",
                fmt_nits(capability.max_nits),
                fmt_nits(capability.max_full_frame_nits),
                fmt_nits(capability.min_nits),
            ));
            ui.push_str(&format!(
                "  gamut hint: {}   source: {}\n\n",
                capability
                    .gamut_hint
                    .map_or_else(|| "—".into(), |gamut| format!("{gamut:?}")),
                source_name(capability.source),
            ));
        }
        None => ui.push_str("  not sensed yet (no capability reported)\n\n"),
    }

    ui.push_str("Pattern:\n");
    for (key, label, this) in [
        ("1", "Peak luminance", CalibrationPattern::PeakLuminance),
        ("2", "Paper white", CalibrationPattern::PaperWhite),
        ("3", "Black level", CalibrationPattern::BlackLevel),
    ] {
        ui.push_str(&format!(
            "({key}) {} {label}\n",
            if *pattern == this { ">" } else { " " }
        ));
    }
    ui.push('\n');

    match *pattern {
        CalibrationPattern::PeakLuminance => ui.push_str(
            "The background checkerboard is at 9000/10000 nits (clipped to the display's\n\
             real peak); the center square tracks the resolved `peak_luminance_nits`. If\n\
             the center square is already invisible, lower the peak (Down) until it\n\
             appears, then raise it (Up) until it just disappears into the background:\n\
             that value is the display's peak luminance. (With peak on Auto, the sensed\n\
             peak drives the center square instead.)\n",
        ),
        CalibrationPattern::PaperWhite => ui.push_str(
            "The card is at exactly 1.0 = paper white, like this UI text. Adjust\n\
             `paper_white_nits` (Left/Right) until it reads as comfortable reference\n\
             white. The lower strip is 203 nits (ITU-R BT.2408 reference white).\n",
        ),
        CalibrationPattern::BlackLevel => ui.push_str(&format!(
            "Near-black steps, left to right (nits): {BLACK_STEP_NITS:?}.\n\
             Set `min_luminance_nits` ([ / ]) to the dimmest step you can tell apart\n\
             from the background. (Calibration metadata only: stored for tone-mapping\n\
             operators that lift shadows above the display's black floor.)\n",
        )),
    }

    ui.push_str(&format!(
        "\n(G) Gran Turismo 7 preview: {}\n",
        if **tonemapping == Tonemapping::GranTurismo7 {
            "ON - full HDR tone-mapping path; patterns are not exact while previewing"
        } else {
            "off - patterns are exact (Tonemapping::None)"
        }
    ));

    if notice.seconds_remaining > 0.0 {
        ui.push_str("\n*** Window moved to a different monitor - recalibration recommended. ***\n");
    }

    text.0 = ui;
}
