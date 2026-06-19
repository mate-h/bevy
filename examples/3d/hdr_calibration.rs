//! HGIG-style HDR display calibration, presented as a three-step guided wizard
//! with optional OS-sensed calibration.
//!
//! Three guided steps - peak luminance, paper white, black level - show one
//! calibration task at a time. Each step leads with a single instruction in the
//! banner, exposes exactly the one value it edits in the value bar, and keeps a
//! fixed data panel on the right that compares your *intent*, the resolved
//! *effective* target with provenance, and what the *display* reports. This is
//! the same flow an in-game "HDR settings" screen would implement, broken into
//! one decision per screen so it is easy to follow.
//!
//! All text - banner, value bar, data panel, legend, and the per-step labels -
//! is plain SDR [`bevy_ui`](bevy::ui). Only the calibration patches reach the
//! display encoder, through a separate [`Tonemapping::None`] camera, so their
//! luminance is never reshaped by tone mapping or UI compositing.
//!
//! **An HDR display is required to calibrate anything real.** scRGB-linear
//! output is reachable on macOS/iOS (Metal), Windows (Vulkan/DX12), and Linux
//! (Wayland + Vulkan, Mesa 25.1+); PQ (HDR10) output on Vulkan/DX12/Metal when
//! the OS has HDR output enabled (press `T` to cycle the requested transfer).
//! On SDR displays (or unsupported backends) the HDR request is downgraded - PQ
//! falls back to scRGB, then to plain SDR - with a warning in the log; the
//! example still runs, but everything brighter than paper white clips on the
//! SDR fallback, so the peak step looks flat. The data panel's "HDR active"
//! badge reports which path is live.
//!
//! # Manual intent versus sensed calibration
//!
//! [`DisplayTarget`] is the user's *intent* and Bevy never overwrites it. The
//! HGIG-style keys edit it directly, and a field whose
//! [`DisplayCalibrationPolicy`] is [`AutoField::Keep`] always renders that
//! authored value (top of the precedence ladder). A field set to
//! [`AutoField::Auto`] instead lets the renderer fill it from sensed display
//! information when the platform reports any - the result, and which provenance
//! won per field, lands in [`EffectiveDisplayTarget`], the target the renderer
//! actually encodes for. Press `A` to flip peak / black level / gamut between
//! `Keep` (manual) and `Auto` (OS-sensed); the data panel's `effective` column
//! and provenance tags change, and a one-line flash announces the swap. Paper
//! white stays manual: it is a viewing-environment preference, not a hardware
//! fact, so the display cannot sense it.
//!
//! The calibration camera uses [`Tonemapping::None`] so the patches reach the
//! display encoder at their exact authored values: a tone-mapping operator
//! would reshape the patches and the readings would calibrate the operator, not
//! the display. Patch brightness is authored in paper-white-relative units
//! (`1.0` = paper white), so a patch meant to show `N` nits is drawn at
//! `N / paper_white_nits`. The patches read the *resolved*
//! [`EffectiveDisplayTarget`], so when peak is on `Auto` the peak pattern tracks
//! the sensed peak. Press `G` to preview the Gran Turismo 7 operator on the same
//! patches (this exercises the full HDR tone-mapping path, but the patches are
//! no longer exact while it is on).

use bevy::{
    camera::{Hdr, ScalingMode},
    color::palettes::css,
    core_pipeline::tonemapping::Tonemapping,
    platform::collections::HashSet,
    prelude::*,
    window::{
        AutoField, DisplayCalibrationPolicy, DisplayInfoSource, DisplayProvenance, DisplayTarget,
        DisplayTransfer, EffectiveDisplayTarget, FieldProvenance, Monitor,
        MonitorDisplayCapability, OnMonitor, PrimaryWindow, WindowDisplayState,
        WindowMonitorChanged, WindowSupportedTransfers,
    },
};

/// The `DisplayTarget` requested at startup (and restored with `R`): a common
/// HDR baseline of 200-nit paper white on a 1000-nit display over scRGB.
///
/// The 1000-nit starting candidate exceeds the real peak of most consumer HDR
/// panels (typically 400-800 nits), so the peak step instructs a two-directional
/// procedure: lower the candidate first if the center square is already
/// invisible. When peak is on [`AutoField::Auto`] and the display reports its
/// peak, the sensed value takes over instead.
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
const BLACK_STEP_NITS: [f32; 7] = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0];

/// Shifts the patch cluster left so it never renders under the right-hand data
/// panel.
const STAGE_SHIFT: f32 = -1.5;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(ClearColor(Color::BLACK))
        .init_resource::<CalibrationPattern>()
        .init_resource::<MonitorChangeNotice>()
        .init_resource::<FlashNotice>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                advance_step,
                adjust_display_target,
                toggle_transfer,
                toggle_auto_sensing,
                toggle_gt7_preview,
                watch_monitor_changes,
                update_pattern_visibility,
                update_patch_levels,
                update_banner,
                update_value_bar,
                update_data_panel,
                update_black_step_labels,
            )
                .chain(),
        )
        .run();
}

/// The current wizard step. [`next`](CalibrationPattern::next) /
/// [`prev`](CalibrationPattern::prev) walk peak -> paper white -> black level
/// (clamped at the ends, no wrap); [`step_index`](CalibrationPattern::step_index)
/// numbers the steps `1..=3` for the banner.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq, Eq)]
enum CalibrationPattern {
    /// Adjust `peak_luminance_nits` until the center square merges with the
    /// clipped surround.
    #[default]
    PeakLuminance,
    /// Adjust `paper_white_nits` until the reference white card sits at a
    /// comfortable "white paper" brightness.
    PaperWhite,
    /// Find the dimmest near-black step distinguishable from black and record
    /// it as `min_luminance_nits`.
    BlackLevel,
}

impl CalibrationPattern {
    /// The next step, clamped at black level (no wrap).
    fn next(self) -> Self {
        match self {
            Self::PeakLuminance => Self::PaperWhite,
            Self::PaperWhite | Self::BlackLevel => Self::BlackLevel,
        }
    }

    /// The previous step, clamped at peak luminance (no wrap).
    fn prev(self) -> Self {
        match self {
            Self::BlackLevel => Self::PaperWhite,
            Self::PaperWhite | Self::PeakLuminance => Self::PeakLuminance,
        }
    }

    /// The step's 1-based number, for the banner.
    fn step_index(self) -> u8 {
        match self {
            Self::PeakLuminance => 1,
            Self::PaperWhite => 2,
            Self::BlackLevel => 3,
        }
    }
}

/// Marks the root entity of one calibration pattern; visibility is toggled from
/// [`CalibrationPattern`].
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

/// A short banner flash (with a countdown) that announces a state change the
/// user just caused with a key, so the cause and its effect are visible
/// together - used by `A` to call out the Keep<->Auto swap.
#[derive(Resource, Default)]
struct FlashNotice {
    text: String,
    seconds_remaining: f32,
}

/// Tags the banner text: the step title and its one instruction.
#[derive(Component)]
struct BannerText;

/// Tags the value-bar text: the single value the current step edits, its
/// marker, and its keys.
#[derive(Component)]
struct ValueBarText;

/// Tags the right-hand data-panel text: the intent / effective / sensed
/// comparison table, the policy line, and the live sensed footers.
#[derive(Component)]
struct DataPanelText;

/// Tags the parent of the black-level nit labels; shown only on step 3.
#[derive(Component)]
struct BlackStepLabels;

/// Tags one black-level nit label by its [`BLACK_STEP_NITS`] index, so its
/// color can brighten at or above the recorded `min_luminance`.
#[derive(Component)]
struct BlackStepLabel(usize);

/// Tags the BT.2408 203-nit label; shown only on step 2.
#[derive(Component)]
struct PaperRefLabel;

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
        // Calibration patches must reach the display encoder unmodified; an
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

    // Step 1: peak luminance. A clipped near-peak surround, a true-black
    // separating frame, and a center square at the candidate peak. The black
    // frame guarantees a hard border so the square is never "touching" the
    // surround; the square merges into the surround *across the black gap* at
    // the display's real peak (HGIG "MaxTML"). The candidate starts at 1000
    // nits (`INITIAL_HDR_TARGET`), which already clips on dimmer panels, so the
    // procedure is two-directional: if the square is already invisible, lower
    // peak until it reappears, then raise it until it just disappears again.
    // With peak on `Auto`, the sensed peak drives the center square instead.
    let peak_root = commands
        .spawn((
            Transform::from_xyz(STAGE_SHIFT, 0.0, 0.0),
            Visibility::Hidden,
            PatternRoot(CalibrationPattern::PeakLuminance),
        ))
        .id();
    patch(
        &mut commands,
        &mut meshes,
        &mut materials,
        peak_root,
        Vec2::new(9.0, 6.0),
        Vec3::new(0.0, 0.0, 0.0),
        Patch::Nits(10000.0),
    );
    patch(
        &mut commands,
        &mut meshes,
        &mut materials,
        peak_root,
        Vec2::splat(2.6),
        Vec3::new(0.0, 0.0, 0.05),
        Patch::Nits(0.0),
    );
    patch(
        &mut commands,
        &mut meshes,
        &mut materials,
        peak_root,
        Vec2::splat(2.0),
        Vec3::new(0.0, 0.0, 0.1),
        Patch::PeakFraction(1.0),
    );

    // Step 2: paper white. A reference white card at paper white over a dim
    // surround, with a BT.2408 203-nit strip and two comfort swatches for
    // context. The card and the UI text both sit at 1.0 = paper white.
    let paper_root = commands
        .spawn((
            Transform::from_xyz(STAGE_SHIFT, 0.0, 0.0),
            Visibility::Hidden,
            PatternRoot(CalibrationPattern::PaperWhite),
        ))
        .id();
    patch(
        &mut commands,
        &mut meshes,
        &mut materials,
        paper_root,
        Vec2::new(9.0, VIEW_HEIGHT),
        Vec3::ZERO,
        Patch::PaperWhiteRelative(0.15),
    );
    patch(
        &mut commands,
        &mut meshes,
        &mut materials,
        paper_root,
        Vec2::new(4.5, 2.6),
        Vec3::new(0.0, 0.5, 0.1),
        Patch::PaperWhiteRelative(1.0),
    );
    patch(
        &mut commands,
        &mut meshes,
        &mut materials,
        paper_root,
        Vec2::new(4.5, 0.7),
        Vec3::new(0.0, -2.2, 0.1),
        Patch::Nits(203.0),
    );
    patch(
        &mut commands,
        &mut meshes,
        &mut materials,
        paper_root,
        Vec2::splat(0.8),
        Vec3::new(-1.4, -1.0, 0.1),
        Patch::PaperWhiteRelative(0.5),
    );
    patch(
        &mut commands,
        &mut meshes,
        &mut materials,
        paper_root,
        Vec2::splat(0.8),
        Vec3::new(1.4, -1.0, 0.1),
        Patch::PaperWhiteRelative(0.75),
    );

    // Step 3: black level. Near-black steps at fixed absolute nit levels over
    // true black. Record the dimmest step you can distinguish as
    // `min_luminance_nits`; the matching nit label brightens at or above it.
    // (On 8-bit SDR the lowest steps quantize to the same code value and are
    // genuinely indistinguishable.)
    let black_root = commands
        .spawn((
            Transform::from_xyz(STAGE_SHIFT, 0.0, 0.0),
            Visibility::Hidden,
            PatternRoot(CalibrationPattern::BlackLevel),
        ))
        .id();
    patch(
        &mut commands,
        &mut meshes,
        &mut materials,
        black_root,
        Vec2::new(9.0, 6.0),
        Vec3::ZERO,
        Patch::Nits(0.0),
    );
    for (index, nits) in BLACK_STEP_NITS.into_iter().enumerate() {
        patch(
            &mut commands,
            &mut meshes,
            &mut materials,
            black_root,
            Vec2::new(1.0, 2.4),
            Vec3::new((index as f32 - 3.0) * 1.3, 0.0, 0.1),
            Patch::Nits(nits),
        );
    }

    // Banner: the step title and its one instruction, across the top.
    commands.spawn((
        BannerText,
        Text::default(),
        TextFont::from_font_size(26.0),
        TextLayout::justify(Justify::Center),
        Node {
            position_type: PositionType::Absolute,
            top: px(12),
            left: px(0),
            right: px(0),
            ..default()
        },
    ));

    // Data panel: the intent / effective / sensed comparison table and the live
    // sensed footers, pinned to the right (default left-justified text).
    commands.spawn((
        DataPanelText,
        Text::default(),
        TextFont::from_font_size(14.0),
        Node {
            position_type: PositionType::Absolute,
            top: px(150),
            right: px(12),
            width: px(440),
            ..default()
        },
    ));

    // Value bar: the single value the current step edits and its keys.
    commands.spawn((
        ValueBarText,
        Text::default(),
        TextFont::from_font_size(20.0),
        TextLayout::justify(Justify::Center),
        Node {
            position_type: PositionType::Absolute,
            bottom: px(44),
            left: px(0),
            right: px(0),
            ..default()
        },
    ));

    // Legend: the full key list, written once (it never changes).
    commands.spawn((
        Text::new(
            "N / Space next  |  P prev  |  1|2|3 jump  |  A auto  |  \
             T transfer  |  G GT7  |  R reset",
        ),
        TextFont::from_font_size(13.0),
        TextLayout::justify(Justify::Center),
        Node {
            position_type: PositionType::Absolute,
            bottom: px(12),
            left: px(0),
            right: px(0),
            ..default()
        },
    ));

    // BT.2408 203-nit label for step 2. It sits in the dim surround just above
    // the reference strip rather than over it, so the white text stays legible
    // instead of washing out against the strip's near-paper-white fill.
    commands.spawn((
        PaperRefLabel,
        Text::new("203 nits - BT.2408 reference white (strip below)"),
        TextFont::from_font_size(14.0),
        TextLayout::justify(Justify::Center),
        Visibility::Hidden,
        Node {
            position_type: PositionType::Absolute,
            bottom: px(200),
            left: px(0),
            right: px(0),
            ..default()
        },
    ));

    // Black-level nit legend for step 3, listing the step values in the same
    // dim-to-bright order they appear on screen. It reads as a centered legend
    // rather than a per-patch overlay, so it needs no world-to-screen projection
    // and does not drift as the window resizes. The value at or above the
    // recorded `min_luminance` brightens in `update_black_step_labels`.
    commands
        .spawn((
            BlackStepLabels,
            Visibility::Hidden,
            Node {
                position_type: PositionType::Absolute,
                bottom: px(96),
                left: px(0),
                right: px(0),
                flex_direction: FlexDirection::Row,
                justify_content: JustifyContent::Center,
                column_gap: px(14),
                ..default()
            },
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("near-black steps (nits), dim to bright:"),
                TextFont::from_font_size(13.0),
                TextColor(css::GRAY.into()),
            ));
            for (index, nits) in BLACK_STEP_NITS.into_iter().enumerate() {
                parent.spawn((
                    BlackStepLabel(index),
                    Text::new(format!("{nits}")),
                    TextFont::from_font_size(13.0),
                    TextColor(css::GRAY.into()),
                ));
            }
        });
}

/// Walks the wizard with `N` / `Space` (next) and `P` (previous), and jumps
/// directly with `1` / `2` / `3`. Stepping is clamped at the ends (no wrap).
fn advance_step(keys: Res<ButtonInput<KeyCode>>, mut pattern: ResMut<CalibrationPattern>) {
    if keys.just_pressed(KeyCode::KeyN) || keys.just_pressed(KeyCode::Space) {
        *pattern = pattern.next();
    }
    if keys.just_pressed(KeyCode::KeyP) {
        *pattern = pattern.prev();
    }
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

/// The transfers `toggle_transfer` steps through, in cycle order: sRGB ->
/// scRGB-linear -> extended-sRGB (encoded, the web HDR path) -> PQ (HDR10) ->
/// sRGB. [`WindowSupportedTransfers`] reports its set in this same order.
const TRANSFER_CYCLE: [DisplayTransfer; 4] = [
    DisplayTransfer::Srgb,
    DisplayTransfer::ScRgbLinear,
    DisplayTransfer::ExtendedSrgb,
    DisplayTransfer::Pq,
];

/// Cycles the requested transfer through only the transfers the surface
/// advertises (via [`WindowSupportedTransfers`]), in the order sRGB ->
/// scRGB-linear -> extended-sRGB (encoded, the web HDR path) -> PQ (HDR10) ->
/// sRGB, skipping any the surface cannot present so `T` never steps into a mode
/// that would silently downgrade to SDR. Before the first surface
/// configuration the component is absent, so it falls back to the full cycle.
/// The surface is renegotiated the same frame. The gamut is left at its default
/// Rec.709 (the encoder coerces it per transfer); the `tonemapping` example
/// demonstrates the wide-gamut Display-P3 path.
///
/// The transfer is the one calibration field that is *never* auto-resolved:
/// changing it renegotiates the swapchain, so [`DisplayCalibrationPolicy`] has
/// no transfer field and sensing never rewrites it.
fn toggle_transfer(
    keys: Res<ButtonInput<KeyCode>>,
    window: Single<(&mut DisplayTarget, Option<&WindowSupportedTransfers>), With<PrimaryWindow>>,
) {
    if !keys.just_pressed(KeyCode::KeyT) {
        return;
    }
    let (mut display_target, supported) = window.into_inner();
    let current = display_target.transfer;
    // Walk the fixed cycle from the current transfer, wrapping, and take the
    // next entry the surface can present. With no capability information yet
    // (pre-first-config) every transfer is treated as a candidate.
    let start = TRANSFER_CYCLE
        .iter()
        .position(|&t| t == current)
        .unwrap_or(0);
    let next = (1..=TRANSFER_CYCLE.len())
        .map(|offset| TRANSFER_CYCLE[(start + offset) % TRANSFER_CYCLE.len()])
        .find(|&t| supported.is_none_or(|s| s.contains(t)))
        .unwrap_or(current);
    display_target.transfer = next;
}

/// Flips peak, black level, and gamut between [`AutoField::Keep`] (the manual
/// HGIG values authored above win) and [`AutoField::Auto`] (the renderer
/// resolves them from sensed display information when the platform reports
/// any), and flashes a one-line notice so the effective column's swap is
/// visible. Paper white is deliberately omitted: it is a viewing preference the
/// display cannot sense, so it stays manual.
fn toggle_auto_sensing(
    keys: Res<ButtonInput<KeyCode>>,
    mut policy: Single<&mut DisplayCalibrationPolicy, With<PrimaryWindow>>,
    mut flash: ResMut<FlashNotice>,
) {
    if keys.just_pressed(KeyCode::KeyA) {
        let next = match policy.peak_luminance {
            AutoField::Keep => AutoField::Auto,
            AutoField::Auto => AutoField::Keep,
        };
        policy.peak_luminance = next;
        policy.min_luminance = next;
        policy.gamut = next;
        flash.text = match next {
            AutoField::Auto => "peak/min/gamut -> Auto: effective now follows the display".into(),
            AutoField::Keep => "peak/min/gamut -> Keep: effective follows your numbers".into(),
        };
        flash.seconds_remaining = 5.0;
    }
}

/// Toggles a Gran Turismo 7 tone-mapping preview. On an HDR display target
/// the operator automatically runs in its HDR mode, plumbing the resolved
/// peak luminance into the tone curve, so this exercises the full HDR
/// tone-mapping path end to end (add `GranTurismo7Params` to tune it). While
/// the preview is on, the patches are *not* exact - calibrate with it off.
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
/// is user-authoritative (Bevy never rewrites it), so suggest recalibrating -
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

/// Shows the patches for the current step and hides the rest, and toggles the
/// step-specific `bevy_ui` labels (the black-level nit row and the 203-nit
/// label) to match.
fn update_pattern_visibility(
    pattern: Res<CalibrationPattern>,
    mut roots: Query<
        (&PatternRoot, &mut Visibility),
        (Without<BlackStepLabels>, Without<PaperRefLabel>),
    >,
    mut black_labels: Single<&mut Visibility, With<BlackStepLabels>>,
    mut paper_label: Single<&mut Visibility, (With<PaperRefLabel>, Without<BlackStepLabels>)>,
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
    **black_labels = visible_if(*pattern == CalibrationPattern::BlackLevel);
    **paper_label = visible_if(*pattern == CalibrationPattern::PaperWhite);
}

/// Maps a boolean to a shown/hidden [`Visibility`].
fn visible_if(shown: bool) -> Visibility {
    if shown {
        Visibility::Inherited
    } else {
        Visibility::Hidden
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

/// Writes the banner: the step title (`Step N of 3 - NAME`), the step's one
/// instruction, and the flash line when `A` (or a monitor change) just fired.
fn update_banner(
    mut text: Single<&mut Text, With<BannerText>>,
    pattern: Res<CalibrationPattern>,
    mut flash: ResMut<FlashNotice>,
    mut monitor_notice: ResMut<MonitorChangeNotice>,
    time: Res<Time>,
) {
    let (name, instruction) = match *pattern {
        CalibrationPattern::PeakLuminance => (
            "PEAK LUMINANCE",
            "Raise PEAK (Up) until the centre square just vanishes into the bright surround, \
             then lower one tap. If it is already invisible, lower PEAK until it reappears first.",
        ),
        CalibrationPattern::PaperWhite => (
            "PAPER WHITE",
            "Adjust PAPER WHITE (Left/Right) until the card looks like comfortable white paper \
             for your room (dark ~100, normal ~200, bright ~250 nits).",
        ),
        CalibrationPattern::BlackLevel => (
            "BLACK LEVEL",
            "Find the dimmest step you can still tell apart from black, then set MIN ([ / ]) to \
             its value. MIN is recorded metadata; the patches do not change, but the matching \
             label brightens.",
        ),
    };

    let mut banner = format!("Step {} of 3 - {name}\n{instruction}", pattern.step_index(),);

    // The flash line announces a swap the user just caused; the monitor notice
    // shares the same below-the-instruction slot.
    flash.seconds_remaining = (flash.seconds_remaining - time.delta_secs()).max(0.0);
    if flash.seconds_remaining > 0.0 {
        banner.push_str(&format!("\n{}", flash.text));
    } else if monitor_notice.seconds_remaining > 0.0 {
        banner.push_str("\nWindow moved to a different monitor - recalibration recommended.");
    }

    // Keep the monitor countdown advancing even when the flash is showing.
    if flash.seconds_remaining > 0.0 {
        monitor_notice.seconds_remaining =
            (monitor_notice.seconds_remaining - time.delta_secs()).max(0.0);
    }

    text.0 = banner;
}

/// Writes the value bar: the single value the current step edits, a marker, and
/// the keys that change it. Reads the *authored* [`DisplayTarget`] (the value
/// the keys move), since that is what the user is editing.
fn update_value_bar(
    mut text: Single<&mut Text, With<ValueBarText>>,
    display_target: Single<&DisplayTarget, With<PrimaryWindow>>,
    pattern: Res<CalibrationPattern>,
) {
    text.0 = match *pattern {
        CalibrationPattern::PeakLuminance => format!(
            "PEAK  *  {:.1} nits   <- editing   (Up / Down,  Shift = 4x)",
            display_target.peak_luminance_nits,
        ),
        CalibrationPattern::PaperWhite => format!(
            "PAPER WHITE  *  {:.1} nits   <- editing   (Left / Right,  Shift = 4x)",
            display_target.paper_white_nits,
        ),
        CalibrationPattern::BlackLevel => {
            let min = display_target.min_luminance_nits;
            let nearest = BLACK_STEP_NITS
                .into_iter()
                .min_by(|a, b| {
                    (a - min)
                        .abs()
                        .partial_cmp(&(b - min).abs())
                        .unwrap_or(core::cmp::Ordering::Equal)
                })
                .unwrap_or(0.0);
            format!(
                "MIN  *  {min:.3} nits   <- editing   ( [ / ] )   | nearest step: {nearest:.2} nits",
            )
        }
    };
}

/// Brightens the black-level nit labels at or above the recorded
/// `min_luminance` and dims the rest, so editing MIN produces a visible change
/// even though MIN is metadata only and the patches stay fixed.
fn update_black_step_labels(
    display_target: Single<&DisplayTarget, With<PrimaryWindow>>,
    mut labels: Query<(&BlackStepLabel, &mut TextColor)>,
) {
    let min = display_target.min_luminance_nits;
    for (label, mut color) in &mut labels {
        let nits = BLACK_STEP_NITS[label.0];
        color.0 = if nits >= min {
            Color::WHITE
        } else {
            css::GRAY.into()
        };
    }
}

/// One-word tag for a field's resolved [`FieldProvenance`], shown beside the
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

/// Renders one optional sensed nits value, or "-" when the platform did not
/// report it.
fn fmt_nits(value: Option<f32>) -> String {
    value.map_or_else(|| "      -".into(), |nits| format!("{nits:7.1}"))
}

/// Writes the right-hand data panel: the SDR-honesty badge, the
/// intent / effective[prov] / sensed comparison table (with a `>` marker on the
/// row the current step edits), the policy line, the live `WindowDisplayState`
/// and `MonitorDisplayCapability` footers, and the provenance legend plus GT7
/// status. Every sensing field the example surfaces lives here.
fn update_data_panel(
    mut text: Single<&mut Text, With<DataPanelText>>,
    window: Single<
        (
            &DisplayTarget,
            &DisplayCalibrationPolicy,
            &EffectiveDisplayTarget,
            Option<&WindowDisplayState>,
            Option<&OnMonitor>,
            Option<&WindowSupportedTransfers>,
        ),
        With<PrimaryWindow>,
    >,
    capabilities: Query<&MonitorDisplayCapability>,
    tonemapping: Single<&Tonemapping, With<Camera3d>>,
    pattern: Res<CalibrationPattern>,
) {
    let (display_target, policy, effective, live, on_monitor, supported) = *window;
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
    // The `>` marker points at the field the current step edits.
    let marker = |this: CalibrationPattern| {
        if *pattern == this {
            ">"
        } else {
            " "
        }
    };

    // Sensed inputs the resolver merges, shown raw in the `sensed` column.
    let sensed_peak = capability.and_then(|c| c.max_nits);
    let sensed_min = capability.and_then(|c| c.min_nits);
    let sensed_gamut = capability
        .and_then(|c| c.gamut_hint)
        .map_or_else(|| "      -".into(), |gamut| format!("{gamut:>7?}"));

    let hdr_active = live
        .and_then(|live| live.hdr_active)
        .map_or_else(|| "unknown".into(), |active| active.to_string());

    // The transfers `T` will cycle through, consumed from the renderer-sensed
    // capability. Absent until the first surface configuration.
    let transfers_available = match supported {
        Some(supported) => supported
            .iter()
            .map(transfer_short_name)
            .collect::<Vec<_>>()
            .join(", "),
        None => "sensing...".into(),
    };

    let mut ui = String::new();
    ui.push_str("DISPLAY CALIBRATION\n");
    ui.push_str(&format!(
        "HDR active: {hdr_active} | transfer: {:?}\n",
        resolved.transfer,
    ));
    ui.push_str(&format!("transfers available: {transfers_available}\n\n"));

    // intent | effective[prov] | sensed comparison table.
    ui.push_str("field      intent    effective[prov]      sensed\n");
    ui.push_str(&format!(
        "{} paper   {:7.1}    {:7.1} [{}]{}{}\n",
        marker(CalibrationPattern::PaperWhite),
        display_target.paper_white_nits,
        resolved.paper_white_nits,
        provenance_tag(provenance.paper_white),
        pad(provenance_tag(provenance.paper_white)),
        // Paper white is never sensed.
        "      -",
    ));
    ui.push_str(&format!(
        "{} peak    {:7.1}    {:7.1} [{}]{}{}\n",
        marker(CalibrationPattern::PeakLuminance),
        display_target.peak_luminance_nits,
        resolved.peak_luminance_nits,
        provenance_tag(provenance.peak_luminance),
        pad(provenance_tag(provenance.peak_luminance)),
        fmt_nits(sensed_peak),
    ));
    ui.push_str(&format!(
        "{} min     {:7.3}    {:7.3} [{}]{}{}\n",
        marker(CalibrationPattern::BlackLevel),
        display_target.min_luminance_nits,
        resolved.min_luminance_nits,
        provenance_tag(provenance.min_luminance),
        pad(provenance_tag(provenance.min_luminance)),
        fmt_nits(sensed_min),
    ));
    ui.push_str(&format!(
        "  gamut   {:>7?}    {:>7?} [{}]{}{}\n\n",
        display_target.gamut,
        resolved.gamut,
        provenance_tag(provenance.gamut),
        pad(provenance_tag(provenance.gamut)),
        sensed_gamut,
    ));

    // Which fields the renderer may auto-resolve.
    ui.push_str(&format!(
        "policy: paper {} | peak {} | min {} | gamut {}\n",
        auto(policy.paper_white),
        auto(policy.peak_luminance),
        auto(policy.min_luminance),
        auto(policy.gamut),
    ));
    ui.push_str("(A toggles peak/min/gamut; paper white is always manual)\n\n");

    // Live window state.
    match live {
        Some(live) => {
            ui.push_str(&format!(
                "Window state (live): HDR {} | src {}\n",
                live.hdr_active
                    .map_or_else(|| "unknown".into(), |active| active.to_string()),
                source_name(live.source),
            ));
            ui.push_str(&format!(
                "  SDR white {} | headroom {}\n",
                fmt_nits(live.sdr_white_nits),
                live.headroom_potential
                    .map_or_else(|| "-".into(), |h| format!("{h:.2}x")),
            ));
        }
        None => ui.push_str("Window state (live): not sensed yet (no successful poll)\n"),
    }

    // Static monitor capability.
    match capability {
        Some(capability) => {
            ui.push_str(&format!(
                "Monitor cap (static): peak {} | full-frame {} | black {}\n",
                fmt_nits(capability.max_nits),
                fmt_nits(capability.max_full_frame_nits),
                fmt_nits(capability.min_nits),
            ));
            ui.push_str(&format!(
                "  gamut {} | src {}\n\n",
                capability
                    .gamut_hint
                    .map_or_else(|| "-".into(), |gamut| format!("{gamut:?}")),
                source_name(capability.source),
            ));
        }
        None => ui.push_str("Monitor cap (static): not sensed yet (no capability reported)\n\n"),
    }

    ui.push_str("prov: user / HGIG / policy / OS-sensed / default  (precedence top->down)\n");
    ui.push_str(&format!(
        "GT7 preview: {}\n",
        if **tonemapping == Tonemapping::GranTurismo7 {
            "on - full HDR path; patches not exact while previewing"
        } else {
            "off - patches exact (Tonemapping::None)"
        }
    ));

    text.0 = ui;
}

/// Pads a provenance tag to a fixed width so the `sensed` column lines up
/// regardless of which tag won.
fn pad(tag: &str) -> String {
    " ".repeat("OS-sensed".len().saturating_sub(tag.len()) + 3)
}

/// A short, human-readable name for a [`DisplayTransfer`], for the
/// `transfers available` line.
fn transfer_short_name(transfer: DisplayTransfer) -> &'static str {
    match transfer {
        DisplayTransfer::Srgb => "sRGB",
        DisplayTransfer::ScRgbLinear => "scRGB",
        DisplayTransfer::ExtendedSrgb => "ext-sRGB",
        DisplayTransfer::Pq => "PQ",
        DisplayTransfer::Hlg => "HLG",
    }
}
