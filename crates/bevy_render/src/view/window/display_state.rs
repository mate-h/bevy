//! Render-side display sensing: polls each surface's live HDR state, collapses
//! every platform's reporting asymmetry into one absolute-nits snapshot
//! ([`normalize`]), smooths it ([`DisplayStateStore`]), and mirrors it back to
//! the main world as [`WindowDisplayState`] / [`MonitorDisplayCapability`].
//!
//! `bevy_window` stays wgpu-free: it holds only the plain-data result types.
//! Everything that touches a wgpu [`DisplayHdrInfo`] lives here.

use bevy_ecs::prelude::*;
use bevy_platform::collections::HashMap;
use bevy_platform::time::Instant;
use bevy_window::{
    DisplayChromaticity, DisplayCoarseRange, DisplayGamut, DisplayInfoSource,
    MonitorDisplayCapability, OnMonitor, WindowDisplayState,
};
use core::time::Duration;
use wgpu::{DisplayGamut as WgpuDisplayGamut, DisplayHdrInfo};

use crate::renderer::RenderAdapter;
use crate::MainWorld;

use super::{ExtractedWindows, WindowSurfaces};

/// A platform-agnostic, absolute-nits snapshot of one surface's display state,
/// produced by [`normalize`]. All cross-platform asymmetry collapses here:
/// callers downstream never see per-backend `Option` shapes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct DisplaySnapshot {
    // Live-state fields (mirrored to `WindowDisplayState`).
    pub hdr_active: Option<bool>,
    pub headroom_current: Option<f32>,
    pub headroom_potential: Option<f32>,
    pub headroom_reference: Option<f32>,
    pub sdr_white_nits: Option<f32>,
    // Capability fields (mirrored to `MonitorDisplayCapability`).
    pub max_nits: Option<f32>,
    pub max_full_frame_nits: Option<f32>,
    pub min_nits: Option<f32>,
    pub chromaticity: Option<DisplayChromaticity>,
    pub gamut_hint: Option<DisplayGamut>,
    pub bits_per_color: Option<u8>,
    pub coarse: Option<DisplayCoarseRange>,
    /// Provenance of this snapshot, shared by the live-state and capability
    /// halves (a single poll has one source).
    pub source: DisplayInfoSource,
}

impl DisplaySnapshot {
    /// The capability half of this snapshot as a [`MonitorDisplayCapability`]
    /// (a field-for-field copy).
    fn capability(&self) -> MonitorDisplayCapability {
        MonitorDisplayCapability {
            max_nits: self.max_nits,
            max_full_frame_nits: self.max_full_frame_nits,
            min_nits: self.min_nits,
            chromaticity: self.chromaticity,
            gamut_hint: self.gamut_hint,
            bits_per_color: self.bits_per_color,
            coarse: self.coarse,
            source: self.source,
        }
    }

    /// The live-state half of this snapshot as a [`WindowDisplayState`]
    /// candidate. `generation` starts at zero; [`commit`] bumps it when the
    /// candidate actually commits.
    fn live_state(&self) -> WindowDisplayState {
        WindowDisplayState {
            hdr_active: self.hdr_active,
            headroom_current: self.headroom_current,
            headroom_potential: self.headroom_potential,
            headroom_reference: self.headroom_reference,
            sdr_white_nits: self.sdr_white_nits,
            source: self.source,
            generation: 0,
        }
    }
}

/// Maps a wgpu coarse [`DisplayGamut`](WgpuDisplayGamut) onto the plain-data
/// [`DisplayGamut`] the window crate carries. `Srgb` and any unrecognized
/// future wgpu variant map to the narrowest known gamut
/// ([`DisplayGamut::Rec709`]), the conservative choice for a capability hint.
fn map_gamut(g: WgpuDisplayGamut) -> DisplayGamut {
    match g {
        WgpuDisplayGamut::DisplayP3 => DisplayGamut::DisplayP3,
        WgpuDisplayGamut::Rec2020 => DisplayGamut::Rec2020,
        // `Srgb` and any unrecognized future variant.
        _ => DisplayGamut::Rec709,
    }
}

/// Returns `Some` only for finite, strictly-positive values; folds wgpu's
/// "reported but garbage" (NaN, infinity, zero, negative) into "not reported".
fn finite_positive(v: Option<f32>) -> Option<f32> {
    v.filter(|x| x.is_finite() && *x > 0.0)
}

/// Reconstructs an absolute peak from a relative headroom: the live SDR white
/// times the potential headroom multiplier, or `None` unless both are present.
/// The anchor is always the live SDR white, never a fixed constant — a display's
/// SDR white moves with its brightness control, so a fixed anchor would
/// mis-scale every highlight.
pub(crate) fn anchored_peak(
    sdr_white_nits: Option<f32>,
    headroom_potential: Option<f32>,
) -> Option<f32> {
    match (sdr_white_nits, headroom_potential) {
        (Some(white), Some(potential)) => Some(white * potential),
        _ => None,
    }
}

/// Mirrors a wgpu chromaticity report onto the plain-data window-crate type.
fn map_chromaticity(c: wgpu::DisplayChromaticity) -> DisplayChromaticity {
    DisplayChromaticity {
        red: c.red,
        green: c.green,
        blue: c.blue,
        white: c.white,
    }
}

/// Mirrors a wgpu coarse-range report onto the plain-data window-crate type,
/// classifying its gamut bucket through [`map_gamut`].
fn map_coarse(c: wgpu::DisplayCoarseRange) -> DisplayCoarseRange {
    DisplayCoarseRange {
        high_dynamic_range: c.high_dynamic_range,
        gamut: c.gamut.map(map_gamut),
    }
}

/// Collapses a wgpu [`DisplayHdrInfo`] into a [`DisplaySnapshot`] in absolute
/// nits.
///
/// The two reporting models are reconciled here:
///
/// - **Absolute (Windows):** `luminance` carries absolute nits directly. Peak
///   is `max_nits`; SDR white is `sdr_white_nits`.
/// - **Relative (Apple EDR):** `headroom` carries unitless multipliers over the
///   *live* SDR white. An absolute peak is reconstructed by anchoring on that
///   live white: `peak = sdr_white_nits * headroom_potential`. The anchor is
///   the live SDR white, never a fixed 100/203-nit constant — Apple's SDR white
///   moves with the brightness slider, so a fixed anchor would mis-scale every
///   highlight.
///
/// `source` is the caller-classified provenance ([`Os`](DisplayInfoSource::Os)
/// for the Windows path,
/// [`DerivedFromHeadroom`](DisplayInfoSource::DerivedFromHeadroom) for the Apple
/// path, [`Web`](DisplayInfoSource::Web) for coarse capability). It determines
/// the live-state source and how downstream hysteresis treats `hdr_active`.
pub(crate) fn normalize(info: &DisplayHdrInfo, source: DisplayInfoSource) -> DisplaySnapshot {
    let luminance = info.luminance;
    let headroom = info.headroom;

    let sdr_white_nits = luminance.and_then(|l| finite_positive(l.sdr_white_nits));
    let headroom_current = headroom.and_then(|h| finite_positive(h.current));
    let headroom_potential = headroom.and_then(|h| finite_positive(h.potential));
    let headroom_reference = headroom.and_then(|h| finite_positive(h.reference));

    // Absolute peak: prefer the directly-reported absolute nits; otherwise
    // reconstruct from live SDR white * potential headroom.
    let max_nits = finite_positive(luminance.and_then(|l| l.max_nits))
        .or(anchored_peak(sdr_white_nits, headroom_potential));
    let max_full_frame_nits = finite_positive(luminance.and_then(|l| l.max_full_frame_nits));
    let min_nits = finite_positive(luminance.and_then(|l| l.min_nits));

    let chromaticity = info.chromaticity.map(map_chromaticity);
    let coarse = info.coarse.map(map_coarse);
    let gamut_hint = coarse.and_then(|c| c.gamut);

    DisplaySnapshot {
        hdr_active: info.hdr_active,
        headroom_current,
        headroom_potential,
        headroom_reference,
        sdr_white_nits,
        max_nits,
        max_full_frame_nits,
        min_nits,
        chromaticity,
        gamut_hint,
        bits_per_color: info.bits_per_color,
        coarse,
        source,
    }
}

/// Classifies which reporting model a [`DisplayHdrInfo`] represents, so callers
/// pass the right [`DisplayInfoSource`] to [`normalize`]. Absolute luminance →
/// [`Os`](DisplayInfoSource::Os); relative headroom →
/// [`DerivedFromHeadroom`](DisplayInfoSource::DerivedFromHeadroom); coarse-only
/// → [`Web`](DisplayInfoSource::Web); nothing usable →
/// [`Unknown`](DisplayInfoSource::Unknown).
pub(crate) fn classify_source(info: &DisplayHdrInfo) -> DisplayInfoSource {
    if info
        .luminance
        .is_some_and(|l| l.max_nits.is_some() || l.sdr_white_nits.is_some())
    {
        DisplayInfoSource::Os
    } else if info.headroom.is_some() {
        DisplayInfoSource::DerivedFromHeadroom
    } else if info.coarse.is_some() {
        DisplayInfoSource::Web
    } else {
        DisplayInfoSource::Unknown
    }
}

/// Relative change below which a continuous field is considered unchanged.
const HYSTERESIS_REL_THRESHOLD: f32 = 0.05;
/// How long a changed continuous field must hold before it commits.
const HYSTERESIS_DWELL: Duration = Duration::from_millis(500);
/// Minimum interval between unforced polls.
const POLL_THROTTLE: Duration = Duration::from_millis(250);

/// The committed (post-hysteresis) [`WindowDisplayState`] for one surface, plus
/// the pending candidate and the instant it was first observed.
#[derive(Debug, Clone, Copy)]
struct CommittedState {
    committed: WindowDisplayState,
    /// A continuous-field candidate awaiting its dwell, and when it appeared.
    pending: Option<(WindowDisplayState, Instant)>,
}

/// Per-surface hysteresis / dwell bookkeeping and the last committed snapshot,
/// keyed by window entity. Render-world only.
#[derive(Resource, Default)]
pub struct DisplayStateStore {
    states: HashMap<Entity, CommittedState>,
    /// The last committed capability per window (so the [`Monitor`] write-back
    /// is insert-on-change).
    ///
    /// [`Monitor`]: bevy_window::Monitor
    capabilities: HashMap<Entity, MonitorDisplayCapability>,
    /// Last unforced-poll instant (throttle tick).
    last_poll: Option<Instant>,
}

/// Whether two optional continuous values differ by more than the relative
/// threshold (a `None`→`Some` or `Some`→`None` transition always counts).
fn rel_changed(old: Option<f32>, new: Option<f32>) -> bool {
    match (old, new) {
        (Some(a), Some(b)) => {
            (a - b).abs() > HYSTERESIS_REL_THRESHOLD * a.abs().max(f32::MIN_POSITIVE)
        }
        (None, None) => false,
        _ => true,
    }
}

/// Folds a fresh snapshot into the committed state for `entity`, applying:
/// discrete commit for [`Os`](DisplayInfoSource::Os)-sourced `hdr_active` (a
/// real OS toggle), dwell + relative-threshold hysteresis for headroom-derived
/// `hdr_active` and every continuous field, and a `generation` bump only when
/// something actually commits. The diagnostic `headroom_current` always tracks
/// the latest poll without bumping `generation`.
fn commit(store: &mut DisplayStateStore, entity: Entity, snap: &DisplaySnapshot, now: Instant) {
    // `generation` is set on commit below; `headroom_current` is diagnostic and
    // passes through unsmoothed.
    let candidate = snap.live_state();

    let entry = store.states.entry(entity).or_insert(CommittedState {
        committed: WindowDisplayState::default(),
        pending: None,
    });
    let committed = entry.committed;

    // `hdr_active`: discrete commit when the source is a real OS toggle;
    // dwell-smoothed (below) when merely derived from headroom.
    let hdr_active_commits = match snap.source {
        DisplayInfoSource::Os | DisplayInfoSource::Web => {
            candidate.hdr_active != committed.hdr_active
        }
        DisplayInfoSource::DerivedFromHeadroom | DisplayInfoSource::Unknown => false,
    };

    // Continuous-field hysteresis: a meaningful change must hold for the dwell.
    // A headroom-derived `hdr_active` flip is smoothed the same way.
    let continuous_changed =
        rel_changed(committed.headroom_potential, candidate.headroom_potential)
            || rel_changed(committed.headroom_reference, candidate.headroom_reference)
            || rel_changed(committed.sdr_white_nits, candidate.sdr_white_nits)
            || (snap.source == DisplayInfoSource::DerivedFromHeadroom
                && candidate.hdr_active != committed.hdr_active);

    let dwell_passed = if continuous_changed {
        match entry.pending {
            Some((_, since)) if now.duration_since(since) >= HYSTERESIS_DWELL => true,
            Some(_) => false,
            None => {
                entry.pending = Some((candidate, now));
                false
            }
        }
    } else {
        entry.pending = None;
        false
    };

    // The diagnostic `headroom_current` always tracks the latest poll without
    // bumping `generation`.
    let diag_changed = entry.committed.headroom_current != candidate.headroom_current;

    if hdr_active_commits || dwell_passed {
        let mut next = candidate;
        next.generation = committed.generation.wrapping_add(1);
        entry.committed = next;
        entry.pending = None;
    } else if diag_changed {
        entry.committed.headroom_current = candidate.headroom_current;
        // No generation bump: `headroom_current` is diagnostic-only.
    }

    // Capability snapshot (no hysteresis; insert-on-change handled at
    // write-back).
    store.capabilities.insert(entity, snap.capability());
}

/// Polls each configured window surface's live HDR state, smooths it through
/// hysteresis, and stores the committed result for write-back.
///
/// Main-thread-pinned on Apple platforms (the relative-headroom query returns
/// `None` off the main thread). Throttled to one unforced poll every
/// [`POLL_THROTTLE`]; a window is force-polled regardless of the throttle on
/// first sight and on surface (re)configuration this frame (a
/// [`display_target_transfer_changed`](super::ExtractedWindow::display_target_transfer_changed)
/// renegotiation). A monitor move is picked up by the next throttled poll: the
/// render world has no main-world monitor-change signal, and the surface's new
/// capability surfaces within [`POLL_THROTTLE`] regardless. A
/// [`None`](DisplayInfoSource::Unknown) poll leaves the committed value
/// untouched — `None` never means "SDR".
pub fn poll_display_state(
    // Apple's relative-headroom query gates on the main thread; pin the system
    // there, matching `create_surfaces`.
    #[cfg(any(target_os = "macos", target_os = "ios"))] _marker: bevy_ecs::system::NonSendMarker,
    window_surfaces: Res<WindowSurfaces>,
    extracted_windows: Res<ExtractedWindows>,
    render_adapter: Res<RenderAdapter>,
    mut store: ResMut<DisplayStateStore>,
) {
    let now = Instant::now();
    let throttled = store
        .last_poll
        .is_some_and(|t| now.duration_since(t) < POLL_THROTTLE);

    // Drop bookkeeping for surfaces that went away.
    store
        .states
        .retain(|e, _| window_surfaces.surfaces.contains_key(e));
    store
        .capabilities
        .retain(|e, _| window_surfaces.surfaces.contains_key(e));

    let mut polled_any = false;

    for (&entity, surface_data) in window_surfaces.surfaces.iter() {
        // Force a poll on first sight or on a fresh (re)configuration this
        // frame; otherwise honor the throttle.
        let first_time = !store.states.contains_key(&entity);
        let forced = first_time
            || extracted_windows
                .get(&entity)
                .is_some_and(|w| w.display_target_transfer_changed);
        if throttled && !forced {
            continue;
        }
        polled_any = true;

        let info = surface_data.surface.display_hdr_info(&render_adapter);
        let source = classify_source(&info);
        if source == DisplayInfoSource::Unknown {
            // "Can't tell": leave the committed value in place.
            continue;
        }
        let snapshot = normalize(&info, source);

        commit(&mut store, entity, &snapshot, now);
    }

    if polled_any && !throttled {
        store.last_poll = Some(now);
    }
}

/// Equality over every [`WindowDisplayState`] field *except* the diagnostic
/// [`headroom_current`](WindowDisplayState::headroom_current). The committed
/// state tracks `headroom_current` on every poll without a `generation` bump, so
/// the write-back compares with this to avoid mirroring (and thus dirtying) a
/// window on bare diagnostic drift.
fn eq_ignoring_diagnostics(a: &WindowDisplayState, b: &WindowDisplayState) -> bool {
    let (mut a, mut b) = (*a, *b);
    a.headroom_current = None;
    b.headroom_current = None;
    a == b
}

/// Mirrors each surface's committed [`WindowDisplayState`] back to its window
/// entity, and its [`MonitorDisplayCapability`] back to the [`Monitor`] entity
/// the window is on (resolved through [`OnMonitor`]). Runs during extraction —
/// the render world's only window into the main world — so the value lags the
/// poll by one frame. Insert-on-change, so
/// [`Changed`](bevy_ecs::prelude::Changed) stays a usable signal.
///
/// [`Monitor`]: bevy_window::Monitor
pub fn write_back_display_state(mut main_world: ResMut<MainWorld>, store: Res<DisplayStateStore>) {
    for (&entity, state) in store.states.iter() {
        let Ok(mut window) = main_world.get_entity_mut(entity) else {
            continue;
        };
        // Write back on any change *except* a bare `headroom_current` drift:
        // that field is diagnostic-only and updates every poll without a
        // `generation` bump, so mirroring it would fire
        // `Changed<WindowDisplayState>` on raw jitter and defeat the generation
        // counter. Comparing everything but `headroom_current` keeps `Changed`
        // tied to genuine, generation-bumping commits.
        let committed = state.committed;
        let changed = match window.get::<WindowDisplayState>() {
            Some(current) => !eq_ignoring_diagnostics(current, &committed),
            None => true,
        };
        if changed {
            window.insert(committed);
        }
    }

    // Resolve each window's Monitor entity through `OnMonitor`, then mirror the
    // capability there (every window on a display shares one record).
    for (&entity, capability) in store.capabilities.iter() {
        let Some(monitor_entity) = main_world
            .get_entity(entity)
            .ok()
            .and_then(|w| w.get::<OnMonitor>())
            .map(|on_monitor| on_monitor.0)
        else {
            continue;
        };
        super::insert_on_change(&mut main_world, monitor_entity, *capability);
    }
}

#[cfg(test)]
mod normalize_tests {
    use super::*;
    use wgpu::{DisplayHeadroom, DisplayLuminance};

    /// Builds a [`DisplayLuminance`] field-by-field. The wgpu types are
    /// `#[non_exhaustive]`, so they cannot be struct-literal-constructed from
    /// outside the crate; start from the default and assign.
    fn luminance(
        max_nits: Option<f32>,
        max_full_frame_nits: Option<f32>,
        min_nits: Option<f32>,
        sdr_white_nits: Option<f32>,
    ) -> DisplayLuminance {
        let mut l = DisplayLuminance::default();
        l.max_nits = max_nits;
        l.max_full_frame_nits = max_full_frame_nits;
        l.min_nits = min_nits;
        l.sdr_white_nits = sdr_white_nits;
        l
    }

    fn headroom(
        current: Option<f32>,
        potential: Option<f32>,
        reference: Option<f32>,
    ) -> DisplayHeadroom {
        let mut h = DisplayHeadroom::default();
        h.current = current;
        h.potential = potential;
        h.reference = reference;
        h
    }

    #[test]
    fn macos_relative_anchors_peak_on_live_sdr_white() {
        // Apple path: no absolute max_nits, headroom potential 5x over live
        // SDR white of 100 nits => reconstructed peak 500 nits.
        let mut info = DisplayHdrInfo::default();
        info.luminance = Some(luminance(None, None, None, Some(100.0)));
        info.headroom = Some(headroom(Some(4.0), Some(5.0), None));
        let snap = normalize(&info, DisplayInfoSource::DerivedFromHeadroom);
        // 100 * 5, NOT a fixed constant.
        assert_eq!(snap.max_nits, Some(500.0));
        assert_eq!(snap.sdr_white_nits, Some(100.0));
        assert_eq!(snap.headroom_potential, Some(5.0));
        assert_eq!(snap.headroom_current, Some(4.0));
        assert_eq!(snap.source, DisplayInfoSource::DerivedFromHeadroom);
    }

    #[test]
    fn windows_absolute_peak_used_directly() {
        let mut info = DisplayHdrInfo::default();
        info.hdr_active = Some(true);
        info.luminance = Some(luminance(
            Some(1000.0),
            Some(600.0),
            Some(0.01),
            Some(200.0),
        ));
        let snap = normalize(&info, DisplayInfoSource::Os);
        // Absolute, not reconstructed.
        assert_eq!(snap.max_nits, Some(1000.0));
        assert_eq!(snap.max_full_frame_nits, Some(600.0));
        assert_eq!(snap.hdr_active, Some(true));
        assert_eq!(snap.min_nits, Some(0.01));
    }

    #[test]
    fn none_stays_none_never_sdr() {
        // All None.
        let info = DisplayHdrInfo::default();
        assert_eq!(classify_source(&info), DisplayInfoSource::Unknown);
        let snap = normalize(&info, DisplayInfoSource::Unknown);
        assert_eq!(snap.hdr_active, None);
        assert_eq!(snap.max_nits, None);
        assert_eq!(snap.sdr_white_nits, None);
    }

    #[test]
    fn non_finite_luminance_filtered_out() {
        let mut info = DisplayHdrInfo::default();
        info.luminance = Some(luminance(Some(f32::NAN), None, None, Some(0.0)));
        info.headroom = Some(headroom(None, Some(f32::INFINITY), None));
        let snap = normalize(&info, DisplayInfoSource::Os);
        // NaN peak and the 0-white reconstruction (with infinite potential
        // also rejected) are both rejected.
        assert_eq!(snap.max_nits, None);
        assert_eq!(snap.sdr_white_nits, None);
        assert_eq!(snap.headroom_potential, None);
    }

    #[test]
    fn classify_prefers_absolute_then_headroom_then_coarse() {
        let mut absolute = DisplayHdrInfo::default();
        absolute.luminance = Some(luminance(Some(1000.0), None, None, None));
        absolute.headroom = Some(headroom(Some(4.0), Some(5.0), None));
        assert_eq!(classify_source(&absolute), DisplayInfoSource::Os);

        let mut relative = DisplayHdrInfo::default();
        relative.headroom = Some(headroom(Some(4.0), Some(5.0), None));
        assert_eq!(
            classify_source(&relative),
            DisplayInfoSource::DerivedFromHeadroom
        );

        let mut coarse = DisplayHdrInfo::default();
        let mut range = wgpu::DisplayCoarseRange::default();
        range.high_dynamic_range = Some(true);
        coarse.coarse = Some(range);
        assert_eq!(classify_source(&coarse), DisplayInfoSource::Web);
    }
}

#[cfg(test)]
mod commit_tests {
    use super::*;

    fn snapshot(source: DisplayInfoSource) -> DisplaySnapshot {
        DisplaySnapshot {
            hdr_active: None,
            headroom_current: None,
            headroom_potential: None,
            headroom_reference: None,
            sdr_white_nits: None,
            max_nits: None,
            max_full_frame_nits: None,
            min_nits: None,
            chromaticity: None,
            gamut_hint: None,
            bits_per_color: None,
            coarse: None,
            source,
        }
    }

    fn committed(store: &DisplayStateStore, entity: Entity) -> WindowDisplayState {
        store.states.get(&entity).unwrap().committed
    }

    #[test]
    fn os_hdr_active_commits_immediately_and_bumps_generation() {
        let mut store = DisplayStateStore::default();
        let entity = Entity::from_raw_u32(1).unwrap();
        let now = Instant::now();

        let mut snap = snapshot(DisplayInfoSource::Os);
        snap.hdr_active = Some(true);
        commit(&mut store, entity, &snap, now);

        let state = committed(&store, entity);
        assert_eq!(state.hdr_active, Some(true));
        assert_eq!(state.generation, 1);
        assert_eq!(state.source, DisplayInfoSource::Os);
    }

    #[test]
    fn headroom_derived_hdr_active_waits_for_dwell() {
        let mut store = DisplayStateStore::default();
        let entity = Entity::from_raw_u32(2).unwrap();
        let start = Instant::now();

        let mut snap = snapshot(DisplayInfoSource::DerivedFromHeadroom);
        snap.hdr_active = Some(true);
        snap.headroom_potential = Some(5.0);

        // First sighting: candidate pending, nothing committed yet.
        commit(&mut store, entity, &snap, start);
        assert_eq!(committed(&store, entity).hdr_active, None);
        assert_eq!(committed(&store, entity).generation, 0);

        // Still within the dwell: still pending.
        commit(&mut store, entity, &snap, start + HYSTERESIS_DWELL / 2);
        assert_eq!(committed(&store, entity).hdr_active, None);

        // Dwell elapsed: commits, generation bumps once.
        commit(&mut store, entity, &snap, start + HYSTERESIS_DWELL);
        let state = committed(&store, entity);
        assert_eq!(state.hdr_active, Some(true));
        assert_eq!(state.headroom_potential, Some(5.0));
        assert_eq!(state.generation, 1);
    }

    #[test]
    fn diagnostic_headroom_current_updates_without_generation_bump() {
        let mut store = DisplayStateStore::default();
        let entity = Entity::from_raw_u32(3).unwrap();
        let now = Instant::now();

        let mut snap = snapshot(DisplayInfoSource::DerivedFromHeadroom);
        snap.headroom_current = Some(2.0);
        commit(&mut store, entity, &snap, now);
        assert_eq!(committed(&store, entity).headroom_current, Some(2.0));
        assert_eq!(committed(&store, entity).generation, 0);

        snap.headroom_current = Some(3.0);
        commit(&mut store, entity, &snap, now);
        assert_eq!(committed(&store, entity).headroom_current, Some(3.0));
        // Diagnostic-only: no generation bump.
        assert_eq!(committed(&store, entity).generation, 0);
    }

    #[test]
    fn write_back_equality_ignores_diagnostic_headroom_current() {
        // Two states that differ only in the diagnostic `headroom_current`
        // compare equal for write-back, so a bare diagnostic drift never fires
        // `Changed<WindowDisplayState>`.
        let base = WindowDisplayState {
            hdr_active: Some(true),
            headroom_current: Some(2.0),
            headroom_potential: Some(5.0),
            sdr_white_nits: Some(100.0),
            source: DisplayInfoSource::DerivedFromHeadroom,
            generation: 3,
            ..Default::default()
        };
        let mut drifted = base;
        drifted.headroom_current = Some(3.5);
        assert!(eq_ignoring_diagnostics(&base, &drifted));

        // A non-diagnostic difference (here the generation bump that accompanies
        // a real commit) compares not-equal, so a genuine transition still
        // writes back.
        let mut committed = base;
        committed.generation = 4;
        assert!(!eq_ignoring_diagnostics(&base, &committed));
    }

    #[test]
    fn sub_threshold_continuous_change_does_not_commit() {
        let mut store = DisplayStateStore::default();
        let entity = Entity::from_raw_u32(4).unwrap();
        let start = Instant::now();

        // Commit an initial potential.
        let mut snap = snapshot(DisplayInfoSource::DerivedFromHeadroom);
        snap.headroom_potential = Some(5.0);
        commit(&mut store, entity, &snap, start);
        commit(&mut store, entity, &snap, start + HYSTERESIS_DWELL);
        assert_eq!(committed(&store, entity).headroom_potential, Some(5.0));
        let generation = committed(&store, entity).generation;

        // A 1% change is below the 5% relative threshold: never commits, even
        // after the dwell would have elapsed.
        snap.headroom_potential = Some(5.05);
        commit(&mut store, entity, &snap, start + HYSTERESIS_DWELL * 4);
        assert_eq!(committed(&store, entity).headroom_potential, Some(5.0));
        assert_eq!(committed(&store, entity).generation, generation);
    }
}
