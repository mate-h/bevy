//! Display-calibration provenance carriers that sit alongside the
//! user-authoritative [`DisplayTarget`](crate::DisplayTarget).
//!
//! [`DisplayTarget`](crate::DisplayTarget) is the *intent*: the calibration the
//! renderer encodes for. These types carry the other three provenances the
//! renderer merges with that intent — what the display can do
//! ([`MonitorDisplayCapability`]), what it is doing right now
//! ([`WindowDisplayState`]), and which fields the engine is allowed to fill in
//! ([`DisplayCalibrationPolicy`]) — and the merged result
//! ([`EffectiveDisplayTarget`]) the render pipeline actually consumes.
//!
//! All of these are plain data: they carry no renderer types, and the defaults
//! reproduce Bevy's SDR behavior exactly. A project that touches none of them
//! renders byte-identically to one that does not know they exist:
//! [`DisplayCalibrationPolicy`] defaults to all-[`Keep`](AutoField::Keep), under
//! which the renderer never overwrites a single [`DisplayTarget`] field.

use bevy_ecs::prelude::Component;

use crate::DisplayTarget;

#[cfg(feature = "bevy_reflect")]
use {
    bevy_ecs::prelude::ReflectComponent,
    bevy_reflect::{std_traits::ReflectDefault, Reflect},
};

#[cfg(all(feature = "serialize", feature = "bevy_reflect"))]
use bevy_reflect::{ReflectDeserialize, ReflectSerialize};

/// Where a piece of display information came from.
///
/// Sensed values flow from the windowing system or the graphics backend; the
/// renderer keeps the source so calibration UIs can distinguish a value the OS
/// reported from one a user or the engine chose, and so the commit policy for
/// live state can depend on how trustworthy a transition is (a real OS toggle
/// commits immediately; a value merely *derived* from headroom is smoothed).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Default, Debug, PartialEq, Hash, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub enum DisplayInfoSource {
    /// No information is available: the platform or the moment cannot tell us.
    /// This is the default and reads as "not sensed", never as "SDR".
    #[default]
    Unknown,
    /// The operating system / windowing system reported the value directly
    /// (for example a Windows HDR enablement flag or absolute luminance from
    /// display metadata).
    Os,
    /// Derived from a relative-headroom signal (the Apple EDR path), where an
    /// absolute luminance is reconstructed by anchoring on the display's live
    /// SDR white. Transitions from this source are smoothed rather than
    /// committed instantly.
    DerivedFromHeadroom,
    /// Reported by the web platform's coarse `dynamic-range` / `color-gamut`
    /// capability query (a capability, not a live mode).
    Web,
}

/// The CIE 1931 *xy* chromaticities of a display's primaries and white point.
///
/// Plain-data mirror of the backend's per-display chromaticity report. Each
/// coordinate is `[x, y]`; absence means that primary was not reported.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Default, Debug, PartialEq, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct DisplayChromaticity {
    /// CIE 1931 `[x, y]` of the red primary.
    pub red: Option<[f32; 2]>,
    /// CIE 1931 `[x, y]` of the green primary.
    pub green: Option<[f32; 2]>,
    /// CIE 1931 `[x, y]` of the blue primary.
    pub blue: Option<[f32; 2]>,
    /// CIE 1931 `[x, y]` of the white point.
    pub white: Option<[f32; 2]>,
}

/// A display's *coarse* dynamic-range and gamut capability, as reported by the
/// web platform's CSS `dynamic-range` / `color-gamut` media features.
///
/// This describes what the display *can* do, not what it is doing now: a
/// monitor may advertise `high_dynamic_range == Some(true)` while HDR output
/// is currently off.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Default, Debug, PartialEq, Hash, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct DisplayCoarseRange {
    /// Whether the display reports high-dynamic-range capability. `None` when
    /// the platform does not report it.
    pub high_dynamic_range: Option<bool>,
    /// The coarse gamut the display reports covering, when reported.
    pub gamut: Option<crate::DisplayGamut>,
}

/// Static-per-display capability of the [`Monitor`](crate::Monitor) a window is
/// presented on: how bright it can get, its primaries, its bit depth.
///
/// Its **absence** means "can't tell" — the renderer never treats a missing
/// component as "SDR". When present, every field is still `Option`: a platform
/// reports whatever subset it can. All luminance fields are absolute nits
/// (candela per square meter), achromatic CIE *Y*.
///
/// # Placement
///
/// Attached to the [`Monitor`](crate::Monitor) entity (resolved from a window
/// through its [`OnMonitor`](crate::OnMonitor) relationship), so every window
/// on the same physical display shares one capability record.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Component, Default, Debug, PartialEq, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct MonitorDisplayCapability {
    /// The maximum luminance, in nits, the display can show on a small window
    /// of the panel (its peak).
    pub max_nits: Option<f32>,
    /// The maximum luminance, in nits, the display can sustain across the full
    /// panel (its full-frame ceiling), typically lower than [`max_nits`].
    ///
    /// [`max_nits`]: Self::max_nits
    pub max_full_frame_nits: Option<f32>,
    /// The minimum luminance, in nits, the display can show (its black level).
    pub min_nits: Option<f32>,
    /// The display's measured primaries and white point, when reported.
    pub chromaticity: Option<DisplayChromaticity>,
    /// A coarse gamut the display covers, when a fine chromaticity report is
    /// unavailable (the web path).
    pub gamut_hint: Option<crate::DisplayGamut>,
    /// The display's bit depth per color channel, when reported.
    pub bits_per_color: Option<u8>,
    /// The web platform's coarse dynamic-range / gamut report, when that is the
    /// only capability source.
    pub coarse: Option<DisplayCoarseRange>,
    /// Where this capability record came from.
    pub source: DisplayInfoSource,
}

impl Default for MonitorDisplayCapability {
    /// An all-`None` capability with an [`Unknown`](DisplayInfoSource::Unknown)
    /// source: "nothing sensed yet", never "SDR".
    fn default() -> Self {
        Self {
            max_nits: None,
            max_full_frame_nits: None,
            min_nits: None,
            chromaticity: None,
            gamut_hint: None,
            bits_per_color: None,
            coarse: None,
            source: DisplayInfoSource::Unknown,
        }
    }
}

/// The live, drifting display state of a window's surface: whether HDR output
/// is active right now and how much headroom is currently available.
///
/// Unlike [`MonitorDisplayCapability`] (static per display) this changes while
/// the window is open — the user flips the OS HDR toggle, drags the window to
/// another monitor, or (on the relative-headroom path) the system EDR headroom
/// shifts with ambient conditions. The renderer mirrors it back to the main
/// world insert-on-change, so [`Changed<WindowDisplayState>`](bevy_ecs::prelude::Changed)
/// is a real signal: [`generation`](Self::generation) is bumped only when a
/// value actually commits past the renderer's hysteresis, never on raw poll
/// jitter.
///
/// Treat it as read-only diagnostics: writing it has no effect on the surface.
/// It is absent until the window's first successful poll.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Component, Default, Debug, PartialEq, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct WindowDisplayState {
    /// Whether HDR output is active on this surface right now, when known.
    pub hdr_active: Option<bool>,
    /// The headroom available *at this instant* (a multiplier over live SDR
    /// white). **Diagnostic only** — it tracks raw, unsmoothed poll values and
    /// must not drive peak-luminance math; use
    /// [`headroom_potential`](Self::headroom_potential) for that.
    pub headroom_current: Option<f32>,
    /// The headroom the display can reach (a multiplier over live SDR white),
    /// the value peak-aware tone mapping targets.
    pub headroom_potential: Option<f32>,
    /// The headroom of the reference white the headroom multipliers are
    /// expressed against, when reported.
    pub headroom_reference: Option<f32>,
    /// The luminance, in nits, of SDR reference white on this surface right
    /// now. On the relative-headroom path this is the anchor that turns a
    /// headroom multiplier into an absolute peak.
    pub sdr_white_nits: Option<f32>,
    /// Where this live state came from.
    pub source: DisplayInfoSource,
    /// Bumped each time a field commits past the renderer's hysteresis, so
    /// [`Changed<WindowDisplayState>`](bevy_ecs::prelude::Changed) signals a
    /// real transition rather than poll jitter.
    pub generation: u32,
}

impl Default for WindowDisplayState {
    /// An all-`None` state with an [`Unknown`](DisplayInfoSource::Unknown)
    /// source and generation `0`: "nothing sensed yet", never "SDR".
    fn default() -> Self {
        Self {
            hdr_active: None,
            headroom_current: None,
            headroom_potential: None,
            headroom_reference: None,
            sdr_white_nits: None,
            source: DisplayInfoSource::Unknown,
            generation: 0,
        }
    }
}

/// Whether the engine may auto-resolve a [`DisplayTarget`] field from sensed
/// display information, or must keep the user's authored value.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Default, Debug, PartialEq, Hash, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub enum AutoField {
    /// Keep the user's authored [`DisplayTarget`] value verbatim; the engine
    /// never overwrites it. This is the default, under which calibration
    /// resolution is an identity pass and output is byte-identical to a project
    /// that never sensed the display.
    #[default]
    Keep,
    /// Let the engine fill this field from sensed display information when a
    /// value is available, falling back through the precedence ladder to the
    /// authored value when nothing is sensed.
    Auto,
}

/// Per-field policy companion to [`DisplayTarget`]: which calibration fields the
/// engine may auto-resolve from sensed display information.
///
/// [`DisplayTarget`] stays user-authoritative — the engine never writes it.
/// This component instead tells the *resolver* which fields of the derived
/// [`EffectiveDisplayTarget`] may diverge from the authored target when the OS
/// or display reports something. The default is all-[`Keep`](AutoField::Keep):
/// the effective target equals the authored target field-for-field, so a
/// project that adds this component (or never does) renders identically.
///
/// [`DisplayTarget::transfer`] is **deliberately absent**: the transfer
/// function is never auto-resolved, because changing it would force a swapchain
/// renegotiation. Sensing fills luminance and gamut only; an OS-reported gamut
/// mismatch may warn but never rewrites the transfer.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Component, Default, Debug, PartialEq, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct DisplayCalibrationPolicy {
    /// Whether to auto-resolve [`DisplayTarget::paper_white_nits`].
    pub paper_white: AutoField,
    /// Whether to auto-resolve [`DisplayTarget::peak_luminance_nits`].
    pub peak_luminance: AutoField,
    /// Whether to auto-resolve [`DisplayTarget::min_luminance_nits`].
    pub min_luminance: AutoField,
    /// Whether to auto-resolve [`DisplayTarget::gamut`].
    pub gamut: AutoField,
}

/// For each [`DisplayTarget`] field, which provenance won the precedence ladder
/// in [`EffectiveDisplayTarget`].
///
/// Calibration UIs read this to show *why* a value is what it is — a user
/// override, an HDR-game-interface live override, an engine policy choice, an
/// OS-sensed value, or the SDR default.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Default, Debug, PartialEq, Hash, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub enum FieldProvenance {
    /// The authored [`DisplayTarget`] value (the field's policy is
    /// [`Keep`](AutoField::Keep), or no higher source applied). This is the
    /// default and the byte-identity case.
    #[default]
    User,
    /// An HDR-game-interface live override took precedence.
    Hgig,
    /// An engine policy value took precedence.
    Policy,
    /// A value sensed from the operating system / display took precedence.
    Os,
    /// The SDR sRGB fallback, used when an [`Auto`](AutoField::Auto) field had
    /// nothing to resolve from.
    Default,
}

/// Which provenance won for each field of an [`EffectiveDisplayTarget`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Default, Debug, PartialEq, Hash, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct DisplayProvenance {
    /// Provenance of [`DisplayTarget::paper_white_nits`].
    pub paper_white: FieldProvenance,
    /// Provenance of [`DisplayTarget::peak_luminance_nits`].
    pub peak_luminance: FieldProvenance,
    /// Provenance of [`DisplayTarget::min_luminance_nits`].
    pub min_luminance: FieldProvenance,
    /// Provenance of [`DisplayTarget::gamut`].
    pub gamut: FieldProvenance,
    /// Provenance of [`DisplayTarget::transfer`]. Always
    /// [`User`](FieldProvenance::User): the transfer is never auto-resolved.
    pub transfer: FieldProvenance,
}

/// The derived display target the render pipeline consumes: the
/// [`DisplayTarget`] after the resolver merges the user's intent with engine
/// policy and sensed display information, plus the per-field
/// [`DisplayProvenance`] of how each value was chosen.
///
/// Computed in the main world (before extraction) so that the identity case —
/// all-[`Keep`](AutoField::Keep) policy, or a user-set target with no sensing —
/// has zero frame lag: a user-set-HDR project shows HDR on its first frame with
/// no SDR pop. The render pipeline reads [`target`](Self::target) in place of
/// the raw [`DisplayTarget`]; when this component is absent (pre-resolve or
/// removed) the pipeline falls back to [`DisplayTarget::SDR_SRGB`], exactly as
/// it falls back for a missing [`DisplayTarget`].
#[derive(Component, Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Component, Default, Debug, PartialEq, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct EffectiveDisplayTarget {
    /// The resolved calibration the renderer encodes for.
    pub target: DisplayTarget,
    /// Per-field provenance of [`target`](Self::target).
    pub provenance: DisplayProvenance,
}

impl Default for EffectiveDisplayTarget {
    /// The SDR sRGB target with all-[`User`](FieldProvenance::User) provenance,
    /// identical to the [`DisplayTarget`] default.
    fn default() -> Self {
        Self {
            target: DisplayTarget::SDR_SRGB,
            provenance: DisplayProvenance::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_default_is_all_keep() {
        let p = DisplayCalibrationPolicy::default();
        assert_eq!(p.paper_white, AutoField::Keep);
        assert_eq!(p.peak_luminance, AutoField::Keep);
        assert_eq!(p.min_luminance, AutoField::Keep);
        assert_eq!(p.gamut, AutoField::Keep);
    }

    #[test]
    fn effective_default_is_sdr_with_user_provenance() {
        let e = EffectiveDisplayTarget::default();
        assert_eq!(e.target, DisplayTarget::SDR_SRGB);
        assert_eq!(e.provenance, DisplayProvenance::default());
        assert_eq!(e.provenance.peak_luminance, FieldProvenance::User);
    }

    #[test]
    fn live_and_capability_defaults_are_unknown_not_sdr() {
        assert_eq!(
            WindowDisplayState::default().source,
            DisplayInfoSource::Unknown
        );
        assert_eq!(WindowDisplayState::default().hdr_active, None);
        assert_eq!(WindowDisplayState::default().generation, 0);
        assert_eq!(
            MonitorDisplayCapability::default().source,
            DisplayInfoSource::Unknown
        );
        assert_eq!(MonitorDisplayCapability::default().max_nits, None);
    }
}
