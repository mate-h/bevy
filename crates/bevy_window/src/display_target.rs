//! Types describing the display a window (or other render target) is presented on.
//!
//! The central type is [`DisplayTarget`], a required component of
//! [`Window`](crate::Window) that captures the calibration of the display the
//! window's swapchain feeds: how bright "white" is, how bright the display can
//! get, which color gamut it covers, and which transfer function the signal
//! should be encoded with.
//!
//! `DisplayTarget` is plain data: it carries no renderer types and changing it
//! never directly mutates GPU state. The renderer reads it during extraction to
//! parameterize tone mapping, gamut mapping, transfer-function encoding, and
//! (eventually) surface format selection. The default value,
//! [`DisplayTarget::SDR_SRGB`], reproduces Bevy's current SDR behavior exactly.

use bevy_ecs::prelude::Component;

#[cfg(feature = "bevy_reflect")]
use {
    bevy_ecs::prelude::ReflectComponent,
    bevy_reflect::{std_traits::ReflectDefault, Reflect},
};

#[cfg(all(feature = "serialize", feature = "bevy_reflect"))]
use bevy_reflect::{ReflectDeserialize, ReflectSerialize};

/// Describes the display device that a [`Window`](crate::Window) (or other
/// render target) is presented on, so the renderer can produce a correctly
/// tone-mapped, gamut-mapped, and transfer-encoded signal for it.
///
/// This component is **user-authoritative**: Bevy never overwrites values you
/// set, even when the window moves to a different monitor. When that happens a
/// [`WindowMonitorChanged`](crate::WindowMonitorChanged) event is emitted so
/// you (or a future auto-resolution system) can decide whether to update this
/// component.
///
/// # Placement
///
/// `DisplayTarget` is a required component of [`Window`](crate::Window):
/// every window entity automatically receives one, defaulting to
/// [`DisplayTarget::SDR_SRGB`], which is byte-identical to Bevy's behavior
/// before this component existed. Multiple cameras rendering to the same
/// window share the window's single `DisplayTarget`.
///
/// Render targets that are not windows (`RenderTarget::Image`,
/// `RenderTarget::TextureView`) have no window entity to host this component;
/// they are looked up in the `ManualDisplayTargets` resource in `bevy_render`
/// instead, and fall back to [`DisplayTarget::SDR_SRGB`] when absent.
///
/// # Units
///
/// All luminance fields are in nits (candela per square meter, cd/m²).
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
pub struct DisplayTarget {
    /// The luminance, in nits, that "paper white" (also called reference or
    /// diffuse white) is displayed at: the brightness of a full-white UI
    /// element or a 100%-diffuse-reflective surface, as opposed to emissive
    /// highlights, which may go brighter (up to [`peak_luminance_nits`]).
    ///
    /// The tone-mapping operator's output is renormalized at the encoder seam
    /// so that a value of `1.0` corresponds to this luminance. On SDR
    /// displays this is the luminance of signal-level white, nominally
    /// `100.0` nits. Typical HDR values range from 100–300 nits depending on
    /// viewing environment; ITU-R BT.2408 recommends 203 nits for HDR
    /// broadcast.
    ///
    /// Note that the scRGB-linear transfer function defines its own reference
    /// white of 80 nits at signal value `1.0`; the encoder accounts for this
    /// by scaling by `paper_white_nits / 80` when encoding for
    /// [`DisplayTransfer::ScRgbLinear`]. No such factor is baked into this
    /// field.
    ///
    /// [`peak_luminance_nits`]: Self::peak_luminance_nits
    pub paper_white_nits: f32,
    /// The maximum luminance, in nits, that the display can show.
    ///
    /// Peak-aware tone-mapping operators compress scene highlights into the
    /// range `[0, peak_luminance_nits]` rather than clipping them. On SDR
    /// displays peak and paper white coincide (nominally `100.0` nits); on
    /// HDR displays the peak is higher (commonly 400–4000 nits), leaving
    /// headroom above paper white for emissive highlights.
    ///
    /// For displays that cannot sustain their peak over the full panel, this
    /// should be the value obtained from OS metadata or HGIG-style
    /// calibration (`MaxTML`), not the marketing peak.
    pub peak_luminance_nits: f32,
    /// The minimum luminance, in nits, that the display can show (its black
    /// level).
    ///
    /// `0.0` is a reasonable default; self-emissive displays (OLED) reach
    /// true zero while backlit panels typically bottom out between 0.01 and
    /// 0.1 nits. Tone-mapping operators may use this to lift shadow detail
    /// above the display's black floor.
    pub min_luminance_nits: f32,
    /// The color gamut (set of primaries) of the display target.
    ///
    /// This controls the gamut transform stage: rendered colors in the
    /// working color space are converted to these primaries (with perceptual
    /// gamut compression for out-of-gamut colors) before transfer encoding.
    pub gamut: DisplayGamut,
    /// The transfer function the signal should be encoded with.
    ///
    /// This is the *requested* transfer. The renderer may be unable to fulfil
    /// it on the current backend/OS (for example PQ is currently not
    /// reachable through wgpu); in that case it degrades and warns rather
    /// than failing. See [`DisplayTransfer`] for per-variant details.
    pub transfer: DisplayTransfer,
}

impl DisplayTarget {
    /// The standard-dynamic-range sRGB display target: paper white and peak
    /// luminance of 100 nits, black level of 0, [`DisplayGamut::Rec709`]
    /// primaries, [`DisplayTransfer::Srgb`] encoding.
    ///
    /// This is the [`Default`] value, and produces output byte-identical to
    /// Bevy's behavior before `DisplayTarget` existed.
    pub const SDR_SRGB: Self = Self {
        paper_white_nits: 100.0,
        peak_luminance_nits: 100.0,
        min_luminance_nits: 0.0,
        gamut: DisplayGamut::Rec709,
        transfer: DisplayTransfer::Srgb,
    };
}

impl Default for DisplayTarget {
    /// Returns [`DisplayTarget::SDR_SRGB`].
    fn default() -> Self {
        Self::SDR_SRGB
    }
}

/// The color gamut (primary chromaticities) of a display target.
///
/// All variants assume a D65 white point.
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
pub enum DisplayGamut {
    /// ITU-R BT.709 primaries, shared by sRGB. The standard-dynamic-range
    /// gamut every display can show, and the default.
    #[default]
    Rec709,
    /// Display P3 primaries (DCI-P3 primaries with a D65 white point), as
    /// used by most Apple displays and many wide-gamut monitors. Wider than
    /// Rec.709, narrower than Rec.2020.
    DisplayP3,
    /// ITU-R BT.2020 primaries, the wide gamut used by HDR10 and most HDR
    /// video standards. Physical displays typically cover only part of this
    /// gamut and apply their own gamut mapping.
    Rec2020,
}

/// The transfer function used to encode the final signal for a display
/// target.
///
/// This is the last stage of the display pipeline: it converts display-linear
/// color (already tone-mapped and gamut-mapped) into the non-linear (or
/// scaled linear) signal values the display expects.
///
/// Note that backend support varies: as of wgpu 29, only [`Srgb`] (all
/// platforms) and [`ScRgbLinear`] (macOS Metal, Windows Vulkan, Wayland
/// Vulkan) are reachable through surface formats. [`Pq`] and [`Hlg`] are
/// defined so APIs and shaders can be written transfer-agnostically, and will
/// light up when the corresponding wgpu surface support lands.
///
/// [`Srgb`]: DisplayTransfer::Srgb
/// [`ScRgbLinear`]: DisplayTransfer::ScRgbLinear
/// [`Pq`]: DisplayTransfer::Pq
/// [`Hlg`]: DisplayTransfer::Hlg
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
pub enum DisplayTransfer {
    /// The sRGB transfer function (IEC 61966-2-1), the standard-dynamic-range
    /// default. Currently applied in hardware via `*UnormSrgb` surface
    /// formats.
    #[default]
    Srgb,
    /// scRGB linear (IEC 61966-2-2): a linear, extended-range encoding where
    /// signal value `1.0` corresponds to 80 nits and values above `1.0` (and
    /// below `0.0`) are valid. Used with `Rgba16Float` surfaces. The encoder
    /// scales by `paper_white_nits / 80` so that scene paper white lands on
    /// the display's configured paper white.
    ScRgbLinear,
    /// The Perceptual Quantizer (SMPTE ST 2084, ITU-R BT.2100), the absolute
    /// transfer function used by HDR10. Encodes absolute luminance normalized
    /// to 10000 nits. Canonically paired with [`DisplayGamut::Rec2020`].
    ///
    /// Not currently reachable through wgpu surfaces on any backend.
    Pq,
    /// Hybrid Log-Gamma (ITU-R BT.2100), the scene-referred HDR transfer
    /// function used in broadcast.
    ///
    /// Not currently reachable through wgpu surfaces on any backend.
    Hlg,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_sdr_srgb() {
        assert_eq!(DisplayTarget::default(), DisplayTarget::SDR_SRGB);
    }

    #[test]
    fn sdr_srgb_constant_values() {
        let sdr = DisplayTarget::SDR_SRGB;
        assert_eq!(sdr.paper_white_nits, 100.0);
        assert_eq!(sdr.peak_luminance_nits, 100.0);
        assert_eq!(sdr.min_luminance_nits, 0.0);
        assert_eq!(sdr.gamut, DisplayGamut::Rec709);
        assert_eq!(sdr.transfer, DisplayTransfer::Srgb);
    }

    #[test]
    fn enum_defaults_match_sdr() {
        assert_eq!(DisplayGamut::default(), DisplayGamut::Rec709);
        assert_eq!(DisplayTransfer::default(), DisplayTransfer::Srgb);
    }
}
