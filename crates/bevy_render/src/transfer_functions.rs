//! CPU mirrors of the display transfer functions in `transfer_functions.wgsl`
//! (importable in WGSL as `bevy_render::transfer_functions`).
//!
//! These convert between *display-linear* light and the encoded signal a
//! display expects. The shader-side versions are the ones the display-encoding
//! pass executes; the functions here are the operation-for-operation `f32`
//! reference used by tests (and by future CPU-side consumers such as readback
//! parity checks). Keep both files in sync.
//!
//! All math uses [`bevy_math::ops`] so results are deterministic across
//! platforms and match the parity policy used by the GT7 CPU reference
//! (`bevy_core_pipeline::tonemapping::Gt7ToneMapping`).

use bevy_math::ops;

/// Luminance, in nits (cd/m²), of scRGB signal value 1.0 (the IEC 61966-2-2
/// D65 reference white).
pub const SCRGB_REFERENCE_WHITE_NITS: f32 = 80.0;

/// Maximum luminance the PQ (SMPTE ST 2084) signal can carry, in nits.
/// PQ luminance is always normalized against this value, not the display peak.
pub const PQ_MAX_LUMINANCE_NITS: f32 = 10000.0;

/// ST-2084 constant `m1` = (2610 / 4096) / 4.
pub const PQ_M1: f32 = 0.159_301_76;
/// ST-2084 constant `m2` = (2523 / 4096) × 128.
pub const PQ_M2: f32 = 78.84375;
/// ST-2084 constant `c1` = 3424 / 4096.
pub const PQ_C1: f32 = 0.8359375;
/// ST-2084 constant `c2` = (2413 / 4096) × 32 = 18.8515625 (exact in `f32`).
pub const PQ_C2: f32 = 18.851_563;
/// ST-2084 constant `c3` = (2392 / 4096) × 32.
pub const PQ_C3: f32 = 18.6875;

/// HLG (ITU-R BT.2100-2 Table 5) constant `a`.
pub const HLG_A: f32 = 0.17883277;
/// HLG constant `b` = 1 − 4a.
pub const HLG_B: f32 = 0.28466892;
/// HLG constant `c` = 0.5 − a·ln(4a) = 0.55991073 per BT.2100-2 Table 5.
pub const HLG_C: f32 = 0.559_910_7;

/// The sRGB (IEC 61966-2-1) OETF (inverse EOTF): display-linear `[0, 1]` →
/// encoded signal.
///
/// `V = 12.92·L` for `L ≤ 0.0031308`, `V = 1.055·L^(1/2.4) − 0.055` otherwise.
/// Negative inputs take the linear segment (extended-sRGB style), mirroring
/// the WGSL version's `pow`-safety behavior.
pub fn srgb_oetf(linear: f32) -> f32 {
    if linear <= 0.003_130_8 {
        12.92 * linear
    } else {
        1.055 * ops::powf(linear, 1.0 / 2.4) - 0.055
    }
}

/// The sRGB EOTF: encoded signal → display-linear. Exact inverse of
/// [`srgb_oetf`]. Negative inputs take the linear segment.
pub fn srgb_eotf(signal: f32) -> f32 {
    if signal <= 0.04045 {
        signal / 12.92
    } else {
        ops::powf((signal + 0.055) / 1.055, 2.4)
    }
}

/// Encodes paper-white-relative display-linear color (1.0 = paper white at
/// the tone-map operator output) as scRGB-linear signal (1.0 = 80 nits):
/// `V = L × paper_white_nits / 80`.
///
/// scRGB is unbounded and permits negative components; no clamping is
/// applied.
pub fn scrgb_encode(color: f32, paper_white_nits: f32) -> f32 {
    color * (paper_white_nits / SCRGB_REFERENCE_WHITE_NITS)
}

/// The PQ (SMPTE ST 2084) inverse EOTF: normalized display-linear luminance
/// (`Y = nits / 10000`) → PQ signal in `[0, 1]`.
///
/// Negative inputs are clamped to zero **before** the `pow` — `powf` with a
/// negative base and the non-integer exponent `m1` is NaN, and (unlike the
/// GT7 operator's internal copy, whose callers guarantee non-negative input)
/// a general-purpose encoder entry point must tolerate slightly-negative
/// values. Inputs above 1.0 are not clamped; the signal exceeds 1.0 and is
/// clamped by the target format on store.
pub fn pq_inverse_eotf(y: f32) -> f32 {
    let y = y.max(0.0);
    let ym = ops::powf(y, PQ_M1);
    // Numerically-stabler form of ((c1 + c2·ym) / (1 + c3·ym))^m2, identical
    // to the GT7 operator's self-contained copy in gt7.wgsl / gt7.rs.
    ops::exp2(PQ_M2 * (ops::log2(PQ_C1 + PQ_C2 * ym) - ops::log2(1.0 + PQ_C3 * ym)))
}

/// [`pq_inverse_eotf`] taking absolute luminance in nits.
/// `pq_inverse_eotf_from_nits(1000.0)` ≈ 0.7518.
pub fn pq_inverse_eotf_from_nits(nits: f32) -> f32 {
    pq_inverse_eotf(nits / PQ_MAX_LUMINANCE_NITS)
}

/// The PQ EOTF: PQ signal (clamped to `[0, 1]`) → normalized display-linear
/// luminance (1.0 = 10000 nits). Inverse of [`pq_inverse_eotf`].
pub fn pq_eotf(signal: f32) -> f32 {
    let n = signal.clamp(0.0, 1.0);
    let np = ops::powf(n, 1.0 / PQ_M2);
    let l = (np - PQ_C1).max(0.0) / (PQ_C2 - PQ_C3 * np);
    ops::powf(l, 1.0 / PQ_M1)
}

/// The HLG OETF (ITU-R BT.2100-2 Table 5): scene-linear `E ∈ [0, 1]` →
/// signal in `[0, 1]`. Negative inputs are clamped to zero.
///
/// HLG is **scene-referred** (the display applies the OOTF) and no wgpu
/// surface can negotiate HLG today: the function is implemented and tested
/// for completeness but is unreachable from the display-encoding pass, which
/// coerces [`DisplayTransfer::Hlg`](bevy_window::DisplayTransfer::Hlg)
/// targets to PQ.
pub fn hlg_oetf(e: f32) -> f32 {
    let e = e.max(0.0);
    if e <= 1.0 / 12.0 {
        ops::sqrt(3.0 * e)
    } else {
        HLG_A * ops::ln(12.0 * e - HLG_B) + HLG_C
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `f64` reference for the PQ inverse EOTF, used to compute exact
    /// expectations independently of the `f32` implementation.
    fn pq_inverse_eotf_f64(y: f64) -> f64 {
        let m1 = 2610.0 / 4096.0 / 4.0;
        let m2 = 2523.0 / 4096.0 * 128.0;
        let c1 = 3424.0 / 4096.0;
        let c2 = 2413.0 / 4096.0 * 32.0;
        let c3 = 2392.0 / 4096.0 * 32.0;
        let ym = y.max(0.0).powf(m1);
        ((c1 + c2 * ym) / (1.0 + c3 * ym)).powf(m2)
    }

    #[test]
    fn pq_constants_are_exact() {
        assert_eq!(PQ_M1, (2610.0f64 / 4096.0 / 4.0) as f32);
        assert_eq!(PQ_M2, (2523.0f64 / 4096.0 * 128.0) as f32);
        assert_eq!(PQ_C1, (3424.0f64 / 4096.0) as f32);
        assert_eq!(PQ_C2, (2413.0f64 / 4096.0 * 32.0) as f32);
        assert_eq!(PQ_C3, (2392.0f64 / 4096.0 * 32.0) as f32);
    }

    #[test]
    fn pq_inverse_eotf_matches_reference_values() {
        // 1000 nits: the canonical check value from the encoder spec (~0.7518).
        let expected_1000 = pq_inverse_eotf_f64(0.1); // 0.75182700871...
        assert!((expected_1000 - 0.751_827).abs() < 1e-6);
        assert!((pq_inverse_eotf_from_nits(1000.0) as f64 - expected_1000).abs() < 1e-5);

        // Endpoint and mid-range sweep against the f64 reference. The f32
        // implementation accumulates ~1 ULP per pow/exp2/log2 step, so allow
        // a slightly wider (still sub-quantization-step) tolerance here.
        for nits in [0.0, 0.1, 1.0, 80.0, 100.0, 203.0, 2000.0, 10000.0] {
            let expected = pq_inverse_eotf_f64(nits as f64 / 10000.0);
            let actual = pq_inverse_eotf_from_nits(nits) as f64;
            assert!(
                (actual - expected).abs() < 5e-5,
                "PQ({nits} nits): {actual} vs {expected}"
            );
        }
        // 10000 nits encodes to exactly 1.0 (within f32 rounding).
        assert!((pq_inverse_eotf(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pq_negative_input_is_clamped_not_nan() {
        // pow(negative, m1) would be NaN; the clamp must make negative inputs
        // behave exactly like zero. (The GT7 cpp's unclamped form NaNs here.)
        let at_zero = pq_inverse_eotf(0.0);
        assert!(at_zero.is_finite());
        for y in [-1e-6, -0.5, -10.0] {
            let v = pq_inverse_eotf(y);
            assert!(v.is_finite(), "PQ({y}) must be finite");
            assert_eq!(v, at_zero, "PQ({y}) must equal PQ(0)");
        }
        // PQ(0) is c1^m2, a tiny positive value, not 0.
        assert!(at_zero > 0.0 && at_zero < 1e-5);
    }

    #[test]
    fn pq_round_trips() {
        for y in [0.0, 1e-4, 0.01, 0.1, 0.5, 1.0] {
            let signal = pq_inverse_eotf(y);
            let back = pq_eotf(signal);
            assert!(
                (back - y).abs() < 2e-4,
                "PQ round trip at {y}: got {back} (signal {signal})"
            );
        }
    }

    #[test]
    fn srgb_round_trips_and_is_continuous() {
        for l in [0.0, 0.001, 0.0031308, 0.004, 0.1, 0.18, 0.5, 0.9, 1.0] {
            let signal = srgb_oetf(l);
            let back = srgb_eotf(signal);
            assert!(
                (back - l).abs() < 1e-6,
                "sRGB round trip at {l}: got {back}"
            );
        }
        // Continuity at the piecewise breakpoint.
        let below = srgb_oetf(0.0031308);
        let above = srgb_oetf(0.0031309);
        assert!((below - above).abs() < 1e-5);
        // Reference values: OETF(1) = 1, OETF(0) = 0.
        assert_eq!(srgb_oetf(0.0), 0.0);
        assert!((srgb_oetf(1.0) - 1.0).abs() < 1e-6);
        // 18% gray encodes to ~0.4613 (well-known sRGB anchor).
        assert!((srgb_oetf(0.18) - 0.461_356).abs() < 1e-4);
        // Negatives use the linear extension (no NaN).
        assert_eq!(srgb_oetf(-0.5), 12.92 * -0.5);
        assert_eq!(srgb_eotf(-0.5), -0.5 / 12.92);
    }

    #[test]
    fn scrgb_scale_is_paper_white_over_80() {
        // At an 80-nit paper white the encoding is the identity.
        assert_eq!(scrgb_encode(1.0, 80.0), 1.0);
        // Default SDR paper white (100 nits): 1.0 → 1.25.
        assert_eq!(scrgb_encode(1.0, 100.0), 1.25);
        // ITU-R BT.2408 reference paper white (203 nits).
        assert!((scrgb_encode(1.0, 203.0) - 2.5375).abs() < 1e-6);
        // Linear in the color value; negatives pass through (scRGB permits
        // them).
        assert_eq!(scrgb_encode(0.5, 100.0), 0.625);
        assert_eq!(scrgb_encode(-0.5, 100.0), -0.625);
    }

    #[test]
    fn hlg_oetf_matches_bt2100_anchors() {
        // E = 1/12 is the piecewise breakpoint: sqrt(3/12) = 0.5 exactly.
        assert!((hlg_oetf(1.0 / 12.0) - 0.5).abs() < 1e-6);
        // The constants are chosen so OETF(1) = 1.
        assert!((hlg_oetf(1.0) - 1.0).abs() < 1e-6);
        // Continuity at the breakpoint.
        let below = hlg_oetf(1.0 / 12.0 - 1e-6);
        let above = hlg_oetf(1.0 / 12.0 + 1e-6);
        assert!((below - above).abs() < 1e-4);
        // Negative inputs clamp to zero (no NaN from sqrt).
        assert_eq!(hlg_oetf(-1.0), 0.0);
        assert_eq!(hlg_oetf(0.0), 0.0);
    }
}
