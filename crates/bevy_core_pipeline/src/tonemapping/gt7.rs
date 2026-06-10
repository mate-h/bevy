//! CPU-side reference implementation of the Gran Turismo 7 tone-mapping operator.
//!
//! This is a native port of Polyphony Digital's reference implementation
//! (`gt7_tone_mapping.cpp`, MIT License, Copyright (c) 2025 Polyphony Digital Inc.),
//! published as part of the SIGGRAPH 2025 course "Physically Based Tone Mapping in
//! Gran Turismo 7".
//!
//! # Unit convention (native)
//!
//! The operator works on linear Rec.2020 RGB "frame buffer values" where `1.0`
//! corresponds to [`REFERENCE_LUMINANCE`] (100 cd/m²) of physical luminance:
//!
//! - In SDR mode the operator tone-maps against Gran Turismo's 250-nit SDR paper
//!   white ([`GRAN_TURISMO_SDR_PAPER_WHITE`]) and rescales the result by
//!   `1 / 2.5` so the output fits `[0, 1]`, ready for the sRGB OETF.
//! - In HDR mode the output range is `[0, peak_nits / 100]`, ready for the PQ
//!   inverse EOTF. HDR peak luminance is valid in the range 250–10000 nits.
//!
//! The GPU implementation lives in `gt7.wgsl` and must be kept in sync with this
//! module; the math here is the parity reference for `cargo test` fixtures
//! (tolerance `1e-4` per channel against the C++ reference output) and for future
//! GPU readback tests (tolerance `1/1024` for fp16 targets).
//!
//! All math is deliberately `f32` to mirror both the C++ reference and the WGSL
//! port operation-for-operation.

use bevy_camera::Camera;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::With,
    reflect::ReflectComponent,
    resource::Resource,
    system::{Commands, Query, Res, ResMut},
};
use bevy_log::warn_once;
use bevy_math::ops;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    extract_component::ExtractComponent,
    render_resource::{DynamicUniformBuffer, ShaderType},
    renderer::{RenderDevice, RenderQueue},
    view::{ExtractedView, ViewDisplayTarget},
};
use bevy_window::DisplayTarget;

use super::Tonemapping;

/// Physical luminance in cd/m² that a linear frame-buffer value of `1.0`
/// corresponds to in Gran Turismo's native unit convention.
pub const REFERENCE_LUMINANCE: f32 = 100.0;

/// The SDR reference (paper) white level used by Gran Turismo's tone mapping,
/// in cd/m². This is Polyphony's artistic calibration, not sRGB's 80/100 nits.
pub const GRAN_TURISMO_SDR_PAPER_WHITE: f32 = 250.0;

/// The lowest HDR peak luminance, in nits, the GT7 operator supports.
///
/// The reference implementation documents 250 nits as the valid lower bound
/// (the curve parameters assume a 250-nit SDR paper white) but does not
/// enforce it; Bevy clamps to it at prepare time and warns.
pub const GT7_MIN_HDR_PEAK_NITS: f32 = 250.0;

/// The highest HDR peak luminance, in nits, the GT7 operator supports
/// (the PQ ceiling). Clamped to at prepare time, with a warning.
pub const GT7_MAX_HDR_PEAK_NITS: f32 = 10000.0;

/// Full-precision (f64-derived) linear Rec.709 → Rec.2020 conversion matrix
/// (row-major, D65 white point, derived per ITU-R BT.2087).
///
/// Used by the in-tree SDR integration seam: Bevy's working space is currently
/// scene-linear Rec.709, while the GT7 operator natively works in linear
/// Rec.2020.
///
/// Each literal is the shortest round-trip representation of the correctly
/// rounded `f32` of the corresponding f64 literal in `gt7.wgsl`
/// (`GT7_REC_709_TO_REC_2020`, note: column-major there); the two must stay
/// bit-identical so this module remains an exact parity reference for the
/// shader.
// TODO: deduplicate with shared color-space matrix constants once they land in
// `bevy_color` / `bevy_render::color_operations` (HDR workstream T2.x).
pub const REC_709_TO_REC_2020: [[f32; 3]; 3] = [
    [0.627_403_9, 0.329_283_03, 0.043_313_067],
    [0.069_097_29, 0.919_540_4, 0.011_362_315],
    [0.016_391_44, 0.088_013_306, 0.895_595_25],
];

/// Full-precision (f64-derived) linear Rec.2020 → Rec.709 conversion matrix
/// (row-major, D65 white point). Inverse of [`REC_709_TO_REC_2020`].
///
/// Bit-identical to `GT7_REC_2020_TO_REC_709` in `gt7.wgsl` (see
/// [`REC_709_TO_REC_2020`] for the sync contract).
pub const REC_2020_TO_REC_709: [[f32; 3]; 3] = [
    [1.660_491, -0.587_641_1, -0.072_849_86],
    [-0.124_550_48, 1.132_899_9, -0.008_349_422],
    [-0.018_150_763, -0.100_578_9, 1.118_729_7],
];

/// Per-camera parameters for the [`Tonemapping::GranTurismo7`] operator.
///
/// Defaults match Polyphony Digital's reference implementation. All parameters
/// are dimensionless except where noted; the curve parameters are expressed in
/// GT7's native frame-buffer units where `1.0` = 100 nits (see the module docs
/// for the unit contract).
///
/// Add this component to a camera that uses [`Tonemapping::GranTurismo7`] to
/// customize the operator. When this component is present (and the camera's
/// tonemapping is `GranTurismo7`), [`prepare_gt7_params_uniforms`] validates
/// the values with [`Self::sanitized`] each frame and uploads a
/// [`Gt7ParamsUniform`] that replaces the shader's baked defaults. Cameras
/// **without** this component keep using the baked SDR defaults, exactly as
/// before.
///
/// The component also selects the operator's mode: on a view whose resolved
/// [`DisplayTarget`] requests an HDR transfer, the uniform is computed in HDR
/// mode (peak taken from
/// [`DisplayTarget::peak_luminance_nits`]); otherwise in
/// SDR mode. See [`Gt7ParamsUniform::new`] for the exact rules.
///
/// [`Tonemapping::GranTurismo7`]: crate::tonemapping::Tonemapping::GranTurismo7
#[derive(Component, Debug, Clone, Copy, PartialEq, Reflect, ExtractComponent)]
#[extract_component_filter(With<Camera>)]
#[reflect(Component, Debug, Default, PartialEq, Clone)]
pub struct GranTurismo7Params {
    /// Mix between the per-channel tone-mapped color and the hue-preserving
    /// UCS (`ICtCp`) processed color. `0.0` = fully per-channel ("camera-like"
    /// skew), `1.0` = fully UCS (hue-stable). Polyphony markets this as the
    /// main artistic dial. Clamped to `[0, 1]`. Default: `0.6`.
    pub blend_ratio: f32,
    /// Start of the highlight chroma fade band, as a fraction of the peak
    /// luminance in UCS (`ICtCp` `I`) units. Original-luminance values above this
    /// begin losing chroma. Clamped to `[0, 1]`. Default: `0.98`.
    pub fade_start: f32,
    /// End of the highlight chroma fade band, as a fraction of the peak
    /// luminance in UCS units. Values above this are fully desaturated.
    /// Intentionally allowed to exceed `1.0` so over-peak colors keep some
    /// chroma; must be greater than [`Self::fade_start`]. Default: `1.16`.
    pub fade_end: f32,
    /// Curvature control for the shoulder region. Must be less than `1.0`.
    /// Default: `0.25`.
    pub alpha: f32,
    /// Gray point in frame-buffer units: the end of the toe→linear blend
    /// region. The curve is exactly linear from here to the shoulder. Must be
    /// greater than `0.0`. Default: `0.538`.
    pub mid_point: f32,
    /// Fraction of the peak intensity at which the linear section ends and the
    /// convergent shoulder begins. Must be less than `1.0`. Default: `0.444`.
    pub linear_section: f32,
    /// Exponent of the toe's power curve. Must be non-negative.
    /// Default: `1.28`.
    pub toe_strength: f32,
}

impl Default for GranTurismo7Params {
    fn default() -> Self {
        Self {
            blend_ratio: 0.6,
            fade_start: 0.98,
            fade_end: 1.16,
            alpha: 0.25,
            mid_point: 0.538,
            linear_section: 0.444,
            toe_strength: 1.28,
        }
    }
}

impl GranTurismo7Params {
    /// The smallest allowed width of the chroma fade band
    /// (`fade_end - fade_start`); prevents a division by zero in the
    /// `smoothstep` underlying the chroma fade.
    const MIN_FADE_BAND: f32 = 1e-4;
    /// Margin keeping `alpha` and `linear_section` strictly below `1.0`,
    /// preventing divisions by zero in the shoulder constant derivation.
    const UNIT_MARGIN: f32 = 1e-3;

    /// Returns a copy with all parameters validated and clamped to safe ranges,
    /// emitting [`warn_once!`] if anything had to be adjusted.
    ///
    /// This implements the prepare-time validation table for the GT7 operator:
    ///
    /// - Any non-finite (NaN/∞) field is reset to its default.
    /// - `blend_ratio` is clamped to `[0, 1]`.
    /// - `fade_start` is clamped to `[0, 1]`.
    /// - `fade_end` is clamped to at least `fade_start + 1e-4`. The upper bound
    ///   is intentionally NOT clamped: `fade_end > 1` lets over-peak colors
    ///   keep some chroma.
    /// - `alpha` and `linear_section` are clamped to `[0, 1 - 1e-3]` (values of
    ///   exactly `1.0` produce divisions by zero in the closed-form shoulder
    ///   constants).
    /// - `mid_point` is clamped to at least `1e-3` (a zero mid point produces
    ///   divisions by zero in the toe).
    /// - `toe_strength` is clamped to be non-negative.
    ///
    /// Called at prepare time by [`prepare_gt7_params_uniforms`] before the
    /// parameters are uploaded to the GPU.
    pub fn sanitized(&self) -> Self {
        let defaults = Self::default();
        let mut sanitized = *self;
        let mut adjusted = false;

        // Reset non-finite fields to their defaults first so the range clamps
        // below operate on real numbers.
        let fields = [
            (&mut sanitized.blend_ratio, defaults.blend_ratio),
            (&mut sanitized.fade_start, defaults.fade_start),
            (&mut sanitized.fade_end, defaults.fade_end),
            (&mut sanitized.alpha, defaults.alpha),
            (&mut sanitized.mid_point, defaults.mid_point),
            (&mut sanitized.linear_section, defaults.linear_section),
            (&mut sanitized.toe_strength, defaults.toe_strength),
        ];
        for (field, default) in fields {
            if !field.is_finite() {
                *field = default;
                adjusted = true;
            }
        }

        let mut clamp = |value: &mut f32, min: f32, max: f32| {
            let clamped = value.clamp(min, max);
            if clamped != *value {
                *value = clamped;
                adjusted = true;
            }
        };

        clamp(&mut sanitized.blend_ratio, 0.0, 1.0);
        clamp(&mut sanitized.fade_start, 0.0, 1.0);
        clamp(&mut sanitized.alpha, 0.0, 1.0 - Self::UNIT_MARGIN);
        clamp(&mut sanitized.linear_section, 0.0, 1.0 - Self::UNIT_MARGIN);
        clamp(&mut sanitized.mid_point, Self::UNIT_MARGIN, f32::MAX);
        clamp(&mut sanitized.toe_strength, 0.0, f32::MAX);
        // No upper clamp on `fade_end`: values past 1.0 are intentional.
        clamp(
            &mut sanitized.fade_end,
            sanitized.fade_start + Self::MIN_FADE_BAND,
            f32::MAX,
        );

        if adjusted {
            warn_once!(
                "GranTurismo7Params contained out-of-range or non-finite values \
                 and was sanitized; see the GranTurismo7Params docs for valid ranges"
            );
        }

        sanitized
    }
}

/// `smoothstep` with the C++ reference's exact semantics: strict comparisons,
/// and the interpolant computed before the range checks (so `edge0 == edge1`
/// yields NaN/∞ rather than a clamp — parameter validation prevents that).
fn smooth_step(x: f32, edge0: f32, edge1: f32) -> f32 {
    let t = (x - edge0) / (edge1 - edge0);
    if x < edge0 {
        return 0.0;
    }
    if x > edge1 {
        return 1.0;
    }
    t * t * (3.0 - 2.0 * t)
}

/// Luminance-driven chroma fade: `1.0` below `a`, falling to `0.0` at `b`.
fn chroma_curve(x: f32, a: f32, b: f32) -> f32 {
    1.0 - smooth_step(x, a, b)
}

// ST-2084 (PQ) constants, SMPTE ST 2084:2014 / ITU-R BT.2100.
const PQ_M1: f32 = 0.159_301_76; // (2610 / 4096) / 4
const PQ_M2: f32 = 78.84375; // (2523 / 4096) * 128
const PQ_C1: f32 = 0.835_937_5; // 3424 / 4096
const PQ_C2: f32 = 18.851_563; // (2413 / 4096) * 32
const PQ_C3: f32 = 18.6875; // (2392 / 4096) * 32
/// Maximum luminance supported by PQ (cd/m²).
const PQ_C: f32 = 10000.0;

/// ST-2084 (PQ) EOTF: normalized PQ signal (clamped to `[0, 1]`) → linear
/// frame-buffer value (`1.0` = 100 nits).
fn eotf_st2084(n: f32) -> f32 {
    let n = n.clamp(0.0, 1.0);
    let np = ops::powf(n, 1.0 / PQ_M2);
    let mut l = np - PQ_C1;
    if l < 0.0 {
        l = 0.0;
    }
    l /= PQ_C2 - PQ_C3 * np;
    l = ops::powf(l, 1.0 / PQ_M1);
    // Convert absolute luminance (cd/m²) into the frame-buffer linear scale.
    l * PQ_C / REFERENCE_LUMINANCE
}

/// ST-2084 (PQ) inverse EOTF: linear frame-buffer value (`1.0` = 100 nits) →
/// normalized PQ signal.
///
/// Deliberately does NOT clamp its input (mirroring the reference): values
/// above 10000 nits encode above `1.0`. Negative inputs would produce NaN;
/// callers clamp at zero before calling (see [`rgb_to_ictcp`]).
fn inverse_eotf_st2084(v: f32) -> f32 {
    let physical = v * REFERENCE_LUMINANCE;
    let y = physical / PQ_C;
    let ym = ops::powf(y, PQ_M1);
    // Numerically-stabler form of ((c1 + c2*ym) / (1 + c3*ym))^m2.
    ops::exp2(PQ_M2 * (ops::log2(PQ_C1 + PQ_C2 * ym) - ops::log2(1.0 + PQ_C3 * ym)))
}

/// Linear Rec.2020 RGB → `ICtCp` (ITU-R BT.2100 / ITU-T T.302).
///
/// Deviation from the C++ reference: the LMS intermediates are clamped at zero
/// before the PQ encode. The reference would produce NaN for inputs saturated
/// enough to drive LMS negative; the clamp is the recommended port policy and
/// matches the WGSL implementation. All parity fixtures keep LMS positive, so
/// fixture outputs are unaffected.
fn rgb_to_ictcp(rgb: [f32; 3]) -> [f32; 3] {
    let l = (rgb[0] * 1688.0 + rgb[1] * 2146.0 + rgb[2] * 262.0) / 4096.0;
    let m = (rgb[0] * 683.0 + rgb[1] * 2951.0 + rgb[2] * 462.0) / 4096.0;
    let s = (rgb[0] * 99.0 + rgb[1] * 309.0 + rgb[2] * 3688.0) / 4096.0;

    let l_pq = inverse_eotf_st2084(l.max(0.0));
    let m_pq = inverse_eotf_st2084(m.max(0.0));
    let s_pq = inverse_eotf_st2084(s.max(0.0));

    [
        (2048.0 * l_pq + 2048.0 * m_pq) / 4096.0,
        (6610.0 * l_pq - 13613.0 * m_pq + 7003.0 * s_pq) / 4096.0,
        (17933.0 * l_pq - 17390.0 * m_pq - 543.0 * s_pq) / 4096.0,
    ]
}

/// `ICtCp` → linear Rec.2020 RGB (ITU-R BT.2100 / ITU-T T.302).
///
/// Output channels are clamped at zero, mirroring the reference. The PQ decode
/// in [`eotf_st2084`] clamps its input to `[0, 1]`, so per-LMS-channel values
/// saturate at 10000 nits.
fn ictcp_to_rgb(ictcp: [f32; 3]) -> [f32; 3] {
    let l = ictcp[0] + 0.00860904 * ictcp[1] + 0.11103 * ictcp[2];
    let m = ictcp[0] - 0.00860904 * ictcp[1] - 0.11103 * ictcp[2];
    let s = ictcp[0] + 0.560031 * ictcp[1] - 0.320627 * ictcp[2];

    let l_lin = eotf_st2084(l);
    let m_lin = eotf_st2084(m);
    let s_lin = eotf_st2084(s);

    [
        (3.43661 * l_lin - 2.50645 * m_lin + 0.0698454 * s_lin).max(0.0),
        (-0.79133 * l_lin + 1.9836 * m_lin - 0.192271 * s_lin).max(0.0),
        (-0.0259499 * l_lin - 0.0989137 * m_lin + 1.12486 * s_lin).max(0.0),
    ]
}

/// The "GT Tone Mapping" curve (V2) with a convergent shoulder, evaluated
/// per channel: a power-curve toe blended into an exactly-linear middle
/// section, followed by a convergent exponential shoulder.
///
/// The shoulder asymptote (`k_a ≈ 1.185 × peak` at default parameters) lies
/// above the peak; it is never visible in operator output because
/// [`Gt7ToneMapping::apply`] clamps at the peak.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gt7ToneMappingCurve {
    /// Display peak intensity in frame-buffer units (`peak_nits / 100`).
    pub peak_intensity: f32,
    /// Shoulder curvature control (must be `< 1`).
    pub alpha: f32,
    /// Gray point: end of the toe→linear blend (frame-buffer units).
    pub mid_point: f32,
    /// Fraction of `peak_intensity` where the shoulder begins.
    pub linear_section: f32,
    /// Toe power-curve exponent.
    pub toe_strength: f32,
    /// Precomputed shoulder asymptote.
    pub k_a: f32,
    /// Precomputed shoulder scale.
    pub k_b: f32,
    /// Precomputed shoulder exponent factor (negative).
    pub k_c: f32,
}

impl Gt7ToneMappingCurve {
    /// Initializes the curve, precomputing the closed-form shoulder constants.
    ///
    /// `monitor_intensity` is the display peak in frame-buffer units
    /// (`peak_nits / 100`); `gray_point` is the curve's mid point. The GT7
    /// defaults are `alpha = 0.25`, `gray_point = 0.538`,
    /// `linear_section = 0.444`, `toe_strength = 1.28`.
    pub fn new(
        monitor_intensity: f32,
        alpha: f32,
        gray_point: f32,
        linear_section: f32,
        toe_strength: f32,
    ) -> Self {
        let k = (linear_section - 1.0) / (alpha - 1.0);
        Self {
            peak_intensity: monitor_intensity,
            alpha,
            mid_point: gray_point,
            linear_section,
            toe_strength,
            k_a: monitor_intensity * linear_section + monitor_intensity * k,
            k_b: -monitor_intensity * k * ops::exp(linear_section / k),
            k_c: -1.0 / (k * monitor_intensity),
        }
    }

    /// Evaluates the curve at `x` (frame-buffer units). Negative inputs map
    /// to zero. With default parameters the regions are: toe→linear blend on
    /// `(0, mid_point)`, exactly linear on `[mid_point, linear_section × peak)`,
    /// shoulder on `[linear_section × peak, ∞)`.
    pub fn evaluate(&self, x: f32) -> f32 {
        if x < 0.0 {
            return 0.0;
        }

        let weight_linear = smooth_step(x, 0.0, self.mid_point);
        let weight_toe = 1.0 - weight_linear;

        // Shoulder mapping for highlights. For extreme inputs `exp(x * k_c)`
        // underflows cleanly to zero (`k_c < 0`), converging on `k_a`.
        let shoulder = self.k_a + self.k_b * ops::exp(x * self.k_c);

        if x < self.linear_section * self.peak_intensity {
            let toe_mapped = self.mid_point * ops::powf(x / self.mid_point, self.toe_strength);
            weight_toe * toe_mapped + weight_linear * x
        } else {
            shoulder
        }
    }
}

/// CPU implementation of the full GT7 tone-mapping pipeline (curve +
/// hue-preserving `ICtCp` branch). See the module docs for the unit contract.
///
/// This mirrors the `GT7ToneMapping` struct in the C++ reference: construct it
/// once per parameter set ([`Self::new_sdr`] / [`Self::new_hdr`]), then call
/// [`Self::apply`] per color.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gt7ToneMapping {
    /// `1 / 2.5` in SDR mode (rescales the 250-nit-referred result into
    /// `[0, 1]`); `1.0` in HDR mode.
    pub sdr_correction_factor: f32,
    /// Display target peak in frame-buffer units (`peak_nits / 100`).
    pub framebuffer_luminance_target: f32,
    /// `ICtCp` `I` of peak white, precomputed at initialization; normalizes the
    /// luminance driving the chroma fade.
    pub framebuffer_luminance_target_ucs: f32,
    /// The per-channel tone-mapping curve.
    pub curve: Gt7ToneMappingCurve,
    /// UCS share of the final per-channel/UCS blend (constant, not
    /// chroma-dependent).
    pub blend_ratio: f32,
    /// Chroma fade band start, as a fraction of peak luminance in UCS units.
    pub fade_start: f32,
    /// Chroma fade band end (may exceed `1.0`).
    pub fade_end: f32,
}

impl Gt7ToneMapping {
    fn new(physical_target_luminance: f32, params: &GranTurismo7Params, sdr: bool) -> Self {
        let framebuffer_luminance_target = physical_target_luminance / REFERENCE_LUMINANCE;
        let curve = Gt7ToneMappingCurve::new(
            framebuffer_luminance_target,
            params.alpha,
            params.mid_point,
            params.linear_section,
            params.toe_strength,
        );
        let peak_ucs = rgb_to_ictcp([
            framebuffer_luminance_target,
            framebuffer_luminance_target,
            framebuffer_luminance_target,
        ])[0];
        Self {
            sdr_correction_factor: if sdr {
                1.0 / framebuffer_luminance_target
            } else {
                1.0
            },
            framebuffer_luminance_target,
            framebuffer_luminance_target_ucs: peak_ucs,
            curve,
            blend_ratio: params.blend_ratio,
            fade_start: params.fade_start,
            fade_end: params.fade_end,
        }
    }

    /// Initializes for SDR output with the default parameters: tone-maps
    /// against Gran Turismo's 250-nit paper white, then rescales the result by
    /// `1 / 2.5` so the output fits `[0, 1]`, ready for the sRGB OETF.
    pub fn new_sdr() -> Self {
        Self::new_sdr_with_params(&GranTurismo7Params::default())
    }

    /// Initializes for SDR output with custom [`GranTurismo7Params`]. The
    /// parameters are used as-is; call [`GranTurismo7Params::sanitized`] first
    /// for untrusted values.
    pub fn new_sdr_with_params(params: &GranTurismo7Params) -> Self {
        Self::new(GRAN_TURISMO_SDR_PAPER_WHITE, params, true)
    }

    /// Initializes for HDR output with the default parameters, targeting the
    /// given display peak luminance in nits (valid range 250–10000; the lower
    /// limit exists because the curve parameters assume a 250-nit SDR paper
    /// white). Output range is `[0, physical_target_luminance / 100]`.
    pub fn new_hdr(physical_target_luminance: f32) -> Self {
        Self::new_hdr_with_params(physical_target_luminance, &GranTurismo7Params::default())
    }

    /// Initializes for HDR output with custom [`GranTurismo7Params`]. The
    /// parameters are used as-is; call [`GranTurismo7Params::sanitized`] first
    /// for untrusted values.
    pub fn new_hdr_with_params(
        physical_target_luminance: f32,
        params: &GranTurismo7Params,
    ) -> Self {
        Self::new(physical_target_luminance, params, false)
    }

    /// Applies the tone-mapping pipeline to a linear Rec.2020 frame-buffer
    /// color (native GT7 units, `1.0` = 100 nits).
    ///
    /// Steps: per-channel curve ("skewed" color); chroma fade driven by the
    /// ORIGINAL color's UCS luminance; recombination of skewed luminance with
    /// faded original chroma; constant 60% UCS / 40% per-channel blend; clamp
    /// at peak; SDR correction factor.
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        // Convert to UCS to separate luminance and chroma.
        let ucs = rgb_to_ictcp(rgb);

        // Per-channel tone mapping ("skewed" color).
        let skewed_rgb = [
            self.curve.evaluate(rgb[0]),
            self.curve.evaluate(rgb[1]),
            self.curve.evaluate(rgb[2]),
        ];

        let skewed_ucs = rgb_to_ictcp(skewed_rgb);

        let chroma_scale = chroma_curve(
            ucs[0] / self.framebuffer_luminance_target_ucs,
            self.fade_start,
            self.fade_end,
        );

        let scaled_ucs = [
            skewed_ucs[0],         // Luminance from the skewed color.
            ucs[1] * chroma_scale, // Chroma from the original color, faded.
            ucs[2] * chroma_scale,
        ];

        let scaled_rgb = ictcp_to_rgb(scaled_ucs);

        // Final blend between per-channel and UCS-scaled results.
        let mut out = [0.0; 3];
        for i in 0..3 {
            let blended =
                (1.0 - self.blend_ratio) * skewed_rgb[i] + self.blend_ratio * scaled_rgb[i];
            out[i] = self.sdr_correction_factor * blended.min(self.framebuffer_luminance_target);
        }
        out
    }

    /// Applies the operator to a Bevy scene-linear Rec.709 color, mirroring the
    /// WGSL `tone_mapping_gran_turismo_7` integration seam used for SDR output
    /// today. This is the CPU parity reference for GPU readback tests.
    ///
    /// Seam contract (SDR, current in-tree integration):
    /// 1. Convert scene-linear Rec.709 → Rec.2020.
    /// 2. Multiply by `2.5`: Bevy's `1.0` (SDR paper white) maps to GT7's
    ///    250-nit paper white (`2.5` frame-buffer units).
    /// 3. Run the operator (SDR mode brings the result back to `[0, 1]` via
    ///    the `1 / 2.5` correction factor).
    /// 4. Convert the display-referred result Rec.2020 → Rec.709 for the
    ///    existing sRGB output chain, clamping to `[0, 1]` (out-of-gamut
    ///    Rec.2020 results have no SDR Rec.709 representation).
    ///
    /// The HDR path (operator output staying in Rec.2020 for the gamut /
    /// transfer-encoding passes) lights up with the encoder workstream.
    pub fn apply_bevy_scene_linear_sdr(&self, rgb: [f32; 3]) -> [f32; 3] {
        let rec2020 = mat3_mul_vec3(&REC_709_TO_REC_2020, rgb);
        let fb = [
            rec2020[0] * (GRAN_TURISMO_SDR_PAPER_WHITE / REFERENCE_LUMINANCE),
            rec2020[1] * (GRAN_TURISMO_SDR_PAPER_WHITE / REFERENCE_LUMINANCE),
            rec2020[2] * (GRAN_TURISMO_SDR_PAPER_WHITE / REFERENCE_LUMINANCE),
        ];
        let mapped = self.apply(fb);
        let rec709 = mat3_mul_vec3(&REC_2020_TO_REC_709, mapped);
        [
            rec709[0].clamp(0.0, 1.0),
            rec709[1].clamp(0.0, 1.0),
            rec709[2].clamp(0.0, 1.0),
        ]
    }
}

fn mat3_mul_vec3(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// GPU uniform feeding the GT7 operator's `Gt7Params` WGSL struct (see
/// `gt7.wgsl`; field order and meaning must stay identical).
///
/// All derived curve constants (`k_a`/`k_b`/`k_c`, `peak_ucs`) are computed
/// CPU-side (closed forms in [`Gt7ToneMappingCurve::new`] /
/// [`Gt7ToneMapping::new_sdr_with_params`] and friends) so the shader stays
/// cheap. Built per view by [`prepare_gt7_params_uniforms`] via
/// [`Gt7ParamsUniform::new`]; bound to the tonemapping pass only when the
/// `GT7_PARAMS_UNIFORM` shader def is pushed. Without that def the shader
/// keeps its baked SDR defaults (`gt7_default_sdr_params()` in `gt7.wgsl`).
#[derive(Clone, Copy, Debug, PartialEq, ShaderType)]
pub struct Gt7ParamsUniform {
    /// Display peak in frame-buffer units (`peak_nits / 100`).
    pub peak: f32,
    /// Precomputed shoulder asymptote ([`Gt7ToneMappingCurve::k_a`]).
    pub k_a: f32,
    /// Precomputed shoulder scale ([`Gt7ToneMappingCurve::k_b`]).
    pub k_b: f32,
    /// Precomputed shoulder exponent factor ([`Gt7ToneMappingCurve::k_c`],
    /// negative).
    pub k_c: f32,
    /// Gray point in frame-buffer units (end of the toe→linear blend).
    pub mid_point: f32,
    /// Fraction of `peak` where the linear section ends and the shoulder
    /// begins.
    pub linear_section: f32,
    /// Exponent of the toe's power curve.
    pub toe_strength: f32,
    /// `ICtCp` `I` of peak white, precomputed at prepare time; normalizes the
    /// luminance driving the chroma fade.
    pub peak_ucs: f32,
    /// UCS share of the final per-channel/UCS blend.
    pub blend_ratio: f32,
    /// Chroma fade band start, as a fraction of `peak_ucs`.
    pub fade_start: f32,
    /// Chroma fade band end (may exceed `1.0`).
    pub fade_end: f32,
    /// Post-clamp output scale. `1 / 2.5` in SDR mode (Polyphony's native
    /// rescale of the 250-nit-referred result into `[0, 1]`);
    /// `100 / paper_white_nits` in HDR mode (the D5 seam renormalization so
    /// `1.0` = paper white at the operator output — identity at the default
    /// 100-nit paper white).
    pub sdr_correction_factor: f32,
}

impl From<&Gt7ToneMapping> for Gt7ParamsUniform {
    fn from(tone_mapping: &Gt7ToneMapping) -> Self {
        Self {
            peak: tone_mapping.framebuffer_luminance_target,
            k_a: tone_mapping.curve.k_a,
            k_b: tone_mapping.curve.k_b,
            k_c: tone_mapping.curve.k_c,
            mid_point: tone_mapping.curve.mid_point,
            linear_section: tone_mapping.curve.linear_section,
            toe_strength: tone_mapping.curve.toe_strength,
            peak_ucs: tone_mapping.framebuffer_luminance_target_ucs,
            blend_ratio: tone_mapping.blend_ratio,
            fade_start: tone_mapping.fade_start,
            fade_end: tone_mapping.fade_end,
            sdr_correction_factor: tone_mapping.sdr_correction_factor,
        }
    }
}

impl Gt7ParamsUniform {
    /// Builds the uniform for a view from its (unsanitized) user parameters
    /// and resolved [`DisplayTarget`], implementing the prepare-time
    /// validation policy:
    ///
    /// - `params` is passed through [`GranTurismo7Params::sanitized`]
    ///   (non-finite fields reset, ranges clamped, one warning).
    /// - If the display target requests an HDR transfer (scRGB-linear, PQ, or
    ///   HLG), the operator is configured in **HDR mode**:
    ///   - non-finite or non-positive `paper_white_nits` is reset to 100,
    ///     non-finite `peak_luminance_nits` to 100 (each with a warning);
    ///   - `paper_white_nits` is clamped to at most
    ///     [`GT7_MAX_HDR_PEAK_NITS`];
    ///   - the peak is clamped to
    ///     [`[GT7_MIN_HDR_PEAK_NITS, GT7_MAX_HDR_PEAK_NITS]`](GT7_MIN_HDR_PEAK_NITS),
    ///     with a warning;
    ///   - a peak below `paper_white_nits` is raised to `paper_white_nits`,
    ///     with a warning;
    ///   - [`Gt7ParamsUniform::sdr_correction_factor`] is set to
    ///     `100 / paper_white_nits`: the seam renormalization that scales the
    ///     operator's native output (`1.0` = 100 nits) so `1.0` = paper white.
    ///     Until the display encoder lands, this output still flows through
    ///     the existing Rec.2020 → Rec.709 + clamp SDR chain, so HDR-mode
    ///     highlights above paper white are clipped on screen for now.
    /// - Otherwise the operator is configured in **SDR mode**, identical to
    ///   the baked defaults except for the user parameters: peak 2.5
    ///   frame-buffer units (Gran Turismo's 250-nit paper white), output
    ///   rescaled into `[0, 1]`.
    pub fn new(display_target: &DisplayTarget, params: &GranTurismo7Params) -> Self {
        let params = params.sanitized();
        // Single-source the HDR predicate with the rest of the display
        // pipeline (`DisplayTransfer::is_hdr`, which also backs
        // `ViewDisplayTarget::is_hdr_transfer`). Callers pass the *resolved*
        // display target, so a downgraded HDR request configures plain SDR
        // mode here too.
        if !display_target.transfer.is_hdr() {
            return Self::from(&Gt7ToneMapping::new_sdr_with_params(&params));
        }

        let mut paper_white = display_target.paper_white_nits;
        if !paper_white.is_finite() || paper_white <= 0.0 {
            warn_once!(
                "DisplayTarget::paper_white_nits is non-finite or non-positive; \
                 GranTurismo7 is using 100 nits instead"
            );
            paper_white = DisplayTarget::SDR_SRGB.paper_white_nits;
        }
        if paper_white > GT7_MAX_HDR_PEAK_NITS {
            warn_once!(
                "DisplayTarget::paper_white_nits exceeds the PQ ceiling of 10000 nits; \
                 GranTurismo7 is clamping it to 10000 nits"
            );
            paper_white = GT7_MAX_HDR_PEAK_NITS;
        }

        let mut peak = display_target.peak_luminance_nits;
        if !peak.is_finite() {
            warn_once!(
                "DisplayTarget::peak_luminance_nits is non-finite; \
                 GranTurismo7 is using 100 nits before range clamping"
            );
            peak = DisplayTarget::SDR_SRGB.peak_luminance_nits;
        }
        let clamped_peak = peak.clamp(GT7_MIN_HDR_PEAK_NITS, GT7_MAX_HDR_PEAK_NITS);
        if clamped_peak != peak {
            warn_once!(
                "DisplayTarget::peak_luminance_nits is outside GranTurismo7's supported \
                 HDR range [250, 10000] nits and was clamped"
            );
            peak = clamped_peak;
        }
        if peak < paper_white {
            warn_once!(
                "DisplayTarget::peak_luminance_nits is below paper_white_nits; \
                 GranTurismo7 is raising the peak to paper white"
            );
            peak = paper_white;
        }

        let mut tone_mapping = Gt7ToneMapping::new_hdr_with_params(peak, &params);
        // D5 seam renormalization: rescale the operator's native output
        // (1.0 = 100 nits) so that 1.0 = paper white. `apply` multiplies by
        // this factor after the peak clamp, so the CPU struct stays a valid
        // parity reference for the shader.
        tone_mapping.sdr_correction_factor = REFERENCE_LUMINANCE / paper_white;
        Self::from(&tone_mapping)
    }
}

/// Resource holding the [`DynamicUniformBuffer`] of per-view
/// [`Gt7ParamsUniform`]s, written each frame by
/// [`prepare_gt7_params_uniforms`].
#[derive(Resource)]
pub struct Gt7ParamsUniforms {
    /// The per-view uniform buffer; entries are addressed with the dynamic
    /// offset stored in each view's [`ViewGt7ParamsUniformOffset`].
    pub uniforms: DynamicUniformBuffer<Gt7ParamsUniform>,
}

impl Default for Gt7ParamsUniforms {
    fn default() -> Self {
        let mut uniforms = DynamicUniformBuffer::default();
        uniforms.set_label(Some("gt7_params_uniforms_buffer"));
        Self { uniforms }
    }
}

/// Render-world component holding a view's dynamic offset into
/// [`Gt7ParamsUniforms`].
///
/// Inserted by [`prepare_gt7_params_uniforms`] on every view whose
/// [`Tonemapping`] is [`Tonemapping::GranTurismo7`] and that has an extracted
/// [`GranTurismo7Params`] — exactly the views whose tonemapping pipeline is
/// specialized with the `GT7_PARAMS_UNIFORM` shader def.
#[derive(Component)]
pub struct ViewGt7ParamsUniformOffset {
    /// The dynamic offset to pass to `set_bind_group`.
    pub offset: u32,
}

/// Prepares a [`Gt7ParamsUniform`] for every view using
/// [`Tonemapping::GranTurismo7`] with a [`GranTurismo7Params`] component,
/// validating the parameters and selecting SDR/HDR mode from the view's
/// resolved [`ViewDisplayTarget`] (see [`Gt7ParamsUniform::new`]).
///
/// Runs in `RenderSystems::PrepareResources`. Views without the component (or
/// with a different operator) are skipped and keep the shader's baked SDR
/// defaults.
pub fn prepare_gt7_params_uniforms(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut gt7_params_uniforms: ResMut<Gt7ParamsUniforms>,
    views: Query<
        (
            Entity,
            &Tonemapping,
            &GranTurismo7Params,
            Option<&ViewDisplayTarget>,
        ),
        With<ExtractedView>,
    >,
) {
    let view_count = views
        .iter()
        .filter(|(_, tonemapping, ..)| **tonemapping == Tonemapping::GranTurismo7)
        .count();
    let Some(mut writer) =
        gt7_params_uniforms
            .uniforms
            .get_writer(view_count, &render_device, &render_queue)
    else {
        return;
    };
    for (entity, tonemapping, params, view_display_target) in &views {
        if *tonemapping != Tonemapping::GranTurismo7 {
            continue;
        }
        let display_target = view_display_target
            .map(|view_display_target| view_display_target.resolved)
            .unwrap_or_default();
        let offset = writer.write(&Gt7ParamsUniform::new(&display_target, params));
        commands
            .entity(entity)
            .insert(ViewGt7ParamsUniformOffset { offset });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Per-channel absolute tolerance for CPU-port-vs-C++-reference parity.
    const TOLERANCE: f32 = 1e-4;

    #[track_caller]
    fn assert_rgb_eq(actual: [f32; 3], expected: [f32; 3]) {
        for i in 0..3 {
            assert!(
                (actual[i] - expected[i]).abs() <= TOLERANCE,
                "channel {i}: actual {:?} vs expected {:?} (diff {:e})",
                actual,
                expected,
                (actual[i] - expected[i]).abs()
            );
        }
    }

    /// Ground-truth fixtures generated by compiling the unmodified C++
    /// reference (`plans/gt7_tone_mapping.cpp`, g++ -O2 -std=c++17) with a
    /// `printf("%.9e")` harness over the canonical `main()` cases
    /// (SDR + HDR 1000/4000/10000 over three inputs) plus branch-coverage
    /// extras. Inputs are linear Rec.2020 frame-buffer values.
    #[test]
    fn cpp_parity_canonical_12() {
        let sdr = Gt7ToneMapping::new_sdr();
        let hdr1000 = Gt7ToneMapping::new_hdr(1000.0);
        let hdr4000 = Gt7ToneMapping::new_hdr(4000.0);
        let hdr10000 = Gt7ToneMapping::new_hdr(10000.0);

        let inputs: [[f32; 3]; 3] = [[0.5, 1.23, 0.75], [12.3, 34.3, 56.9], [1504.7, 64.51, 0.5]];

        // SDR (peak 250 nits internal, output rescaled into [0, 1]).
        assert_rgb_eq(
            sdr.apply(inputs[0]),
            [1.996_225e-1, 4.907_029_6e-1, 2.995_677_6e-1],
        );
        assert_rgb_eq(sdr.apply(inputs[1]), [1.0, 1.0, 1.0]);
        assert_rgb_eq(sdr.apply(inputs[2]), [1.0, 1.0, 7.387_512e-1]);

        // HDR, 1000-nit peak.
        assert_rgb_eq(
            hdr1000.apply(inputs[0]),
            [4.998_231_8e-1, 1.230_000_7, 7.499_952_3e-1],
        );
        assert_rgb_eq(hdr1000.apply(inputs[1]), [10.0, 10.0, 10.0]);
        assert_rgb_eq(hdr1000.apply(inputs[2]), [10.0, 10.0, 6.706_747_5]);

        // HDR, 4000-nit peak.
        assert_rgb_eq(
            hdr4000.apply(inputs[0]),
            [4.998_231_8e-1, 1.230_000_7, 7.499_952_3e-1],
        );
        assert_rgb_eq(hdr4000.apply(inputs[1]), [11.354_192, 30.071_972, 40.0]);
        assert_rgb_eq(hdr4000.apply(inputs[2]), [40.0, 40.0, 23.847_712]);

        // HDR, 10000-nit peak (peak UCS is exactly 1.0: PQ(10000 nits) = 1).
        assert_rgb_eq(
            hdr10000.apply(inputs[0]),
            [4.998_231_8e-1, 1.230_000_7, 7.499_952_3e-1],
        );
        assert_rgb_eq(
            hdr10000.apply(inputs[1]),
            [12.277_842, 34.240_51, 56.395_88],
        );
        assert_rgb_eq(
            hdr10000.apply(inputs[2]),
            [91.726_71, 68.024_31, 42.575_134],
        );
    }

    #[test]
    fn cpp_parity_branch_coverage_extras() {
        let sdr = Gt7ToneMapping::new_sdr();
        let hdr1000 = Gt7ToneMapping::new_hdr(1000.0);

        // Mid-band chroma fade (chromaScale ≈ 0.55): one clamped + two
        // unclamped channels.
        assert_rgb_eq(
            hdr1000.apply([20.0, 15.0, 5.0]),
            [10.0, 9.912_016, 5.249_325_3],
        );

        // Achromatic in the SDR fade band: UCS path is ~no-op on gray (tiny
        // channel asymmetry comes from the f32 ICtCp matrices, itself a
        // parity probe).
        assert_rgb_eq(
            sdr.apply([3.0, 3.0, 3.0]),
            [9.179_522e-1, 9.179_485_4e-1, 9.179_471e-1],
        );

        // Exact seam values: R == mid_point, G == linear_section * peak
        // (shoulder branch via strict <), B == linear_section.
        assert_rgb_eq(
            sdr.apply([0.538, 1.11, 0.444]),
            [2.151_980_8e-1, 4.439_385_5e-1, 1.772_700_5e-1],
        );

        // Negative channel: curve's x < 0 branch; LMS of this input stays
        // positive so the (deviating) LMS clamp is not engaged and the result
        // matches the unmodified reference.
        assert_rgb_eq(
            sdr.apply([-0.1, 0.2, 0.1]),
            [0.0, 7.863_592e-2, 3.668_311_2e-2],
        );

        // Bevy SDR paper-white anchor: 2.5 fb gray (250 nits).
        assert_rgb_eq(
            sdr.apply([2.5, 2.5, 2.5]),
            [8.351_579e-1, 8.351_547e-1, 8.351_534e-1],
        );

        // Tiny gray: toe pow + PQ near-zero (`l < 0` clamp in the EOTF).
        assert_rgb_eq(
            hdr1000.apply([1e-5, 1e-5, 1e-5]),
            [4.735_695_7e-7, 4.735_677_5e-7, 4.735_67e-7],
        );
    }

    #[test]
    fn cpp_parity_custom_params_identity_region() {
        // blend = 0 (pure per-channel), toe_strength = 1 makes the toe exactly
        // linear, so inputs below the shoulder seam (0.3 * 10 = 3.0 fb) pass
        // through unchanged.
        let params = GranTurismo7Params {
            blend_ratio: 0.0,
            alpha: 0.5,
            mid_point: 0.4,
            linear_section: 0.3,
            toe_strength: 1.0,
            ..Default::default()
        };
        let custom = Gt7ToneMapping::new_hdr_with_params(1000.0, &params);
        assert_rgb_eq(custom.apply([0.5, 1.23, 0.75]), [0.5, 1.23, 0.75]);
    }

    #[test]
    fn cpp_parity_init_products() {
        // Closed-form curve constants and precomputed peak UCS, from the C++
        // reference harness (%.9e).
        let sdr = Gt7ToneMapping::new_sdr();
        assert!((sdr.curve.k_a - 2.963_333_1).abs() < 1e-5);
        assert!((sdr.curve.k_b - -3.373_351).abs() < 1e-5);
        assert!((sdr.curve.k_c - -5.395_683_6e-1).abs() < 1e-6);
        assert!((sdr.framebuffer_luminance_target_ucs - 6.025_607_6e-1).abs() < 1e-5);
        assert!((sdr.sdr_correction_factor - 0.4).abs() < 1e-7);

        let hdr10000 = Gt7ToneMapping::new_hdr(10000.0);
        // PQ(10000 nits) = 1 exactly.
        assert!((hdr10000.framebuffer_luminance_target_ucs - 1.0).abs() < 1e-6);
        assert_eq!(hdr10000.sdr_correction_factor, 1.0);
    }

    #[test]
    fn curve_matches_reference_values() {
        // Curve-only values from the C++ reference harness; covers the spec's
        // numeric check table (toe, exact-linear region, shoulder, asymptote).
        let curve25 = Gt7ToneMappingCurve::new(2.5, 0.25, 0.538, 0.444, 1.28);
        let curve10 = Gt7ToneMappingCurve::new(10.0, 0.25, 0.538, 0.444, 1.28);
        let curve40 = Gt7ToneMappingCurve::new(40.0, 0.25, 0.538, 0.444, 1.28);

        let cases25 = [
            (0.0, 0.0),
            (0.1, 6.583_988e-2),
            (0.25, 2.233_054_3e-1),
            (0.444, 4.421_193e-1),
            (0.538, 0.538),
            (1.0, 1.0),
            (1.11, 1.109_999_9),
            (2.5, 2.087_880_6),
            (4.0, 2.573_628_7),
            (10.0, 2.948_031_2),
            (1504.7, 2.963_333_1), // exp underflow → k_a
        ];
        for (x, expected) in cases25 {
            assert!(
                (curve25.evaluate(x) - expected).abs() <= TOLERANCE,
                "peak 2.5, x = {x}"
            );
        }
        // Exactly-linear region and shoulder for higher peaks.
        assert!((curve10.evaluate(2.5) - 2.5).abs() <= TOLERANCE);
        assert!((curve10.evaluate(4.44) - 4.439_999_6).abs() <= TOLERANCE);
        assert!((curve10.evaluate(10.0) - 8.351_522).abs() <= TOLERANCE);
        assert!((curve40.evaluate(40.0) - 33.406_09).abs() <= 1e-3);

        // x < 0 branch.
        assert_eq!(curve25.evaluate(-1.0), 0.0);
    }

    #[test]
    fn bevy_sdr_seam_gray_anchor() {
        // Bevy scene-linear white (1, 1, 1) maps through the seam to GT7's
        // 250-nit paper white (2.5 fb gray): the matrices are
        // gray-preserving, so the result matches the native 2.5-fb fixture.
        let sdr = Gt7ToneMapping::new_sdr();
        let out = sdr.apply_bevy_scene_linear_sdr([1.0, 1.0, 1.0]);
        assert_rgb_eq(out, [8.351_579e-1, 8.351_547e-1, 8.351_534e-1]);

        // Black stays black; the output stays in [0, 1].
        assert_rgb_eq(sdr.apply_bevy_scene_linear_sdr([0.0, 0.0, 0.0]), [0.0; 3]);
        let bright = sdr.apply_bevy_scene_linear_sdr([100.0, 100.0, 100.0]);
        for c in bright {
            assert!((0.0..=1.0).contains(&c));
        }
    }

    #[test]
    fn matrices_are_gray_preserving_inverses() {
        for v in [
            mat3_mul_vec3(&REC_709_TO_REC_2020, [1.0, 1.0, 1.0]),
            mat3_mul_vec3(&REC_2020_TO_REC_709, [1.0, 1.0, 1.0]),
        ] {
            for c in v {
                assert!((c - 1.0).abs() < 1e-6);
            }
        }
        let round_trip = mat3_mul_vec3(
            &REC_2020_TO_REC_709,
            mat3_mul_vec3(&REC_709_TO_REC_2020, [0.25, 0.5, 0.75]),
        );
        assert_rgb_eq(round_trip, [0.25, 0.5, 0.75]);
    }

    #[test]
    fn params_sanitized_clamps_and_resets() {
        // NaN/Inf reset to defaults.
        let nan_params = GranTurismo7Params {
            blend_ratio: f32::NAN,
            fade_end: f32::INFINITY,
            ..Default::default()
        };
        let sanitized = nan_params.sanitized();
        assert_eq!(sanitized, GranTurismo7Params::default());

        // Range clamps.
        let out_of_range = GranTurismo7Params {
            blend_ratio: 1.5,
            fade_start: -0.25,
            fade_end: -1.0,
            alpha: 1.0,
            mid_point: 0.0,
            linear_section: 1.0,
            toe_strength: -3.0,
        };
        let sanitized = out_of_range.sanitized();
        assert_eq!(sanitized.blend_ratio, 1.0);
        assert_eq!(sanitized.fade_start, 0.0);
        assert!(sanitized.fade_end >= sanitized.fade_start + 1e-4);
        assert!(sanitized.alpha < 1.0);
        assert!(sanitized.mid_point > 0.0);
        assert!(sanitized.linear_section < 1.0);
        assert!(sanitized.toe_strength >= 0.0);

        // fade_end > 1 is intentional: must NOT be clamped down.
        let wide_fade = GranTurismo7Params {
            fade_end: 4.0,
            ..Default::default()
        };
        assert_eq!(wide_fade.sanitized().fade_end, 4.0);

        // Defaults pass through untouched.
        let defaults = GranTurismo7Params::default();
        assert_eq!(defaults.sanitized(), defaults);

        // Sanitized params must produce finite output everywhere the raw
        // params would have produced NaN.
        let tm = Gt7ToneMapping::new_sdr_with_params(&out_of_range.sanitized());
        for c in tm.apply([0.5, 1.23, 0.75]) {
            assert!(c.is_finite());
        }
    }

    use bevy_window::{DisplayGamut, DisplayTransfer};

    fn hdr_target(peak: f32, paper_white: f32, transfer: DisplayTransfer) -> DisplayTarget {
        DisplayTarget {
            paper_white_nits: paper_white,
            peak_luminance_nits: peak,
            min_luminance_nits: 0.0,
            gamut: DisplayGamut::Rec2020,
            transfer,
        }
    }

    /// SDR-target uniform must reproduce the C++ reference's SDR init
    /// products (the same fixtures as `cpp_parity_init_products`), i.e. the
    /// values baked into `gt7_default_sdr_params()` in gt7.wgsl.
    #[test]
    fn uniform_sdr_mode_matches_init_fixtures() {
        let uniform =
            Gt7ParamsUniform::new(&DisplayTarget::SDR_SRGB, &GranTurismo7Params::default());
        assert_eq!(uniform.peak, 2.5);
        assert!((uniform.k_a - 2.963_333_1).abs() < 1e-5);
        assert!((uniform.k_b - -3.373_351).abs() < 1e-5);
        assert!((uniform.k_c - -5.395_683_6e-1).abs() < 1e-6);
        assert!((uniform.peak_ucs - 6.025_607_6e-1).abs() < 1e-5);
        assert!((uniform.sdr_correction_factor - 0.4).abs() < 1e-7);
        assert_eq!(uniform.mid_point, 0.538);
        assert_eq!(uniform.linear_section, 0.444);
        assert_eq!(uniform.toe_strength, 1.28);
        assert_eq!(uniform.blend_ratio, 0.6);
        assert_eq!(uniform.fade_start, 0.98);
        assert_eq!(uniform.fade_end, 1.16);

        // A non-HDR transfer stays SDR mode no matter how bright the target
        // claims to be.
        let bright_sdr = Gt7ParamsUniform::new(
            &DisplayTarget {
                peak_luminance_nits: 4000.0,
                ..DisplayTarget::SDR_SRGB
            },
            &GranTurismo7Params::default(),
        );
        assert_eq!(bright_sdr.peak, 2.5);
        assert!((bright_sdr.sdr_correction_factor - 0.4).abs() < 1e-7);
    }

    /// HDR-target uniform must match the C++ reference's HDR init products
    /// and apply the D5 seam renormalization (`100 / paper_white`).
    #[test]
    fn uniform_hdr_mode_matches_init_fixtures() {
        let params = GranTurismo7Params::default();
        let uniform = Gt7ParamsUniform::new(
            &hdr_target(1000.0, 100.0, DisplayTransfer::ScRgbLinear),
            &params,
        );
        let reference = Gt7ToneMapping::new_hdr(1000.0);
        assert_eq!(uniform.peak, 10.0);
        assert_eq!(uniform.k_a, reference.curve.k_a);
        assert_eq!(uniform.k_b, reference.curve.k_b);
        assert_eq!(uniform.k_c, reference.curve.k_c);
        assert_eq!(uniform.peak_ucs, reference.framebuffer_luminance_target_ucs);
        // Seam renormalization is identity at the default 100-nit paper white
        // (native HDR-mode factor).
        assert_eq!(uniform.sdr_correction_factor, 1.0);

        // Non-default paper white scales the output so 1.0 = paper white.
        let uniform =
            Gt7ParamsUniform::new(&hdr_target(1000.0, 200.0, DisplayTransfer::Pq), &params);
        assert_eq!(uniform.peak, 10.0);
        assert!((uniform.sdr_correction_factor - 0.5).abs() < 1e-7);

        // The 10000-nit peak hits PQ's exact ceiling (peak UCS == 1).
        let uniform = Gt7ParamsUniform::new(
            &hdr_target(10000.0, 100.0, DisplayTransfer::ScRgbLinear),
            &params,
        );
        assert_eq!(uniform.peak, 100.0);
        assert!((uniform.peak_ucs - 1.0).abs() < 1e-6);
    }

    /// The D6 clamp table for HDR-mode peak/paper-white selection.
    #[test]
    fn uniform_hdr_mode_clamp_table() {
        let params = GranTurismo7Params::default();

        // Peak below the documented 250-nit lower bound: clamped up.
        let uniform = Gt7ParamsUniform::new(
            &hdr_target(100.0, 100.0, DisplayTransfer::ScRgbLinear),
            &params,
        );
        assert_eq!(uniform.peak, 2.5);
        assert_eq!(uniform.sdr_correction_factor, 1.0);

        // Peak above the 10000-nit PQ ceiling: clamped down.
        let uniform =
            Gt7ParamsUniform::new(&hdr_target(20000.0, 100.0, DisplayTransfer::Pq), &params);
        assert_eq!(uniform.peak, 100.0);

        // Peak below paper white: raised to paper white.
        let uniform = Gt7ParamsUniform::new(
            &hdr_target(400.0, 600.0, DisplayTransfer::ScRgbLinear),
            &params,
        );
        assert_eq!(uniform.peak, 6.0);
        assert!((uniform.sdr_correction_factor - 100.0 / 600.0).abs() < 1e-7);

        // Non-finite peak: reset to 100 nits, then range-clamped to 250.
        let uniform = Gt7ParamsUniform::new(
            &hdr_target(f32::NAN, 100.0, DisplayTransfer::ScRgbLinear),
            &params,
        );
        assert_eq!(uniform.peak, 2.5);

        // Non-finite / non-positive paper white: reset to 100 nits.
        for paper_white in [f32::NAN, f32::INFINITY, 0.0, -50.0] {
            let uniform = Gt7ParamsUniform::new(
                &hdr_target(1000.0, paper_white, DisplayTransfer::ScRgbLinear),
                &params,
            );
            assert_eq!(uniform.peak, 10.0);
            assert_eq!(uniform.sdr_correction_factor, 1.0);
        }

        // Absurd paper white above the PQ ceiling: clamped to 10000, and the
        // peak follows it up.
        let uniform = Gt7ParamsUniform::new(
            &hdr_target(1000.0, 20000.0, DisplayTransfer::ScRgbLinear),
            &params,
        );
        assert_eq!(uniform.peak, 100.0);
        assert!((uniform.sdr_correction_factor - 0.01).abs() < 1e-9);

        // HLG counts as an HDR transfer (defined but unreachable through
        // wgpu surfaces today).
        let uniform =
            Gt7ParamsUniform::new(&hdr_target(1000.0, 100.0, DisplayTransfer::Hlg), &params);
        assert_eq!(uniform.peak, 10.0);
    }

    /// User params flow through `sanitized()` before reaching the uniform.
    #[test]
    fn uniform_sanitizes_user_params() {
        let out_of_range = GranTurismo7Params {
            blend_ratio: 7.5,
            ..Default::default()
        };
        let uniform = Gt7ParamsUniform::new(&DisplayTarget::SDR_SRGB, &out_of_range);
        assert_eq!(uniform.blend_ratio, 1.0);

        let nan_params = GranTurismo7Params {
            mid_point: f32::NAN,
            ..Default::default()
        };
        let uniform = Gt7ParamsUniform::new(&DisplayTarget::SDR_SRGB, &nan_params);
        assert_eq!(uniform.mid_point, GranTurismo7Params::default().mid_point);
    }

    /// The uniform round-trips `Gt7ToneMapping` exactly, so the CPU struct
    /// (with the same correction-factor override) remains the parity
    /// reference for GPU readback tests of the uniform path.
    #[test]
    fn uniform_mirrors_cpu_tone_mapping() {
        let params = GranTurismo7Params::default();
        let mut reference = Gt7ToneMapping::new_hdr_with_params(1000.0, &params);
        reference.sdr_correction_factor = REFERENCE_LUMINANCE / 200.0;
        let uniform = Gt7ParamsUniform::new(
            &hdr_target(1000.0, 200.0, DisplayTransfer::ScRgbLinear),
            &params,
        );
        assert_eq!(uniform, Gt7ParamsUniform::from(&reference));
    }
}
