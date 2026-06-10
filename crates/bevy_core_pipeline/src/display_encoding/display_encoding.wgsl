// Display-encoding pass: paper-white-relative display-linear color (the
// tone-map operator's output space, with UI already composited) → encoded
// display signal.
//
// Stages, per the separated display pipeline:
//   1. (optional) decode the view's compositing space back to linear
//      (`SRGB_TO_LINEAR` / `OKLAB_TO_LINEAR`, normally done by the upscaling
//      blit — which passes encoded output through untouched instead when this
//      pass ran),
//   2. gamut transform from the working primaries (linear Rec.709 today) to
//      the display *signal* primaries (`DISPLAY_GAMUT_REC2020`, identity for
//      Rec.709) — only reachable together with `DISPLAY_TRANSFER_PQ`: scRGB
//      signals are definitionally expressed in (extended) Rec.709 coordinates
//      whatever the panel's physical gamut, so prepare coerces the scRGB
//      encoding gamut to Rec.709 (the compositor maps to the panel itself),
//   3. out-of-gamut handling: ACES-RGC-style hue-approximate chroma
//      compression toward the achromatic axis (`DISPLAY_GAMUT_COMPRESSION`),
//      with the plain hue-shifting per-channel clip as the debug fallback
//      (`DISPLAY_GAMUT_CLIP_DEBUG`), followed by an always-on `max(0)` safety
//      clip (PQ requires non-negative input),
//   4. transfer encoding (`DISPLAY_TRANSFER_SCRGB` / `DISPLAY_TRANSFER_PQ`).
//
// This shader is never specialized for sRGB targets: the exact sRGB OETF is
// hardware-applied on the upscaling blit's `*UnormSrgb` writeback, byte-
// identical to Bevy's behavior before this pass existed.

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_render::display_target::DisplayTargetUniform
#import bevy_render::transfer_functions::{scrgb_encode, pq_inverse_eotf_from_nits}
#ifdef SRGB_TO_LINEAR
#import bevy_render::color_operations::srgb_to_linear
#endif
#ifdef OKLAB_TO_LINEAR
#import bevy_render::color_operations::oklab_to_linear_rgb
#endif

@group(0) @binding(0) var in_texture: texture_2d<f32>;
@group(0) @binding(1) var in_sampler: sampler;
// Per-view display-target calibration (paper white / peak / gamut / transfer
// indices). Gamut and transfer are compile-time shader defs here; only the
// luminance fields are read at runtime. `paper_white_nits` is sanitized by
// the uniform writer (`prepare_display_target_uniforms`: finite, positive,
// <= 10000) with the same rules the tone-map operators fold at prepare time,
// so the seam scale factors cancel exactly.
@group(0) @binding(2) var<uniform> display_target: DisplayTargetUniform;

#ifdef DISPLAY_GAMUT_REC2020
// Full-precision linear Rec.709 → Rec.2020 matrix (D65, per ITU-R BT.2087;
// f64-derived via the Lindbloom primaries→XYZ method, matching
// `bevy_color::primaries::rgb_to_rgb_matrix(RgbPrimaries::BT709,
// RgbPrimaries::BT2020)`). Identical literals to `GT7_REC_709_TO_REC_2020`
// in gt7.wgsl / `REC_709_TO_REC_2020` in gt7.rs.
// TODO: deduplicate with shared color-space constants once they land in
// `bevy_render::color_operations` (HDR workstream follow-up).
const REC_709_TO_REC_2020 = mat3x3<f32>(
    0.627403895934699, 0.06909728935823199, 0.016391438875150228,   // column 0
    0.32928303837788375, 0.919540395075459, 0.08801330787722578,    // column 1
    0.043313065687417246, 0.011362315566309154, 0.895595253247624,  // column 2
);
#endif

#ifdef DISPLAY_GAMUT_COMPRESSION
// Out-of-gamut chroma compression in the style of the ACES 1.3 Reference
// Gamut Compression (Academy S-2020-001 / ACES "RGC", aces-dev
// lib/RGC_common.ctl): per-channel distance from the achromatic axis,
// `dist = (ach - c) / ach` with `ach = max(r, g, b)`, smoothly compressed
// with the parametric power curve so that `dist == limit` lands exactly on
// the gamut boundary (`dist == 1`, i.e. channel value 0) while distances
// below the threshold pass through bit-identically.
//
// Thresholds and power are the published ACES RGC values (cyan 0.815,
// magenta 0.803, yellow 0.880, power 1.2). The ACES *limits*
// (1.147 / 1.264 / 1.312) were derived from digital-cinema camera gamuts and
// under-cover the Rec.2020 → Rec.709 contraction this pass performs (the
// Rec.2020 hull reaches a distance of ~1.594 in the cyan direction when
// expressed in Rec.709 coordinates), so the limits below are re-derived from
// the Rec.2020 hull maxima (~1.594 / ~1.087 / ~1.117) plus headroom.
// CPU mirror + tests: `gamut_compression.rs` next to this shader — keep both
// in sync.
const GAMUT_COMPRESSION_THRESHOLD = vec3<f32>(0.815, 0.803, 0.880);
const GAMUT_COMPRESSION_POWER: f32 = 1.2;
// Limits (the distance that maps exactly onto the gamut boundary); kept for
// documentation/derivation only — the shader consumes the precomputed scales.
const GAMUT_COMPRESSION_LIMIT = vec3<f32>(1.62, 1.10, 1.13);
// scale = (limit - thr) / (((1 - thr) / (limit - thr))^(-power) - 1)^(1/power),
// evaluated in f64 (see `compression_scale` in gamut_compression.rs; a test
// locks these literals to the closed form).
const GAMUT_COMPRESSION_SCALE = vec3<f32>(0.21634937, 0.43270176, 0.18745117);

// The ACES RGC parametric compression curve, vectorized over the three
// chroma directions. Identity below the threshold (the `max` keeps `pow`
// away from negative bases; callers select the original value there anyway),
// monotonically increasing above it, mapping `limit` to 1 and approaching
// `threshold + scale` asymptotically.
fn gamut_compress_distance(dist: vec3<f32>) -> vec3<f32> {
    let nd = max(dist - GAMUT_COMPRESSION_THRESHOLD, vec3(0.0)) / GAMUT_COMPRESSION_SCALE;
    let p = pow(nd, vec3(GAMUT_COMPRESSION_POWER));
    return GAMUT_COMPRESSION_THRESHOLD
        + GAMUT_COMPRESSION_SCALE * nd / pow(1.0 + p, vec3(1.0 / GAMUT_COMPRESSION_POWER));
}

// Compresses out-of-gamut colors (negative components) toward the achromatic
// axis at constant `max(r, g, b)`. Channels whose distance is below the
// threshold are returned bit-identically; colors whose distance does not
// exceed the limit land inside the target gamut. Hue is approximately (not
// exactly) preserved — the per-channel formulation is the standard
// cost/robustness trade-off of the ACES RGC (no boundary search, no
// iteration, monotonic, NaN-free for finite inputs).
fn gamut_compress(rgb: vec3<f32>) -> vec3<f32> {
    let achromatic = max(rgb.r, max(rgb.g, rgb.b));
    if achromatic <= 0.0 {
        // No positive channel to compress toward; the final safety clip
        // below handles all-non-positive colors (exactly like the previous
        // clip-only behavior).
        return rgb;
    }
    let dist = (vec3(achromatic) - rgb) / achromatic;
    let compressed = vec3(achromatic) - gamut_compress_distance(dist) * achromatic;
    // Bit-identical pass-through for in-gamut channels under the threshold.
    return select(compressed, rgb, dist < GAMUT_COMPRESSION_THRESHOLD);
}
#endif

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    var color = textureSample(in_texture, in_sampler, in.uv);

    // 1. Decode the compositing space, if the main texture is not already
    // display-linear. (Same defs and math as the upscaling blit.)
#ifdef SRGB_TO_LINEAR
    color = vec4(srgb_to_linear(color.rgb), color.a);
#endif
#ifdef OKLAB_TO_LINEAR
    color = vec4(oklab_to_linear_rgb(color.rgb), color.a);
#endif

    var rgb = color.rgb;

    // 2. Gamut transform: tone-map-output primaries → display primaries.
    //
    // Input-contract note for the working-space axis (T2.5): the tone
    // mapping pass emits Rec.709 display-linear under EVERY
    // `WorkingColorSpace` — Rec.709-fit operators receive a Rec.2020 →
    // Rec.709 conversion at the pass entry, and the GT7 operator keeps its
    // own Rec.2020 → Rec.709 back-conversion. So this stage's input is
    // Rec.709 regardless of the working space, and the defs below are
    // unchanged by `WorkingColorSpace::Rec2020`. When the GT7 HDR-native
    // output path lands (operator output staying Rec.2020 for HDR targets),
    // this stage becomes a true working → display transform: Rec.2020 input
    // + PQ/Rec.2020 display = identity, Rec.2020 input + scRGB (Rec.709
    // coordinates) = the inverse REC_2020_TO_REC_709 matrix — a gamut
    // *contraction*, for which prepare keys in the out-of-gamut compression
    // below (`DISPLAY_GAMUT_COMPRESSION`; see `DisplayGamutCompression` and
    // `encoder_input_gamut` in mod.rs).
#ifdef DISPLAY_GAMUT_REC2020
    rgb = REC_709_TO_REC_2020 * rgb;
#endif

    // 3. Out-of-gamut handling (DECISIONS.md D3: perceptual compression,
    // with the per-channel clip as the debug fallback). The compression def
    // is pushed only when the gamut stage can actually produce out-of-gamut
    // colors (a gamut *contraction*, or `DisplayGamutCompression::Always`);
    // expansions and identity transforms keep the plain clip below, which is
    // a no-op for their in-gamut-by-construction inputs.
#ifdef DISPLAY_GAMUT_COMPRESSION
    rgb = gamut_compress(rgb);
#endif
    // Final safety clip of negative components (PQ additionally requires
    // non-negative input before its `pow`). After compression this only
    // catches floating-point residue and scene-referred negatives that did
    // not come from the gamut stage (compressed colors land in-gamut by
    // construction); under DISPLAY_GAMUT_CLIP_DEBUG — or when no compression
    // is active — it IS the entire out-of-gamut handling: the hue-shifting
    // per-channel clip, kept for A/B comparison against the compression.
    rgb = max(rgb, vec3(0.0));

    // 4. Transfer encoding. Input is paper-white-relative display-linear
    // (1.0 = paper white at the operator output, the D1/D5 convention).
#ifdef DISPLAY_TRANSFER_SCRGB
    // scRGB-linear: 1.0 = 80 nits, so scale by paper_white / 80.
    rgb = scrgb_encode(rgb, display_target.paper_white_nits);
#else ifdef DISPLAY_TRANSFER_PQ
    // PQ encodes absolute luminance normalized to 10000 nits: convert
    // paper-white-relative values to nits first.
    rgb = pq_inverse_eotf_from_nits(rgb * display_target.paper_white_nits);
#endif

    // Alpha passes through for the multi-camera alpha-blended upscale path.
    return vec4(rgb, color.a);
}
