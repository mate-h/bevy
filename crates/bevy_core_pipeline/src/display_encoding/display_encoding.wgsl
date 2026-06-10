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
//      the display primaries (`DISPLAY_GAMUT_REC2020`, identity for Rec.709),
//   3. out-of-gamut handling (currently a per-channel clip stub),
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
// luminance fields are read at runtime.
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

    // 2. Gamut transform: working primaries → display primaries.
#ifdef DISPLAY_GAMUT_REC2020
    rgb = REC_709_TO_REC_2020 * rgb;
#endif

    // 3. Out-of-gamut handling stub: per-channel clip of negative components
    // (PQ additionally requires non-negative input before its `pow`). This
    // also discards the negative-component wide-gamut headroom scRGB could
    // carry — acceptable while the working space is Rec.709.
    // TODO(T2.6): replace with hue-preserving gamut compression in ICtCp
    // (per-channel clip stays as the debug fallback), per DECISIONS.md D3.
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
