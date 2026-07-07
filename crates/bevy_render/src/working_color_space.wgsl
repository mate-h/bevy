// Working-color-space conversion helpers.
//
// Bevy's scene-referred working color space is linear Rec.709 by default and
// linear Rec.2020 when `WorkingColorSpace::Rec2020` is configured on
// `RenderPlugin` (see `bevy_render::working_color_space` on the Rust side).
// When the working space is Rec.2020, working-space-aware pipelines receive
// the `WORKING_COLOR_SPACE_REC2020` shader def; call sites are expected to
// gate their use of these helpers behind that def so that default (Rec.709)
// projects compose byte-identically to shaders that predate this module.

#define_import_path bevy_render::working_color_space

// Full-precision linear Rec.709 -> Rec.2020 matrix (D65, per ITU-R BT.2087).
// The f64 literals round to f32 values bit-identical to
// `bevy_render::working_color_space::REC709_TO_REC2020` on the Rust side
// (and to `GT7_REC_709_TO_REC_2020` in gt7.wgsl); keep them in sync.
const REC709_TO_REC2020 = mat3x3<f32>(
    0.627403895934699, 0.06909728935823199, 0.016391438875150228,   // column 0
    0.32928303837788375, 0.919540395075459, 0.08801330787722578,    // column 1
    0.043313065687417246, 0.011362315566309154, 0.895595253247624,  // column 2
);

// Inverse of the above (linear Rec.2020 -> Rec.709). Bit-identical to
// `bevy_render::working_color_space::REC2020_TO_REC709` on the Rust side.
const REC2020_TO_REC709 = mat3x3<f32>(
    1.6604910021084347, -0.12455047452159052, -0.01815076335490522, // column 0
    -0.5876411387885496, 1.1328998971259598, -0.10057889800800739,  // column 1
    -0.07284986331988484, -0.008349422604369487, 1.1187296613629125, // column 2
);

// Converts a linear Rec.709 color into the linear Rec.2020 working space.
// Out-of-gamut inputs (negative components) convert linearly like any other
// value; no clamping is applied.
fn rec709_to_rec2020(color: vec3<f32>) -> vec3<f32> {
    return REC709_TO_REC2020 * color;
}

// Converts a linear Rec.2020 working-space color to linear Rec.709.
// Colors outside the Rec.709 gamut produce negative components; callers that
// feed Rec.709-fit consumers (tone mapping operators, LUTs) are responsible
// for clamping.
fn rec2020_to_rec709(color: vec3<f32>) -> vec3<f32> {
    return REC2020_TO_REC709 * color;
}
