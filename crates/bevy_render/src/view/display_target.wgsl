// Per-view display-target calibration uniform.
//
// Mirrors `DisplayTargetUniform` in
// `bevy_render::view::display_target_uniform` (Rust); the two must stay
// field-for-field in sync. Only the display-encoding pass binds and reads the
// uniform, and only on HDR-transfer targets; SDR pipelines never reference
// this module.

#define_import_path bevy_render::display_target

// `DisplayTargetUniform::gamut` indices. Keep in sync with the
// `DISPLAY_GAMUT_*` constants in `display_target_uniform.rs`.
const DISPLAY_GAMUT_REC709: u32 = 0u;
const DISPLAY_GAMUT_DISPLAY_P3: u32 = 1u;
const DISPLAY_GAMUT_REC2020: u32 = 2u;

// `DisplayTargetUniform::transfer` indices. Keep in sync with the
// `DISPLAY_TRANSFER_*` constants in `display_target_uniform.rs`.
const DISPLAY_TRANSFER_SRGB: u32 = 0u;
const DISPLAY_TRANSFER_SCRGB_LINEAR: u32 = 1u;
const DISPLAY_TRANSFER_PQ: u32 = 2u;
// Index 3 is reserved for a future HLG transfer; `ExtendedSrgb` keeps index 4.
// const DISPLAY_TRANSFER_HLG: u32 = 3u;
const DISPLAY_TRANSFER_EXTENDED_SRGB: u32 = 4u;

// The resolved calibration of the display a view is presented on.
//
// Luminance fields are in nits (cd/m²). `gamut` and `transfer` hold the
// `DISPLAY_GAMUT_*` / `DISPLAY_TRANSFER_*` indices above. Gamut conversion
// matrices are not part of this uniform; the gamut-transform pass derives
// them per pipeline.
struct DisplayTargetUniform {
    // Luminance of "paper white" (1.0 at the tone-map operator output), nits.
    paper_white_nits: f32,
    // Maximum luminance of the display, nits.
    peak_luminance_nits: f32,
    // Black level of the display, nits.
    min_luminance_nits: f32,
    // Display gamut as a DISPLAY_GAMUT_* index.
    gamut: u32,
    // Resolved transfer function as a DISPLAY_TRANSFER_* index.
    transfer: u32,
}
