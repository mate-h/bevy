//! Manson & Sloan (EGSR 2015) — Activision polynomial coefficient table for specular filtering.
//!
//! The auxiliary data stores **`coeffs[level][band][poly_term][index]`** as `float4`
//! (four trilinear sub-taps per super-tap), matching `filter_using_table_128` from the paper materials.
//! Levels are **7** (128² … 1² cubemap faces at `BASE_RESOLUTION 128`).
//!
//! ## Regenerating `manson_sloan_poly_table.bin`
//!
//! From `coeffs_quad_32.txt` (`static const float4 coeffs[7][5][3][24]`):
//!
//! ```text
//! python3 crates/bevy_pbr/assets/build_manson_sloan_poly_table.py \
//!   path/to/coeffs_quad_32.txt crates/bevy_pbr/assets/manson_sloan_poly_table.bin
//! ```
//!
//! ## Resolution
//!
//! Sample LODs in the table assume **128²** reference faces; the filter pass adds
//! [`crate::light_probe::generate::FilteringConstants::lod_resolution_bias`] =
//! **`log2(face_size / 128)`** when sampling the mip chain (paper §6).

/// Cubemap mip rows in the polynomial table (128 → 1 in seven steps).
pub const SPECULAR_TABLE_LEVELS: usize = 7;
/// `dir0`, `dir1`, `dir2`, `lod`, `weight` polynomial bands.
pub const POLY_BANDS: usize = 5;
/// Constant, θ², φ² terms per band.
pub const POLY_TERMS: usize = 3;
/// `(NUM_TAPS / 4) * 3` axes — **32** taps / **8** super-taps per axis.
pub const POLY_INDICES: usize = 24;

pub const FILTER_TABLE_VEC4_COUNT: usize =
    SPECULAR_TABLE_LEVELS * POLY_BANDS * POLY_TERMS * POLY_INDICES;

/// GPU storage buffer size: `vec4<f32>` × [`FILTER_TABLE_VEC4_COUNT`].
pub const FILTER_TABLE_SIZE: usize = FILTER_TABLE_VEC4_COUNT * 16;

pub const FILTER_TABLE_BYTES: &[u8] = include_bytes!("../../assets/manson_sloan_poly_table.bin");

const _: () = assert!(FILTER_TABLE_BYTES.len() == FILTER_TABLE_SIZE);
