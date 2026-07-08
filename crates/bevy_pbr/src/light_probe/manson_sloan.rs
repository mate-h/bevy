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
//!
//! Runtime shading pairs with `bevy_light::SpecularEnvironmentIntegration::MansonSloan` (Listing 22
//! disabled; reflection vector `R`; same `F_ab` as GGX until an MS-specific DFG exists).
//!
//! ## Residual drift vs. importance-sampled / glTF IBL
//!
//! - **Table rows:** Seven optimized roughness levels; `environment_filter.wgsl` interpolates
//!   adjacent rows (`ms_poly_lerp`) so gather coefficients vary continuously with roughness.
//! - **`lod_resolution_bias`:** `log2(face_size / 128)` aligns paper §5 / eq. 4 mip–Jacobian with
//!   non-128 sources; mis-tuning shifts how detail collapses across roughness.
//! - **Three axial frames + θ,φ (Fig. 5–6):** Overlapping polar frames are inherently anisotropic
//!   on cube faces; handoff regions can differ per face from spherical GGX preintegration.
//! - **Pass-1 boundaries:** Quadratic downsample clamps the 4×4 B-spline footprint at face edges,
//!   asymmetrically reweighting boundary texels vs. the paper’s interior-only bilinear dots (Fig. 2).

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
