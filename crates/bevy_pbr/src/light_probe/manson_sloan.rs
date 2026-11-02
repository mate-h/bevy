//! Manson–Sloan (EGSR 2015) polynomial coefficients for specular environment filtering.
//!
//! Regenerate `assets/manson_sloan_poly_table.bin` with `assets/build_manson_sloan_poly_table.py`.

/// Roughness levels in the polynomial table (128² … 1² cubemap faces).
pub const SPECULAR_TABLE_LEVELS: usize = 7;
/// Polynomial bands per table row: direction, LOD, and weight.
pub const POLY_BANDS: usize = 5;
/// Polynomial terms per band.
pub const POLY_TERMS: usize = 3;
/// Tap indices per axis.
pub const POLY_INDICES: usize = 24;

pub const FILTER_TABLE_VEC4_COUNT: usize =
    SPECULAR_TABLE_LEVELS * POLY_BANDS * POLY_TERMS * POLY_INDICES;

pub const FILTER_TABLE_SIZE: usize = FILTER_TABLE_VEC4_COUNT * 16;

pub const FILTER_TABLE_BYTES: &[u8] = include_bytes!("../../assets/manson_sloan_poly_table.bin");

const _: () = assert!(FILTER_TABLE_BYTES.len() == FILTER_TABLE_SIZE);
