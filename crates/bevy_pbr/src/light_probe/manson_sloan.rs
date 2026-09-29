//! Manson–Sloan (EGSR 2015) polynomial coefficients for specular environment filtering.
//!
//! The embedded binary is the EsotericaFilterFitter `const_8` cubemap table fitted for a
//! **linear perceptual-roughness** schedule (`--curve esoterica`): table row `i` is the
//! filter for √α = i/6 (stored width α = (i/6)²). Source:
//! <https://github.com/bazhenovc/EsotericaFilterFitter>
//! (`ReflectionProbeTable_ggx_esoterica_cube.bin`, FTBL payload only).
//!
//! Generated MS environment maps use the **same** encoding as that curve: mip `i` of an
//! `n`-mip chain stores the filter for perceptual roughness `i/(n-1)` (α = perceptual²).
//! Runtime LOD is therefore `perceptual_roughness * maxMip` for both MS-generated maps and
//! prefiltered `EnvironmentMapLight` maps from glTF-IBL-Sampler / Karis (which also bake
//! mips on a linear perceptual schedule and square only inside the GGX NDF).
//!
//! Layout: `float4 coeffs[7][5][6]` — 7 levels × 5 parameters (dir xyz, LOD, weight) ×
//! 6 indices (3 axes × 8 taps / 4). Constant polynomial (no θ²/φ² terms).
//!
//! Regenerate via `assets/export_esoterica_filter_table.py` after fitting with
//! `FilterFitter --fit --curve esoterica --write-binary`.

/// Roughness levels in the filter table (128² … 2² cubemap faces).
pub const SPECULAR_TABLE_LEVELS: usize = 7;
/// Parameters per tap: direction xyz, mip level, weight.
pub const POLY_BANDS: usize = 5;
/// Active polynomial terms. `const_8` stores only the constant term.
pub const POLY_TERMS: usize = 1;
/// Tap indices: 3 axes × 8 taps / 4.
pub const POLY_INDICES: usize = 6;
/// Taps per axial frame (`const_8`).
pub const TAPS_PER_AXIS: usize = 8;

pub const FILTER_TABLE_VEC4_COUNT: usize =
    SPECULAR_TABLE_LEVELS * POLY_BANDS * POLY_TERMS * POLY_INDICES;

pub const FILTER_TABLE_SIZE: usize = FILTER_TABLE_VEC4_COUNT * 16;

pub const FILTER_TABLE_BYTES: &[u8] = include_bytes!("../../assets/manson_sloan_poly_table.bin");

const _: () = assert!(FILTER_TABLE_BYTES.len() == FILTER_TABLE_SIZE);
const _: () = assert!(FILTER_TABLE_VEC4_COUNT == 210);
const _: () = assert!(POLY_INDICES == (3 * TAPS_PER_AXIS) / 4);

/// Perceptual roughness (√α) at each table row. Linear across levels.
pub const TABLE_PERCEPTUAL_ROUGHNESS: [f32; SPECULAR_TABLE_LEVELS] =
    [0.0, 1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0, 4.0 / 6.0, 5.0 / 6.0, 1.0];

/// Maps linear perceptual roughness to a continuous table row index in `[0, 6]`.
pub fn perceptual_roughness_to_table_t(perceptual_roughness: f32) -> f32 {
    (perceptual_roughness.clamp(0.0, 1.0) * (SPECULAR_TABLE_LEVELS - 1) as f32).clamp(0.0, 6.0)
}

/// Perceptual roughness stored in MS-generated mip `mip` of an `num_mips`-level chain.
///
/// Matches Esoterica `--curve esoterica`: linear in perceptual roughness across mips.
pub fn mip_perceptual_roughness(mip: u32, num_mips: u32) -> f32 {
    let denom = num_mips.saturating_sub(1).max(1) as f32;
    (mip as f32 / denom).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_schedule_is_linear_perceptual_roughness() {
        for mip in 0..SPECULAR_TABLE_LEVELS {
            let expected = mip as f32 / (SPECULAR_TABLE_LEVELS - 1) as f32;
            let actual = TABLE_PERCEPTUAL_ROUGHNESS[mip];
            assert!(
                (expected - actual).abs() < 1e-6,
                "mip {mip}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn remap_is_identity_on_linear_schedule() {
        assert_eq!(perceptual_roughness_to_table_t(0.0), 0.0);
        assert_eq!(perceptual_roughness_to_table_t(1.0), 6.0);
        for i in 0..SPECULAR_TABLE_LEVELS {
            let r = TABLE_PERCEPTUAL_ROUGHNESS[i];
            let t = perceptual_roughness_to_table_t(r);
            assert!(
                (t - i as f32).abs() < 1e-5,
                "row {i} roughness {r} should map to {i}, got {t}"
            );
        }
    }

    #[test]
    fn mip_schedule_matches_esoterica_linear_perceptual() {
        // 7-level chain: mip 1 → pr = 1/6, α = (1/6)² (gentle first step).
        let pr = mip_perceptual_roughness(1, 7);
        assert!((pr - 1.0 / 6.0).abs() < 1e-6);
        let alpha = pr * pr;
        assert!((alpha - (1.0 / 6.0) * (1.0 / 6.0)).abs() < 1e-6);
        let table_t = perceptual_roughness_to_table_t(pr);
        assert!((table_t - 1.0).abs() < 1e-5);
    }

    #[test]
    fn embedded_table_has_expected_size() {
        assert_eq!(FILTER_TABLE_BYTES.len(), 3360);
    }
}
