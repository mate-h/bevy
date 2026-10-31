//! Exact GPU accumulation capture decoding. Alpha and RGB travel in the same copy.
#[derive(Clone, Copy)]
pub(crate) struct ReferenceConfig {
    pub spp: u32,
    pub seed: u32,
    pub timeout_secs: f32,
}
impl Default for ReferenceConfig {
    fn default() -> Self {
        Self {
            spp: 512,
            seed: 0,
            timeout_secs: 3600.0,
        }
    }
}
impl ReferenceConfig {
    pub fn from_env() -> Result<Self, String> {
        let read = |name: &str, fallback: &str| match std::env::var(name) {
            Ok(v) => Ok(v),
            Err(std::env::VarError::NotPresent) => Ok(fallback.to_string()),
            Err(_) => Err(format!("{name} must be valid Unicode")),
        };
        Self::parse(
            &read("IBL_REFERENCE_SPP", "512")?,
            &read("IBL_REFERENCE_SEED", "0")?,
            &read("IBL_REFERENCE_TIMEOUT_SECS", "3600")?,
        )
    }
    fn parse(spp: &str, seed: &str, timeout: &str) -> Result<Self, String> {
        let spp = spp
            .parse::<u32>()
            .map_err(|_| "IBL_REFERENCE_SPP must be a u32")?;
        if !(1..=16_777_216).contains(&spp) {
            return Err("IBL_REFERENCE_SPP must be in 1..=16777216".into());
        }
        let seed = seed
            .parse::<u32>()
            .map_err(|_| "IBL_REFERENCE_SEED must be a u32")?;
        let timeout_secs = timeout
            .parse::<f32>()
            .map_err(|_| "Invalid IBL_REFERENCE_TIMEOUT_SECS")?;
        if !timeout_secs.is_finite() || timeout_secs <= 0.0 {
            return Err("IBL_REFERENCE_TIMEOUT_SECS must be finite and positive".into());
        }
        Ok(Self {
            spp,
            seed,
            timeout_secs,
        })
    }
}
pub(crate) fn accepts_snapshot(active_run: u64, snapshot_run: u64, waiting: bool) -> bool {
    waiting && active_run == snapshot_run
}

pub(crate) fn stride(width: u32) -> usize {
    (width as usize * 16).div_ceil(256) * 256
}
fn pixel(bytes: &[u8], width: u32, x: u32, y: u32) -> Option<[f32; 4]> {
    let start = y as usize * stride(width) + x as usize * 16;
    let bytes = bytes.get(start..start + 16)?;
    let value =
        std::array::from_fn(|i| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()));
    Some(value)
}
pub(crate) fn reference_rgb(bytes: &[u8], width: u32, x: u32, y: u32) -> Option<[f32; 3]> {
    let p = pixel(bytes, width, x, y)?;
    let rgb = [p[0], p[1], p[2]];
    rgb.iter().all(|v| v.is_finite()).then_some(rgb)
}
/// False means an incomplete snapshot with valid layout and first-pixel count.
/// Only completed snapshots receive full-image validation; no CPU frame counter
/// is involved, and incomplete data is never accepted as a reference.
pub(crate) fn validate_reference_capture(
    bytes: &[u8],
    width: u32,
    height: u32,
    requested: u32,
) -> Result<bool, &'static str> {
    if width == 0 || height == 0 || bytes.len() != stride(width) * height as usize {
        return Err("invalid RGBA32Float padded layout");
    }
    let first = pixel(bytes, width, 0, 0).unwrap()[3];
    if !first.is_finite() || first < 0.0 || first.fract() != 0.0 || first > requested as f32 {
        return Err("invalid GPU spp");
    }
    if first < requested as f32 {
        return Ok(false);
    }
    for y in 0..height {
        for x in 0..width {
            let p = pixel(bytes, width, x, y).unwrap();
            if p.iter().any(|v| !v.is_finite()) {
                return Err("nonfinite accumulation");
            }
            if p[3] != first {
                return Err("nonuniform GPU spp");
            }
        }
    }
    Ok(first == requested as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn capture(width: u32, height: u32, spp: f32) -> Vec<u8> {
        let mut bytes = vec![0; stride(width) * height as usize];
        for y in 0..height {
            for x in 0..width {
                let start = y as usize * stride(width) + x as usize * 16;
                for (i, value) in [1.25_f32, 0.5, 2.0, spp].iter().enumerate() {
                    bytes[start + i * 4..start + i * 4 + 4].copy_from_slice(&value.to_le_bytes());
                }
            }
        }
        bytes
    }
    #[test]
    fn config_and_stale_snapshot() {
        let config = ReferenceConfig::parse("8192", "4294967295", "3600").unwrap();
        assert_eq!(config.seed, u32::MAX);
        assert_eq!(config.spp, 8192);
        assert_eq!(config.timeout_secs, 3600.0);
        for spp in ["0", "16777217", "-1", "1.5"] {
            assert!(ReferenceConfig::parse(spp, "0", "1").is_err());
        }
        for seed in ["4294967296", "-1", "abc"] {
            assert!(ReferenceConfig::parse("512", seed, "1").is_err());
        }
        for timeout in ["0", "NaN", "inf", "-1"] {
            assert!(ReferenceConfig::parse("512", "0", timeout).is_err());
        }
        assert!(accepts_snapshot(2, 2, true));
        assert!(!accepts_snapshot(2, 1, true));
        assert!(!accepts_snapshot(2, 2, false));
    }
    #[test]
    fn padded_layout_and_exact_count() {
        let bytes = capture(17, 2, 8192.0);
        assert_eq!(stride(17), 512);
        assert_eq!(validate_reference_capture(&bytes, 17, 2, 8192), Ok(true));
        assert_eq!(validate_reference_capture(&bytes, 17, 2, 16384), Ok(false));
        assert!(validate_reference_capture(&bytes[..bytes.len() - 1], 17, 2, 8192).is_err());
    }
    #[test]
    fn rejects_mismatched_fractional_and_nonfinite_samples() {
        for alpha in [3.5, f32::NAN, f32::INFINITY, -1.0, 9.0] {
            assert!(validate_reference_capture(&capture(2, 1, alpha), 2, 1, 8).is_err());
        }
        let mut bytes = capture(2, 1, 8.0);
        bytes[28..32].copy_from_slice(&7_f32.to_le_bytes());
        assert!(validate_reference_capture(&bytes, 2, 1, 8).is_err());
        bytes[28..32].copy_from_slice(&8_f32.to_le_bytes());
        bytes[0..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(validate_reference_capture(&bytes, 2, 1, 8).is_err());
    }

    #[test]
    fn incomplete_snapshots_are_deferred_but_final_snapshots_are_fully_checked() {
        let mut bytes = capture(2, 1, 4.0);
        bytes[16..20].copy_from_slice(&f32::NAN.to_le_bytes());
        bytes[28..32].copy_from_slice(&3_f32.to_le_bytes());
        assert_eq!(validate_reference_capture(&bytes, 2, 1, 8), Ok(false));
        bytes[12..16].copy_from_slice(&8_f32.to_le_bytes());
        assert!(validate_reference_capture(&bytes, 2, 1, 8).is_err());
        bytes[16..20].copy_from_slice(&1_f32.to_le_bytes());
        assert!(validate_reference_capture(&bytes, 2, 1, 8).is_err());
        bytes[28..32].copy_from_slice(&8_f32.to_le_bytes());
        assert_eq!(validate_reference_capture(&bytes, 2, 1, 8), Ok(true));
    }
}
