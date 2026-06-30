---
title: "`Bloom` and `BloomPrefilter` have new fields"
pull_requests: []
---

`Bloom` gained a `scatter: BloomScatterModel` field, and its nested
`BloomPrefilter` gained a `threshold_nits: Option<f32>` field. `..default()`,
functional update (`..Bloom::NATURAL`), and preset construction are unaffected;
exhaustive struct literals need both new fields:

```rust
// 0.19
Bloom {
    intensity: 0.2,
    low_frequency_boost: 0.7,
    low_frequency_boost_curvature: 0.95,
    high_pass_frequency: 1.0,
    prefilter: BloomPrefilter {
        threshold: 0.6,
        threshold_softness: 0.2,
    },
    composite_mode: BloomCompositeMode::EnergyConserving,
    max_mip_dimension: 512,
    scale: Vec2::ONE,
}

// 0.20
Bloom {
    intensity: 0.2,
    low_frequency_boost: 0.7,
    low_frequency_boost_curvature: 0.95,
    high_pass_frequency: 1.0,
    prefilter: BloomPrefilter {
        threshold: 0.6,
        threshold_nits: None,
        threshold_softness: 0.2,
    },
    composite_mode: BloomCompositeMode::EnergyConserving,
    max_mip_dimension: 512,
    scale: Vec2::ONE,
    scatter: BloomScatterModel::Aesthetic,
}
```

- `scatter` selects how the blur pyramid is weighted during compositing.
  `BloomScatterModel::Aesthetic` is the existing parametric curve and the
  default; all presets (`NATURAL`, `ANAMORPHIC`, `OLD_SCHOOL`, `SCREEN_BLUR`)
  use it, so their rendering is unchanged. `BloomScatterModel::Gt7Glare { f_number }`
  is physically based veiling glare derived from a camera aperture's diffraction
  pattern, inspired by Gran Turismo 7 (also the `Bloom::GT7_GLARE` preset). Under
  `Gt7Glare` the `prefilter` thresholds are ignored (a physical glare PSF has no
  brightness cutoff) and `composite_mode` is forced to `EnergyConserving`; both
  are warned about at runtime where relevant.
- `threshold_nits` expresses the bloom cutoff as a physical luminance in nits
  (default `None`). When set, it takes precedence over `threshold` and is divided
  by `DisplayTarget::paper_white_nits` (100 for SDR targets) at prepare time,
  anchoring the cutoff to a fixed brightness on HDR displays.
