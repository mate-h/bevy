---
title: "`Bloom` has a new `scatter` field"
pull_requests: []
---

`Bloom` gained a `scatter: BloomScatterModel` field selecting how the blur
pyramid is weighted during compositing:

- `BloomScatterModel::Aesthetic` — the existing parametric curve. This is
  the default; all presets (`NATURAL`, `ANAMORPHIC`, `OLD_SCHOOL`,
  `SCREEN_BLUR`) use it, and rendering is unchanged.
- `BloomScatterModel::Gt7Glare { f_number }` — physically based veiling
  glare derived from the diffraction pattern of a camera aperture, inspired
  by Gran Turismo 7 (also available as the `Bloom::GT7_GLARE` preset).

Code constructing `Bloom` with functional update syntax (`..default()`,
`..Bloom::NATURAL`) or via the presets is unaffected; exhaustive struct
literals need the new field:

```rust
// 0.18
Bloom {
    intensity: 0.2,
    low_frequency_boost: 0.7,
    low_frequency_boost_curvature: 0.95,
    high_pass_frequency: 1.0,
    prefilter: BloomPrefilter::default(),
    composite_mode: BloomCompositeMode::EnergyConserving,
    max_mip_dimension: 512,
    scale: Vec2::ONE,
}

// 0.19
Bloom {
    intensity: 0.2,
    low_frequency_boost: 0.7,
    low_frequency_boost_curvature: 0.95,
    high_pass_frequency: 1.0,
    prefilter: BloomPrefilter::default(),
    composite_mode: BloomCompositeMode::EnergyConserving,
    max_mip_dimension: 512,
    scale: Vec2::ONE,
    scatter: BloomScatterModel::Aesthetic,
}
```

Note that under `BloomScatterModel::Gt7Glare` the `prefilter` thresholds are
ignored (a physical glare PSF has no brightness cutoff) and `composite_mode`
is effectively `EnergyConserving`; both behaviors are documented on the enum
and warned about at runtime where relevant.
