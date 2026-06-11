---
title: "`AutoExposure` has new `metering_bias` and `physiological` fields"
pull_requests: []
---

The `AutoExposure` component gained two new public fields:

```rust
pub metering_bias: f32,
pub physiological: Option<PhysiologicalAdaptation>,
```

Both default to a no-op (`0.0` and `None`), so the metering behavior is unchanged
unless you set them — see the "Physiological two-stage auto exposure" release note
for what they do.

If you construct `AutoExposure` with `..default()` (or `AutoExposure::default()`),
no change is needed. If you build it with a full struct literal, add the two
fields:

```rust
// Before
AutoExposure {
    range: -8.0..=8.0,
    filter: 0.10..=0.90,
    speed_brighten: 3.0,
    speed_darken: 1.0,
    exponential_transition_distance: 1.5,
    metering_mask: default(),
    compensation_curve: default(),
};

// After
AutoExposure {
    range: -8.0..=8.0,
    filter: 0.10..=0.90,
    speed_brighten: 3.0,
    speed_darken: 1.0,
    exponential_transition_distance: 1.5,
    metering_mask: default(),
    compensation_curve: default(),
    metering_bias: 0.0,
    physiological: None,
};
```
