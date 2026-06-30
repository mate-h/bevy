---
title: "Anti-aliasing and bloom are now correct on HDR display targets"
authors: ["@stuartparmenter"]
pull_requests: []
---

On an HDR display target (a window `DisplayTarget` with an scRGB-linear, PQ, or
extended-range sRGB transfer), the tone-mapped image post-processing operates on is no longer
confined to `[0, 1]`: values are paper-white-relative display-linear, and
highlights legitimately reach `peak_luminance_nits / paper_white_nits` (for
example 10.0 on a 1000-nit display at 100-nit paper white). Several
post-process effects were tuned for a `[0, 1]` signal and are now HDR-aware.
Each change is gated per view on the *resolved* display target, so cameras
presenting to SDR targets compile byte-identical shaders and render exactly as
before.

**Contrast adaptive sharpening (CAS)** assumed the image never went brighter
than paper white, producing fireflies and inverted sharpening around highlights
on HDR input. On HDR-target views the sharpening math now runs in a remapped
range where its assumptions hold, bounded so it can never invent out-of-scene
highlights.

**FXAA and SMAA** detect edges with luma thresholds calibrated for `[0, 1]`. On
HDR-target views the luma used for edge detection is saturated to `[0, 1]`:
content at or below paper white anti-aliases as on SDR, and brighter edges are
detected at their paper-white-clamped contrast. Only the edge metric is
clamped; the blended output still reads the unmodified HDR pixels, so no
brightness is lost.

**Bloom** gained two HDR refinements:

- On HDR-target views the internal bloom pyramid is `Rgba16Float` instead of
  `Rg11b10Ufloat`. The packed format's coarse mantissa above 1.0 causes visible
  quantization banding in the firefly-weighted downsample sums; fp16 has uniform
  precision across the range at twice the memory cost, paid only by views that
  can show the difference. SDR views keep `Rg11b10Ufloat` bit-for-bit.
- `BloomPrefilter` has a new `threshold_nits: Option<f32>` field. The existing
  `threshold` is a raw framebuffer value whose physical meaning rescales with
  paper white; `threshold_nits`, when set, takes precedence and is divided by
  the paper white of the view's resolved `DisplayTarget` at prepare time,
  anchoring the cutoff to a fixed physical brightness on any display.

```rust
Bloom {
    prefilter: BloomPrefilter {
        // Bloom only above 250 nits, on any display calibration.
        threshold_nits: Some(250.0),
        threshold_softness: 0.2,
        ..default()
    },
    composite_mode: BloomCompositeMode::Additive,
    ..Bloom::OLD_SCHOOL
}
```
