---
title: "Anti-aliasing and bloom are now correct on HDR display targets"
authors: ["@stuartparmenter"]
pull_requests: []
---

With HDR display output (a window `DisplayTarget` with an scRGB-linear or PQ
transfer), the tone-mapped image that post-processing operates on is no longer
confined to `[0, 1]`: values are paper-white-relative display-linear, and
highlights legitimately reach `peak_luminance_nits / paper_white_nits` (for
example 10.0 on a 1000-nit display at 100-nit paper white). Several
post-process effects were tuned for a `[0, 1]` signal; they have been made
HDR-aware. All of these changes are gated per view on the *resolved* display
target, so cameras presenting to SDR targets compile byte-identical shaders
and render exactly as before.

**Contrast adaptive sharpening (CAS)** was algorithmically broken on HDR
input: the RCAS limiter constants are clip solves that assume the signal
peaks at 1.0, and with brighter input the limiter could divide by zero, flip
sign, or massively under/overshoot — fireflies and inverted sharpening. On
HDR-target views the shader now range-compresses the 5-tap neighborhood into
`[0, 1)` with the reversible Reinhard `x / (1 + x)`, runs the unmodified RCAS
math where its assumptions hold, decompresses the result, and bounds
overshoot by `max(local_max, paper_white)` so sharpening can never
manufacture out-of-scene highlights.

**FXAA and SMAA** run after tone mapping and detect edges with luma
thresholds calibrated for `[0, 1]`. On HDR-target views the luma used for
edge detection is now saturated to `[0, 1]`: content at or below paper white
anti-aliases exactly as on SDR displays, and edges involving brighter values
are detected at their paper-white-clamped contrast. Only the edge metric is
clamped — the blended output still reads the unmodified HDR pixels, so no
brightness is lost.

**Bloom** gained two HDR refinements:

- On HDR-target views the internal bloom pyramid is now `Rgba16Float` instead
  of `Rg11b10Ufloat`. The packed format's coarse mantissa above 1.0 causes
  visible quantization banding in the firefly-weighted downsample sums
  exactly where HDR output makes it visible; fp16 has uniform precision
  across the range at twice the memory cost, which is only paid by views that
  can show the difference. SDR views keep the packed format bit-for-bit.
- `BloomPrefilter` has a new `threshold_nits: Option<f32>` field. The
  existing `threshold` is a raw framebuffer value whose physical meaning
  silently rescales with the display's paper white; `threshold_nits`, when
  set, takes precedence and is divided by the paper white of the view's
  resolved `DisplayTarget` at prepare time, anchoring the cutoff to a fixed
  physical brightness on any display.

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
