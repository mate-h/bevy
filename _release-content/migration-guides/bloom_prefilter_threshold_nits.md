---
title: "`BloomPrefilter` has a new `threshold_nits` field"
pull_requests: []
---

`BloomPrefilter` gained a `threshold_nits: Option<f32>` field, which lets the
bloom threshold be expressed as a physical luminance in nits instead of a raw
framebuffer value. When set, it takes precedence over `threshold` and is
divided by the paper white of the view's resolved display target
(`DisplayTarget::paper_white_nits` on the window; 100 nits for plain SDR
targets) at prepare time. This anchors the cutoff to a fixed physical
brightness on HDR displays, where the meaning of a raw framebuffer value
re-scales with the configured paper white.

The default is `None`, which preserves the existing `threshold` behavior
exactly. Code constructing `BloomPrefilter` with functional update syntax or
`Default` is unaffected; exhaustive struct literals need the new field:

```rust
// 0.18
BloomPrefilter {
    threshold: 0.6,
    threshold_softness: 0.2,
}

// 0.19
BloomPrefilter {
    threshold: 0.6,
    threshold_nits: None,
    threshold_softness: 0.2,
}
```
