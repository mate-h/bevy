---
title: "Display-encoding pass: gamut transform and HDR transfer functions"
authors: ["@stuartparmenter"]
pull_requests: []
---

When a camera presents to an HDR display, the tone-mapped image has to be
converted into the exact signal the display expects — the right primaries and
the right transfer function — or highlights and saturated colors come out
wrong. Bevy now does this in a dedicated **display-encoding pass**, so HDR
output is encoded correctly without any per-camera setup on your part.

To use it, request an HDR transfer on the window's `DisplayTarget` and give the
camera an HDR-aware tone-mapping operator (see the "HDR display output
(scRGB-linear and HDR10/PQ)" release note). Everything below happens
automatically for that view.

The pass runs after the UI pass (UI keeps compositing in display-linear,
paper-white-relative space, so a white UI panel lands exactly at
`DisplayTarget::paper_white_nits`) and before the final upscaling blit. It
reads the view's `DisplayTarget` calibration and performs, in order:

- a full-precision gamut transform from the tone-map operator's output
  primaries to the display signal's primaries (for example, a Rec.2020 →
  Rec.709 contraction when `Tonemapping::GranTurismo7` drives an scRGB signal,
  or a Rec.709 → Display-P3 expansion for an `ExtendedDisplayP3` signal);
- out-of-gamut handling: smooth, ACES-RGC-style perceptual gamut compression
  when the transform can push colors outside the display gamut, so wide-gamut
  colors compress gracefully instead of clipping (see the display gamut
  compression release note);
- the display transfer function — scRGB-linear, PQ (SMPTE ST 2084), or the
  encoded extended-range sRGB OETF (the `ExtendedSrgb` / `ExtendedDisplayP3`
  color spaces) — selected by the target's `DisplayTransfer`.

This also rounds out a cleaner tone-mapping path: tone mapping always runs as
the post-process tonemapping pass now, so every camera with an active
`Tonemapping` operator renders into a scene-linear `Rgba16Float` intermediate,
blends transparents in linear light, and tone-maps once. See the migration
guides for the visual implications on previously-SDR cameras.

The transfer functions themselves — the exact piecewise sRGB OETF/EOTF, the
odd-symmetric extended-range sRGB OETF/EOTF, scRGB scaling, PQ
inverse-EOTF/EOTF, and the HLG OETF — live in a new importable WGSL library,
`bevy_render::transfer_functions`, with CPU mirrors (and parity tests) in the
matching Rust module, so you can reuse them in your own shaders.

**Nothing changes for SDR rendering.** Views on the default
`DisplayTarget::SDR_SRGB` (or any sRGB-transfer target) never run the pass — it
records no GPU work, and the exact sRGB encode remains the free hardware
conversion on the swapchain writeback, byte-identical to previous releases. The
pass only activates for display targets whose resolved transfer is HDR
(`ScRgbLinear`, `Pq`, or `ExtendedSrgb`), which is exactly when surface
selection has configured a matching HDR swapchain — see the scRGB HDR output
release note (including its note on the currently required wgpu patch).
