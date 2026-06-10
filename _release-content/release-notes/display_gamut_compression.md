---
title: "Perceptual out-of-gamut compression in the display-encoding pass"
authors: ["@pavlov"]
pull_requests: []
---

The display-encoding pass's gamut stage now performs perceptual out-of-gamut
compression instead of a bare per-channel clip. When the transform from the
working primaries to the display primaries is a *contraction* (a wider source
gamut than the display can show), colors land outside the display gamut as
negative RGB components; clipping those per channel collapses chroma unevenly
and shifts hue (the classic `(1500, 1200, 500) → (1000, 1000, 500)` problem).

The new handling is chroma compression toward the achromatic axis in the
style of the ACES 1.3 Reference Gamut Compression (Academy S-2020-001): each
channel's distance from the achromatic axis (`max(r, g, b)`) is compressed
with a smooth parametric knee, using the published ACES thresholds
(0.815 / 0.803 / 0.880) and power (1.2), with the per-direction limits
re-derived from the Rec.2020 hull so that every Rec.2020 color compresses
into Rec.709 (the ACES camera-gamut limits under-cover Rec.2020's cyan
direction). Key properties:

- colors below the threshold (everything that isn't highly saturated) pass
  through **bit-identically**;
- the mapping is smooth and monotonic — no banding or gradient reversal at
  the gamut boundary, and distinct out-of-gamut colors stay distinct instead
  of collapsing onto the gamut hull;
- `max(r, g, b)` is preserved exactly, and hue is approximately preserved
  (the cheap, closed-form trade-off ACES standardized; an exact
  constant-hue ICtCp boundary search is a possible future upgrade);
- a `max(0)` safety clip still runs afterwards (PQ encoding requires
  non-negative input).

The behavior is controlled by the new `DisplayGamutCompression` resource:
`Auto` (default — compress exactly when the gamut stage can produce
out-of-gamut colors), `Always` (force it on), or `Clip` (the hue-shifting
per-channel clip, kept as a debug fallback for A/B comparison via the
`DISPLAY_GAMUT_CLIP_DEBUG` shader def).

Under `Auto` nothing changes today: the encoder's input is Rec.709
display-linear in every configuration, so both reachable gamut stages
(identity for scRGB, Rec.709 → Rec.2020 expansion for PQ) cannot produce
out-of-gamut colors and the compression stays off. It becomes load-bearing
when tone-mapping operators start emitting wide-gamut output for HDR targets
(the Rec.2020-working → scRGB path). A CPU mirror of the shader algorithm,
with constants, citations, and tests, lives in
`bevy_core_pipeline::display_encoding::gamut_compression`.

SDR rendering is untouched: views on sRGB display targets never run the
display-encoding pass at all.
