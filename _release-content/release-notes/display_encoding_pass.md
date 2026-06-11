---
title: "Display-encoding pass: gamut transform and HDR transfer functions"
authors: ["@pavlov"]
pull_requests: []
---

Bevy's display pipeline is now structured as the industry-standard separated
chain: tone mapping (scene-referred → display-linear, where `1.0` = paper
white) → gamut transform (working primaries → display primaries) → transfer
encoding (linear light → display signal). The last two stages are implemented
by a new **display-encoding pass** in `bevy_core_pipeline::display_encoding`,
scheduled after the UI pass (UI keeps compositing in display-linear,
paper-white-relative space, so a white UI panel lands exactly at
`DisplayTarget::paper_white_nits`) and before the final upscaling blit.

The first stage is now unified, too: tone mapping always runs as the
post-process tonemapping pass — the legacy in-shader (`TONEMAP_IN_SHADER`)
path that non-`Hdr` cameras used is gone, so every camera with an active
`Tonemapping` operator renders into a scene-linear `Rgba16Float` intermediate,
blends transparents in linear light, and tone-maps once. This also made the
screen-space-transmission background snapshot exact for every camera (the old
`approximate_inverse_tone_mapping` hack has been deleted). See the migration
guides for the visual implications on previously-SDR cameras.

The pass reads the per-view `DisplayTarget` calibration and performs:

- a full-precision gamut transform from the view's *source* primaries — the
  tone-map operator's output gamut, resolved per view: Rec.2020 for
  `Tonemapping::GranTurismo7` on HDR targets, whether authored or substituted
  for an SDR-only operator (the operator emits its native Rec.2020 output
  directly, with no intermediate Rec.709 conversion or clamp), Rec.709
  otherwise — to the display signal's
  primaries: a Rec.709 → Rec.2020 expansion for PQ, the Rec.2020 → Rec.709
  contraction for GT7-on-scRGB, identity otherwise,
- out-of-gamut handling: ACES-RGC-style perceptual gamut compression when the
  gamut transform can produce out-of-gamut colors (exactly the GT7
  Rec.2020 → scRGB contraction under the default setting), with a
  per-channel-clip debug fallback (see the display gamut compression release
  note),
- the display transfer function: scRGB-linear (`× paper_white_nits / 80`) or
  PQ/ST-2084 (absolute nits normalized to 10000), selected by
  `DisplayTransfer`.

The tonemapping pipeline and the encoder derive the per-view source gamut
from a single shared predicate (`tonemap_output_gamut`), so the two passes
can never disagree about the buffer's primaries. One documented limitation
of the Rec.2020-native path: UI composites Rec.709-authored colors into the
post-tonemap buffer unconverted, so on GT7-HDR views (a Rec.2020 buffer)
saturated UI colors oversaturate slightly; grays and whites are unaffected
(shared D65 white point). Per-view UI gamut conversion is deferred to the UI
HDR follow-up.

The shader-side transfer functions themselves — exact piecewise sRGB
OETF/EOTF, scRGB scaling, PQ inverse-EOTF/EOTF, and the HLG OETF — live in a
new importable WGSL library, `bevy_render::transfer_functions`, with CPU
mirrors (and parity tests) in the `bevy_render::transfer_functions` Rust
module.

**Nothing changes for SDR rendering.** Views on the default
`DisplayTarget::SDR_SRGB` (or any sRGB-transfer target) never run the pass —
the node early-returns without recording GPU work, and the exact sRGB encode
remains the free hardware conversion on the swapchain writeback, byte-identical
to previous releases. The pass only activates for display targets whose
*resolved* transfer is HDR (`ScRgbLinear`, or `Pq` once wgpu can negotiate it),
which is exactly when surface selection has configured a matching HDR
swapchain — see the scRGB HDR output release note.
