---
title: "Perceptual out-of-gamut compression in the display-encoding pass"
authors: ["@stuartparmenter"]
pull_requests: []
---

When the display-encoding pass maps colors from a wide working gamut into a
narrower display gamut, some colors simply don't fit — they land outside what
the display can show. Previously those colors were clipped per channel, which
collapses saturation unevenly and shifts hue: a vivid orange can read as a
duller, slightly different orange (the classic
`(1500, 1200, 500) → (1000, 1000, 500)` problem).

The pass now compresses out-of-gamut colors smoothly toward the achromatic axis
instead, in the style of the ACES 1.3 Reference Gamut Compression (Academy
S-2020-001). In practice this means:

- colors already inside the display gamut (everything that isn't highly
  saturated) pass through unchanged;
- the most saturated colors are eased back to the gamut boundary along a smooth
  curve, so there's no banding or sudden reversal at the edge, and distinct
  bright colors stay distinct instead of all collapsing onto the same boundary;
- brightness and hue are preserved as closely as the closed-form mapping allows,
  so the result looks like the intended color, just a touch less saturated.

You control this with the new `DisplayGamutCompression` resource: `Auto` (the
default — compress only when the gamut stage can actually produce out-of-gamut
colors), `Always` (force it on), or `Clip` (the old hue-shifting per-channel
clip, kept as a debug fallback for A/B comparison).

Under `Auto`, compression kicks in for exactly one configuration: a
`Tonemapping::GranTurismo7` camera presenting to an scRGB HDR target, where the
operator's wide Rec.2020 output has to be fit into the Rec.709-coordinate scRGB
signal. Every other reachable path can't produce out-of-gamut colors, so it
keeps the plain clip (a no-op there).

SDR rendering is untouched: views on sRGB display targets never run the
display-encoding pass at all.
