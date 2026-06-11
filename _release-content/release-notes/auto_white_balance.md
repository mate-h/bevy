---
title: Automatic white balance
authors: ["@stuartparmenter"]
pull_requests: []
---

Bevy now ships an automatic white balance: it estimates the scene's dominant illuminant from
what the camera sees and slowly shifts the white point toward neutral, so a scene lit by warm
tungsten or cool daylight no longer carries a permanent color cast. It is modeled on the system
Polyphony Digital presented for Gran Turismo 7 at SIGGRAPH 2025 ("Physically Based Tone Mapping
in Gran Turismo 7").

Add the new `AutoWhiteBalance` component to a camera (with the `AutoExposurePlugin`), and the
renderer estimates the scene's dominant illuminant and slowly adapts the white point towards
neutral, the way your eyes (and your phone camera) do:

```rust
commands.spawn((
    Camera3d::default(),
    AutoWhiteBalance::default(),
));
```

Following Gran Turismo 7's design:

* **Shared metering** — the measurement rides along in the auto exposure compute pass: the
  same metering-mask weights that build the luminance histogram also accumulate a
  luminance-weighted average of the scene chromaticity in Yxy space (the blend space
  Polyphony found to tune best). One dispatch serves both adaptations; `AutoWhiteBalance`
  also works without `AutoExposure` on the camera.
* **Dark-scene stability** — a faint *virtual light* (an ideal D65 source with a configurable
  luminance) is blended into the measurement as one more luminance-weighted reference. Its
  influence scales with the inverse of the scene luminance, so near-black scenes calmly
  anchor at neutral instead of chasing measurement noise.
* **Bounded output** — the adapted chromaticity is converted to a correlated color
  temperature (McCamy's approximation) plus an off-locus tint, and the temperature is clamped
  to the 2500 K–7000 K range of typical real camera AWB specifications, so deliberate extreme
  lighting is never fully "corrected" away.

The correction is applied through Bevy's existing CAM16 white-balance machinery and *composes*
with manual `ColorGrading` temperature/tint: the automatic correction neutralizes the scene
first, and the artist's grade applies on top — a deliberate warm look stays warm.

Try it in the updated `auto_exposure` example: press `L` to switch the room light to tungsten,
then `B` to watch the camera adapt the orange cast away.
