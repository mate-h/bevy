---
title: Physically based veiling glare (GT7-style) as a bloom scatter model
authors: ["@pavlov"]
pull_requests: []
---

`Bloom` has a new `scatter` field selecting how the blurred pyramid levels
are weighted when composited back onto the image:

- `BloomScatterModel::Aesthetic` (the default) is the hand-tuned parametric
  curve Bevy's bloom has always used — existing scenes render exactly as
  before.
- `BloomScatterModel::Gt7Glare { f_number }` replaces the curve with
  per-level weights derived from the far-field (Fraunhofer) diffraction
  point-spread function of a camera aperture — the physically based veiling
  glare design Polyphony Digital presented for Gran Turismo 7 at SIGGRAPH
  2025, where glare *is* the bloom and the aperture F-number drives how the
  light spreads.

```rust
commands.spawn((
    Camera3d::default(),
    // or: Bloom { scatter: BloomScatterModel::Gt7Glare { f_number: 8.0 }, ..default() }
    Bloom::GT7_GLARE,
));
```

The per-level weights are the photopically weighted energy of the
polychromatic Airy pattern integrated over each pyramid level's annulus of
blur radii, precomputed for the standard f/1–f/22 full-stop ladder and
interpolated per F-stop in between. Small F-numbers (wide apertures) give a
tight glare that falls off steeply around bright sources; stopping down to
f/22 spreads the energy into a wide, soft veil, exactly as diffraction does
on a real camera. GT7's own 240 hand-calibrated weights are unpublished, so
Bevy derives its table from the documented physical model rather than
cloning constants — see the `bloom::glare` module documentation for the full
derivation and references.

Because a physical point-spread function applies to *all* light, the glare
model is threshold-free: any configured `BloomPrefilter` is ignored (with a
warning), and compositing is forced to energy-conserving blending.
`Bloom::intensity` keeps its meaning as the total fraction of energy
scattered out of the sharp image, so the artistic exposure dial and the
physical distribution dial stay independent. The whole effect reuses the
existing bloom downsample/upsample pyramid — switching models costs nothing
beyond different blend constants.

Try it in the `bloom_3d` example: `B` toggles the scatter model and `O`/`L`
walk the aperture through the F-stop ladder.
