---
title: Physiological two-stage auto exposure
authors: ["@stuartparmenter"]
pull_requests: []
---

Bevy's `AutoExposure` can now model how human vision adapts to brightness on two time scales,
following the two-stage model Polyphony Digital presented for Gran Turismo 7 at SIGGRAPH 2025
("Physically Based Tone Mapping in Gran Turismo 7"). A short-term stage (pupil and neural gain)
covers a few EV within seconds; a long-term stage (receptor sensitivity and photopigment
bleaching) covers the rest of the ~12 EV range over minutes, asymmetrically — adapting to light
is much faster than to darkness. The long-term stage *bounds* the short-term one: walk from
daylight into a cave and the scene brightens a little immediately but stays dark until you've
truly adapted.

Bevy's existing smoothing is the short-term stage. The new opt-in `physiological` setting adds
the long-term stage: a slow, asymmetric adaptation envelope tracked per camera on the GPU that
clamps the short-term exposure to a bounded range around itself.

```rust
commands.spawn((
    Camera3d::default(),
    AutoExposure {
        physiological: Some(PhysiologicalAdaptation::default()),
        ..default()
    },
));
```

Defaults are tuned to real physiological time scales; games will often want faster values.
`PhysiologicalAdaptation` exposes the long-term speeds (`speed_brighten`/`speed_darken`, EV per
second per direction), the bounding range (`bound_brighten`/`bound_darken`, EV below/above the
envelope), and an optional `initial_long_term_ev` to start a camera already adapted to a known
environment.

`AutoExposure` also gains the first piece of GT7-style multi-reference metering: a
`metering_bias` EV offset applied at the metering seam, plus an optional
`AutoExposureExternalReference` component that blends an externally computed scene-brightness
estimate (e.g. from sun illuminance or an environment map) into the frame-buffer histogram as a
weighted average. Engine-computed sky/sun/light-probe references are planned follow-ups feeding
this same seam.

Try it in the `auto_exposure` example with the `P` key.
