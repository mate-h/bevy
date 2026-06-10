---
title: Physiological two-stage auto exposure
authors: ["@pavlov"]
pull_requests: []
---

Bevy's `AutoExposure` can now model how human vision *actually* adapts to brightness changes,
following the two-stage physiological model Polyphony Digital presented for Gran Turismo 7 at
SIGGRAPH 2025 ("Physically Based Tone Mapping in Gran Turismo 7").

Human adaptation happens on two very different time scales: a short-term stage (pupil and
neural gain) that covers a few EV within seconds, and a long-term stage (receptor sensitivity
and photopigment bleaching) that covers the rest of the roughly 12 EV adaptation range over
minutes to tens of minutes — and asymmetrically, since adapting to light is much faster than
adapting to darkness. Crucially, the long-term stage *bounds* the short-term stage: walking
from daylight into a cave, your eyes brighten the scene a little right away, but the cave
stays dark until you've truly adapted.

Bevy's existing auto exposure smoothing is the short-term stage. The new opt-in
`physiological` setting adds the long-term stage on top: a slow, asymmetric adaptation
envelope tracked per camera on the GPU, which clamps the short-term exposure into a bounded
range around itself. Because the long-term speeds are asymmetric, the same scene brightness
reads differently at dawn than at dusk — the eye hasn't recovered on one side.

```rust
commands.spawn((
    Camera3d::default(),
    AutoExposure {
        physiological: Some(PhysiologicalAdaptation::default()),
        ..default()
    },
));
```

The defaults are tuned to real physiological time scales; games will often want faster,
gameplay-friendly values, and `PhysiologicalAdaptation` exposes the long-term speeds
(EV per second, per direction), the bounding range (EV above/below the envelope), and an
optional initial adaptation state (e.g. to start a camera already adapted to bright
daylight). When disabled (the default), the auto exposure behavior is unchanged.

`AutoExposure` also gains the first piece of GT7-style multi-reference metering: a
`metering_bias` EV offset, plus a new optional `AutoExposureExternalReference` component
that blends an externally computed scene-brightness estimate (e.g. derived from sun
illuminance or an environment map) into the frame-buffer histogram measurement as a weighted
average. Engine-computed sky/sun/light-probe references are planned follow-ups that will
feed this same seam.

Try it in the `auto_exposure` example with the `P` key.
