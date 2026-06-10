---
title: Gran Turismo 7 tonemapping
authors: ["@pavlov"]
pull_requests: []
---

Bevy has a new tone-mapping operator: `Tonemapping::GranTurismo7`, a native port of the
operator Polyphony Digital uses in Gran Turismo 7, published with their SIGGRAPH 2025 course
"Physically Based Tone Mapping in Gran Turismo 7" (reference implementation MIT licensed,
Copyright (c) 2025 Polyphony Digital Inc.).

The operator blends a per-channel filmic curve — a power-curve toe, an exactly-linear middle
section, and a convergent exponential shoulder — with a hue-preserving ICtCp branch
(60% hue-preserving / 40% per-channel by default), plus a luminance-driven chroma fade so
colors gracefully desaturate as they approach peak white instead of clipping to hard white.
The result is a "camera-like" highlight response that still keeps hues stable.

```rust
commands.spawn((
    Camera3d::default(),
    Tonemapping::GranTurismo7,
));
```

Unlike the LUT-based operators (AgX, TonyMcMapface, BlenderFilmic), GranTurismo7 is fully
algorithmic and does not require the `tonemapping_luts` cargo feature.

The operator is natively peak-luminance aware and designed to drive both SDR and HDR
displays; today Bevy wires up its SDR mode (tone-mapping against Gran Turismo's 250-nit
paper-white calibration and rescaling into the sRGB range), with the full HDR output path
arriving together with Bevy's broader HDR display support.

A new `GranTurismo7Params` component exposes the operator's artistic dials (`blend_ratio`,
the chroma fade band, and the curve shape parameters). Adding it to a camera that uses
`Tonemapping::GranTurismo7` uploads the (validated) parameters to the GPU each frame,
replacing the operator's baked defaults:

```rust
commands.spawn((
    Camera3d::default(),
    Tonemapping::GranTurismo7,
    GranTurismo7Params {
        blend_ratio: 1.0, // fully hue-preserving
        ..Default::default()
    },
));
```

When the camera renders to a target whose `DisplayTarget` requests an HDR transfer
(scRGB-linear, PQ, or HLG), the operator is additionally configured in its HDR mode: the
tone curve is rebuilt around the display's `peak_luminance_nits` (clamped to the operator's
supported 250–10000 nit range, and to at least `paper_white_nits`), and the output is
rescaled so `1.0` equals the display's paper white. Note that until the HDR display encoder
lands, this HDR-mode output is still converted back to Rec.709 and clamped for the existing
sRGB output chain, so highlights above paper white are clipped on screen for now. Cameras
without the component (or on plain SDR targets without it) are completely unaffected.
