---
title: Gran Turismo 7 tonemapping
authors: ["@stuartparmenter"]
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
displays, and Bevy wires up both modes end to end: on SDR targets it tone-maps against
Gran Turismo's 250-nit paper-white calibration and rescales into the sRGB range; on HDR
targets it runs in HDR mode and feeds the display-encoding pass directly (see below).

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

When the camera renders to a target whose resolved `DisplayTarget` requests an HDR transfer
(scRGB-linear, or PQ once wgpu can negotiate it), the operator is configured in its HDR mode
— with or without the `GranTurismo7Params` component (the defaults are used if it is
absent), matching the source implementation, which initializes HDR mode directly from the
target's peak: the tone curve is rebuilt around the display's `peak_luminance_nits` (clamped
to the operator's supported 250–10000 nit range, and to at least `paper_white_nits`), and
the output is rescaled so `1.0` equals the display's paper white.

On those HDR views the full end-to-end pipeline is now active: the operator emits its
**native linear Rec.2020 display-referred output** — unclamped, `[0, peak / paper_white]` —
straight into the display-encoding pass (no intermediate Rec.709 conversion and no clamp, so
no highlight or wide-gamut information is lost between the operator and the display signal).
The encoder's gamut stage then becomes a true source → display transform:

- **PQ / Rec.2020 targets**: identity — GT7's output is already in the signal's primaries.
- **scRGB targets** (Rec.709-coordinate by definition): a full-precision Rec.2020 → Rec.709
  conversion, with the ACES-RGC-style perceptual out-of-gamut compression now active by
  default (`DisplayGamutCompression::Auto`) so wide-gamut colors compress gracefully instead
  of clipping per channel.

One documented limitation of the Rec.2020-native path: Bevy UI composites its
Rec.709-authored colors into the post-tonemap buffer unconverted, so on a GT7-HDR view
saturated UI colors are reinterpreted in the wider primaries and oversaturate slightly
(grays and whites are unaffected — the white point is shared). Per-view UI gamut conversion
is planned as part of the UI HDR follow-up (see `plans/ui-hdr-rfc.md` in the working notes).

On plain SDR targets, cameras without the component are completely unaffected, and SDR
output remains byte-identical.
