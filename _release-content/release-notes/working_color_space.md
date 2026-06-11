---
title: "Wide working color space (Rec.2020, opt-in)"
authors: ["@stuartparmenter"]
pull_requests: []
---

Bevy's scene-referred rendering has always used linear Rec.709 (the sRGB
primaries) as its working color space — implicitly. That axis is now explicit
and configurable: `RenderPlugin` has a new `working_color_space` field of type
`bevy_render::working_color_space::WorkingColorSpace`.

```rust
use bevy::render::{working_color_space::WorkingColorSpace, RenderPlugin};

App::new().add_plugins(DefaultPlugins.set(RenderPlugin {
    working_color_space: WorkingColorSpace::Rec2020,
    ..default()
}));
```

The default, `WorkingColorSpace::Rec709`, is bit-for-bit identical to previous
Bevy releases. Opting into `WorkingColorSpace::Rec2020` switches the
scene-referred buffers and lighting math to the wide-gamut ITU-R BT.2020
primaries (D65 white point throughout), which cover roughly twice as much of
the visible gamut as Rec.709 — including most real-world saturated colors
(car paints, neon lights, lasers) that Rec.709 simply cannot represent with
non-negative components. Rec.2020 is also the native space of the new
`Tonemapping::GranTurismo7` operator, which consumes the wide working space
directly without any conversion.

This is a project-global, startup-time setting (Unreal-style "working color
space" project setting): shared assets and buffers make per-camera working
spaces impractical, so all cameras share one set of primaries. It is read
once when `RenderPlugin` builds; mutating the resource afterwards has no
effect.

What happens under `Rec2020`:

- Light colors (point/spot/directional/rect), ambient light, distance fog and
  clear colors convert Rec.709 → Rec.2020 on the CPU at their extract/prepare
  seams, through one shared helper
  (`bevy_render::working_color_space::linear_rgba_rec709_to_working`).
- Color quantities composed in shaders from Rec.709 factors — PBR base color
  and emissive (material factor × texture × vertex color), lightmap samples,
  environment-map radiance, skybox samples, sprite / `ColorMaterial` /
  tilemap colors — convert once in the shader, after composition, under the
  new global `WORKING_COLOR_SPACE_REC2020` shader def. All sampled color
  textures are assumed to be authored against Rec.709 (the overwhelmingly
  common case; this also covers compressed textures, which cannot be
  converted on the CPU). Textures stamped with wide primaries via
  `Image::source_primaries` do not yet have a per-texture escape hatch — the
  stamp is now propagated to `GpuImage::source_primaries` so one can be added
  later.
- The Gran Turismo 7 tone mapping operator runs natively on the Rec.2020
  scene values. Every other operator (and the color-grading stack) is
  Rec.709-fit — the AgX / Tony McMapface / Blender Filmic LUTs have no
  algorithmic source to rebake from — so the tone mapping pass converts
  Rec.2020 → Rec.709 at its entry for them, clipping colors outside the
  Rec.709 gamut.
- The tone mapping pass outputs Rec.709 display-linear for every operator
  **except** `Tonemapping::GranTurismo7` on an HDR-transfer display target:
  there the operator emits its native linear Rec.2020 output straight into
  the display encoding pass, which transforms it per the display's gamut —
  identity for PQ/Rec.2020 signals, a hue-preserving gamut compression for
  Rec.709-coordinate scRGB signals (see the Gran Turismo 7 tonemapping,
  display encoding pass, and display gamut compression release notes, all in
  this release). So the full wide-gamut display payoff — Rec.2020 values
  flowing through to the HDR display signal — is live on GT7 HDR views,
  while every other configuration keeps Rec.709 output and UI compositing,
  FXAA/SMAA, and the display encoding pass behave as before. On GT7 HDR
  views, UI composites its Rec.709-authored colors unconverted and saturated
  UI colors can oversaturate (see the GT7 note for the documented
  limitation).

Known caveats under `Rec2020`: `Tonemapping::None` cameras (the `Camera2d`
default) skip the conversion and render oversaturated (a `warn_once` fires);
`CompositingSpace::Oklab` and bloom luminance weights remain Rec.709-fit;
gizmos, UI and atmosphere-generated sky values are not converted.
