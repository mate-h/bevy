---
title: "Tone mapping render path: node-based tone mapping and the `Rgba16Float` intermediate"
pull_requests: []
---

In 0.19, cameras with the `Hdr` marker tone-mapped in the post-process
tonemapping pass and rendered to an `Rgba16Float` main texture, while every
other ("SDR") camera applied its `Tonemapping` operator inside the sprite,
mesh2d, and PBR fragment shaders (the `TONEMAP_IN_SHADER` shader def) on an
8-bit (`Rgba8UnormSrgb` / `Rgba8Unorm`) main texture.

That in-shader fast path still exists. A tone-mapped camera (`Tonemapping` other
than `Tonemapping::None`) keeps it — per-fragment in-shader tone mapping, its
8-bit main texture, and no tonemapping node — as long as all of these hold
(`eligible_in_shader_tonemap` in `bevy_render/src/camera.rs`):

- tone mapping enabled and no `Hdr`,
- no scene-linear post-processing (`NeedsSceneLinearPost`) or reprojecting
  anti-aliasing such as TAA (`NeedsSceneLinearAa`),
- no `NeedsNodeTonemapping` (e.g. `Tonemapping::GranTurismo7` with a
  `GranTurismo7Params` component),
- `CompositingSpace` absent or `Linear`,
- resolved `DisplayTarget` is `DisplayTarget::SDR_SRGB`,
- the render target is a `Window`,
- it is the sole active camera on that target.

The default `Camera3d` (`TonyMcMapface`, no `Hdr`, sole camera on a plain SDR
sRGB window) meets these conditions: it keeps its 8-bit path and memory
footprint and is unchanged from 0.19.

A tone-mapped camera that fails any condition instead tone-maps in the
post-process pass and renders to a scene-linear `Rgba16Float` intermediate.
Triggers: the `Hdr` marker, scene-linear post-processing or anti-aliasing,
`NeedsNodeTonemapping`, a non-`Linear` `CompositingSpace`, an HDR display
target, a non-window render target, or sharing the target with another active
camera. For those cameras, expect small visual differences:

- Transparents blend in scene-linear before tone mapping, instead of
  tone-mapped values being alpha-blended.
- Everything the camera renders is tone-mapped, including gizmos and custom
  materials that did not call `tone_mapping()` themselves. Custom material
  shaders should output scene-linear color and let the pass tone-map.
- `DebandDither` is applied once in the pass, on the blended image, rather than
  per fragment.
- Stacked cameras rendering to the same target (`ClearColorConfig::None`, no
  viewport) compose in scene-linear and tone-map once on the last camera;
  earlier cameras' operators do not run (Bevy warns if they differ).
- The fp16 intermediate also removes the banding the old 8-bit intermediate
  introduced.

Whether a camera tone-maps at all is tracked by a new auto-managed marker,
`bevy_camera::TonemappingEnabled` (present whenever `Tonemapping` is not
`Tonemapping::None`), kept in sync each frame in `PostUpdate`. Treat it as
read-only; do not insert or remove it. If you match custom render graphs against
the exact set of components on a camera entity, account for the new marker. On
WebGL2 the fp16 path requires `EXT_color_buffer_float` (already required for
`Hdr` cameras and widely supported).

`Tonemapping::None` is a true passthrough: no tonemapping pass runs and no
`ColorGrading` exposure or post-saturation is applied. If you used `ColorGrading`
or `AutoExposure` with `Tonemapping::None`, switch to `Tonemapping::Linear`,
which applies grading and dither with no tone curve.
