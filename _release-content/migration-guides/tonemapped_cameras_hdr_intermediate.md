---
title: Cameras with active tone mapping now render to an `Rgba16Float` intermediate
pull_requests: []
---

Cameras whose `Tonemapping` is anything other than `Tonemapping::None` now
render to an `Rgba16Float` intermediate main texture, even without the `Hdr`
marker component. Previously only `Hdr` cameras got the high-precision
intermediate; SDR cameras rendered through an 8-bit (`Rgba8UnormSrgb` /
`Rgba8Unorm`) main texture.

This is a **deliberate rendering change** in preparation for HDR display
output: tone mapping and display encoding now run as node-side post-process
passes that need a high-precision, scene-linear-capable buffer for every
tone-mapped camera (see the "Tone mapping always runs as a post-process pass"
guide for the tone-mapping move itself). One visible consequence is reduced
intermediate quantization: banding that was previously introduced by an 8-bit
intermediate disappears.

What this means in practice:

- **Camera3d** (default `Tonemapping::TonyMcMapface`): now uses an
  `Rgba16Float` main texture. GPU memory for the camera's ping-pong main
  textures doubles (8 → 16 bytes per pixel). If you relied on the 8-bit
  intermediate, add `Tonemapping::None` (and tone-map yourself) or use
  `CompositingSpace::Srgb` without an active operator (see below).
- **Camera2d** (default `Tonemapping::None`): unchanged. Pixel-art and UI
  cameras keep their existing 8-bit path exactly.
- **Cameras with `Hdr`**: unchanged (already `Rgba16Float`).
- **Cameras with an explicit `CompositingSpace::Srgb`** and no active
  tone-mapping operator: unchanged — they keep the sRGB-compositing
  `Rgba8Unorm` main texture. With an active operator the camera gets the
  `Rgba16Float` intermediate too: shaders still write sRGB-encoded values and
  blending still happens in the encoded space, but fp16 storage keeps
  scene-referred values above 1.0 intact for the tonemapping pass (which
  decodes, tone-maps, and re-encodes).

The selection is carried by a new auto-managed marker component,
`bevy_camera::TonemappingEnabled`, which `TonemappingPlugin` keeps in sync
with each camera's `Tonemapping` component every frame (in `PostUpdate`). Do
not insert or remove it manually; treat it as read-only state. If you spawn
cameras with custom render graphs and matched against the exact set of
components on a camera entity, account for the new marker.

On WebGL2 this requires `EXT_color_buffer_float` (already required for `Hdr`
cameras and widely supported); platforms without renderable/blendable
`Rgba16Float` were already unable to use Bevy's `Hdr` path.
