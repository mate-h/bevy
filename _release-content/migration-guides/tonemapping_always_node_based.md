---
title: Tone mapping always runs as a post-process pass; the in-shader tonemapping path is removed
pull_requests: []
---

Bevy previously had two tone-mapping paths: cameras with the `Hdr` marker were
tone-mapped by the post-process tonemapping pass, while all other ("SDR")
cameras applied the `Tonemapping` operator *inside* the sprite, mesh2d, and PBR
fragment shaders (the `TONEMAP_IN_SHADER` shader def). The in-shader path has
been removed: **every camera whose `Tonemapping` is not `Tonemapping::None`
now tone-maps in the post-process pass**, with or without `Hdr`.

This is a **deliberate rendering change**, made so that tone mapping is a
single, well-defined stage (scene-referred input, display-referred output)
that the HDR display-output pipeline can build on. Expect small visual
differences on previously-SDR cameras with an active operator (the Camera3d
default):

- **Transparents now blend in scene-linear HDR, before tone mapping.**
  Previously each fragment was tone-mapped first and the *tone-mapped* values
  were alpha-blended. Blending before tone mapping is how `Hdr` cameras (and
  other engines) already behaved; bright backgrounds showing through
  semi-transparent geometry can look slightly different.
- **Everything a camera renders is now tone-mapped**, including gizmos,
  custom materials, and 2D primitives that previously did not call
  `tone_mapping()` themselves and so bypassed the operator on SDR cameras.
  Custom material shaders should output scene-linear color and let the pass
  tone-map.
- **Deband dithering (`DebandDither`) is applied once in the post-process
  pass** instead of per-fragment in material shaders. The dither math is
  unchanged; it now acts on the blended image.
- **`Tonemapping::None` is now a true passthrough.** The in-shader path used
  to apply `ColorGrading` exposure and post-saturation even with
  `Tonemapping::None`; now no color grading at all is applied without an
  active operator. If you used `ColorGrading` (or `AutoExposure`) with
  `Tonemapping::None`, pick an operator (`Tonemapping::Reinhard` is the
  cheapest) — grading is part of the tonemapping pass.
- **Screen-space transmission is more accurate.** The transmission pass
  snapshots the main texture before tone mapping, which is now scene-linear
  for every camera, so the crude `approximate_inverse_tone_mapping()`
  correction (which could not recover values above ~2 and was meaningless for
  LUT-based operators) has been deleted. Rough/refractive materials on
  previously-SDR cameras render with correct background radiance now.
- Cameras with an active operator render to an `Rgba16Float` intermediate
  (see the "Cameras with active tone mapping now render to an `Rgba16Float`
  intermediate" guide). With this change that also applies to cameras with an
  explicit `CompositingSpace::Srgb`: shaders still write sRGB-encoded values
  and blending still happens in the encoded space, but the storage is fp16 so
  scene-referred values above 1.0 survive until the tonemapping pass decodes,
  tone-maps, and re-encodes them.

There is no switch to get the old "tone-map-then-blend" look back. The
`Tonemapping::None` opt-out remains, and a camera with `Tonemapping::None`
keeps its previous main-texture format and runs no tonemapping pass at all;
if you need per-fragment tone mapping you can apply an operator in your own
material and set the camera to `Tonemapping::None`.

Performance notes: 2D cameras keep `Tonemapping::None` by default and are
unaffected. A 2D camera that opts into an operator now pays one fullscreen
post-process pass (plus the fp16 main-texture bandwidth) instead of a few
extra ALU instructions per fragment; for 3D cameras the pass replaces
per-fragment tone mapping and is usually cheaper on overdraw-heavy scenes.

## Removed API

The in-shader mechanism's plumbing was removed. If you maintain custom render
pipelines, the following are gone:

- `MeshPipelineKey::TONEMAP_IN_SHADER`, `MeshPipelineKey::DEBAND_DITHER`, and
  all `MeshPipelineKey::TONEMAP_METHOD_*` constants (and the same flags on
  `Mesh2dPipelineKey` and `SpritePipelineKey`).
- `bevy_pbr::tonemapping_pipeline_key` and
  `bevy_sprite_render::tonemapping_pipeline_key`.
- `MeshPipelineViewLayoutKey::TONEMAP_IN_SHADER` and the
  `TONEMAPPING_LUT_TEXTURE_BINDING_INDEX` / `TONEMAPPING_LUT_SAMPLER_BINDING_INDEX`
  constants in `bevy_pbr` — the tonemapping LUT (bindings 18/19) is no longer
  part of the mesh view bind group.
- The tonemapping LUT entries in the sprite and mesh2d view bind group
  layouts: the sprite view layout is now just the view uniform, and the
  mesh2d view layout is the view + globals uniforms.

WGSL changes:

- `sprite.wgsl`, `mesh2d.wgsl`, `color_material.wgsl`, `sprite_material.wgsl`,
  `pbr_functions.wgsl`, and `transmission.wgsl` no longer call
  `tone_mapping()` (or reference `TONEMAP_IN_SHADER` / the in-shader
  `DEBAND_DITHER` variant).
- `dt_lut_texture` / `dt_lut_sampler` were removed from
  `bevy_pbr::mesh_view_bindings`, `bevy_sprite::sprite_view_bindings`, and
  `bevy_sprite::mesh2d_view_bindings`.
- `approximate_inverse_tone_mapping()` was removed from
  `bevy_core_pipeline::tonemapping`.
- Custom material shaders that import `bevy_core_pipeline::tonemapping` and
  call `tone_mapping()` themselves will now be tone-mapped twice, and must
  also supply the `TONEMAPPING_LUT_TEXTURE_BINDING_INDEX` /
  `TONEMAPPING_LUT_SAMPLER_BINDING_INDEX` shader defs and matching LUT
  bindings themselves (the material pipelines no longer push them). The
  recommended migration is to delete the call and output scene-linear color.
