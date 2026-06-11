---
title: "`RenderPlugin::working_color_space` and `GpuImage::source_primaries`"
pull_requests: []
---

`RenderPlugin` has a new public field, `working_color_space:
bevy_render::working_color_space::WorkingColorSpace`. If you construct
`RenderPlugin` with functional-record-update syntax (`..default()`), no change
is needed. If you construct it field-by-field, add the new field; the default
`WorkingColorSpace::Rec709` preserves the previous rendering bit-for-bit.

A `WorkingColorSpace` resource is now present in both the main and render
worlds. It is read exactly once, when `RenderPlugin` builds; changing the
resource at runtime has no effect (render pipelines are specialized against it
at startup).

Several render-world systems and pipeline resources grew a
`working_color_space` parameter/field. If you construct any of these directly
(rare), add the new field/argument:

- `MeshPipeline`, `PrepassPipeline`, `Mesh2dPipeline`, `SpritePipeline` (new
  `working_color_space` field).
- `GpuImage` has a new `source_primaries: bevy_image::SourceColorPrimaries`
  field, propagated from `Image::source_primaries` at `prepare_asset` time.
  When constructing a `GpuImage` manually, stamp the source image's value (or
  `Default::default()`, which is `Bt709`).

If you opt into `WorkingColorSpace::Rec2020` (new, opt-in — nothing changes
otherwise):

- Scene-linear buffers, light/fog/clear colors and composed material colors
  hold linear Rec.2020 values; custom materials and custom render passes that
  inject Rec.709 colors into the scene should convert them with
  `bevy_render::working_color_space::linear_rgba_rec709_to_working` (CPU) or
  the `bevy_render::working_color_space` WGSL module's `rec709_to_rec2020`
  under the `WORKING_COLOR_SPACE_REC2020` shader def (pushed globally into
  the mesh/2D/sprite/tonemapping pipelines when Rec.2020 is active).
- Every camera needs an active `Tonemapping` operator: the Rec.2020 → display
  conversion runs in the tonemapping pass, so `Tonemapping::None` cameras
  (the `Camera2d` default) render reinterpreted, desaturated colors (a
  `warn_once` diagnoses this). Use `Tonemapping::Linear` for the conversion
  with no tone curve — the usual choice for 2D and UI cameras.
- Tone mapping operators other than `Tonemapping::GranTurismo7` are
  Rec.709-fit and clip working-space colors outside the Rec.709 gamut at the
  tone mapping pass entry.
- Some parts of the renderer are not yet converted and stay Rec.709-fit:
  `CompositingSpace::Oklab` and the bloom luminance weights; clustered decals
  and irradiance volumes; the `specular_tint` and clearcoat tint material
  inputs; `bevy_solari`; and gizmos, UI, and atmosphere-generated sky values.
- `LinearRgba` (and the rest of `bevy_color`) remains defined as linear
  Rec.709; the conversion to the working space happens exactly once, at the
  render-world seams above. Do not pre-convert colors you hand to standard
  Bevy APIs.
