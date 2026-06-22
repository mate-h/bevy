---
title: "Camera compositing resolves once per frame: `ViewTarget::compositing_space` removed, stacks share one space"
pull_requests: []
---

A camera's `CompositingSpace` is a *request*, just like the transfer function on
its `DisplayTarget`. Several render stages used to each re-derive, on their own,
what that request meant for a camera once its neighbors on a shared render target
were taken into account. Bevy now resolves all of it **once per frame**, in two
phases, into state that every consumer reads instead of re-deriving:

- `ResolvedCompositionSpaces` (a render-world resource, built in
  `RenderSystems::CreateViews` after `sort_cameras`) holds one
  `Option<CompositingSpace>` per camera view, after grouping cameras that share a
  main-texture ping-pong.
- `ViewStackContract` (a render-world component, built in
  `RenderSystems::PrepareViews` after `prepare_view_targets` and
  `prepare_view_display_targets`) holds each view's tone-map and display-encode
  roles, its blit disposition, its resolved compositing space, the source gamut
  the encoder should use, and the resolved encode parameters.

Most projects need no changes: a solo camera resolves to its own request, so its
pipeline keys and rendered pixels are byte-identical. The changes below affect
custom render code that read the old per-view state, and projects that used
camera stacks with a non-default compositing space or HDR output.

## Removed API

- **`ViewTarget::compositing_space` is removed.** Read the resolved space from
  `ResolvedCompositionSpaces` (keyed by the render-world view entity) in
  `bevy_render`, or from the `ViewStackContract::compositing_space` field in
  `bevy_core_pipeline`. The raw per-camera request is still available as
  `ExtractedCamera::compositing_space`, but it feeds only the extract-time
  main-texture format choice and should not be consulted for pass or pipeline-key
  decisions.

  ```rust
  // Before:
  // let space = view_target.compositing_space;

  // After (in bevy_render, needs `Res<ResolvedCompositionSpaces>` and the view entity):
  let space = resolved_spaces.get(entity, camera.compositing_space);

  // After (in bevy_core_pipeline, from the per-view contract):
  let space = contract.compositing_space;
  ```

- **`encoder_input_gamut` is removed** from `bevy_core_pipeline::display_encoding`.
  The encoder's source gamut now comes from `ViewStackContract::source_gamut`.
  `tonemap_output_gamut` remains the single source of truth for an operator's
  output gamut and is what the resolver calls.

## Behavior changes

These follow the deliberate-rendering-change tier already used by the
node-based-tonemapping work; they affect non-default compositing-space and HDR
configurations, never default SDR cameras.

- **Camera stacks resolve one compositing space per shared buffer.** Cameras that
  share a main texture and form a stack (each later camera composites full-screen
  over the previous output) now resolve to a single compositing space. Previously
  each camera's request was honored per-view, which could silently mis-decode the
  shared buffer; now conflicting requests, or any non-2D-camera member, resolve to
  linear with a warning naming the misconfiguration. Give every member of a stack
  the same `CompositingSpace`, or none.

- **Mixed-HDR camera stacks on one target order deterministically.** A
  default-tone-mapped 3D camera and an `Hdr` 2D overlay rendering to the same
  target previously both received sorted-camera index 0, producing a
  nondeterministic stack order and replace-over-replace upscaling blits (the upper
  camera overwrote the lower instead of compositing). Sorted-camera indices now
  count per render target, so the stack orders deterministically and the overlay
  composites over the base.

- **Stack members below the finalizer no longer run upscaling blits.** When a
  stack defers its tone-map or display-encode pass to a finalizing camera, every
  member ordered below that finalizer — deferred members and pass-disabled
  members alike — now skips its upscaling blit, and the finalizer presents once,
  with replace, owning the output texture. Previously each deferred member blit
  its un-finalized (un-tone-mapped, and un-encoded on HDR) buffer and the
  finalizer alpha-blended its encoded result over those pixels, which was visible
  wherever the composed alpha was below 1. On HDR display targets the out-texture
  clear color is now CPU-encoded through the resolved transfer, gamut, and paper
  white, so regions no camera covers present a correct clear instead of a raw
  linear value (which read as full-peak nits). A finalizer with
  `CameraOutputMode::Skip` cancels the skipping for its group, keeping the
  previous blits.

- **GT7 stacked with an overlay encodes correctly on HDR output.** When
  `Tonemapping::GranTurismo7` drives a 3D camera that is composited under a
  `Tonemapping::None` 2D overlay on a PQ display target, the encoder previously
  keyed its source gamut off the *overlay* camera's operator (Rec.709), applying a
  Rec.709 → Rec.2020 expansion to a buffer the GT7 camera had already produced in
  Rec.2020 — a double expansion that oversaturated the image. The encoder now keys
  off the camera that actually tone-mapped the buffer, so the expansion does not
  fire. See the composition-contract release note for the before/after capture.

- **UI on a `CompositingSpace::Srgb` or `CompositingSpace::Oklab` camera now
  encodes into that space.** UI runs after tone mapping, which leaves the buffer
  encoded in the camera's compositing space; UI previously wrote linear values
  into that encoded buffer, so the terminal decode mis-decoded them (darkened or
  corrupted output). Each UI fragment shader now writer-encodes its final color
  into the resolved compositing space, the same way the sprite shader does. UI
  over 3D cameras is unaffected (non-2D cameras always resolve to linear).

- **UI and gizmos now convert Rec.709-authored colors to the buffer's working
  gamut on a Rec.2020 (GT7) HDR view.** A UI or gizmo color authored in Rec.709
  used to be written verbatim into a post-tone-map buffer that held Rec.2020
  primaries, so saturated colors landed in wider primaries and oversaturated
  (grays and whites, sharing the D65 white point, were unaffected). UI fragment
  shaders, and the 2D and 3D gizmo line/joint shaders, now apply
  `rec709_to_rec2020` (the shared `WORKING_COLOR_SPACE_REC2020` writer-encode)
  before the compositing-space encode. UI keys this per view off the buffer's
  `source_gamut`; gizmos render pre-tone-map and key off the global
  `WorkingColorSpace` like sprites and meshes. SDR and Rec.709 views are
  byte-identical (no shader def, no conversion).

- **2D gizmos now writer-encode into the camera's compositing space.** Like UI
  and sprites, 2D gizmos blend into a `CompositingSpace::Srgb`/`Oklab` 2D buffer,
  so they now encode their output to match it; previously they wrote linear values
  and could look slightly off on those views. 3D gizmos render pre-tone-map and
  are unaffected (no compositing space). SDR/linear views are byte-identical.

- **FXAA on an Oklab-compositing view uses the Oklab L channel for edge luma.**
  FXAA's edge-detection luma was a Rec.601-weighted dot, which can go negative (and
  NaN under the square root) on an Oklab buffer's signed chroma channels. On a view
  that resolves to `CompositingSpace::Oklab`, FXAA now reads the Oklab L channel
  directly. Other compositing spaces are unchanged.
