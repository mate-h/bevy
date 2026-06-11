---
title: "`DisplayTarget`: per-display calibration for windows"
authors: ["@pavlov"]
pull_requests: []
---

Bevy now has a first-class description of the display a window is presented
on: the new `DisplayTarget` component in `bevy_window`.

`DisplayTarget` captures the calibration of a display target — how bright
"paper white" is, the peak luminance the display can reach, its black level,
its color gamut (`DisplayGamut`: Rec.709, Display P3, or Rec.2020), and the
transfer function the final signal should be encoded with (`DisplayTransfer`:
sRGB, scRGB-linear, PQ, or HLG). It is a required component of `Window`, so
every window automatically gets one. The default, `DisplayTarget::SDR_SRGB`
(paper white and peak of 100 nits, Rec.709, sRGB), reproduces Bevy's existing
SDR output exactly. It is the foundation that this release's HDR output
support is parameterized by: peak-aware tone mapping
(`Tonemapping::GranTurismo7`), wide-gamut Rec.2020 output through the display
encoding pass, and the `Rgba16Float` scRGB swapchain format (see the "HDR
display output (scRGB-linear)" release note for the end-to-end path).

In the render world, every camera view resolves its target's calibration into
a `ViewDisplayTarget` component (carrying both the `requested` calibration
and the `resolved` one after surface negotiation — see the scRGB HDR output
release note), and a per-view `DisplayTargetUniform`
(luminance values plus gamut/transfer indices, importable in WGSL as
`bevy_render::display_target`) is prepared each frame. The tonemapping pass
binds it only for views whose display target differs from the SDR default (or
whose operator needs it, like `Tonemapping::GranTurismo7` with per-camera
params) — views on default SDR targets keep pipelines byte-identical to
previous releases. The uniform parameterizes the GT7 operator's HDR mode and
the display-encoding (gamut-mapping and transfer-encoding) pass.

Render targets that aren't windows, such as `RenderTarget::Image` and
`RenderTarget::TextureView` (used by OpenXR), have no window entity to host
the component. For those, the new `ManualDisplayTargets` resource in
`bevy_render` maps a `NormalizedRenderTarget` to its `DisplayTarget`; targets
without an entry fall back to `DisplayTarget::SDR_SRGB`.

`DisplayTarget` is user-authoritative: Bevy never overwrites values you set.
To support reacting to a window being dragged to a different display, there is
also a new `WindowMonitorChanged` message, sent whenever the monitor a window
is on changes (including when it first becomes known):

```rust
fn react_to_monitor_change(
    mut events: MessageReader<WindowMonitorChanged>,
    monitors: Query<&Monitor>,
) {
    for event in events.read() {
        if let Some(monitor) = event.monitor.and_then(|m| monitors.get(m).ok()) {
            // Inspect the new monitor and decide whether to update the
            // window's `DisplayTarget`.
        }
    }
}
```
