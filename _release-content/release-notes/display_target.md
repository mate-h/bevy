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
SDR output exactly. It is the foundation that upcoming HDR output support
(peak-aware tone mapping, wide-gamut output, and HDR swapchain formats) will
be parameterized by.

In the render world, every camera view resolves its target's calibration into
a `ViewDisplayTarget` component, and a per-view `DisplayTargetUniform`
(luminance values plus gamut/transfer indices, importable in WGSL as
`bevy_render::display_target`) is prepared each frame. The tonemapping pass
binds it only for views whose display target differs from the SDR default (or
whose operator needs it, like `Tonemapping::GranTurismo7` with per-camera
params) — views on default SDR targets keep pipelines byte-identical to
previous releases. Today the uniform parameterizes the GT7 operator's HDR
mode; the upcoming gamut-mapping and transfer-encoding passes consume it
next.

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
