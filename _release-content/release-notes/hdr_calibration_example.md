---
title: "HDR calibration example"
authors: ["@stuartparmenter"]
pull_requests: []
---

HDR output is only as good as the calibration values in the window's
`DisplayTarget`, and no operating system reliably reports them (neither winit
nor wgpu exposes display luminance metadata yet). The new
`examples/3d/hdr_calibration.rs` example shows the practical answer — an
in-app, HGIG-style calibration flow — and doubles as a reference for shipping
an "HDR settings" screen in your own game:

- **Peak luminance**: a near-peak checkerboard behind a center patch that
  tracks `peak_luminance_nits`; if the patch is already invisible, lower the
  value until it appears, then raise it until it just disappears into the
  clipped background.
- **Paper white**: a reference white card at exactly `1.0` (with a 203-nit
  ITU-R BT.2408 strip for comparison) while adjusting `paper_white_nits`.
- **Black level**: near-black steps at fixed absolute luminances for picking
  `min_luminance_nits`.

All adjustments mutate the primary window's `DisplayTarget` live — including
cycling the requested transfer with the `T` key (sRGB → scRGB-linear → PQ),
which renegotiates the swapchain on the fly — and a Gran Turismo 7 preview
toggle runs the full HDR tone-mapping path on the same patterns. The patterns
are rendered with `Tonemapping::None` and unlit materials so they reach the
display encoder at exact paper-white-relative values. The example also listens
for the new `WindowMonitorChanged` message and suggests recalibrating when the
window lands on a different monitor.

`DisplayTarget` gained matching builder-style helpers for deriving calibrated
targets from a base value:

```rust
let hdr = DisplayTarget::SDR_SRGB
    .with_paper_white(200.0)
    .with_peak(1000.0)
    .with_transfer(DisplayTransfer::ScRgbLinear);
```

An HDR display is required to calibrate anything real (macOS/iOS Metal,
Windows Vulkan, or Wayland Vulkan with Mesa 25.1+); on SDR systems the example
still runs on the documented warn-and-degrade path.
