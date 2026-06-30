---
title: "HDR display output (scRGB-linear and HDR10/PQ)"
authors: ["@stuartparmenter"]
pull_requests: []
---

Bevy can now present real high-dynamic-range output. Set the window's
`DisplayTarget` to request an HDR transfer, and give the camera an HDR-aware
tone-mapping operator:

```rust
fn enable_hdr_output(
    mut commands: Commands,
    mut window: Single<&mut DisplayTarget, With<PrimaryWindow>>,
    camera: Single<Entity, With<Camera>>,
) {
    **window = DisplayTarget {
        paper_white_nits: 200.0,
        peak_luminance_nits: 1000.0,
        // Or `DisplayTransfer::Pq` for HDR10 output.
        transfer: DisplayTransfer::ScRgbLinear,
        ..DisplayTarget::SDR_SRGB
    };
    // On HDR targets GT7 runs in its HDR mode automatically, driven by the
    // display target's peak luminance. Add a `GranTurismo7Params` component
    // to customize the operator's artistic dials.
    commands.entity(*camera).insert(Tonemapping::GranTurismo7);
}
```

Surface negotiation now selects a real (format, color space) pair through
wgpu's surface color-space API:

- **`DisplayTransfer::ScRgbLinear`** configures an `Rgba16Float` swapchain in
  the extended-sRGB-**linear** color space (1.0 = 80 nits, values above 1.0
  reach into the display's HDR headroom). Available on macOS/iOS (Metal EDR),
  Windows (Vulkan/DX12), and Wayland (Vulkan, Mesa 25.1+ color management) —
  this is **native-only** (browser WebGPU cannot present a linear-transfer
  canvas; use `ExtendedSrgb` there).
- **`DisplayTransfer::ExtendedSrgb`** is the *encoded* (gamma) sibling of
  scRGB-linear: the same 1.0 = 80 nits normalization, but the signal is run
  through the odd-symmetric extended sRGB OETF instead of staying linear. It is
  the **web HDR path** and is also available on Metal and Vulkan. Its surface
  color space follows `DisplayTarget::gamut`: `DisplayGamut::Rec709` configures
  the `ExtendedSrgb` color space, and `DisplayGamut::DisplayP3` configures
  `ExtendedDisplayP3` — wide-gamut HDR where the encoder converts the
  tone-mapped output into P3 primaries before encoding (Metal and browser
  WebGPU). This is the first transfer for which `DisplayGamut::DisplayP3` is a
  real, non-coerced encoder target.
- **`DisplayTransfer::Pq`** configures an HDR10 swapchain — `Rgb10a2Unorm`
  preferred (PQ's native 10-bit container), `Rgba16Float` where that is what
  the backend advertises — carrying the PQ (SMPTE ST 2084) signal in Rec.2020
  primaries. Available on Vulkan, DX12, and Metal when the OS has HDR output
  enabled. With the Gran Turismo 7 operator this is a fully native path: GT7's
  HDR mode emits linear Rec.2020 directly, the display-encoding pass applies
  the PQ OETF with no intermediate gamut round-trip, and the encoded signal is
  presented as-is.

The display-encoding pass writes the encoded signal (scRGB scaled by
`paper_white_nits / 80`; PQ from absolute nits) and the final blit hands it to
the surface unchanged — these formats have no hardware sRGB encode. With the
GT7 operator running in its HDR mode, highlights above paper white finally
make it to the panel. Press `O` in the `tonemapping` example (cycles sRGB → scRGB → PQ) or
`T` in the `hdr_calibration` example to try it on an HDR-capable display.

SDR-only tone-mapping operators — everything except `GranTurismo7` and `None`,
including the `Camera3d` default `TonyMcMapface` — cap their output at paper
white, which would leave an HDR display's headroom permanently unused. A
camera using one on an HDR-transfer target therefore degrades gracefully
instead of silently rendering an SDR-capped image: the view runs
`Tonemapping::GranTurismo7` instead (with the camera's `GranTurismo7Params`
if present, otherwise the defaults) and Bevy warns once. The camera's
`Tonemapping` component itself is never modified; set
`Tonemapping::GranTurismo7` explicitly to adopt the substitute and silence
the warning, or switch back to an SDR display target to keep the authored
operator. (`Tonemapping::None` is not substituted — it is a deliberate
pass-through — but also warns on HDR targets.)

When a request cannot be fulfilled it is **downgraded with a warning at each
step**: PQ falls back to scRGB-linear where available, and any HDR request
falls back to plain SDR sRGB — byte-identical to a default window — when the
surface offers nothing better (SDR displays, OS HDR disabled, X11, GLES). The
outcome is visible in the render world: `ViewDisplayTarget` carries both the
`requested` and the `resolved` display target, and every consumer (the
encoding pass, the upscaling blit, GT7's HDR mode, the display-target uniform)
keys on `resolved`, so an unfulfilled HDR request can never mis-encode the
image. A cross-HDR downgrade (PQ → scRGB) keeps your calibration values and
swaps only the transfer. Bevy also renegotiates defensively if the
capabilities change at runtime (e.g. the OS HDR toggle is flipped) rather
than failing surface validation.

Changing `DisplayTarget::transfer` at runtime reconfigures the surface with
fresh (format, color space) negotiation (and invalidates the window's view
targets), so HDR output can be toggled from a settings menu. A
`DisplayTarget::gamut` change is treated the same way when the transfer is
`ExtendedSrgb` (it selects `ExtendedSrgb` vs `ExtendedDisplayP3`); paper white,
peak, and every other gamut change flow through uniforms without any surface
work.

Screenshots understand the new surfaces too: scRGB (`Rgba16Float`) captures
read back as display-linear floats, HDR10 captures are decoded from the PQ
signal through the PQ EOTF, and encoded extended-range sRGB captures are
decoded through the extended sRGB EOTF (converting `ExtendedDisplayP3` back to
Rec.709) — all to the same display-linear Rec.709 scale (1.0 = 80 nits).
`save_to_disk` writes float images losslessly to float-capable containers
(OpenEXR `.exr` with Bevy's `exr` feature, Radiance `.hdr`); saving to an
8-bit format clamps, sRGB-encodes, and warns.

**Caveat: this currently requires a patched wgpu.** The surface color-space
API (`SurfaceColorSpace`, per-format `format_capabilities`,
`SurfaceConfiguration::color_space`) is not in a released wgpu; Bevy's
workspace carries a temporary `[patch.crates-io]` section pointing the wgpu
family of crates at the fork implementing it, until the API lands upstream
(tracked under [wgpu#2920](https://github.com/gfx-rs/wgpu/issues/2920)).
Cargo patches only apply at the workspace root of the final binary, so an
application building against this Bevy must copy the same `[patch.crates-io]`
entries into its own root `Cargo.toml`. The patch — and this caveat — go away
once the API ships in a wgpu release.
