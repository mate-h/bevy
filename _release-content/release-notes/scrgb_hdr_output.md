---
title: "HDR display output (scRGB-linear)"
authors: ["@pavlov"]
pull_requests: []
---

Bevy can now present real high-dynamic-range output. Set the window's
`DisplayTarget` to request the scRGB-linear transfer, and give the camera an
HDR-aware tone-mapping operator plus its params component:

```rust
fn enable_hdr_output(
    mut commands: Commands,
    mut window: Single<&mut DisplayTarget, With<PrimaryWindow>>,
    camera: Single<Entity, With<Camera>>,
) {
    *window = DisplayTarget {
        paper_white_nits: 200.0,
        peak_luminance_nits: 1000.0,
        transfer: DisplayTransfer::ScRgbLinear,
        ..DisplayTarget::SDR_SRGB
    };
    commands.entity(*camera).insert((
        Tonemapping::GranTurismo7,
        // Required for GT7's HDR mode: the prepared params uniform is what
        // plumbs the display target's peak luminance into the operator.
        // Without it, GT7 stays in SDR mode and warns.
        GranTurismo7Params::default(),
    ));
}
```

When the backend offers it, surface selection now configures an `Rgba16Float`
swapchain, which wgpu's Metal and Vulkan backends present as extended-range
scRGB-linear (1.0 = 80 nits, values above 1.0 reach into the display's HDR
headroom). The display-encoding pass writes the encoded signal (scaled by
`paper_white_nits / 80`) and the final blit hands it to the surface unchanged —
float surfaces have no hardware sRGB encode. With the GT7 operator running in
its HDR mode (`Tonemapping::GranTurismo7` + `GranTurismo7Params`), highlights
above paper white finally make it to the panel. Press `O` in the `tonemapping`
example to try it on an HDR-capable display.

Supported today: macOS/iOS (Metal, EDR), Windows (Vulkan), and Wayland
(Vulkan, Mesa 25.1+ color management). On other backends (DX12, X11, GLES,
WebGPU) — and for the PQ and HLG transfers everywhere — wgpu cannot yet
communicate an HDR color space to the OS
([wgpu#2920](https://github.com/gfx-rs/wgpu/issues/2920)), so the request is
**downgraded**: Bevy warns once and renders plain SDR sRGB, byte-identical to
a default window. The outcome of this negotiation is visible in the render
world: `ViewDisplayTarget` now carries both the `requested` and the `resolved`
display target, and every consumer (the encoding pass, the upscaling blit,
GT7's HDR mode, the display-target uniform) keys on `resolved`, so an
unfulfilled HDR request can never mis-encode the image.

Changing `DisplayTarget::transfer` at runtime reconfigures the surface with
fresh format selection (and invalidates the window's view targets), so HDR
output can be toggled from a settings menu. Changes to the other calibration
fields (paper white, peak, gamut) flow through uniforms without any surface
work.

Screenshots understand the new surfaces too: `Image::try_into_dynamic` now
converts `Rgba16Float` (and `Rgba32Float`) images to floating-point dynamic
images, and `save_to_disk` writes them losslessly to float-capable containers
(OpenEXR `.exr` with Bevy's `exr` feature, Radiance `.hdr`). Saving an HDR
screenshot to an 8-bit format clamps, sRGB-encodes, and warns.
