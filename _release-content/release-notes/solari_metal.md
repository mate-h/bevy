---
title: Solari on Metal
authors: ["@mate-h"]
pull_requests: [25123]
---

Ray tracing on Metal has been available since wgpu 29, and with bindless storage buffers landing in wgpu 30, Solari now runs on Apple Silicon Macs.

Run the Solari example on a compatible Mac:

```sh
cargo run --example solari --features bevy_solari,https,free_camera
```

For MetalFX ray reconstruction denoising (macOS 26+ Apple Silicon with Temporal Denoised Scaler support):

```
cargo run --example solari --features bevy_solari,metal_fx,https,free_camera
```

When supported, MetalFX Temporal Denoised Scaler is enabled automatically (toggle with `3`), mirroring the DLSS Ray Reconstruction path on NVIDIA.
