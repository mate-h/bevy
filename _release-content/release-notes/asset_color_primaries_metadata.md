---
title: Color-primaries metadata for image assets
authors: ["@stuartparmenter"]
pull_requests: []
---

Image assets now carry their source gamut explicitly: `Image` has a new
`source_primaries: SourceColorPrimaries` field (`Bt709`, `Bt2020`, or `DisplayP3`),
and the image loaders fill it in from real file metadata:

- **KTX2**: the `colorPrimaries` field of the data format descriptor. The loader
  also warns (once) when the file's declared transfer function contradicts the
  `is_srgb` loader setting, or when the file declares an HDR transfer function such
  as PQ or HLG.
- **Radiance HDR (`.hdr`)**: `PRIMARIES=` header lines.
- **OpenEXR (`.exr`)**: the standardized `chromaticities` header attribute.
- **glTF**: textures are stamped BT.709, as the glTF 2.0 specification mandates.

You can also pin the primaries per asset via the new optional `source_primaries`
field on `ImageLoaderSettings`, `HdrTextureLoaderSettings`, and
`ExrTextureLoaderSettings` (in code or in `.meta` files). The resolution order is:
explicit setting, then file metadata, then the BT.709 default.

For now this is pure metadata — decoding and rendering are unchanged, and untagged
assets behave as before. It is the foundation for the configurable wide working
color space: once rendering can run in Rec. 2020, this stamp is what lets a Rec. 709
HDRI, a Display P3 texture, and a Rec. 2020 video frame all land in the working space
with their colors intact instead of silently shifting saturation.
