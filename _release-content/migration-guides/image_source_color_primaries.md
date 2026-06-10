---
title: "`Image` has a new `source_primaries` field"
pull_requests: []
---

`Image` (in `bevy_image`, available as `bevy::image::Image`) has a new public field:

```rust
pub source_primaries: SourceColorPrimaries,
```

It records which color primaries (gamut) the image data is expressed in — `Bt709`
(the sRGB primaries, and the default), `Bt2020`, or `DisplayP3`. This is metadata
only: it does not change how any image is decoded, stored, or rendered today. It
exists so the upcoming configurable wide working color space can convert texture
data from its source gamut correctly instead of silently reinterpreting it.

If you construct `Image` with a struct literal, you must now provide the field;
`SourceColorPrimaries::default()` (BT.709) preserves the previous behavior:

```rust
// 0.18
let image = Image {
    data,
    texture_descriptor,
    // ...
    copy_on_resize: false,
};

// 0.19
let image = Image {
    data,
    texture_descriptor,
    // ...
    copy_on_resize: false,
    source_primaries: Default::default(),
};
```

All of `Image`'s constructors (`Image::new`, `Image::new_fill`, `Image::default`,
`Image::from_buffer`, `Image::from_dynamic`, ...) initialize the field to
`SourceColorPrimaries::Bt709`, so code using them is unaffected.

Asset loaders now stamp this field when loading:

- `ImageLoaderSettings`, `HdrTextureLoaderSettings`, and `ExrTextureLoaderSettings`
  gained an optional `source_primaries: Option<SourceColorPrimaries>` setting
  (defaults to `None`; existing `.meta` files keep working). `Some` forces the
  stamped value.
- With the default `None`, the loaders trust color-primary metadata carried by the
  file itself — the KTX2 data format descriptor's `colorPrimaries`, Radiance HDR
  `PRIMARIES=` header lines, and the OpenEXR `chromaticities` attribute — falling
  back to BT.709 when absent or unrecognized. Decoded pixel data and resolved
  texture formats are unchanged in all cases.
- The KTX2 loader also inspects the file's declared transfer function and logs a
  one-time warning when it contradicts the loader's `is_srgb` setting (the setting
  still wins, byte-for-byte identical to before) or when the file declares an HDR
  transfer function (PQ/HLG), which is still loaded as-is.
- The glTF loader stamps `Bt709` explicitly on all textures, as mandated by the
  glTF 2.0 specification.
