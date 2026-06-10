---
title: Wide-gamut Rec. 2020 color support in `bevy_color`
authors: ["@pavlov"]
pull_requests: []
---

`bevy_color` can now represent wide-gamut colors as first-class citizens. The new
`LinearRec2020` color space is a linear RGB space using the ITU-R BT.2020
(Rec. 2020) primaries — the standard container gamut for HDR displays and video —
covering roughly twice the area of the sRGB gamut. It converts to and from every
other color space in the crate, has a `Color::LinearRec2020` variant, and supports
all the standard color operations (`Mix`, `Luminance` with the correct BT.2020
luminance weights, splines via `VectorSpace`, reflection, and serialization).

Authoring wide-gamut and HDR colors no longer requires hand-computed negative sRGB
components:

```rust
// A vivid Rec. 2020 red, far outside the sRGB gamut:
let red = Color::rec2020(1.0, 0.0, 0.0);
// A Display P3 color, exactly as shown in a macOS/CSS color picker:
let p3 = Color::display_p3(1.0, 0.2, 0.1);
// Any visible chromaticity via CIE xyY coordinates, e.g. D65 white at 5× paper white:
let bright = Color::xy_y(0.3127, 0.3290, 5.0);
```

Underneath, the new `primaries` module provides the building blocks the renderer's
upcoming wide-gamut working-space support is built on: `Chromaticity` (CIE 1931 xy
coordinates), `RgbPrimaries` (primary sets with constants for `BT709`, `BT2020`,
`DISPLAY_P3`, and `ACES_CG`), and `rgb_to_rgb_matrix` for deriving conversion
matrices between any two primary sets at runtime.

Alongside this, the existing color operations were made HDR-safe: values brighter
than standard white or outside the sRGB gamut now survive operations like
`lighter`, `with_luminance`, and the `Laba`/`Lcha` conversions instead of being
silently clamped, while SDR colors behave exactly as before.
