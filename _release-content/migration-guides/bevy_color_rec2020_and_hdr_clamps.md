---
title: "`bevy_color`: new `Color::LinearRec2020` variant and HDR-safe clamping"
pull_requests: []
---

## New `Color::LinearRec2020` variant

`Color` has a new variant, `Color::LinearRec2020(LinearRec2020)`, a first-class linear
RGB color space with wide-gamut Rec. 2020 (ITU-R BT.2020) primaries. If you match
exhaustively on `Color`, you must handle the new variant (or add a wildcard arm).
Conversions to and from every other color space are provided via `From`/`Into`, so
`color.into()` / `Color::to_linear` / `Color::to_srgba` continue to work for the new
variant.

## Clamping in color operations is now HDR-aware

Several operations in `bevy_color` used to clamp their results into the standard
`[0.0, 1.0]` (SDR) range, silently destroying HDR (brighter-than-white) and
wide-gamut (out-of-sRGB-gamut) values. They now preserve the documented clamping
behavior **only for SDR inputs**, and pass HDR / out-of-range values through:

- `LinearRgba::with_luminance` (and the new `LinearRec2020::with_luminance`) no
  longer clamps each channel to `[0, 1]` when the input color or the target
  luminance is outside the SDR range. Inputs with all channels in `[0, 1]` and an
  SDR target luminance behave exactly as before.
- `Luminance::lighter` for `LinearRgba` and `LinearRec2020` still clamps to white
  for SDR colors, but HDR colors (any channel or the luminance above `1.0`) now
  keep getting brighter while preserving their chromaticity, instead of being
  pulled toward white. For example, a saturated emissive red like
  `LinearRgba::new(4.0, 0.0, 0.0, 1.0)` is no longer desaturated.
- `Luminance::lighter` for `Xyza`, `Laba`, `Lcha`, `Oklaba`, `Oklcha`, `Hsla`, and
  `Okhsla` still clamps to white for colors whose luminance/lightness is at most
  `1.0`, but colors whose luminance/lightness already exceeds `1.0` keep getting
  brighter instead of being pulled down to `1.0`.
- `Luminance::darker` for `Xyza`, `Hsla`, and `Okhsla` no longer clamps the result
  to an upper bound of `1.0` (it still clamps to black). This only affects inputs
  that were already brighter than white, or negative `amount` values outside the
  documented `[0.0, 1.0]` range.
- The `Laba` → `Lcha` conversion no longer clamps chroma to `[0.0, 1.5]`. All colors
  within the sRGB gamut have chroma below `1.5`, so this only changes results for
  wide-gamut inputs, which now round-trip losslessly.

Quantization methods (`ColorToPacked::to_u8_array` and friends) still clamp to
`[0, 1]`, since `u8` output has no headroom for HDR values.

If you relied on these operations to force colors into SDR range, clamp explicitly
instead, e.g. `LinearRgba { red: c.red.clamp(0., 1.), .. }` or convert through
`ColorToPacked`.

## Oklab conversion constants at full precision

The `Oklaba` ↔ `LinearRgba` conversion matrices now use the full-precision constants
from the reference Oklab implementation instead of truncated literals. Results may
differ from previous versions by about one `f32` ULP (~1e-7); exact float
comparisons against stored Oklab values may need a small tolerance.
