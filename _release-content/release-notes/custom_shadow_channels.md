---
title: Custom shadow factors for any light
authors: ["@mate-h"]
pull_requests: []
---

Materials that call `apply_pbr_lighting` can now attenuate specific point, spot and directional lights with their own shadowing,
such as a custom shadow map, a deep opacity map for hair, or a ray-marched heightfield.

Add a `CustomShadowChannel` to the lights that should receive the custom shadow, and write a visibility factor for that channel
into the new `PbrInput::custom_shadow_factors` field in your material's fragment shader:

```rust
commands.spawn((PointLight::default(), CustomShadowChannel(0)));
```

```wgsl
var pbr_input = pbr_input_new();
// ...
pbr_input.custom_shadow_factors[0] = my_custom_shadow(pbr_input.world_position.xyz);
let color = apply_pbr_lighting(pbr_input);
```

Up to four channels are available, so different lights can be shadowed by different custom shadow maps.
The factor is multiplied with Bevy's own shadow maps and contact shadows,
and applies even to lights that have Bevy shadow maps disabled.
Lights without a `CustomShadowChannel` are unaffected.

The existing `PbrInput::directional_shadow_factor`, which applies to every shadow-mapped directional light, keeps working as before.
