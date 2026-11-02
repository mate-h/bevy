---
title: Material-supplied shadows for any light
authors: ["@mate-h"]
pull_requests: []
---

Materials that call `apply_pbr_lighting` can now attenuate specific point, spot and directional lights with their own shadowing,
such as a custom shadow map, a deep opacity map for hair, or a ray-marched heightfield.

Add a `MaterialShadowChannel` to the lights the material should shadow, and write a visibility factor for that channel
into the new `PbrInput::material_shadow_factors` field in your material's fragment shader:

```rust
commands.spawn((PointLight::default(), MaterialShadowChannel(0)));
```

```wgsl
var pbr_input = pbr_input_new();
// ...
pbr_input.material_shadow_factors[0] = my_shadow_map(pbr_input.world_position.xyz);
let color = apply_pbr_lighting(pbr_input);
```

Up to four channels are available, so a material can shadow different lights with different maps.
The factor is multiplied with Bevy's own shadow maps and contact shadows,
and applies even to lights that have Bevy shadow maps disabled.
Lights without a `MaterialShadowChannel` are unaffected.

The existing `PbrInput::directional_shadow_factor`, which applies to every shadow-mapped directional light, keeps working as before.
