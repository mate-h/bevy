use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;

/// The number of distinct [`CustomShadowChannel`]s a material can supply shadow factors for.
pub const MAX_CUSTOM_SHADOW_CHANNELS: u8 = 4;

/// Routes a material-supplied shadow factor to this light.
///
/// Add this component to a [`PointLight`](crate::PointLight),
/// [`SpotLight`](crate::SpotLight) or [`DirectionalLight`](crate::DirectionalLight) to let
/// materials attenuate it with their own shadowing, for example from a custom shadow map, a
/// deep opacity map or a heightfield ray march.
///
/// In the PBR shader, `PbrInput::custom_shadow_factors` holds one factor per channel (all `1.0`
/// by default). A material that calls `apply_pbr_lighting` can write a visibility factor in
/// `[0, 1]` into the component matching this channel, and every light with this channel will be
/// multiplied by it:
///
/// ```wgsl
/// var pbr_input = pbr_input_new();
/// // ...
/// pbr_input.custom_shadow_factors[0] = my_custom_shadow(pbr_input.world_position.xyz);
/// let color = apply_pbr_lighting(pbr_input);
/// ```
///
/// The factor is applied whether or not the light has Bevy shadow maps enabled, and is multiplied
/// with Bevy's shadow maps and contact shadows. Like Bevy's own shadows, it attenuates direct
/// light on meshes that receive shadows (meshes without
/// [`NotShadowReceiver`](crate::NotShadowReceiver)), and diffuse transmitted light on meshes that
/// also have [`TransmittedShadowReceiver`](crate::TransmittedShadowReceiver).
///
/// Lights without this component are unaffected by `custom_shadow_factors`.
///
/// The factor is only available to materials that shade in the forward pass. With deferred
/// rendering the lighting pass reconstructs `PbrInput` from the G-buffer, so every channel reads
/// as `1.0` there.
#[derive(Clone, Copy, Component, Default, Debug, PartialEq, Eq, Hash, Reflect)]
#[reflect(Component, Default, Debug, Clone, PartialEq, Hash)]
pub struct CustomShadowChannel(pub u8);

impl CustomShadowChannel {
    /// Returns the channel index, or `None` if it is not less than
    /// [`MAX_CUSTOM_SHADOW_CHANNELS`].
    pub fn index(self) -> Option<u8> {
        (self.0 < MAX_CUSTOM_SHADOW_CHANNELS).then_some(self.0)
    }
}
