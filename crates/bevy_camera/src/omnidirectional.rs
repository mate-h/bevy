//! Omnidirectional cameras that render the scene into a cubemap.

use bevy_ecs::prelude::*;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bitflags::bitflags;

use crate::{
    primitives::CubemapFrusta, visibility::OmnidirectionalVisibleEntities, Camera3d,
    OmnidirectionalProjection,
};

bitflags! {
    /// Specifies which sides of an omnidirectional camera will be rendered to
    /// this frame.
    ///
    /// Enabling a flag will cause the renderer to refresh the corresponding
    /// cubemap side on this frame.
    #[repr(transparent)]
    #[derive(Clone, Copy, Component, Reflect, PartialEq, Eq, Hash, Debug)]
    #[reflect(opaque)]
    #[reflect(Component, Default, Hash, PartialEq, Debug, Clone)]
    pub struct ActiveCubemapSides: u8 {
        /// The +X face.
        const X = 0x01;
        /// The -X face.
        const NEG_X = 0x02;
        /// The +Y face.
        const Y = 0x04;
        /// The -Y face.
        const NEG_Y = 0x08;
        /// The +Z face (forward in Bevy's left-handed conventions).
        const NEG_Z = 0x10;
        /// The -Z face (backward in Bevy's left-handed conventions).
        const Z = 0x20;
    }
}

impl Default for ActiveCubemapSides {
    fn default() -> ActiveCubemapSides {
        ActiveCubemapSides::all()
    }
}

/// A camera that renders the scene into all six faces of a cubemap image.
///
/// These cubemap images are typically attached to an environment map light on a
/// light probe, in order to achieve real-time reflective surfaces.
///
/// Internally, these cameras become up to six sub-cameras, one for each side of
/// the cube. Consequently, omnidirectional cameras are quite expensive by
/// default. The [`ActiveCubemapSides`] bitfield may be used to reduce this load
/// by rendering to only a subset of the cubemap faces each frame. A common
/// technique is to render to only one cubemap face per frame, cycling through
/// the faces in a round-robin fashion.
///
/// # Usage
///
/// ```
/// # use bevy_asset::Handle;
/// # use bevy_camera::prelude::*;
/// # use bevy_camera::{
/// #     ActiveCubemapSides, ImageRenderTarget, OmnidirectionalCamera,
/// #     OmnidirectionalProjection, RenderTarget,
/// # };
/// # use bevy_ecs::prelude::*;
/// # use bevy_image::Image;
/// # use bevy_transform::prelude::Transform;
/// # fn spawn(mut commands: Commands, cubemap: Handle<Image>) {
/// commands.spawn((
///     OmnidirectionalCamera,
///     Camera3d::default(),
///     OmnidirectionalProjection::default(),
///     ActiveCubemapSides::default(),
///     RenderTarget::Image(ImageRenderTarget {
///         handle: cubemap,
///         scale_factor: 1.0,
///         array_layer: None,
///     }),
///     Transform::default(),
/// ));
/// # }
/// ```
#[derive(Component, Reflect, Clone, Default, Debug)]
#[reflect(Component, Default, Debug, Clone)]
#[require(
    Camera3d,
    OmnidirectionalProjection,
    ActiveCubemapSides,
    CubemapFrusta,
    OmnidirectionalVisibleEntities
)]
pub struct OmnidirectionalCamera;
