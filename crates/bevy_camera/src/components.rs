use crate::{primitives::Frustum, Camera, CameraProjection, OrthographicProjection, Projection};
use bevy_ecs::prelude::*;
use bevy_log::warn;
use bevy_reflect::{std_traits::ReflectDefault, Reflect, ReflectDeserialize, ReflectSerialize};
use bevy_transform::prelude::{GlobalTransform, Transform};
use serde::{Deserialize, Serialize};
use wgpu_types::{LoadOp, TextureUsages};

/// A 2D camera component. Enables the 2D render graph for a [`Camera`].
#[derive(Component, Default, Reflect, Clone)]
#[reflect(Component, Default, Clone)]
#[require(
    Camera,
    Projection::Orthographic(OrthographicProjection::default_2d()),
    Frustum = OrthographicProjection::default_2d().compute_frustum(&GlobalTransform::from(Transform::default())),
)]
pub struct Camera2d;

/// A 3D camera component. Enables the main 3D render graph for a [`Camera`].
///
/// The camera coordinate space is right-handed X-right, Y-up, Z-back.
/// This means "forward" is -Z.
#[derive(Component, Reflect, Clone)]
#[reflect(Component, Default, Clone)]
#[require(Camera, Projection)]
pub struct Camera3d {
    /// The depth clear operation to perform for the main 3d pass.
    pub depth_load_op: Camera3dDepthLoadOp,
    /// The texture usages for the depth texture created for the main 3d pass.
    pub depth_texture_usages: Camera3dDepthTextureUsage,
}

impl Default for Camera3d {
    fn default() -> Self {
        Self {
            depth_load_op: Default::default(),
            depth_texture_usages: TextureUsages::RENDER_ATTACHMENT.into(),
        }
    }
}

#[derive(Clone, Copy, Reflect, Serialize, Deserialize)]
#[reflect(Serialize, Deserialize, Clone)]
pub struct Camera3dDepthTextureUsage(pub u32);

impl From<TextureUsages> for Camera3dDepthTextureUsage {
    fn from(value: TextureUsages) -> Self {
        Self(value.bits())
    }
}

impl From<Camera3dDepthTextureUsage> for TextureUsages {
    fn from(value: Camera3dDepthTextureUsage) -> Self {
        Self::from_bits_truncate(value.0)
    }
}

/// The depth clear operation to perform for the main 3d pass.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug)]
#[reflect(Serialize, Deserialize, Clone, Default)]
pub enum Camera3dDepthLoadOp {
    /// Clear with a specified value.
    /// Note that 0.0 is the far plane due to bevy's use of reverse-z projections.
    Clear(f32),
    /// Load from memory.
    Load,
}

impl Default for Camera3dDepthLoadOp {
    fn default() -> Self {
        Camera3dDepthLoadOp::Clear(0.0)
    }
}

impl From<Camera3dDepthLoadOp> for LoadOp<f32> {
    fn from(config: Camera3dDepthLoadOp) -> Self {
        match config {
            Camera3dDepthLoadOp::Clear(x) => LoadOp::Clear(x),
            Camera3dDepthLoadOp::Load => LoadOp::Load,
        }
    }
}

/// If this component is added to a camera, the camera will use an intermediate "high dynamic range" render texture.
/// This allows rendering with a wider range of lighting values. However, this does *not* affect
/// whether the camera will render with hdr display output (which bevy does not support currently)
/// and only affects the intermediate render texture.
#[derive(Component, Default, Copy, Clone, Reflect, PartialEq, Eq, Hash, Debug)]
#[reflect(Component, Default, PartialEq, Hash, Debug)]
pub struct Hdr;

/// Controls the color space used for alpha compositing during rendering.
///
/// Alpha blending mixes colors from overlapping semi-transparent surfaces. The result
/// depends on whether blending happens in linear light space or sRGB (gamma-encoded) space.
///
/// - **Linear**: Physically correct. Matches tools like ColorAide with `--space srgb-linear`.
///   Requires [`Hdr`] on the camera; a validation warning is emitted if missing.
/// - **Srgb**: Matches many image editors (e.g. Photoshop) that blend in gamma-encoded space.
///   Uses the default sRGB render target.
/// - **Oklab**: Perceptually uniform compositing. Often produces smoother gradients than
///   sRGB or linear. Requires [`Hdr`] on the camera; a validation warning is emitted if missing.
#[derive(Component, Copy, Clone, Reflect, PartialEq, Eq, Hash, Debug, Default)]
#[reflect(Component, PartialEq, Hash, Debug, Default)]
pub enum CompositingSpace {
    /// sRGB compositing. Matches many image editors.
    /// Uses the default sRGB render target; blending behavior may vary by GPU/driver.
    #[default]
    Srgb,
    /// Linear light compositing. Physically correct.
    /// Requires [`Hdr`] to be present on the camera.
    Linear,
    /// Oklab compositing. Perceptually uniform; often produces smoother gradients.
    /// Requires [`Hdr`] to be present on the camera.
    Oklab,
}

/// Validates that cameras using [`CompositingSpace::Linear`] or [`CompositingSpace::Oklab`]
/// have the [`Hdr`] component. Emits a warning if not.
pub fn validate_compositing_space_requires_hdr(
    query: Query<(Entity, &CompositingSpace), Without<Hdr>>,
) {
    for (entity, compositing_space) in &query {
        if matches!(
            compositing_space,
            CompositingSpace::Linear | CompositingSpace::Oklab
        ) {
            warn!(
                "Camera entity {entity:?} uses CompositingSpace::Linear or CompositingSpace::Oklab \
                but is missing the Hdr component. Add Hdr to the camera for correct rendering."
            );
        }
    }
}
