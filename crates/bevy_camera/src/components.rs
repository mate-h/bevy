use crate::{primitives::Frustum, Camera, CameraProjection, OrthographicProjection, Projection};
use bevy_ecs::prelude::*;
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
/// This allows rendering with a wider range of lighting values. Note that this only affects the
/// intermediate render texture, not the signal sent to the display: to output HDR to an
/// HDR-capable display, request an HDR transfer on the window's
/// [`DisplayTarget`](bevy_window::DisplayTarget) (cameras rendering to such a target get the
/// high-precision intermediate automatically).
#[derive(Component, Default, Copy, Clone, Reflect, PartialEq, Eq, Hash, Debug)]
#[reflect(Component, Default, PartialEq, Hash, Debug)]
pub struct Hdr;

/// Marker mirroring "this camera has an active tone-mapping operator"
/// (`Tonemapping` other than `Tonemapping::None`) into a crate the renderer's
/// camera extraction can see.
///
/// **Managed automatically** — do not insert or remove this manually. The
/// `Tonemapping` component lives in `bevy_core_pipeline`, which `bevy_camera`
/// and `bevy_render` cannot depend on, so `bevy_core_pipeline`'s
/// `TonemappingPlugin` keeps this marker in sync with it every frame (in
/// [`PostUpdate`](https://docs.rs/bevy/latest/bevy/app/struct.PostUpdate.html)).
///
/// Cameras with this marker render to an `Rgba16Float` intermediate main
/// texture (like [`Hdr`] and like cameras on HDR display targets, which both
/// force it independently), so that the node-side tone-mapping and
/// display-encoding post-process passes operate over a high-precision buffer
/// instead of quantizing through an 8-bit intermediate. This includes cameras with an explicit
/// [`CompositingSpace::Srgb`]/[`CompositingSpace::Oklab`]: shaders still
/// write encoded values (blending stays in the encoded space), but the
/// storage is fp16 so scene-referred values above 1.0 survive until the
/// tone-mapping pass decodes them.
///
/// To force the high-precision intermediate on a camera *without* an active
/// tone-mapping operator, add [`Hdr`] instead.
#[derive(Component, Default, Copy, Clone, Reflect, PartialEq, Eq, Hash, Debug)]
#[reflect(Component, Default, PartialEq, Hash, Debug)]
pub struct TonemappingEnabled;

/// Marker placed on a camera that has a post-process effect requiring a
/// scene-linear (pre-tone-mapping) HDR buffer: bloom, auto exposure, depth of
/// field, or motion blur.
///
/// **Managed automatically** — do not insert or remove this manually. Those
/// effect components live in `bevy_post_process`, which `bevy_camera` and
/// `bevy_render` cannot depend on, so `bevy_post_process` keeps this marker in
/// sync with them every frame (in
/// [`PostUpdate`](https://docs.rs/bevy/latest/bevy/app/struct.PostUpdate.html)).
///
/// Its presence vetoes the SDR in-shader tone-mapping fast path in camera
/// extraction (which would otherwise keep the camera on an 8-bit intermediate
/// and fold tone mapping into the material shaders): these effects read the
/// scene-referred buffer before tone mapping, so the camera must keep its
/// `Rgba16Float` intermediate. See [`NeedsSceneLinearAa`] for the
/// anti-aliasing equivalent.
#[derive(Component, Default, Copy, Clone, Reflect, PartialEq, Eq, Hash, Debug)]
#[reflect(Component, Default, PartialEq, Hash, Debug)]
pub struct NeedsSceneLinearPost;

/// Marker placed on a camera that has an anti-aliasing mode requiring a
/// scene-linear (pre-tone-mapping) HDR buffer: temporal anti-aliasing or DLSS.
///
/// **Managed automatically** — do not insert or remove this manually. Those
/// components live in `bevy_anti_alias`, which `bevy_camera` and `bevy_render`
/// cannot depend on, so `bevy_anti_alias` keeps this marker in sync with them
/// every frame (in
/// [`PostUpdate`](https://docs.rs/bevy/latest/bevy/app/struct.PostUpdate.html)).
///
/// Its presence vetoes the SDR in-shader tone-mapping fast path in camera
/// extraction, exactly like [`NeedsSceneLinearPost`]: these modes reproject
/// and resolve the scene-referred buffer before tone mapping, so the camera
/// must keep its `Rgba16Float` intermediate. (FXAA and SMAA operate on the
/// tone-mapped image and therefore do *not* set this marker.)
#[derive(Component, Default, Copy, Clone, Reflect, PartialEq, Eq, Hash, Debug)]
#[reflect(Component, Default, PartialEq, Hash, Debug)]
pub struct NeedsSceneLinearAa;

/// Marker placed on a camera whose tone-mapping operator configuration cannot
/// be reproduced by the in-shader fast path and must run node-side.
///
/// Currently set when the operator is `Tonemapping::GranTurismo7` together with
/// a `GranTurismo7Params` component: the in-shader fold has no path to bind the
/// per-view GT7 params uniform, so it would silently fall back to the baked SDR
/// defaults and discard the user's custom parameters. Keeping such a camera on
/// the node-side path lets it honor them.
///
/// **Managed automatically** — do not insert or remove this manually. The
/// `Tonemapping` and `GranTurismo7Params` components live in
/// `bevy_core_pipeline`, which `bevy_camera` and `bevy_render` cannot depend
/// on, so `bevy_core_pipeline` keeps this marker in sync with them every frame
/// (in [`PostUpdate`](https://docs.rs/bevy/latest/bevy/app/struct.PostUpdate.html)).
///
/// Its presence vetoes the SDR in-shader tone-mapping fast path in camera
/// extraction, exactly like [`NeedsSceneLinearPost`] / [`NeedsSceneLinearAa`].
#[derive(Component, Default, Copy, Clone, Reflect, PartialEq, Eq, Hash, Debug)]
#[reflect(Component, Default, PartialEq, Hash, Debug)]
pub struct NeedsNodeTonemapping;

/// Color space for alpha compositing. Affects how overlapping semi-transparent layers blend.
#[derive(Component, Copy, Clone, Reflect, PartialEq, Eq, Hash, Debug, Default)]
#[reflect(Component, PartialEq, Hash, Debug, Default)]
pub enum CompositingSpace {
    /// Gamma-encoded blending. Matches most image editors. Uses default sRGB target.
    #[default]
    Srgb,
    /// Linear light blending. Physically correct.
    Linear,
    /// Perceptually uniform blending. Often smoother gradients. Requires [`Hdr`] because its value can be outside [0, 1].
    Oklab,
}
