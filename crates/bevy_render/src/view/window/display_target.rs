//! Render-world plumbing for [`DisplayTarget`].
//!
//! [`DisplayTarget`] lives on [`Window`](bevy_window::Window) entities as a
//! required component (see `bevy_window`). This module covers the two cases a
//! window component cannot:
//!
//! - non-entity render targets ([`RenderTarget::Image`] and
//!   [`RenderTarget::TextureView`]), which are described by the
//!   [`ManualDisplayTargets`] resource, and
//! - the render world, where [`resolve_display_target`] provides a single
//!   lookup that view-preparation systems can use to find the
//!   [`DisplayTarget`] for any [`NormalizedRenderTarget`].
//!
//! [`RenderTarget::Image`]: bevy_camera::RenderTarget::Image
//! [`RenderTarget::TextureView`]: bevy_camera::RenderTarget::TextureView

use bevy_camera::NormalizedRenderTarget;
use bevy_ecs::{entity::ContainsEntity, reflect::ReflectResource, resource::Resource};
use bevy_platform::collections::HashMap;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render_macros::ExtractResource;
use bevy_window::DisplayTarget;

use super::ExtractedWindows;

/// Resource that stores the [`DisplayTarget`] for render targets that are not
/// backed by a [`Window`](bevy_window::Window) entity.
///
/// [`RenderTarget::Window`] targets carry their [`DisplayTarget`] as a
/// (required) component on the window entity; [`RenderTarget::Image`] and
/// [`RenderTarget::TextureView`] targets have no entity to host the
/// component, so consumers look them up here instead, keyed by their
/// [`NormalizedRenderTarget`] (the same keying used by `ViewTargetAttachments`).
/// Targets without an entry — including [`RenderTarget::None`] — use the
/// default of [`DisplayTarget::SDR_SRGB`].
///
/// This type dereferences to a `HashMap<NormalizedRenderTarget, DisplayTarget>`.
/// Insert into it from the main world; it is extracted (cloned) into the
/// render world every frame, where [`resolve_display_target`] consults it.
///
/// This is the "resource sidecar" half of the hybrid `DisplayTarget`
/// placement: most users only ever touch the window component, while
/// offscreen and XR-style texture-view targets opt in through this map.
///
/// [`RenderTarget::Window`]: bevy_camera::RenderTarget::Window
/// [`RenderTarget::Image`]: bevy_camera::RenderTarget::Image
/// [`RenderTarget::TextureView`]: bevy_camera::RenderTarget::TextureView
/// [`RenderTarget::None`]: bevy_camera::RenderTarget::None
#[derive(Default, Clone, Debug, PartialEq, Resource, ExtractResource, Reflect)]
#[reflect(Resource, Default, Debug, PartialEq, Clone)]
pub struct ManualDisplayTargets(HashMap<NormalizedRenderTarget, DisplayTarget>);

impl core::ops::Deref for ManualDisplayTargets {
    type Target = HashMap<NormalizedRenderTarget, DisplayTarget>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::DerefMut for ManualDisplayTargets {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Resolves the [`DisplayTarget`] for a render target in the render world.
///
/// This is the seam view-preparation code should use to parameterize
/// per-display work (tone mapping, gamut mapping, transfer encoding) for a
/// view: pass the view's [`NormalizedRenderTarget`] (e.g.
/// `ExtractedCamera::target`) together with the [`ExtractedWindows`] and
/// [`ManualDisplayTargets`] resources. All cameras rendering to the same
/// target resolve to the same `DisplayTarget`.
///
/// Resolution rules:
/// - [`NormalizedRenderTarget::Window`]: the window's extracted
///   `DisplayTarget` (from the `DisplayTarget` component on the
///   [`Window`](bevy_window::Window) entity).
/// - [`NormalizedRenderTarget::Image`] / [`NormalizedRenderTarget::TextureView`]:
///   looked up in [`ManualDisplayTargets`].
/// - [`NormalizedRenderTarget::None`], `target == None`, or any missing
///   entry: [`DisplayTarget::SDR_SRGB`].
pub fn resolve_display_target(
    target: Option<&NormalizedRenderTarget>,
    extracted_windows: &ExtractedWindows,
    manual_display_targets: &ManualDisplayTargets,
) -> DisplayTarget {
    match target {
        Some(NormalizedRenderTarget::Window(window_ref)) => extracted_windows
            .get(&window_ref.entity())
            .map(|window| window.display_target)
            .unwrap_or_default(),
        Some(
            target @ (NormalizedRenderTarget::Image(_) | NormalizedRenderTarget::TextureView(_)),
        ) => manual_display_targets
            .get(target)
            .copied()
            .unwrap_or_default(),
        Some(NormalizedRenderTarget::None { .. }) | None => DisplayTarget::SDR_SRGB,
    }
}
