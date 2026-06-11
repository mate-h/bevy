//! Shared logic for deciding which camera in a stack runs a fullscreen
//! post-process pass.
//!
//! Cameras that render to the same target share one main-texture ping-pong
//! (see `prepare_view_targets`), so a fullscreen pass run by an earlier
//! camera feeds already-processed pixels into every later camera that
//! composites on top with [`ClearColorConfig::None`]. Running the pass per
//! camera would then apply it twice to the earlier camera's pixels — e.g.
//! tone mapping the lower camera's output a second time, or alpha-blending
//! PQ-encoded signal. Instead, the pass is deferred to the last enabled
//! camera in the stack, which processes the composed buffer exactly once.
//!
//! [`ClearColorConfig::None`]: bevy_camera::ClearColorConfig

use bevy_ecs::entity::{Entity, EntityHashMap};
use bevy_platform::collections::HashMap;
use core::hash::Hash;

/// A view participating in the stack analysis for one fullscreen pass.
pub(crate) struct StackView<K> {
    pub entity: Entity,
    /// Identity of the main-texture ping-pong the view renders into. Views
    /// only form a stack when they share it.
    pub texture: K,
    /// The camera's position in its render target's sorted camera order.
    pub sorted_index: usize,
    /// Whether the fullscreen pass would run for this view.
    pub enabled: bool,
    /// Whether this view composites over the previous camera's output and
    /// covers the whole target: [`ClearColorConfig::None`] and no viewport.
    ///
    /// [`ClearColorConfig::None`]: bevy_camera::ClearColorConfig
    pub composites_fullscreen: bool,
}

/// Returns the views whose fullscreen pass must be skipped because a later
/// camera in the same stack runs it on the composed buffer, mapped to that
/// finalizing camera's entity.
///
/// Within each group of views sharing a main texture, the pass is deferred
/// to the last enabled view if and only if every enabled view after the
/// first composites fullscreen (loads the previous content and covers the
/// whole target with its output). Any other arrangement — clearing cameras,
/// viewport-scoped cameras — keeps the per-camera behavior, where each
/// camera's pass only feeds its own region of the final image.
pub(crate) fn stack_deferred_views<K: Copy + Eq + Hash>(
    views: impl IntoIterator<Item = StackView<K>>,
) -> EntityHashMap<Entity> {
    let mut groups: HashMap<K, Vec<StackView<K>>> = HashMap::default();
    for view in views {
        groups.entry(view.texture).or_default().push(view);
    }

    let mut deferred = EntityHashMap::default();
    for group in groups.values_mut() {
        group.sort_unstable_by_key(|view| view.sorted_index);
        let enabled: Vec<&StackView<K>> = group.iter().filter(|view| view.enabled).collect();
        let Some((finalizer, earlier)) = enabled.split_last() else {
            continue;
        };
        if earlier.is_empty() || !enabled[1..].iter().all(|view| view.composites_fullscreen) {
            continue;
        }
        for view in earlier {
            deferred.insert(view.entity, finalizer.entity);
        }
    }
    deferred
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(
        raw: u32,
        texture: u32,
        index: usize,
        enabled: bool,
        composites: bool,
    ) -> StackView<u32> {
        StackView {
            entity: Entity::from_raw_u32(raw).unwrap(),
            texture,
            sorted_index: index,
            enabled,
            composites_fullscreen: composites,
        }
    }

    #[test]
    fn single_camera_is_never_deferred() {
        let deferred = stack_deferred_views([view(1, 0, 0, true, false)]);
        assert!(deferred.is_empty());
    }

    #[test]
    fn lower_camera_defers_to_compositing_upper_camera() {
        let deferred =
            stack_deferred_views([view(1, 0, 0, true, false), view(2, 0, 1, true, true)]);
        assert_eq!(deferred.len(), 1);
        assert_eq!(
            deferred.get(&Entity::from_raw_u32(1).unwrap()),
            Some(&Entity::from_raw_u32(2).unwrap())
        );
    }

    #[test]
    fn three_camera_stack_defers_to_the_last() {
        let deferred = stack_deferred_views([
            view(1, 0, 0, true, false),
            view(2, 0, 1, true, true),
            view(3, 0, 2, true, true),
        ]);
        assert_eq!(deferred.len(), 2);
        let finalizer = Entity::from_raw_u32(3).unwrap();
        assert_eq!(
            deferred.get(&Entity::from_raw_u32(1).unwrap()),
            Some(&finalizer)
        );
        assert_eq!(
            deferred.get(&Entity::from_raw_u32(2).unwrap()),
            Some(&finalizer)
        );
    }

    #[test]
    fn clearing_upper_camera_keeps_per_camera_passes() {
        // The upper camera clears (does not composite), so no deferral.
        let deferred =
            stack_deferred_views([view(1, 0, 0, true, false), view(2, 0, 1, true, false)]);
        assert!(deferred.is_empty());
    }

    #[test]
    fn viewport_scoped_cameras_keep_per_camera_passes() {
        // Split screen: the later camera only covers its viewport, so each
        // camera must keep its own pass.
        let deferred = stack_deferred_views([
            view(1, 0, 0, true, true),
            view(2, 0, 1, true, false),
            view(3, 0, 2, true, true),
        ]);
        assert!(deferred.is_empty());
    }

    #[test]
    fn disabled_views_do_not_participate() {
        // The overlay camera has the pass disabled (e.g. `Tonemapping::None`);
        // the lower camera keeps its own pass.
        let deferred =
            stack_deferred_views([view(1, 0, 0, true, false), view(2, 0, 1, false, true)]);
        assert!(deferred.is_empty());
    }

    #[test]
    fn separate_textures_form_separate_stacks() {
        let deferred =
            stack_deferred_views([view(1, 0, 0, true, false), view(2, 1, 1, true, true)]);
        assert!(deferred.is_empty());
    }

    #[test]
    fn sorted_index_orders_the_stack_not_insertion_order() {
        let deferred =
            stack_deferred_views([view(2, 0, 1, true, true), view(1, 0, 0, true, false)]);
        assert_eq!(
            deferred.get(&Entity::from_raw_u32(1).unwrap()),
            Some(&Entity::from_raw_u32(2).unwrap())
        );
    }
}
