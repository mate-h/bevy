use super::{prepare::PathtracerAccumulationTexture, Pathtracer};
use bevy_camera::Camera;
use bevy_ecs::{
    change_detection::DetectChanges,
    system::{Commands, Query},
    world::Ref,
};
use bevy_render::{sync_world::RenderEntity, Extract};
use bevy_transform::components::GlobalTransform;

pub fn extract_pathtracer(
    cameras_3d: Extract<
        Query<(
            RenderEntity,
            &Camera,
            Ref<GlobalTransform>,
            Option<&Pathtracer>,
        )>,
    >,
    previous: Query<&Pathtracer>,
    mut commands: Commands,
) {
    for (entity, camera, global_transform, pathtracer) in &cameras_3d {
        let mut entity_commands = commands
            .get_entity(entity)
            .expect("Camera entity wasn't synced.");
        if let Some(pathtracer) = pathtracer
            && camera.is_active
        {
            let mut pathtracer = pathtracer.clone();
            pathtracer.reset |= global_transform.is_changed()
                || accumulation_needs_reset(&pathtracer, previous.get(entity).ok());
            entity_commands.insert(pathtracer);
        } else {
            entity_commands.remove::<(Pathtracer, PathtracerAccumulationTexture)>();
        }
    }
}

// Keep pending GPU clears sticky until the render node consumes them. Sample-limit
// changes intentionally retain history, allowing a completed checkpoint to resume.
fn accumulation_needs_reset(current: &Pathtracer, previous: Option<&Pathtracer>) -> bool {
    previous.is_none_or(|old| {
        old.reset
            || old.seed != current.seed
            || old.accumulation_output != current.accumulation_output
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_asset::Handle;
    use bevy_image::Image;

    #[test]
    fn reset_stays_pending_until_dispatch_consumes_it() {
        let current = Pathtracer::default();
        assert!(accumulation_needs_reset(&current, None));
        let mut previous = Pathtracer {
            reset: true,
            ..Default::default()
        };
        // Even unchanged input must preserve a clear deferred by an unavailable pipeline.
        assert!(accumulation_needs_reset(&current, Some(&previous)));
        previous.reset = false; // Render node successfully encoded clear + dispatch.
        assert!(!accumulation_needs_reset(&current, Some(&previous)));
    }

    #[test]
    fn seed_and_output_changes_reset_but_sample_limits_retain_history() {
        let previous = Pathtracer::default();
        let mut current = previous.clone();
        current.max_samples = 512;
        assert!(!accumulation_needs_reset(&current, Some(&previous)));
        current.max_samples = 2048;
        assert!(!accumulation_needs_reset(&current, Some(&previous)));
        current.max_samples = 0;
        assert!(!accumulation_needs_reset(&current, Some(&previous)));
        current.seed = 7;
        assert!(accumulation_needs_reset(&current, Some(&previous)));
        current.seed = 0;
        current.accumulation_output = Some(Handle::<Image>::default());
        assert!(accumulation_needs_reset(&current, Some(&previous)));
    }
}
