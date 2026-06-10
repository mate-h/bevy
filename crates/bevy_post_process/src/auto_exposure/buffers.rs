use bevy_ecs::{entity::EntityHashMap, prelude::*};
use bevy_platform::collections::hash_map::Entry;
use bevy_render::{
    render_resource::{StorageBuffer, UniformBuffer},
    renderer::{RenderDevice, RenderQueue},
    sync_world::RenderEntity,
    Extract,
};
use bevy_utils::once;
use tracing::warn;

use super::{
    pipeline::{AutoExposureState, AutoExposureUniform},
    settings::PhysiologicalAdaptation,
    AutoExposure, AutoExposureExternalReference,
};

#[derive(Resource, Default)]
pub(super) struct AutoExposureBuffers {
    pub(super) buffers: EntityHashMap<AutoExposureBuffer>,
}

pub(super) struct AutoExposureBuffer {
    pub(super) state: StorageBuffer<AutoExposureState>,
    pub(super) settings: UniformBuffer<AutoExposureUniform>,
}

#[derive(Resource)]
pub(super) struct ExtractedStateBuffers {
    changed: Vec<(Entity, AutoExposure, Option<AutoExposureExternalReference>)>,
    removed: Vec<Entity>,
}

pub(super) fn extract_buffers(
    mut commands: Commands,
    changed: Extract<
        Query<
            (
                RenderEntity,
                &AutoExposure,
                Option<&AutoExposureExternalReference>,
            ),
            Or<(
                Changed<AutoExposure>,
                Changed<AutoExposureExternalReference>,
            )>,
        >,
    >,
    mut removed: Extract<RemovedComponents<AutoExposure>>,
    mut removed_references: Extract<RemovedComponents<AutoExposureExternalReference>>,
    cameras: Extract<
        Query<(
            RenderEntity,
            &AutoExposure,
            Option<&AutoExposureExternalReference>,
        )>,
    >,
) {
    let mut changed: Vec<_> = changed
        .iter()
        .map(|(entity, settings, reference)| (entity, settings.clone(), reference.copied()))
        .collect();

    // Removing only the `AutoExposureExternalReference` component does not trigger the
    // `Changed` filters above, but the settings uniform must still be rebuilt without
    // the reference. Read the live component state instead of assuming the reference is
    // gone: a remove + re-insert within the same frame still buffers a removal event, and
    // unconditionally pushing `None` here would override the freshly inserted reference
    // (the `changed` entries above are processed first, in order).
    for entity in removed_references.read() {
        if let Ok((render_entity, settings, reference)) = cameras.get(entity) {
            changed.push((render_entity, settings.clone(), reference.copied()));
        }
    }

    commands.insert_resource(ExtractedStateBuffers {
        changed,
        removed: removed.read().collect(),
    });
}

pub(super) fn prepare_buffers(
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut extracted: ResMut<ExtractedStateBuffers>,
    mut buffers: ResMut<AutoExposureBuffers>,
) {
    for (entity, settings, reference) in extracted.changed.drain(..) {
        let uniform = build_uniform(&settings, reference.as_ref());

        match buffers.buffers.entry(entity) {
            Entry::Occupied(mut entry) => {
                // Update the settings buffer, but skip updating the state buffer.
                // The state buffer is skipped so that the animation stays continuous.
                let value = entry.get_mut();
                value.settings.set(uniform);
                value.settings.write_buffer(&device, &queue);
            }
            Entry::Vacant(entry) => {
                let value = entry.insert(AutoExposureBuffer {
                    state: StorageBuffer::from(initial_state(&settings)),
                    settings: UniformBuffer::from(uniform),
                });

                value.state.write_buffer(&device, &queue);
                value.settings.write_buffer(&device, &queue);
            }
        }
    }

    for entity in extracted.removed.drain(..) {
        buffers.buffers.remove(&entity);
    }
}

/// Builds the settings uniform for one view from the [`AutoExposure`] component and the
/// optional [`AutoExposureExternalReference`] component, sanitizing invalid values.
///
/// When [`AutoExposure::physiological`] is `None`, the long-term envelope parameters are
/// still filled in (with their defaults) because the compute shader keeps the envelope
/// tracking the short-term exposure even while it is disabled; only the
/// `physiological` flag controls whether the envelope actually bounds the exposure.
pub(super) fn build_uniform(
    settings: &AutoExposure,
    reference: Option<&AutoExposureExternalReference>,
) -> AutoExposureUniform {
    let (min_log_lum, max_log_lum) = settings.range.clone().into_inner();
    let (low_percent, high_percent) = settings.filter.clone().into_inner();

    let metering_bias = if settings.metering_bias.is_finite() {
        settings.metering_bias
    } else {
        once!(warn!(
            "AutoExposure::metering_bias must be finite; ignoring the configured value"
        ));
        0.0
    };

    let (external_reference_ev, external_reference_weight) =
        reference.map_or((0.0, 0.0), sanitize_external_reference);

    let adaptation = settings
        .physiological
        .as_ref()
        .map(PhysiologicalAdaptation::sanitized)
        .unwrap_or_default();

    AutoExposureUniform {
        min_log_lum,
        inv_log_lum_range: 1.0 / (max_log_lum - min_log_lum),
        log_lum_range: max_log_lum - min_log_lum,
        low_percent,
        high_percent,
        speed_up: settings.speed_brighten,
        speed_down: settings.speed_darken,
        exponential_transition_distance: settings.exponential_transition_distance,
        metering_bias,
        external_reference_ev,
        external_reference_weight,
        long_term_speed_up: adaptation.speed_brighten,
        long_term_speed_down: adaptation.speed_darken,
        long_term_bound_up: adaptation.bound_darken,
        long_term_bound_down: adaptation.bound_brighten,
        physiological: settings.physiological.is_some() as u32,
    }
}

/// Builds the initial per-view adaptation state for one view.
///
/// The short-term exposure starts at the neutral value the classic implementation used;
/// the long-term envelope starts at
/// [`PhysiologicalAdaptation::initial_long_term_ev`] if configured, and at the same
/// neutral value otherwise.
pub(super) fn initial_state(settings: &AutoExposure) -> AutoExposureState {
    let (min_log_lum, max_log_lum) = settings.range.clone().into_inner();
    let exposure = 0.0f32.clamp(min_log_lum, max_log_lum);

    let long_term = settings
        .physiological
        .as_ref()
        .map(PhysiologicalAdaptation::sanitized)
        .and_then(|adaptation| adaptation.initial_long_term_ev)
        .unwrap_or(exposure);

    AutoExposureState {
        exposure,
        long_term,
    }
}

/// Returns the sanitized `(ev, weight)` pair for an external metering reference,
/// ignoring the reference entirely (weight 0) if any value is invalid.
fn sanitize_external_reference(reference: &AutoExposureExternalReference) -> (f32, f32) {
    if !reference.ev.is_finite() || !reference.weight.is_finite() || reference.weight < 0.0 {
        once!(warn!(
            "AutoExposureExternalReference::ev and ::weight must be finite, and the weight \
            non-negative; ignoring the reference"
        ));
        (0.0, 0.0)
    } else {
        (reference.ev, reference.weight)
    }
}
