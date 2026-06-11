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
    pipeline::{AutoExposureState, AutoExposureUniform, ViewAutoExposurePipeline},
    settings::PhysiologicalAdaptation,
    AutoExposure, AutoExposureExternalReference, AutoWhiteBalance,
};

/// The CIE 1931 *xy* chromaticity of the D65 white point, matching `D65_XY`
/// in `bevy_render::view` and `AWB_D65_XY` in `auto_exposure.wgsl`; keep them
/// in sync.
const D65_XY: (f32, f32) = (0.31272, 0.32903);

#[derive(Resource, Default)]
pub(super) struct AutoExposureBuffers {
    pub(super) buffers: EntityHashMap<AutoExposureBuffer>,
}

pub(super) struct AutoExposureBuffer {
    pub(super) state: StorageBuffer<AutoExposureState>,
    pub(super) settings: UniformBuffer<AutoExposureUniform>,
}

/// One extracted camera that needs its settings uniform (re)built. Cameras
/// carry [`AutoExposure`], [`AutoWhiteBalance`], or both; the shared metering
/// pass serves whichever is present.
type ExtractedCamera = (
    Entity,
    Option<AutoExposure>,
    Option<AutoExposureExternalReference>,
    Option<AutoWhiteBalance>,
);

#[derive(Resource)]
pub(super) struct ExtractedStateBuffers {
    changed: Vec<ExtractedCamera>,
    /// Render-world entities whose camera stopped metering entirely (all
    /// metering components removed while the camera itself stays alive).
    /// These are *render* entities because [`AutoExposureBuffers`] is keyed
    /// by render entities; cameras that were despawned outright are handled
    /// by the liveness sweep in [`prepare_buffers`] instead.
    removed: Vec<Entity>,
}

pub(super) fn extract_buffers(
    mut commands: Commands,
    changed: Extract<
        Query<
            (
                RenderEntity,
                Option<&AutoExposure>,
                Option<&AutoExposureExternalReference>,
                Option<&AutoWhiteBalance>,
            ),
            (
                Or<(With<AutoExposure>, With<AutoWhiteBalance>)>,
                Or<(
                    Changed<AutoExposure>,
                    Changed<AutoExposureExternalReference>,
                    Changed<AutoWhiteBalance>,
                )>,
            ),
        >,
    >,
    mut removed: Extract<RemovedComponents<AutoExposure>>,
    mut removed_references: Extract<RemovedComponents<AutoExposureExternalReference>>,
    mut removed_white_balance: Extract<RemovedComponents<AutoWhiteBalance>>,
    cameras: Extract<
        Query<
            (
                RenderEntity,
                Option<&AutoExposure>,
                Option<&AutoExposureExternalReference>,
                Option<&AutoWhiteBalance>,
            ),
            Or<(With<AutoExposure>, With<AutoWhiteBalance>)>,
        >,
    >,
    render_entities: Extract<Query<RenderEntity>>,
) {
    let mut changed: Vec<ExtractedCamera> = changed
        .iter()
        .map(|(entity, settings, reference, white_balance)| {
            (
                entity,
                settings.cloned(),
                reference.copied(),
                white_balance.copied(),
            )
        })
        .collect();
    let mut fully_removed = Vec::new();

    // Removing one of the components does not trigger the `Changed` filters
    // above, but the settings uniform must still be rebuilt from whatever is
    // left on the camera. Read the live component state instead of assuming
    // the removed component is gone: a remove + re-insert within the same
    // frame still buffers a removal event, and unconditionally pushing the
    // component as absent here would override the freshly inserted value
    // (the `changed` entries above are processed first, in order). Only when
    // the camera no longer matches the metering query at all is the buffer
    // torn down.
    {
        let mut handle_removal = |entity: Entity| {
            if let Ok((render_entity, settings, reference, white_balance)) = cameras.get(entity) {
                changed.push((
                    render_entity,
                    settings.cloned(),
                    reference.copied(),
                    white_balance.copied(),
                ));
            } else if let Ok(render_entity) = render_entities.get(entity) {
                // The camera is still alive but no longer meters: tear its
                // buffer down by its *render*-world key — the buffer map is
                // keyed by render entities, so pushing the main-world entity
                // here would silently leave the buffer (and the metering
                // dispatches consuming it) alive forever.
                fully_removed.push(render_entity);
            }
            // Otherwise the camera was despawned outright; its render entity
            // is torn down by the sync machinery and `prepare_buffers`'
            // liveness sweep drops the buffer.
        };

        for entity in removed.read() {
            handle_removal(entity);
        }
        for entity in removed_white_balance.read() {
            handle_removal(entity);
        }
        for entity in removed_references.read() {
            // The reference alone never owns a buffer; only re-push live
            // state for cameras that still meter.
            if let Ok((render_entity, settings, reference, white_balance)) = cameras.get(entity) {
                changed.push((
                    render_entity,
                    settings.cloned(),
                    reference.copied(),
                    white_balance.copied(),
                ));
            }
        }
    }

    commands.insert_resource(ExtractedStateBuffers {
        changed,
        removed: fully_removed,
    });
}

pub(super) fn prepare_buffers(
    mut commands: Commands,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    mut extracted: ResMut<ExtractedStateBuffers>,
    mut buffers: ResMut<AutoExposureBuffers>,
    live_entities: Query<()>,
) {
    for (entity, settings, reference, white_balance) in extracted.changed.drain(..) {
        let uniform = build_uniform(
            settings.as_ref(),
            reference.as_ref(),
            white_balance.as_ref(),
        );

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
                    state: StorageBuffer::from(initial_state(settings.as_ref())),
                    settings: UniformBuffer::from(uniform),
                });

                value.state.write_buffer(&device, &queue);
                value.settings.write_buffer(&device, &queue);
            }
        }
    }

    for entity in extracted.removed.drain(..) {
        buffers.buffers.remove(&entity);

        // Also drop the cached per-view pipeline component: the queue system
        // only ever *inserts* it, so without this a camera that stopped
        // metering would keep dispatching the metering pass every frame
        // against the (now stale) buffer.
        if let Ok(mut entity_commands) = commands.get_entity(entity) {
            entity_commands.remove::<ViewAutoExposurePipeline>();
        }
    }

    // Cameras that are despawned outright never make it into the `removed`
    // list (their main-world entity is gone before the removal events are
    // read), so sweep out buffers whose render-world entity no longer exists.
    buffers
        .buffers
        .retain(|&entity, _| live_entities.contains(entity));
}

/// Builds the settings uniform for one view from the optional [`AutoExposure`],
/// [`AutoExposureExternalReference`], and [`AutoWhiteBalance`] components,
/// sanitizing invalid values.
///
/// When [`AutoExposure::physiological`] is `None`, the long-term envelope parameters are
/// still filled in (with their defaults) because the compute shader keeps the envelope
/// tracking the short-term exposure even while it is disabled; only the
/// `physiological` flag controls whether the envelope actually bounds the exposure.
///
/// When the camera has no [`AutoExposure`] component at all (auto white
/// balance running alone), the default metering configuration is used with
/// both adaptation speeds forced to zero: the histogram still runs (the white
/// balance measurement reuses its metering range and total weight), but the
/// exposure state never moves from its neutral initial value and the
/// write-back adds an exact `0.0`.
pub(super) fn build_uniform(
    settings: Option<&AutoExposure>,
    reference: Option<&AutoExposureExternalReference>,
    white_balance: Option<&AutoWhiteBalance>,
) -> AutoExposureUniform {
    let neutral;
    let settings = match settings {
        Some(settings) => settings,
        None => {
            neutral = AutoExposure {
                speed_brighten: 0.0,
                speed_darken: 0.0,
                ..Default::default()
            };
            &neutral
        }
    };

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

    let white_balance = white_balance.map(AutoWhiteBalance::sanitized);

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
        awb_speed: white_balance.map_or(0.0, |wb| wb.speed),
        awb_anchor: white_balance.map_or(0.0, |wb| wb.virtual_light_anchor),
        awb_enabled: white_balance.is_some() as u32,
        awb_pad: 0,
    }
}

/// Builds the initial per-view adaptation state for one view.
///
/// The short-term exposure starts at the neutral value the classic implementation used;
/// the long-term envelope starts at
/// [`PhysiologicalAdaptation::initial_long_term_ev`] if configured, and at the same
/// neutral value otherwise. The adapted white-balance chromaticity always
/// starts at the neutral D65 white point.
pub(super) fn initial_state(settings: Option<&AutoExposure>) -> AutoExposureState {
    let (min_log_lum, max_log_lum) = settings
        .map(|settings| settings.range.clone().into_inner())
        .unwrap_or_else(|| AutoExposure::default().range.into_inner());
    let exposure = 0.0f32.clamp(min_log_lum, max_log_lum);

    let long_term = settings
        .and_then(|settings| settings.physiological.as_ref())
        .map(PhysiologicalAdaptation::sanitized)
        .and_then(|adaptation| adaptation.initial_long_term_ev)
        .unwrap_or(exposure);

    AutoExposureState {
        exposure,
        long_term,
        chroma_x: D65_XY.0,
        chroma_y: D65_XY.1,
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
