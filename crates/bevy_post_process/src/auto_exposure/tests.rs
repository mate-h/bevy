//! CPU-side tests for the auto exposure temporal adaptation math.
//!
//! The functions below are an exact Rust mirror of the metering-reference blend and the
//! temporal smoothing / two-stage clamp at the end of `compute_average` in
//! `auto_exposure.wgsl`, operation for operation. If the shader math changes, these mirrors
//! must be updated to match.

use super::{
    buffers::{build_uniform, initial_state},
    pipeline::{AutoExposureState, AutoExposureUniform},
    AutoExposure, AutoExposureExternalReference, PhysiologicalAdaptation,
};

/// Mirror of the external-reference blend and metering bias in `compute_average`.
fn blend_references(avg_lum: f32, settings: &AutoExposureUniform) -> f32 {
    let mut avg_lum = avg_lum;
    if settings.external_reference_weight > 0.0 {
        avg_lum = (avg_lum + settings.external_reference_ev * settings.external_reference_weight)
            / (1.0 + settings.external_reference_weight);
    }
    if settings.metering_bias != 0.0 {
        avg_lum += settings.metering_bias;
    }
    avg_lum
}

/// Mirror of the temporal smoothing, long-term envelope update, and two-stage clamp in
/// `compute_average`. Returns the exposure that would be applied to the view.
fn adaptation_step(
    state: &mut AutoExposureState,
    target_exposure: f32,
    delta_time: f32,
    settings: &AutoExposureUniform,
) -> f32 {
    let mut exposure = state.exposure;
    let delta = target_exposure - exposure;
    if target_exposure > exposure {
        let speed_down = settings.speed_down * delta_time;
        let exp_down = speed_down / settings.exponential_transition_distance;
        exposure += f32::min(speed_down, delta * exp_down);
    } else {
        let speed_up = settings.speed_up * delta_time;
        let exp_up = speed_up / settings.exponential_transition_distance;
        exposure += f32::max(-speed_up, delta * exp_up);
    }

    let mut long_term = state.long_term;
    let long_term_delta = exposure - long_term;
    if exposure > long_term {
        let long_term_speed_down = settings.long_term_speed_down * delta_time;
        let long_term_exp_down = long_term_speed_down / settings.exponential_transition_distance;
        long_term += f32::min(long_term_speed_down, long_term_delta * long_term_exp_down);
    } else {
        let long_term_speed_up = settings.long_term_speed_up * delta_time;
        let long_term_exp_up = long_term_speed_up / settings.exponential_transition_distance;
        long_term += f32::max(-long_term_speed_up, long_term_delta * long_term_exp_up);
    }

    if settings.physiological != 0 {
        exposure = exposure.clamp(
            long_term - settings.long_term_bound_down,
            long_term + settings.long_term_bound_up,
        );
    }

    state.exposure = exposure;
    state.long_term = long_term;

    exposure
}

/// The single-stage smoothing exactly as it existed before the two-stage model was added.
/// Used to prove that the disabled path is numerically identical to the old behavior.
fn legacy_single_stage_step(
    exposure: f32,
    target_exposure: f32,
    delta_time: f32,
    settings: &AutoExposureUniform,
) -> f32 {
    let delta = target_exposure - exposure;
    if target_exposure > exposure {
        let speed_down = settings.speed_down * delta_time;
        let exp_down = speed_down / settings.exponential_transition_distance;
        exposure + f32::min(speed_down, delta * exp_down)
    } else {
        let speed_up = settings.speed_up * delta_time;
        let exp_up = speed_up / settings.exponential_transition_distance;
        exposure + f32::max(-speed_up, delta * exp_up)
    }
}

const DT: f32 = 1.0 / 60.0;

fn enabled_settings(adaptation: PhysiologicalAdaptation) -> AutoExposureUniform {
    build_uniform(
        &AutoExposure {
            physiological: Some(adaptation),
            ..Default::default()
        },
        None,
    )
}

/// A deterministic, wandering target sequence exercising both adaptation directions and
/// both the linear and exponential smoothing regions.
fn target_sequence(step: usize) -> f32 {
    match (step / 120) % 4 {
        0 => 6.0,
        1 => -4.0,
        2 => 0.25,
        _ => -7.5,
    }
}

#[test]
fn gpu_struct_layouts_match_the_wgsl_structs() {
    use bevy_render::render_resource::ShaderType;

    // naga computes span=64 for the WGSL `AutoExposure` uniform struct (16 sequential
    // 4-byte scalars) and span=8 for `AutoExposureState`; the encase layouts must agree.
    assert_eq!(AutoExposureUniform::min_size().get(), 64);
    assert_eq!(AutoExposureState::min_size().get(), 8);
}

#[test]
fn default_component_is_configuration_identical_to_legacy() {
    let settings = AutoExposure::default();

    // The pre-existing fields keep their documented legacy defaults...
    assert_eq!(settings.range, -8.0..=8.0);
    assert_eq!(settings.filter, 0.10..=0.90);
    assert_eq!(settings.speed_brighten, 3.0);
    assert_eq!(settings.speed_darken, 1.0);
    assert_eq!(settings.exponential_transition_distance, 1.5);

    // ...and the new fields default to their neutral values.
    assert_eq!(settings.metering_bias, 0.0);
    assert!(settings.physiological.is_none());
}

#[test]
fn default_uniform_is_neutral() {
    let uniform = build_uniform(&AutoExposure::default(), None);

    // The values the legacy implementation uploaded, unchanged.
    assert_eq!(uniform.min_log_lum, -8.0);
    assert_eq!(uniform.inv_log_lum_range, 1.0 / 16.0);
    assert_eq!(uniform.log_lum_range, 16.0);
    assert_eq!(uniform.low_percent, 0.10);
    assert_eq!(uniform.high_percent, 0.90);
    assert_eq!(uniform.speed_up, 3.0);
    assert_eq!(uniform.speed_down, 1.0);
    assert_eq!(uniform.exponential_transition_distance, 1.5);

    // The new fields are at their neutral values, so the shader skips both the
    // reference blend and the two-stage clamp.
    assert_eq!(uniform.metering_bias, 0.0);
    assert_eq!(uniform.external_reference_weight, 0.0);
    assert_eq!(uniform.physiological, 0);

    // The initial GPU state matches the legacy initial state (exposure 0, clamped into
    // range), with the envelope starting at the same neutral value.
    let state = initial_state(&AutoExposure::default());
    assert_eq!(state.exposure, 0.0);
    assert_eq!(state.long_term, 0.0);
}

#[test]
fn disabled_two_stage_is_bit_identical_to_legacy_single_stage() {
    let settings = build_uniform(&AutoExposure::default(), None);
    assert_eq!(settings.physiological, 0);

    let mut state = initial_state(&AutoExposure::default());
    let mut legacy_exposure = state.exposure;

    for step in 0..1200 {
        let target = target_sequence(step);
        let applied = adaptation_step(&mut state, target, DT, &settings);
        legacy_exposure = legacy_single_stage_step(legacy_exposure, target, DT, &settings);

        assert_eq!(
            applied.to_bits(),
            legacy_exposure.to_bits(),
            "exposure diverged from the legacy math at step {step}"
        );
    }
}

#[test]
fn long_term_envelope_bounds_short_term_exposure() {
    let adaptation = PhysiologicalAdaptation::default();
    let settings = enabled_settings(adaptation);
    let mut state = AutoExposureState {
        exposure: 0.0,
        long_term: 0.0,
    };

    // The scene suddenly becomes 10 EV darker; the short-term stage wants to raise the
    // exposure all the way to +10 EV at 1 EV/s.
    let target = 10.0;
    for _ in 0..(10.0 / DT) as usize {
        let applied = adaptation_step(&mut state, target, DT, &settings);
        assert!(
            applied <= state.long_term + adaptation.bound_darken + 1e-4,
            "exposure {applied} escaped the long-term bound {}",
            state.long_term + adaptation.bound_darken
        );
    }

    // After 10 simulated seconds, the unbounded single-stage exposure would be at ~8.5 EV;
    // the bounded exposure must still sit at the envelope's upper bound instead
    // (envelope ≈ 10 s × 0.01 EV/s = 0.1 EV, plus the 2 EV bound).
    assert!(
        state.exposure < 3.0,
        "exposure {} was not bounded by the long-term envelope",
        state.exposure
    );
    assert!(
        (state.exposure - (state.long_term + adaptation.bound_darken)).abs() < 1e-4,
        "exposure {} should be saturated at the upper bound {}",
        state.exposure,
        state.long_term + adaptation.bound_darken
    );
}

#[test]
fn bounded_exposure_converges_once_the_envelope_catches_up() {
    let settings = enabled_settings(PhysiologicalAdaptation::default());
    let mut state = AutoExposureState {
        exposure: 0.0,
        long_term: 0.0,
    };

    // Simulate 30 minutes of dark adaptation at 10 Hz.
    let target = 10.0;
    let dt = 0.1;
    for _ in 0..18000 {
        adaptation_step(&mut state, target, dt, &settings);
    }

    assert!(
        (state.exposure - target).abs() < 0.05,
        "exposure {} did not converge to the target {target}",
        state.exposure
    );
    assert!(
        (state.long_term - target).abs() < 0.5,
        "long-term envelope {} did not follow the exposure to {target}",
        state.long_term
    );
}

#[test]
fn convergence_within_bounds_is_unaffected_by_the_envelope() {
    // A target within the bounding range of the (converged) envelope adapts exactly as
    // fast as the legacy single-stage path.
    let settings = enabled_settings(PhysiologicalAdaptation::default());
    let mut state = AutoExposureState {
        exposure: 0.0,
        long_term: 0.0,
    };
    let mut legacy_exposure = 0.0;

    let target = 1.5;
    for _ in 0..600 {
        adaptation_step(&mut state, target, DT, &settings);
        legacy_exposure = legacy_single_stage_step(legacy_exposure, target, DT, &settings);
    }

    assert!(
        (state.exposure - legacy_exposure).abs() < 1e-4,
        "in-bounds adaptation ({}) should match the single-stage path ({legacy_exposure})",
        state.exposure
    );
    assert!((state.exposure - target).abs() < 0.05);
}

#[test]
fn light_adaptation_is_faster_than_dark_adaptation() {
    let adaptation = PhysiologicalAdaptation::default();
    let settings = enabled_settings(adaptation);

    // Fully adapted to a dark environment, the scene becomes 8 EV brighter.
    let mut state = AutoExposureState {
        exposure: 8.0,
        long_term: 8.0,
    };
    let mut steps_to_brighten = None;
    for step in 0..1_000_000 {
        adaptation_step(&mut state, 0.0, DT, &settings);
        if (state.exposure - 0.0).abs() < 0.25 {
            steps_to_brighten = Some(step);
            break;
        }
    }

    // Fully adapted to a bright environment, the scene becomes 8 EV darker.
    let mut state = AutoExposureState {
        exposure: 0.0,
        long_term: 0.0,
    };
    let mut steps_to_darken = None;
    for step in 0..1_000_000 {
        adaptation_step(&mut state, 8.0, DT, &settings);
        if (state.exposure - 8.0).abs() < 0.25 {
            steps_to_darken = Some(step);
            break;
        }
    }

    let (steps_to_brighten, steps_to_darken) = (
        steps_to_brighten.expect("light adaptation never converged"),
        steps_to_darken.expect("dark adaptation never converged"),
    );
    assert!(
        steps_to_brighten < steps_to_darken,
        "light adaptation ({steps_to_brighten} steps) must be faster than dark adaptation \
        ({steps_to_darken} steps)"
    );
}

#[test]
fn long_term_envelope_is_rate_limited() {
    let adaptation = PhysiologicalAdaptation::default();
    let settings = enabled_settings(adaptation);
    let mut state = AutoExposureState {
        exposure: 0.0,
        long_term: 0.0,
    };

    let max_speed = adaptation.speed_brighten.max(adaptation.speed_darken);
    for step in 0..2400 {
        let previous = state.long_term;
        adaptation_step(&mut state, target_sequence(step), DT, &settings);
        assert!(
            (state.long_term - previous).abs() <= max_speed * DT + 1e-6,
            "the envelope moved faster than its speed limit at step {step}"
        );
    }
}

#[test]
fn external_reference_blend_math() {
    let neutral = build_uniform(&AutoExposure::default(), None);

    // Without a reference, the metered value passes through bit-identically.
    let avg = -3.7f32;
    assert_eq!(blend_references(avg, &neutral).to_bits(), avg.to_bits());

    // A reference with weight 1.0 averages equally with the histogram.
    let settings = build_uniform(
        &AutoExposure::default(),
        Some(&AutoExposureExternalReference {
            ev: 5.0,
            weight: 1.0,
        }),
    );
    assert_eq!(blend_references(3.0, &settings), 4.0);

    // A dominant weight pulls the metered value to the reference.
    let settings = build_uniform(
        &AutoExposure::default(),
        Some(&AutoExposureExternalReference {
            ev: 5.0,
            weight: 1e6,
        }),
    );
    assert!((blend_references(3.0, &settings) - 5.0).abs() < 1e-4);

    // The metering bias is applied after the references are fused.
    let settings = build_uniform(
        &AutoExposure {
            metering_bias: 1.0,
            ..Default::default()
        },
        Some(&AutoExposureExternalReference {
            ev: 5.0,
            weight: 1.0,
        }),
    );
    assert_eq!(blend_references(3.0, &settings), 5.0);
}

#[test]
fn invalid_inputs_are_sanitized() {
    // A non-finite metering bias is ignored.
    let settings = build_uniform(
        &AutoExposure {
            metering_bias: f32::NAN,
            ..Default::default()
        },
        None,
    );
    assert_eq!(settings.metering_bias, 0.0);

    // Invalid external references are ignored entirely.
    for reference in [
        AutoExposureExternalReference {
            ev: f32::NAN,
            weight: 1.0,
        },
        AutoExposureExternalReference {
            ev: 1.0,
            weight: f32::INFINITY,
        },
        AutoExposureExternalReference {
            ev: 1.0,
            weight: -1.0,
        },
    ] {
        let settings = build_uniform(&AutoExposure::default(), Some(&reference));
        assert_eq!(settings.external_reference_ev, 0.0);
        assert_eq!(settings.external_reference_weight, 0.0);
    }

    // Invalid physiological fields are reset to their defaults; valid ones are kept.
    let defaults = PhysiologicalAdaptation::default();
    let settings = enabled_settings(PhysiologicalAdaptation {
        speed_brighten: f32::NAN,
        speed_darken: -1.0,
        bound_brighten: 4.0,
        bound_darken: f32::INFINITY,
        initial_long_term_ev: None,
    });
    assert_eq!(settings.physiological, 1);
    assert_eq!(settings.long_term_speed_up, defaults.speed_brighten);
    assert_eq!(settings.long_term_speed_down, defaults.speed_darken);
    assert_eq!(settings.long_term_bound_down, 4.0);
    assert_eq!(settings.long_term_bound_up, defaults.bound_darken);
}

#[test]
fn initial_state_seeds_the_long_term_envelope() {
    // Without physiological settings, the envelope starts at the neutral exposure.
    let state = initial_state(&AutoExposure::default());
    assert_eq!(state.long_term, state.exposure);

    // An explicit initial envelope value is honored.
    let state = initial_state(&AutoExposure {
        physiological: Some(PhysiologicalAdaptation {
            initial_long_term_ev: Some(-4.5),
            ..Default::default()
        }),
        ..Default::default()
    });
    assert_eq!(state.exposure, 0.0);
    assert_eq!(state.long_term, -4.5);

    // A non-finite initial envelope value falls back to the neutral exposure.
    let state = initial_state(&AutoExposure {
        physiological: Some(PhysiologicalAdaptation {
            initial_long_term_ev: Some(f32::NAN),
            ..Default::default()
        }),
        ..Default::default()
    });
    assert_eq!(state.long_term, 0.0);
}
