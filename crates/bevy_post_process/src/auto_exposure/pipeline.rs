use super::compensation_curve::{
    AutoExposureCompensationCurve, AutoExposureCompensationCurveUniform,
};
use bevy_asset::{load_embedded_asset, prelude::*};
use bevy_ecs::prelude::*;
use bevy_image::Image;
use bevy_render::{
    globals::GlobalsUniform,
    render_resource::{binding_types::*, *},
    view::ViewUniform,
};
use bevy_shader::Shader;
use bevy_utils::default;
use core::num::NonZero;

#[derive(Resource)]
pub struct AutoExposurePipeline {
    pub histogram_layout: BindGroupLayoutDescriptor,
    pub histogram_shader: Handle<Shader>,
}

#[derive(Component)]
pub struct ViewAutoExposurePipeline {
    pub histogram_pipeline: CachedComputePipelineId,
    pub mean_luminance_pipeline: CachedComputePipelineId,
    pub compensation_curve: Handle<AutoExposureCompensationCurve>,
    pub metering_mask: Handle<Image>,
}

/// CPU mirror of the `AutoExposure` settings uniform in `auto_exposure.wgsl`.
/// The field order and types must match the WGSL struct exactly.
#[derive(ShaderType, Clone, Copy)]
pub struct AutoExposureUniform {
    pub(super) min_log_lum: f32,
    pub(super) inv_log_lum_range: f32,
    pub(super) log_lum_range: f32,
    pub(super) low_percent: f32,
    pub(super) high_percent: f32,
    pub(super) speed_up: f32,
    pub(super) speed_down: f32,
    pub(super) exponential_transition_distance: f32,
    pub(super) metering_bias: f32,
    pub(super) external_reference_ev: f32,
    pub(super) external_reference_weight: f32,
    pub(super) long_term_speed_up: f32,
    pub(super) long_term_speed_down: f32,
    pub(super) long_term_bound_up: f32,
    pub(super) long_term_bound_down: f32,
    pub(super) physiological: u32,
}

/// CPU mirror of the per-view `AutoExposureState` storage buffer in `auto_exposure.wgsl`.
/// The field order and types must match the WGSL struct exactly.
#[derive(ShaderType, Clone, Copy, Debug, PartialEq)]
pub struct AutoExposureState {
    /// The smoothed short-term exposure correction, in EV.
    pub(super) exposure: f32,
    /// The long-term physiological adaptation envelope, in EV.
    pub(super) long_term: f32,
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub enum AutoExposurePass {
    Histogram,
    Average,
}

pub const HISTOGRAM_BIN_COUNT: u64 = 64;

pub fn init_auto_exposure_pipeline(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.insert_resource(AutoExposurePipeline {
        histogram_layout: BindGroupLayoutDescriptor::new(
            "compute histogram bind group",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    uniform_buffer::<GlobalsUniform>(false),
                    uniform_buffer::<AutoExposureUniform>(false),
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    texture_1d(TextureSampleType::Float { filterable: false }),
                    uniform_buffer::<AutoExposureCompensationCurveUniform>(false),
                    storage_buffer_sized(false, NonZero::<u64>::new(HISTOGRAM_BIN_COUNT * 4)),
                    storage_buffer::<AutoExposureState>(false),
                    storage_buffer::<ViewUniform>(true),
                ),
            ),
        ),
        histogram_shader: load_embedded_asset!(asset_server.as_ref(), "auto_exposure.wgsl"),
    });
}

impl SpecializedComputePipeline for AutoExposurePipeline {
    type Key = AutoExposurePass;

    fn specialize(&self, pass: AutoExposurePass) -> ComputePipelineDescriptor {
        ComputePipelineDescriptor {
            label: Some("luminance compute pipeline".into()),
            layout: vec![self.histogram_layout.clone()],
            shader: self.histogram_shader.clone(),
            shader_defs: vec![],
            entry_point: Some(match pass {
                AutoExposurePass::Histogram => "compute_histogram".into(),
                AutoExposurePass::Average => "compute_average".into(),
            }),
            ..default()
        }
    }
}
