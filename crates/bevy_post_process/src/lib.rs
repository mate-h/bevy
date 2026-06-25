#![expect(missing_docs, reason = "Not all docs are written yet, see #3492.")]
#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(
    html_logo_url = "https://bevy.org/assets/icon.png",
    html_favicon_url = "https://bevy.org/assets/icon.png"
)]

pub mod auto_exposure;
pub mod bloom;
pub mod dof;
pub mod effect_stack;
pub mod motion_blur;
pub mod msaa_writeback;

use crate::{
    auto_exposure::AutoExposure, bloom::Bloom, bloom::BloomPlugin, dof::DepthOfField,
    dof::DepthOfFieldPlugin, effect_stack::EffectStackPlugin, motion_blur::MotionBlur,
    motion_blur::MotionBlurPlugin, msaa_writeback::MsaaWritebackPlugin,
};
use bevy_app::{App, Plugin, PostUpdate};
use bevy_camera::{Camera, NeedsSceneLinearPost};
use bevy_ecs::prelude::*;
use bevy_shader::load_shader_library;

/// Adds bloom, motion blur, depth of field, and chromatic aberration support.
#[derive(Default)]
pub struct PostProcessPlugin;

impl Plugin for PostProcessPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "gaussian_blur.wgsl");

        app.add_plugins((
            MsaaWritebackPlugin,
            BloomPlugin,
            MotionBlurPlugin,
            DepthOfFieldPlugin,
            EffectStackPlugin,
        ))
        .add_systems(PostUpdate, sync_needs_scene_linear_post);
    }
}

/// Keeps the auto-managed [`NeedsSceneLinearPost`] marker in sync with the
/// scene-linear-requiring post-process effects on each camera (bloom, auto
/// exposure, depth of field, motion blur).
///
/// The marker lets `bevy_render`'s camera extraction (which cannot depend on
/// this crate) veto the SDR in-shader tone-mapping fast path: these effects
/// sample the scene-referred buffer before tone mapping, so the camera must
/// keep its `Rgba16Float` intermediate. Runs in [`PostUpdate`]; effects added
/// after that point are seen the next frame.
fn sync_needs_scene_linear_post(
    mut commands: Commands,
    cameras: Query<
        (
            Entity,
            Has<Bloom>,
            Has<AutoExposure>,
            Has<DepthOfField>,
            Has<MotionBlur>,
            Has<NeedsSceneLinearPost>,
        ),
        With<Camera>,
    >,
) {
    for (entity, bloom, auto_exposure, dof, motion_blur, has_marker) in &cameras {
        let needs = bloom || auto_exposure || dof || motion_blur;
        if needs && !has_marker {
            commands.entity(entity).insert(NeedsSceneLinearPost);
        } else if !needs && has_marker {
            commands.entity(entity).remove::<NeedsSceneLinearPost>();
        }
    }
}
