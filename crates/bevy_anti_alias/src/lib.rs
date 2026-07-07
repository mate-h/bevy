#![expect(missing_docs, reason = "Not all docs are written yet, see #3492.")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(
    html_logo_url = "https://bevy.org/assets/icon.png",
    html_favicon_url = "https://bevy.org/assets/icon.png"
)]

use bevy_app::{Plugin, PostUpdate};
use bevy_camera::{Camera, NeedsSceneLinearAa};
use bevy_ecs::prelude::*;
use contrast_adaptive_sharpening::CasPlugin;
use fxaa::FxaaPlugin;
use smaa::SmaaPlugin;
use taa::{TemporalAntiAliasPlugin, TemporalAntiAliasing};

pub mod contrast_adaptive_sharpening;
#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
pub mod dlss;
pub mod fxaa;
pub mod smaa;
pub mod taa;

/// Adds fxaa, smaa, taa, contrast aware sharpening, and optional dlss support.
#[derive(Default)]
pub struct AntiAliasPlugin;

impl Plugin for AntiAliasPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_plugins((
            FxaaPlugin,
            SmaaPlugin,
            TemporalAntiAliasPlugin,
            CasPlugin,
            #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
            dlss::DlssPlugin,
        ))
        .add_systems(PostUpdate, sync_needs_scene_linear_aa);
    }
}

/// Keeps the auto-managed [`NeedsSceneLinearAa`] marker in sync with the
/// scene-linear-requiring anti-aliasing modes on each camera (temporal
/// anti-aliasing and DLSS).
///
/// The marker lets `bevy_render`'s camera extraction (which cannot depend on
/// this crate) veto the SDR in-shader tone-mapping fast path: these modes
/// reproject and resolve the scene-referred buffer before tone mapping, so the
/// camera must keep its `Rgba16Float` intermediate. FXAA and SMAA run on the
/// tone-mapped image and so do not participate. Runs in [`PostUpdate`].
fn sync_needs_scene_linear_aa(
    mut commands: Commands,
    cameras: Query<(Entity, Has<TemporalAntiAliasing>, Has<NeedsSceneLinearAa>), With<Camera>>,
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))] dlss_cameras: Query<
        (),
        With<dlss::Dlss>,
    >,
) {
    for (entity, taa, has_marker) in &cameras {
        #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
        let needs = taa || dlss_cameras.contains(entity);
        #[cfg(not(all(feature = "dlss", not(feature = "force_disable_dlss"))))]
        let needs = taa;
        if needs && !has_marker {
            commands.entity(entity).insert(NeedsSceneLinearAa);
        } else if !needs && has_marker {
            commands.entity(entity).remove::<NeedsSceneLinearAa>();
        }
    }
}
