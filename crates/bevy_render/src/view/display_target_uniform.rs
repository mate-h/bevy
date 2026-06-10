//! Per-view display-target plumbing: the render-world [`ViewDisplayTarget`]
//! component and the [`DisplayTargetUniform`] GPU uniform.
//!
//! [`DisplayTarget`] is authored in the main world (as a required component of
//! `Window`, or via [`ManualDisplayTargets`] for non-window targets). This
//! module resolves it per view each frame:
//!
//! 1. [`prepare_view_display_targets`] runs in
//!    [`RenderSystems::PrepareViews`](crate::RenderSystems::PrepareViews) and
//!    inserts a [`ViewDisplayTarget`] on every extracted camera view, using
//!    [`resolve_display_target`] for the *requested* target and the window
//!    surface's negotiated transfer
//!    ([`ExtractedWindow::resolved_transfer`](super::window::ExtractedWindow::resolved_transfer))
//!    for the *resolved* one: when the surface could not fulfil the requested
//!    transfer (e.g. scRGB-linear on a backend without `Rgba16Float`
//!    surfaces), the resolved target degrades to
//!    [`DisplayTarget::SDR_SRGB`], so downgraded views take the plain SDR
//!    path bit-for-bit. Views whose target cannot be resolved fall back to
//!    [`DisplayTarget::SDR_SRGB`]. All cameras rendering to the same surface
//!    resolve to the same value by construction.
//! 2. [`prepare_display_target_uniforms`] runs in
//!    [`RenderSystems::PrepareResources`](crate::RenderSystems::PrepareResources)
//!    and writes a [`DisplayTargetUniform`] per view into the
//!    [`DisplayTargetUniforms`] dynamic uniform buffer, inserting a
//!    [`ViewDisplayTargetUniformOffset`] on each view (mirroring how
//!    [`ViewUniforms`](super::ViewUniforms) /
//!    [`ViewUniformOffset`](super::ViewUniformOffset) work).
//!
//! Passes bind the uniform **conditionally**: the binding (and the
//! `DISPLAY_TARGET_UNIFORM` shader def guarding it) is only added to a
//! pipeline when the view's display target is not the plain
//! [`DisplayTarget::SDR_SRGB`] default, or when an active operator needs it.
//! Views on default SDR targets specialize to pipelines that are byte-identical
//! to the ones produced before this machinery existed.
//!
//! The matching WGSL struct lives in `display_target.wgsl` and is importable as
//! `bevy_render::display_target`.

use bevy_camera::NormalizedRenderTarget;
use bevy_ecs::prelude::*;
use bevy_window::{DisplayGamut, DisplayTarget, DisplayTransfer};

use super::{
    window::{
        display_target::{resolve_display_target, ManualDisplayTargets},
        ExtractedWindows,
    },
    ExtractedView,
};
use crate::{
    camera::ExtractedCamera,
    render_resource::{DynamicUniformBuffer, ShaderType},
    renderer::{RenderDevice, RenderQueue},
};

/// Render-world component holding the [`DisplayTarget`] of the surface
/// (window, image, or manual texture view) a view renders to, in both its
/// *requested* and *resolved* forms.
///
/// Inserted by [`prepare_view_display_targets`] on every extracted view that
/// has an [`ExtractedCamera`]; falls back to [`DisplayTarget::SDR_SRGB`] when
/// the camera's render target has no explicit display target. Views without a
/// camera (e.g. shadow views) do not receive this component; consumers should
/// treat a missing component as [`DisplayTarget::SDR_SRGB`].
///
/// Prepare-time systems (tonemapping pipeline specialization, operator uniform
/// preparation, the display-encoding pass, and the upscaling blit) read the
/// [`resolved`](Self::resolved) target instead of re-resolving the render
/// target themselves, so they always agree on whether a view takes the HDR
/// path — and they key on what the surface can actually show, never on an
/// unfulfilled request.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct ViewDisplayTarget {
    /// The display target as authored by the user (the `DisplayTarget`
    /// window component or `ManualDisplayTargets` entry).
    ///
    /// Useful for diagnostics and for re-resolution logic; rendering systems
    /// should use [`resolved`](Self::resolved).
    pub requested: DisplayTarget,
    /// The display target after surface negotiation.
    ///
    /// Equal to [`requested`](Self::requested) when the surface fulfils the
    /// requested transfer (or the target is not a window surface, where the
    /// user owns the texture format). When the requested transfer had to be
    /// downgraded (see `select_surface_format` in `view::window`), this is
    /// [`DisplayTarget::SDR_SRGB`], so the downgraded view takes the plain
    /// SDR path bit-for-bit.
    pub resolved: DisplayTarget,
}

impl ViewDisplayTarget {
    /// Creates a `ViewDisplayTarget` whose resolved target equals the
    /// requested one (no surface-side downgrade).
    pub fn fulfilled(target: DisplayTarget) -> Self {
        Self {
            requested: target,
            resolved: target,
        }
    }

    /// Returns `true` if this view's **resolved** display target is exactly
    /// the default [`DisplayTarget::SDR_SRGB`].
    ///
    /// This is the negative of the `DISPLAY_TARGET_UNIFORM` shader-def
    /// predicate: plain-SDR views (including views whose HDR request was
    /// downgraded at surface negotiation) push no new shader defs and keep
    /// their pipelines byte-identical to Bevy's pre-`DisplayTarget` output.
    pub fn is_plain_sdr_srgb(&self) -> bool {
        self.resolved == DisplayTarget::SDR_SRGB
    }

    /// Returns `true` if the **resolved** transfer function is a high dynamic
    /// range transfer ([`DisplayTransfer::ScRgbLinear`],
    /// [`DisplayTransfer::Pq`], or [`DisplayTransfer::Hlg`]; see
    /// [`DisplayTransfer::is_hdr`]).
    ///
    /// This gates the display-encoding pass and the upscaling blit's
    /// pass-through mode, and HDR-capable operators (e.g.
    /// `Tonemapping::GranTurismo7`) use it to pick their HDR mode at prepare
    /// time. Because it reads the resolved transfer, a view whose HDR request
    /// was downgraded behaves exactly like a plain SDR view.
    pub fn is_hdr_transfer(&self) -> bool {
        self.resolved.transfer.is_hdr()
    }
}

/// Index of [`DisplayGamut::Rec709`] in [`DisplayTargetUniform::gamut`].
pub const DISPLAY_GAMUT_REC709: u32 = 0;
/// Index of [`DisplayGamut::DisplayP3`] in [`DisplayTargetUniform::gamut`].
pub const DISPLAY_GAMUT_DISPLAY_P3: u32 = 1;
/// Index of [`DisplayGamut::Rec2020`] in [`DisplayTargetUniform::gamut`].
pub const DISPLAY_GAMUT_REC2020: u32 = 2;

/// Index of [`DisplayTransfer::Srgb`] in [`DisplayTargetUniform::transfer`].
pub const DISPLAY_TRANSFER_SRGB: u32 = 0;
/// Index of [`DisplayTransfer::ScRgbLinear`] in [`DisplayTargetUniform::transfer`].
pub const DISPLAY_TRANSFER_SCRGB_LINEAR: u32 = 1;
/// Index of [`DisplayTransfer::Pq`] in [`DisplayTargetUniform::transfer`].
pub const DISPLAY_TRANSFER_PQ: u32 = 2;
/// Index of [`DisplayTransfer::Hlg`] in [`DisplayTargetUniform::transfer`].
pub const DISPLAY_TRANSFER_HLG: u32 = 3;

/// Returns the [`DisplayTargetUniform::gamut`] index for a [`DisplayGamut`].
pub const fn display_gamut_index(gamut: DisplayGamut) -> u32 {
    match gamut {
        DisplayGamut::Rec709 => DISPLAY_GAMUT_REC709,
        DisplayGamut::DisplayP3 => DISPLAY_GAMUT_DISPLAY_P3,
        DisplayGamut::Rec2020 => DISPLAY_GAMUT_REC2020,
    }
}

/// Returns the [`DisplayTargetUniform::transfer`] index for a
/// [`DisplayTransfer`].
pub const fn display_transfer_index(transfer: DisplayTransfer) -> u32 {
    match transfer {
        DisplayTransfer::Srgb => DISPLAY_TRANSFER_SRGB,
        DisplayTransfer::ScRgbLinear => DISPLAY_TRANSFER_SCRGB_LINEAR,
        DisplayTransfer::Pq => DISPLAY_TRANSFER_PQ,
        DisplayTransfer::Hlg => DISPLAY_TRANSFER_HLG,
    }
}

/// GPU uniform carrying a view's resolved [`DisplayTarget`] calibration.
///
/// The WGSL counterpart is `DisplayTargetUniform` in
/// `bevy_render::display_target` (`display_target.wgsl`); the two must stay
/// field-for-field in sync.
///
/// The [`gamut`](Self::gamut) and [`transfer`](Self::transfer) enums are
/// encoded as `u32` indices (see the `DISPLAY_GAMUT_*` /
/// `DISPLAY_TRANSFER_*` constants, also mirrored in WGSL):
///
/// | `gamut` | meaning | | `transfer` | meaning |
/// |---|---|---|---|---|
/// | 0 | Rec.709 | | 0 | sRGB |
/// | 1 | Display P3 | | 1 | scRGB linear |
/// | 2 | Rec.2020 | | 2 | PQ (ST 2084) |
/// | | | | 3 | HLG |
///
/// Gamut conversion matrices are deliberately **not** part of this uniform;
/// the gamut-transform pass of the display pipeline derives them per pipeline
/// (they arrive with the encoder workstream).
///
/// Values are copied verbatim from the resolved [`DisplayTarget`]; no
/// validation or clamping is applied here. Consumers that have hard numeric
/// requirements (e.g. the GT7 operator's HDR peak range) sanitize at their own
/// prepare step.
#[derive(Clone, Copy, Debug, PartialEq, ShaderType)]
pub struct DisplayTargetUniform {
    /// [`DisplayTarget::paper_white_nits`]: the luminance, in nits, that
    /// `1.0` at the tone-map operator output corresponds to.
    pub paper_white_nits: f32,
    /// [`DisplayTarget::peak_luminance_nits`]: the display's maximum
    /// luminance in nits.
    pub peak_luminance_nits: f32,
    /// [`DisplayTarget::min_luminance_nits`]: the display's black level in
    /// nits.
    pub min_luminance_nits: f32,
    /// The display gamut as a `DISPLAY_GAMUT_*` index (see the type docs).
    pub gamut: u32,
    /// The (resolved) transfer function as a `DISPLAY_TRANSFER_*` index (see
    /// the type docs).
    pub transfer: u32,
}

impl From<DisplayTarget> for DisplayTargetUniform {
    fn from(target: DisplayTarget) -> Self {
        Self {
            paper_white_nits: target.paper_white_nits,
            peak_luminance_nits: target.peak_luminance_nits,
            min_luminance_nits: target.min_luminance_nits,
            gamut: display_gamut_index(target.gamut),
            transfer: display_transfer_index(target.transfer),
        }
    }
}

/// Resource holding the [`DynamicUniformBuffer`] of per-view
/// [`DisplayTargetUniform`]s, written each frame by
/// [`prepare_display_target_uniforms`].
///
/// Bind the whole buffer with a dynamic offset and index it per view with
/// [`ViewDisplayTargetUniformOffset`], exactly like
/// [`ViewUniforms`](super::ViewUniforms) /
/// [`ViewUniformOffset`](super::ViewUniformOffset).
#[derive(Resource)]
pub struct DisplayTargetUniforms {
    /// The per-view uniform buffer; entries are addressed with the dynamic
    /// offset stored in each view's [`ViewDisplayTargetUniformOffset`].
    pub uniforms: DynamicUniformBuffer<DisplayTargetUniform>,
}

impl Default for DisplayTargetUniforms {
    fn default() -> Self {
        let mut uniforms = DynamicUniformBuffer::default();
        uniforms.set_label(Some("display_target_uniforms_buffer"));
        Self { uniforms }
    }
}

/// Render-world component holding a view's dynamic offset into
/// [`DisplayTargetUniforms`].
///
/// Inserted by [`prepare_display_target_uniforms`] on every view that has a
/// [`ViewDisplayTarget`].
#[derive(Component)]
pub struct ViewDisplayTargetUniformOffset {
    /// The dynamic offset to pass to `set_bind_group`.
    pub offset: u32,
}

/// Resolves and inserts a [`ViewDisplayTarget`] on every extracted view that
/// has an [`ExtractedCamera`].
///
/// Runs in [`RenderSystems::PrepareViews`](crate::RenderSystems::PrepareViews)
/// — after `create_surfaces`, so the window surface's negotiated transfer is
/// fresh — so later prepare systems (pipeline specialization, uniform
/// preparation) can rely on the component being present.
///
/// Resolution policy:
/// - **Window targets** go through surface negotiation: when the surface
///   reports it cannot carry the requested transfer
///   ([`ExtractedWindow::resolved_transfer`](super::window::ExtractedWindow::resolved_transfer)
///   is [`DisplayTransfer::Srgb`] while an HDR transfer was requested), the
///   resolved target degrades to [`DisplayTarget::SDR_SRGB`] (D6: warn +
///   degrade; the warning is emitted at format-selection time in
///   `create_surfaces`). Otherwise resolved == requested.
/// - **Image / manual-texture-view targets** resolve to the requested value
///   unchanged: there is no surface negotiation, the user owns the texture
///   and its format.
pub fn prepare_view_display_targets(
    mut commands: Commands,
    extracted_windows: Res<ExtractedWindows>,
    manual_display_targets: Res<ManualDisplayTargets>,
    views: Query<(Entity, &ExtractedCamera), With<ExtractedView>>,
) {
    for (entity, camera) in &views {
        let requested = resolve_display_target(
            camera.target.as_ref(),
            &extracted_windows,
            &manual_display_targets,
        );

        let surface_transfer = match camera.target.as_ref() {
            Some(NormalizedRenderTarget::Window(window_ref)) => extracted_windows
                .get(&window_ref.entity())
                .and_then(|window| window.resolved_transfer),
            _ => None,
        };
        let resolved = match surface_transfer {
            // The surface negotiation downgraded the requested transfer:
            // degrade the whole target to the plain SDR default so the view
            // takes the SDR path bit-for-bit.
            Some(DisplayTransfer::Srgb) if requested.transfer != DisplayTransfer::Srgb => {
                DisplayTarget::SDR_SRGB
            }
            // Surface fulfils the request, the target is not a window, or
            // the surface is not configured yet (transient): no downgrade.
            _ => requested,
        };

        commands.entity(entity).insert(ViewDisplayTarget {
            requested,
            resolved,
        });
    }
}

/// Writes one [`DisplayTargetUniform`] per view into
/// [`DisplayTargetUniforms`] and inserts the matching
/// [`ViewDisplayTargetUniformOffset`].
///
/// Runs in
/// [`RenderSystems::PrepareResources`](crate::RenderSystems::PrepareResources),
/// mirroring [`prepare_view_uniforms`](super::prepare_view_uniforms).
pub fn prepare_display_target_uniforms(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut display_target_uniforms: ResMut<DisplayTargetUniforms>,
    views: Query<(Entity, &ViewDisplayTarget)>,
) {
    let views_iter = views.iter();
    let view_count = views_iter.len();
    let Some(mut writer) =
        display_target_uniforms
            .uniforms
            .get_writer(view_count, &render_device, &render_queue)
    else {
        return;
    };
    for (entity, view_display_target) in &views {
        let offset = writer.write(&DisplayTargetUniform::from(view_display_target.resolved));
        commands
            .entity(entity)
            .insert(ViewDisplayTargetUniformOffset { offset });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enum_indices_are_stable() {
        // These indices are baked into shaders (display_target.wgsl); they
        // must never change for existing variants.
        assert_eq!(display_gamut_index(DisplayGamut::Rec709), 0);
        assert_eq!(display_gamut_index(DisplayGamut::DisplayP3), 1);
        assert_eq!(display_gamut_index(DisplayGamut::Rec2020), 2);
        assert_eq!(display_transfer_index(DisplayTransfer::Srgb), 0);
        assert_eq!(display_transfer_index(DisplayTransfer::ScRgbLinear), 1);
        assert_eq!(display_transfer_index(DisplayTransfer::Pq), 2);
        assert_eq!(display_transfer_index(DisplayTransfer::Hlg), 3);
    }

    #[test]
    fn uniform_copies_display_target_verbatim() {
        let target = DisplayTarget {
            paper_white_nits: 203.0,
            peak_luminance_nits: 1000.0,
            min_luminance_nits: 0.005,
            gamut: DisplayGamut::Rec2020,
            transfer: DisplayTransfer::ScRgbLinear,
        };
        let uniform = DisplayTargetUniform::from(target);
        assert_eq!(uniform.paper_white_nits, 203.0);
        assert_eq!(uniform.peak_luminance_nits, 1000.0);
        assert_eq!(uniform.min_luminance_nits, 0.005);
        assert_eq!(uniform.gamut, DISPLAY_GAMUT_REC2020);
        assert_eq!(uniform.transfer, DISPLAY_TRANSFER_SCRGB_LINEAR);
    }

    #[test]
    fn default_target_uniform_is_sdr_srgb() {
        let uniform = DisplayTargetUniform::from(DisplayTarget::SDR_SRGB);
        assert_eq!(uniform.paper_white_nits, 100.0);
        assert_eq!(uniform.peak_luminance_nits, 100.0);
        assert_eq!(uniform.min_luminance_nits, 0.0);
        assert_eq!(uniform.gamut, DISPLAY_GAMUT_REC709);
        assert_eq!(uniform.transfer, DISPLAY_TRANSFER_SRGB);
    }

    #[test]
    fn view_display_target_predicates() {
        let sdr = ViewDisplayTarget::fulfilled(DisplayTarget::SDR_SRGB);
        assert!(sdr.is_plain_sdr_srgb());
        assert!(!sdr.is_hdr_transfer());

        // Any field deviation makes the target non-plain.
        let brighter = ViewDisplayTarget::fulfilled(DisplayTarget {
            paper_white_nits: 203.0,
            ..DisplayTarget::SDR_SRGB
        });
        assert!(!brighter.is_plain_sdr_srgb());
        assert!(!brighter.is_hdr_transfer());

        for transfer in [
            DisplayTransfer::ScRgbLinear,
            DisplayTransfer::Pq,
            DisplayTransfer::Hlg,
        ] {
            let hdr = ViewDisplayTarget::fulfilled(DisplayTarget {
                transfer,
                ..DisplayTarget::SDR_SRGB
            });
            assert!(!hdr.is_plain_sdr_srgb());
            assert!(hdr.is_hdr_transfer());
        }
    }

    #[test]
    fn downgraded_target_takes_the_plain_sdr_path() {
        // A view whose HDR request was downgraded at surface negotiation
        // (resolved = SDR_SRGB) must be indistinguishable from a plain SDR
        // view to every predicate, regardless of what was requested.
        let downgraded = ViewDisplayTarget {
            requested: DisplayTarget {
                paper_white_nits: 200.0,
                peak_luminance_nits: 1000.0,
                transfer: DisplayTransfer::ScRgbLinear,
                ..DisplayTarget::SDR_SRGB
            },
            resolved: DisplayTarget::SDR_SRGB,
        };
        assert!(downgraded.is_plain_sdr_srgb());
        assert!(!downgraded.is_hdr_transfer());
        assert_eq!(
            DisplayTargetUniform::from(downgraded.resolved),
            DisplayTargetUniform::from(DisplayTarget::SDR_SRGB)
        );
    }
}
