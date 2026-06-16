use crate::camera::extract_cameras;
use crate::extract_resource::ExtractResourcePlugin;
use crate::renderer::WgpuWrapper;
use crate::{
    render_resource::{SurfaceTexture, TextureView},
    renderer::{RenderAdapter, RenderDevice, RenderInstance},
    Extract, ExtractSchedule, GpuResourceAppExt, MainWorld, Render, RenderApp, RenderSystems,
};
use bevy_app::{App, Plugin};
use bevy_ecs::entity::EntityHashSet;
use bevy_ecs::{entity::EntityHashMap, prelude::*};
use bevy_log::{debug, info, warn, warn_once};
use bevy_utils::default;
use bevy_window::{
    CompositeAlphaMode, DisplayGamut, DisplayTarget, DisplayTransfer, PresentMode, PrimaryWindow,
    RawHandleWrapper, Window, WindowClosing, WindowResolvedTransfer,
};
use core::{
    num::NonZero,
    ops::{Deref, DerefMut},
};
use wgpu::{
    SurfaceColorSpace, SurfaceColorSpaces, SurfaceConfiguration, SurfaceFormatCapabilities,
    SurfaceTargetUnsafe, TextureFormat, TextureUsages, TextureViewDescriptor,
};

pub mod display_target;
pub mod screenshot;

pub use display_target::*;
use screenshot::ScreenshotPlugin;

pub struct WindowRenderPlugin;

impl Plugin for WindowRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ScreenshotPlugin,
            ExtractResourcePlugin::<ManualDisplayTargets>::default(),
        ))
        .init_resource::<ManualDisplayTargets>();

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<ManualDisplayTargets>()
                .init_gpu_resource::<ExtractedWindows>()
                .init_gpu_resource::<WindowSurfaces>()
                .add_systems(
                    ExtractSchedule,
                    (
                        extract_windows.before(extract_cameras),
                        write_back_resolved_transfers.after(extract_windows),
                    ),
                )
                .add_systems(
                    Render,
                    create_surfaces
                        .run_if(need_surface_configuration)
                        .before(prepare_windows),
                )
                .add_systems(Render, prepare_windows.in_set(RenderSystems::PrepareViews));
        }
    }
}

pub struct ExtractedWindow {
    /// An entity that contains the components in [`Window`].
    pub entity: Entity,
    pub handle: RawHandleWrapper,
    pub physical_width: u32,
    pub physical_height: u32,
    pub present_mode: PresentMode,
    pub desired_maximum_frame_latency: Option<NonZero<u32>>,
    /// Note: this will not always be the swap chain texture view. When taking a screenshot,
    /// this will point to an alternative texture instead to allow for copying the render result
    /// to CPU memory.
    pub swap_chain_texture_view: Option<TextureView>,
    pub swap_chain_texture: Option<SurfaceTexture>,
    pub swap_chain_texture_format: Option<TextureFormat>,
    /// This is an srgb view of [`ExtractedWindow::swap_chain_texture_format`]
    /// so that in shaders we are always in linear space. For formats without
    /// an sRGB pair (e.g. the `Rgba16Float` scRGB surface or the
    /// `Rgb10a2Unorm` HDR10 surface), it is the surface format itself: there
    /// is no hardware encode, and shaders write (already-encoded) signal
    /// values directly.
    pub swap_chain_texture_view_format: Option<TextureFormat>,
    pub size_changed: bool,
    pub present_mode_changed: bool,
    pub alpha_mode: CompositeAlphaMode,
    /// The [`DisplayTarget`] extracted from the window entity, describing the
    /// display this window's surface feeds (paper white, peak luminance,
    /// gamut, transfer).
    ///
    /// This is the per-surface calibration shared by every camera rendering
    /// to this window; use [`resolve_display_target`] to look it up uniformly
    /// across render-target kinds. [`create_surfaces`] uses its
    /// [`transfer`](DisplayTarget::transfer) to select the surface format
    /// (e.g. `Rgba16Float` for [`DisplayTransfer::ScRgbLinear`]); the outcome
    /// is reported back in [`resolved_transfer`](Self::resolved_transfer).
    pub display_target: DisplayTarget,
    /// Whether a [`DisplayTarget`] change since the last frame requires surface
    /// renegotiation — a [`DisplayTarget::transfer`] change, or a
    /// [`DisplayTarget::gamut`] change under the
    /// [`DisplayTransfer::ExtendedSrgb`] transfer (the one transfer whose
    /// surface color space depends on the gamut: Rec.709 → `ExtendedSrgb`,
    /// Display-P3 → `ExtendedDisplayP3`).
    ///
    /// Such a change requires the surface to be reconfigured with fresh
    /// format selection (and the window's [`ViewTarget`](crate::view::ViewTarget)s
    /// invalidated, since the output attachment's format changes). Analogous
    /// to [`size_changed`](Self::size_changed) /
    /// [`present_mode_changed`](Self::present_mode_changed). Changes to paper
    /// white and peak (and to the gamut under any other transfer, where it is
    /// coerced away at negotiation) flow through uniforms / pipeline
    /// specialization and never set this flag.
    pub display_target_transfer_changed: bool,
    /// The [`DisplayTransfer`] the configured surface can actually carry,
    /// written by [`create_surfaces`] after (format, color space)
    /// negotiation.
    ///
    /// `Some(transfer)` equals the requested [`DisplayTarget::transfer`] when
    /// the surface negotiation succeeded (e.g. an HDR10 color space was
    /// available for [`DisplayTransfer::Pq`], or an encoded extended-range sRGB
    /// space for [`DisplayTransfer::ExtendedSrgb`] — whose Rec.709 and
    /// Display-P3 gamuts both resolve to `ExtendedSrgb`). It differs on a
    /// downgrade: [`DisplayTransfer::Srgb`] for a full SDR downgrade,
    /// [`DisplayTransfer::ScRgbLinear`] when a PQ request fell back to the
    /// extended-sRGB-linear color space, and [`DisplayTransfer::Pq`] for an
    /// [`DisplayTransfer::Hlg`] request (always fulfilled as PQ/HDR10; see
    /// `negotiate_surface_format`). `None` until the surface has been
    /// configured.
    ///
    /// `prepare_view_display_targets` reads this to build each view's
    /// resolved [`ViewDisplayTarget`](crate::view::ViewDisplayTarget), so the
    /// encoding pass and HDR operator modes always key on what the surface
    /// can show, never on an unfulfilled request.
    pub resolved_transfer: Option<DisplayTransfer>,
    /// Whether this window needs an initial buffer commit.
    ///
    /// On Wayland, windows must present at least once before they are shown.
    /// See <https://wayland.app/protocols/xdg-shell#xdg_surface>
    pub needs_initial_present: bool,
}

impl ExtractedWindow {
    fn set_swapchain_texture(
        &mut self,
        frame: wgpu::SurfaceTexture,
        texture_view_format: Option<TextureFormat>,
    ) {
        // `texture_view_format` is the sRGB view format the negotiation
        // stored in `SurfaceData` (`None` for HDR transfers and formats
        // without an sRGB pair); recomputing `add_srgb_suffix` here would
        // re-attach a hardware sRGB encode that the negotiation decided
        // against.
        self.swap_chain_texture_view_format =
            Some(texture_view_format.unwrap_or_else(|| frame.texture.format()));
        let texture_view_descriptor = TextureViewDescriptor {
            format: self.swap_chain_texture_view_format,
            ..default()
        };
        self.swap_chain_texture_view = Some(TextureView::from(
            frame.texture.create_view(&texture_view_descriptor),
        ));
        self.swap_chain_texture = Some(SurfaceTexture::from(frame));
    }

    fn has_swapchain_texture(&self) -> bool {
        self.swap_chain_texture_view.is_some() && self.swap_chain_texture.is_some()
    }

    pub fn present(&mut self, queue: &wgpu::Queue) {
        if let Some(surface_texture) = self.swap_chain_texture.take() {
            // TODO(clean): winit docs recommends calling pre_present_notify before this.
            // though `present()` doesn't present the frame, it schedules it to be presented
            // by wgpu.
            // https://docs.rs/winit/0.29.9/wasm32-unknown-unknown/winit/window/struct.Window.html#method.pre_present_notify
            surface_texture.present(queue);
        }
    }
}

#[derive(Default, Resource)]
pub struct ExtractedWindows {
    pub primary: Option<Entity>,
    pub windows: EntityHashMap<ExtractedWindow>,
}

impl Deref for ExtractedWindows {
    type Target = EntityHashMap<ExtractedWindow>;

    fn deref(&self) -> &Self::Target {
        &self.windows
    }
}

impl DerefMut for ExtractedWindows {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.windows
    }
}

fn extract_windows(
    mut extracted_windows: ResMut<ExtractedWindows>,
    mut closing: Extract<MessageReader<WindowClosing>>,
    windows: Extract<
        Query<(
            Entity,
            &Window,
            Option<&DisplayTarget>,
            &RawHandleWrapper,
            Option<&PrimaryWindow>,
        )>,
    >,
    mut removed: Extract<RemovedComponents<RawHandleWrapper>>,
    mut window_surfaces: ResMut<WindowSurfaces>,
) {
    for (entity, window, display_target, handle, primary) in windows.iter() {
        // `DisplayTarget` is a required component of `Window`, but tolerate
        // its removal by falling back to the SDR default rather than
        // dropping the window from extraction.
        let display_target = display_target.copied().unwrap_or_default();
        if primary.is_some() {
            extracted_windows.primary = Some(entity);
        }

        let (new_width, new_height) = (
            window.resolution.physical_width().max(1),
            window.resolution.physical_height().max(1),
        );

        let extracted_window = extracted_windows.entry(entity).or_insert(ExtractedWindow {
            entity,
            handle: handle.clone(),
            physical_width: new_width,
            physical_height: new_height,
            present_mode: window.present_mode,
            desired_maximum_frame_latency: window.desired_maximum_frame_latency,
            swap_chain_texture: None,
            swap_chain_texture_view: None,
            size_changed: false,
            swap_chain_texture_format: None,
            swap_chain_texture_view_format: None,
            present_mode_changed: false,
            alpha_mode: window.composite_alpha_mode,
            display_target,
            display_target_transfer_changed: false,
            resolved_transfer: None,
            needs_initial_present: true,
        });

        // Keep the extracted `DisplayTarget` in sync every frame, diffing the
        // fields that affect surface negotiation: a transfer change requires
        // surface reconfiguration with fresh format selection
        // (`need_surface_configuration` / `create_surfaces`) and `ViewTarget`
        // invalidation (`cleanup_view_targets_for_resize`). The gamut affects
        // the negotiated surface color space ONLY for the encoded
        // extended-range sRGB transfer (Rec.709 -> `ExtendedSrgb`,
        // DisplayP3 -> `ExtendedDisplayP3`); for every other transfer it is
        // coerced away at negotiation, so a gamut change there flows through
        // uniforms / pipeline specialization without surface work, exactly as
        // paper white and peak always do.
        let previous = extracted_window.display_target;
        let transfer_changed = previous.transfer != display_target.transfer;
        let extended_srgb_gamut_changed = (previous.transfer == DisplayTransfer::ExtendedSrgb
            || display_target.transfer == DisplayTransfer::ExtendedSrgb)
            && previous.gamut != display_target.gamut;
        extracted_window.display_target_transfer_changed =
            transfer_changed || extended_srgb_gamut_changed;
        extracted_window.display_target = display_target;

        if extracted_window.swap_chain_texture.is_none() {
            // If we called present on the previous swap-chain texture last update,
            // then drop the swap chain frame here, otherwise we can keep it for the
            // next update as an optimization. `prepare_windows` will only acquire a new
            // swap chain texture if needed.
            extracted_window.swap_chain_texture_view = None;
        }
        extracted_window.size_changed = new_width != extracted_window.physical_width
            || new_height != extracted_window.physical_height;
        extracted_window.present_mode_changed =
            window.present_mode != extracted_window.present_mode;

        if extracted_window.size_changed {
            debug!(
                "Window size changed from {}x{} to {}x{}",
                extracted_window.physical_width,
                extracted_window.physical_height,
                new_width,
                new_height
            );
            extracted_window.physical_width = new_width;
            extracted_window.physical_height = new_height;
        }

        if extracted_window.present_mode_changed {
            debug!(
                "Window Present Mode changed from {:?} to {:?}",
                extracted_window.present_mode, window.present_mode
            );
            extracted_window.present_mode = window.present_mode;
        }
    }

    for closing_window in closing.read() {
        extracted_windows.remove(&closing_window.window);
        window_surfaces.remove(&closing_window.window);
    }
    for removed_window in removed.read() {
        extracted_windows.remove(&removed_window);
        window_surfaces.remove(&removed_window);
    }
}

/// Mirrors each window surface's negotiated transfer back to the main world
/// as a [`WindowResolvedTransfer`] component, so apps can detect downgraded
/// (or later-renegotiated) HDR requests. Runs during extraction — the render
/// world's only window into the main world — so the value lags the
/// negotiation it reports by one frame.
fn write_back_resolved_transfers(
    mut main_world: ResMut<MainWorld>,
    window_surfaces: Res<WindowSurfaces>,
) {
    for (&entity, surface_data) in window_surfaces.surfaces.iter() {
        let Ok(mut window) = main_world.get_entity_mut(entity) else {
            continue;
        };
        // Insert only on change, so `Changed<WindowResolvedTransfer>` stays
        // a usable signal.
        if window
            .get::<WindowResolvedTransfer>()
            .map(|resolved| resolved.0)
            != Some(surface_data.resolved_transfer)
        {
            window.insert(WindowResolvedTransfer(surface_data.resolved_transfer));
        }
    }
}

struct SurfaceData {
    // TODO: what lifetime should this be?
    surface: WgpuWrapper<wgpu::Surface<'static>>,
    configuration: SurfaceConfiguration,
    texture_view_format: Option<TextureFormat>,
    /// The [`DisplayTransfer`] the configured (format, color space) pair
    /// carries; differs from the requested transfer when the request was
    /// downgraded (see [`negotiate_surface_format`]). Mirrored into
    /// [`ExtractedWindow::resolved_transfer`] by [`create_surfaces`].
    resolved_transfer: DisplayTransfer,
}

impl SurfaceData {
    /// The format the renderer's final blit writes through: the sRGB view of
    /// the surface format when one exists, otherwise the surface format
    /// itself (float/HDR formats have no sRGB view; shaders write signal
    /// values directly).
    fn view_format(&self) -> TextureFormat {
        self.texture_view_format
            .unwrap_or(self.configuration.format)
    }

    /// Applies a fresh [`negotiate_surface_format`] outcome to the stored
    /// configuration: surface format, color space, the sRGB view format (only
    /// 8-bit non-sRGB formats on the plain sRGB transfer get one), and the
    /// resolved transfer.
    ///
    /// The sRGB view is gated on the *resolved transfer*, not just the
    /// format: a last-resort HDR10 negotiation can land on an 8-bit format,
    /// whose stores carry already-PQ-encoded signal — a hardware sRGB
    /// encode on top would double-encode it.
    fn apply_negotiated(&mut self, negotiated: NegotiatedSurface) {
        let view_format = negotiated.format.add_srgb_suffix();
        self.configuration.format = negotiated.format;
        self.configuration.color_space = negotiated.color_space;
        self.texture_view_format = (negotiated.resolved_transfer == DisplayTransfer::Srgb
            && view_format != negotiated.format)
            .then_some(view_format);
        self.configuration.view_formats = match self.texture_view_format {
            Some(format) => vec![format],
            None => vec![],
        };
        self.resolved_transfer = negotiated.resolved_transfer;
    }
}

#[derive(Resource, Default)]
pub struct WindowSurfaces {
    surfaces: EntityHashMap<SurfaceData>,
    /// List of windows that we have already called the initial `configure_surface` for
    configured_windows: EntityHashSet,
}

impl WindowSurfaces {
    fn remove(&mut self, window: &Entity) {
        self.surfaces.remove(window);
        self.configured_windows.remove(window);
    }
}

/// (re)configures window surfaces, and obtains a swapchain texture for rendering.
///
/// NOTE: `get_current_texture` in `prepare_windows` can take a long time if the GPU workload is
/// the performance bottleneck. This can be seen in profiles as multiple prepare-set systems all
/// taking an unusually long time to complete, and all finishing at about the same time as the
/// `prepare_windows` system. Improvements in bevy are planned to avoid this happening when it
/// should not but it will still happen as it is easy for a user to create a large GPU workload
/// relative to the GPU performance and/or CPU workload.
/// This can be caused by many reasons, but several of them are:
/// - GPU workload is more than your current GPU can manage
/// - Error / performance bug in your custom shaders
/// - wgpu was unable to detect a proper GPU hardware-accelerated device given the chosen
///   [`Backends`](crate::settings::Backends), [`WgpuLimits`](crate::settings::WgpuLimits),
///   and/or [`WgpuFeatures`](crate::settings::WgpuFeatures). For example, on Windows currently
///   `DirectX 11` is not supported by wgpu 0.12 and so if your GPU/drivers do not support Vulkan,
///   it may be that a software renderer called "Microsoft Basic Render Driver" using `DirectX 12`
///   will be chosen and performance will be very poor. This is visible in a log message that is
///   output during renderer initialization.
///   Another alternative is to try to use [`ANGLE`](https://github.com/gfx-rs/wgpu#angle) and
///   [`Backends::GL`](crate::settings::Backends::GL) with the `gles` feature enabled if your
///   GPU/drivers support `OpenGL 4.3` / `OpenGL ES 3.0` or later.
pub fn prepare_windows(
    mut windows: ResMut<ExtractedWindows>,
    mut window_surfaces: ResMut<WindowSurfaces>,
    render_device: Res<RenderDevice>,
    render_adapter: Res<RenderAdapter>,
    sorted_cameras: Res<crate::camera::SortedCameras>,
    #[cfg(target_os = "linux")] render_instance: Res<RenderInstance>,
) {
    for window in windows.windows.values_mut() {
        // Skip acquiring a swap-chain texture for windows that no camera
        // targets. This avoids a wasted clear pass in
        // `handle_uncovered_swap_chains` that triggers a DMA-fence fd leak on
        // Adreno 740 (Quest 3). The exception is windows that still need their
        // initial present (required on Wayland).
        let is_camera_target = sorted_cameras.0.iter().any(|c| {
            matches!(
                &c.target,
                Some(bevy_camera::NormalizedRenderTarget::Window(w)) if w.entity() == window.entity
            ) && matches!(c.output_mode, bevy_camera::CameraOutputMode::Write { .. })
        });
        if !is_camera_target && !window.needs_initial_present {
            continue;
        }

        let window_surfaces = window_surfaces.deref_mut();
        let Some(surface_data) = window_surfaces.surfaces.get_mut(&window.entity) else {
            continue;
        };

        // We didn't present the previous frame, so we can keep using our existing swapchain texture.
        if window.has_swapchain_texture()
            && !window.size_changed
            && !window.present_mode_changed
            && !window.display_target_transfer_changed
        {
            continue;
        }

        // A recurring issue is hitting `wgpu::SurfaceError::Timeout` on certain Linux
        // mesa driver implementations. This seems to be a quirk of some drivers.
        // We'd rather keep panicking when not on Linux mesa, because in those case,
        // the `Timeout` is still probably the symptom of a degraded unrecoverable
        // application state.
        // see https://github.com/bevyengine/bevy/pull/5957
        // and https://github.com/gfx-rs/wgpu/issues/1218
        #[cfg(target_os = "linux")]
        let may_erroneously_timeout = || {
            bevy_tasks::IoTaskPool::get().scope(|scope| {
                scope.spawn(async {
                    render_instance
                        .enumerate_adapters(wgpu::Backends::VULKAN)
                        .await
                        .iter()
                        .any(|adapter| {
                            let name = adapter.get_info().name;
                            name.starts_with("Radeon")
                                || name.starts_with("AMD")
                                || name.starts_with("Intel")
                        })
                });
            })[0]
        };

        let surface = &surface_data.surface;
        match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                window.set_swapchain_texture(surface_texture, surface_data.texture_view_format);
            }
            #[cfg(target_os = "linux")]
            wgpu::CurrentSurfaceTexture::Timeout if may_erroneously_timeout() => {
                bevy_log::trace!(
                    "Couldn't get swap chain texture. This is probably a quirk \
                        of your Linux GPU driver, so it can be safely ignored."
                );
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                // Defensive (HDR surfaces only — `to_flag()` is `None` for
                // the SDR path's `Auto`): a surface can go outdated because
                // the OS-level color capabilities changed (e.g. the HDR
                // toggle was flipped), not just because of a resize. If the
                // stored explicit color space is no longer advertised,
                // reconfiguring with it would fail wgpu validation
                // (`ConfigureSurfaceError::UnsupportedColorSpace`), so
                // renegotiate first. The display-encoding systems already ran
                // this frame with the old resolved transfer (one frame of
                // incorrectly-encoded output); the updated
                // `resolved_transfer` corrects them from the next frame.
                if let Some(flag) = surface_data.configuration.color_space.to_flag() {
                    let caps = surface_data.surface.get_capabilities(&render_adapter);
                    if !caps
                        .color_spaces(surface_data.configuration.format)
                        .contains(flag)
                    {
                        warn!(
                            "The configured surface color space ({:?}) is no longer \
                            supported for {:?} (did the OS HDR setting change?); \
                            renegotiating the swapchain.",
                            surface_data.configuration.color_space,
                            surface_data.configuration.format
                        );
                        surface_data.apply_negotiated(negotiate_surface_format(
                            &caps.formats,
                            &caps.format_capabilities,
                            window.display_target.transfer,
                            window.display_target.gamut,
                        ));
                        window.resolved_transfer = Some(surface_data.resolved_transfer);
                    }
                }
                let surface = &surface_data.surface;
                render_device.configure_surface(surface, &surface_data.configuration);
                let frame = match surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(surface_texture)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => surface_texture,
                    variant => {
                        // This is a common occurrence on X11 and Xwayland with NVIDIA drivers
                        // when opening and resizing the window.
                        warn!(
                            "Couldn't get swap chain texture after configuring. Cause: '{variant:?}'"
                        );
                        continue;
                    }
                };
                window.set_swapchain_texture(frame, surface_data.texture_view_format);
            }
            wgpu::CurrentSurfaceTexture::Occluded => {}
            other => {
                bevy_log::error!("Couldn't get swap chain texture: {other:?}");
            }
        }
        window.swap_chain_texture_format = Some(surface_data.configuration.format);
    }
}

pub fn need_surface_configuration(
    windows: Res<ExtractedWindows>,
    window_surfaces: Res<WindowSurfaces>,
) -> bool {
    for window in windows.windows.values() {
        if !window_surfaces.configured_windows.contains(&window.entity)
            || window.size_changed
            || window.present_mode_changed
            || window.display_target_transfer_changed
        {
            return true;
        }
    }
    false
}

/// The outcome of [`negotiate_surface_format`]: the (format, color space)
/// pair [`create_surfaces`] configures the surface with, together with the
/// [`DisplayTransfer`] that pair actually carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NegotiatedSurface {
    /// The swapchain texture format.
    format: TextureFormat,
    /// The color space the presentation engine interprets the swapchain in
    /// ([`SurfaceConfiguration::color_space`]).
    ///
    /// Always either [`SurfaceColorSpace::Auto`] (the SDR path) or an
    /// explicit color space taken from the surface's advertised
    /// `format_capabilities`, so configuring the pair can never fail wgpu's
    /// validation (`ConfigureSurfaceError::UnsupportedColorSpace`) against
    /// the capabilities the negotiation ran on.
    color_space: SurfaceColorSpace,
    /// The transfer the configured surface carries — the *resolved* transfer
    /// reported back through [`ExtractedWindow::resolved_transfer`].
    resolved_transfer: DisplayTransfer,
}

/// Mirror of `wgpu::SurfaceCapabilities::color_spaces`, over the raw
/// capability slice so [`negotiate_surface_format`] stays unit-testable
/// without a live surface.
fn advertised_color_spaces(
    format_capabilities: &[SurfaceFormatCapabilities],
    format: TextureFormat,
) -> SurfaceColorSpaces {
    format_capabilities
        .iter()
        .filter(|fc| fc.format == format)
        .fold(SurfaceColorSpaces::empty(), |acc, fc| acc | fc.color_spaces)
}

/// Negotiates an `Rgba16Float` swapchain in the extended-sRGB-linear (scRGB)
/// color space, if the surface advertises the pair.
///
/// scRGB (IEC 61966-2-2) is definitionally an extended-range float encoding
/// against Rec.709/sRGB primaries, so only `Rgba16Float` is considered.
fn negotiate_scrgb_linear(
    format_capabilities: &[SurfaceFormatCapabilities],
) -> Option<NegotiatedSurface> {
    advertised_color_spaces(format_capabilities, TextureFormat::Rgba16Float)
        .contains(SurfaceColorSpaces::EXTENDED_SRGB_LINEAR)
        .then_some(NegotiatedSurface {
            format: TextureFormat::Rgba16Float,
            color_space: SurfaceColorSpace::ExtendedSrgbLinear,
            resolved_transfer: DisplayTransfer::ScRgbLinear,
        })
}

/// Negotiates a swapchain in the HDR10 (PQ / SMPTE ST 2084, Rec.2020) color
/// space, if the surface advertises it for any format.
///
/// Format preference follows wgpu's canonical HDR example: `Rgb10a2Unorm`
/// (HDR10's native 10-bit container; what DX12 and most Vulkan drivers
/// expose) first, `Rgba16Float` second (advertised on Metal), then any other
/// format the surface lists with HDR10 support, in capability order.
fn negotiate_hdr10(format_capabilities: &[SurfaceFormatCapabilities]) -> Option<NegotiatedSurface> {
    const PREFERRED: &[TextureFormat] = &[TextureFormat::Rgb10a2Unorm, TextureFormat::Rgba16Float];
    let preferred = PREFERRED.iter().copied().filter(|&format| {
        advertised_color_spaces(format_capabilities, format).contains(SurfaceColorSpaces::HDR10)
    });
    // Formats that are themselves sRGB-encoded are excluded: their stores
    // bake in the sRGB OETF, which would re-encode the already-PQ-encoded
    // signal regardless of which view the blit writes through.
    let any = format_capabilities
        .iter()
        .filter(|fc| fc.color_spaces.contains(SurfaceColorSpaces::HDR10) && !fc.format.is_srgb())
        .map(|fc| fc.format);
    preferred.chain(any).next().map(|format| NegotiatedSurface {
        format,
        color_space: SurfaceColorSpace::Hdr10,
        resolved_transfer: DisplayTransfer::Pq,
    })
}

/// Negotiates a swapchain in one of the two encoded extended-range sRGB color
/// spaces (`ExtendedSrgb` for Rec.709, `ExtendedDisplayP3` for Display-P3), if
/// the surface advertises `flag` for any format.
///
/// `Rgba16Float` (the natural extended-range container, advertised by Metal
/// and browser WebGPU) is preferred, then any other non-sRGB format the
/// surface lists with the color space, in capability order. sRGB formats are
/// excluded: their hardware encode would re-encode the already-gamma signal on
/// store. Both color spaces resolve to [`DisplayTransfer::ExtendedSrgb`] — the
/// gamut rides [`DisplayTarget::gamut`], not the resolved transfer.
fn negotiate_extended_srgb_space(
    format_capabilities: &[SurfaceFormatCapabilities],
    flag: SurfaceColorSpaces,
    color_space: SurfaceColorSpace,
) -> Option<NegotiatedSurface> {
    let preferred = core::iter::once(TextureFormat::Rgba16Float)
        .filter(|&format| advertised_color_spaces(format_capabilities, format).contains(flag));
    let any = format_capabilities
        .iter()
        .filter(|fc| fc.color_spaces.contains(flag) && !fc.format.is_srgb())
        .map(|fc| fc.format);
    preferred.chain(any).next().map(|format| NegotiatedSurface {
        format,
        color_space,
        resolved_transfer: DisplayTransfer::ExtendedSrgb,
    })
}

/// Negotiates an encoded extended-range sRGB swapchain in the `ExtendedSrgb`
/// (Rec.709) color space — the web's HDR path (browser WebGPU cannot present a
/// linear-transfer canvas), also advertised by Metal and Vulkan.
fn negotiate_extended_srgb(
    format_capabilities: &[SurfaceFormatCapabilities],
) -> Option<NegotiatedSurface> {
    negotiate_extended_srgb_space(
        format_capabilities,
        SurfaceColorSpaces::EXTENDED_SRGB,
        SurfaceColorSpace::ExtendedSrgb,
    )
}

/// Negotiates an encoded extended-range Display-P3 swapchain in the
/// `ExtendedDisplayP3` color space (wide-gamut HDR), advertised by Metal and
/// browser WebGPU on HDR-capable displays.
fn negotiate_extended_display_p3(
    format_capabilities: &[SurfaceFormatCapabilities],
) -> Option<NegotiatedSurface> {
    negotiate_extended_srgb_space(
        format_capabilities,
        SurfaceColorSpaces::EXTENDED_DISPLAY_P3,
        SurfaceColorSpace::ExtendedDisplayP3,
    )
}

/// Negotiates the (format, color space) pair for a window surface from the
/// surface's capabilities, honoring the requested [`DisplayTransfer`] when
/// possible.
///
/// `auto_formats` is `SurfaceCapabilities::formats` (the formats configurable
/// with [`SurfaceColorSpace::Auto`], in preference order) and
/// `format_capabilities` is `SurfaceCapabilities::format_capabilities` (every
/// format together with the color spaces the surface supports it in, a
/// superset of `auto_formats`). `requested_gamut` keys only the
/// [`DisplayTransfer::ExtendedSrgb`] arm (the one transfer whose surface color
/// space depends on the gamut). Policy per requested transfer:
///
/// - [`DisplayTransfer::Srgb`] (the default): the plain SDR
///   selection — the first of `Rgba8UnormSrgb` / `Bgra8UnormSrgb` in
///   capability order, else the surface's first `Auto`-configurable format —
///   paired with [`SurfaceColorSpace::Auto`]. `Auto` lets wgpu pick the
///   color space (sRGB for the 8-bit formats; extended-sRGB-linear for a
///   first-listed `Rgba16Float` where supported) and is always
///   valid for formats in `auto_formats`, whereas an explicit
///   [`SurfaceColorSpace::Srgb`] would fail validation on drivers that do
///   not advertise the sRGB color space by name.
/// - [`DisplayTransfer::ScRgbLinear`]: `Rgba16Float` +
///   [`SurfaceColorSpace::ExtendedSrgbLinear`] when advertised (macOS/iOS
///   Metal EDR, Windows Vulkan/DX12, Wayland Vulkan) — this is native-only, so
///   the web requests [`DisplayTransfer::ExtendedSrgb`] instead; else warn +
///   SDR downgrade.
/// - [`DisplayTransfer::ExtendedSrgb`]: the requested gamut picks the surface
///   color space — [`DisplayGamut::DisplayP3`](bevy_window::DisplayGamut)
///   negotiates [`SurfaceColorSpace::ExtendedDisplayP3`] (wide-gamut HDR, Metal
///   and browser WebGPU), and any other gamut (the coerced Rec.709/Rec.2020)
///   negotiates [`SurfaceColorSpace::ExtendedSrgb`] (Metal, Vulkan, browser
///   WebGPU — the web HDR path). There is no cross-gamut downgrade: a Display-P3
///   request that cannot get `ExtendedDisplayP3` falls straight to SDR rather
///   than to the Rec.709 `ExtendedSrgb` surface (which would mismatch the
///   encoder's resolved P3 gamut, since the returned transfer carries no gamut).
/// - [`DisplayTransfer::Pq`]: HDR10 ([`SurfaceColorSpace::Hdr10`]) on the
///   advertised formats (see [`negotiate_hdr10`]; requires the OS to have
///   HDR output enabled on DX12/Vulkan). When unavailable, the downgrade
///   chain applies, each step with its own warning:
///   PQ → scRGB-linear → SDR sRGB.
/// - [`DisplayTransfer::Hlg`]: fulfilled as **PQ/HDR10**, never as an HLG
///   swapchain, even where the backend advertises
///   [`SurfaceColorSpace::Hlg`]: HLG is scene-referred (the display applies
///   the OOTF), and Bevy's display-encoding pass deliberately refuses to
///   HLG-encode its display-referred tone-mapped output (it would
///   double-tone-map; the pass coerces HLG to PQ). Negotiating an HLG
///   surface and presenting PQ-encoded signal into it would display
///   incorrectly. The HLG → `SurfaceColorSpace::Hlg` mapping can light up
///   once a scene-referred HLG encoder path exists.
///
/// Gamut interaction: the negotiation keys on the transfer (and, for
/// `ExtendedSrgb`, the gamut), consistent with the encoder's gamut coercions —
/// HDR10 *is* Rec.2020 (the encoder coerces PQ targets to a Rec.2020 encode),
/// extended-sRGB-linear *is* Rec.709-coordinate (wide gamut rides out-of-range
/// component values), and the encoded extended-sRGB surfaces carry Rec.709
/// (`ExtendedSrgb`) or Display-P3 (`ExtendedDisplayP3`) primaries to match the
/// requested gamut.
///
/// Every returned pair is taken from (or, for `Auto`, guaranteed valid
/// against) the passed capabilities, so `create_surfaces` can configure it
/// without tripping `ConfigureSurfaceError::UnsupportedColorSpace`. A full
/// downgrade resolves to [`DisplayTransfer::Srgb`]; the caller propagates
/// this so downgraded views take the same plain SDR path as a natively-SDR
/// view.
fn negotiate_surface_format(
    auto_formats: &[TextureFormat],
    format_capabilities: &[SurfaceFormatCapabilities],
    requested_transfer: DisplayTransfer,
    requested_gamut: DisplayGamut,
) -> NegotiatedSurface {
    match requested_transfer {
        DisplayTransfer::Srgb => {}
        DisplayTransfer::ScRgbLinear => {
            if let Some(negotiated) = negotiate_scrgb_linear(format_capabilities) {
                return negotiated;
            }
            warn_once!(
                "DisplayTransfer::ScRgbLinear was requested, but this surface does not \
                support an Rgba16Float swapchain in the extended-sRGB-linear color \
                space. Downgrading to SDR sRGB output. scRGB-linear output requires an \
                HDR-capable display on macOS/iOS (Metal), Windows (Vulkan/DX12), or \
                Wayland (Vulkan); on the web, request DisplayTransfer::ExtendedSrgb \
                (the encoded sibling) instead."
            );
        }
        DisplayTransfer::ExtendedSrgb => {
            // The requested gamut selects the surface color space:
            // Display-P3 -> ExtendedDisplayP3, Rec.709 (and the coerced
            // Rec.2020) -> ExtendedSrgb. There is no cross-gamut downgrade — a
            // Display-P3 request that cannot get ExtendedDisplayP3 degrades
            // straight to SDR rather than to the Rec.709 ExtendedSrgb surface,
            // which would mismatch the encoder's resolved P3 gamut (the
            // resolved transfer carries no gamut; the encoder reads
            // DisplayTarget::gamut).
            if requested_gamut == DisplayGamut::DisplayP3 {
                if let Some(negotiated) = negotiate_extended_display_p3(format_capabilities) {
                    return negotiated;
                }
                warn_once!(
                    "DisplayTransfer::ExtendedSrgb with DisplayGamut::DisplayP3 was \
                    requested, but this surface does not advertise the ExtendedDisplayP3 \
                    color space (wide-gamut HDR, available on Metal and browser WebGPU on \
                    HDR-capable displays). Downgrading to SDR sRGB output."
                );
            } else {
                if let Some(negotiated) = negotiate_extended_srgb(format_capabilities) {
                    return negotiated;
                }
                warn_once!(
                    "DisplayTransfer::ExtendedSrgb was requested, but this surface does \
                    not advertise the encoded extended-range sRGB color space (available \
                    on Metal, Vulkan, and browser WebGPU on HDR-capable displays). \
                    Downgrading to SDR sRGB output."
                );
            }
        }
        DisplayTransfer::Pq | DisplayTransfer::Hlg => {
            if requested_transfer == DisplayTransfer::Hlg {
                warn_once!(
                    "DisplayTransfer::Hlg was requested, but Bevy's display pipeline \
                    cannot produce a correct HLG signal (HLG is scene-referred; \
                    encoding tone-mapped display-referred output with the HLG OETF \
                    would double-tone-map), so an HLG swapchain is never negotiated. \
                    Fulfilling the request with PQ (HDR10) where available."
                );
            }
            if let Some(negotiated) = negotiate_hdr10(format_capabilities) {
                return negotiated;
            }
            warn_once!(
                "DisplayTransfer::{requested_transfer:?} was requested, but this \
                surface does not advertise the HDR10 (PQ) color space — the OS may \
                have HDR output disabled, or the backend lacks support. Downgrading \
                to scRGB-linear if available, else SDR sRGB."
            );
            if let Some(negotiated) = negotiate_scrgb_linear(format_capabilities) {
                return negotiated;
            }
            warn_once!(
                "DisplayTransfer::{requested_transfer:?} could not be downgraded to \
                scRGB-linear either (no Rgba16Float extended-sRGB-linear support); \
                downgrading to SDR sRGB output."
            );
        }
    }

    // SDR path: prefer sRGB formats for surfaces, but fall back to the first
    // available format if no sRGB formats are available. `Auto` lets wgpu
    // pick the color space for whichever format wins.
    if let Some(first) = auto_formats.first() {
        let mut format = *first;
        for available_format in auto_formats {
            // Rgba8UnormSrgb and Bgra8UnormSrgb and the only sRGB formats wgpu exposes that we can use for surfaces.
            if *available_format == TextureFormat::Rgba8UnormSrgb
                || *available_format == TextureFormat::Bgra8UnormSrgb
            {
                format = *available_format;
                break;
            }
        }
        return NegotiatedSurface {
            format,
            color_space: SurfaceColorSpace::Auto,
            resolved_transfer: DisplayTransfer::Srgb,
        };
    }

    // Defensive: some drivers report formats ONLY in explicit-opt-in
    // (wide-gamut / HDR) color spaces when the OS is in HDR mode, leaving
    // `SurfaceCapabilities::formats` (the `Auto`-configurable set) empty.
    // Configuring such a format with `Auto` fails wgpu validation, so pick
    // the first advertised pair we can drive correctly instead of panicking.
    for (flag, color_space, resolved_transfer) in [
        (
            SurfaceColorSpaces::SRGB,
            SurfaceColorSpace::Srgb,
            DisplayTransfer::Srgb,
        ),
        (
            SurfaceColorSpaces::EXTENDED_DISPLAY_P3,
            SurfaceColorSpace::ExtendedDisplayP3,
            DisplayTransfer::ExtendedSrgb,
        ),
        (
            SurfaceColorSpaces::EXTENDED_SRGB,
            SurfaceColorSpace::ExtendedSrgb,
            DisplayTransfer::ExtendedSrgb,
        ),
        (
            SurfaceColorSpaces::EXTENDED_SRGB_LINEAR,
            SurfaceColorSpace::ExtendedSrgbLinear,
            DisplayTransfer::ScRgbLinear,
        ),
        (
            SurfaceColorSpaces::HDR10,
            SurfaceColorSpace::Hdr10,
            DisplayTransfer::Pq,
        ),
    ] {
        // The encoded-extended-P3 surface only matches a Display-P3 request and
        // the encoded-extended-sRGB (Rec.709) surface only a non-P3 request, so
        // the negotiated surface gamut always equals the gamut the encoder
        // emits (it reads `DisplayTarget::gamut`, which negotiation never
        // changes — only the transfer is reported back).
        if color_space == SurfaceColorSpace::ExtendedDisplayP3
            && requested_gamut != DisplayGamut::DisplayP3
        {
            continue;
        }
        if color_space == SurfaceColorSpace::ExtendedSrgb
            && requested_gamut == DisplayGamut::DisplayP3
        {
            continue;
        }
        if let Some(fc) = format_capabilities.iter().find(|fc| {
            // An sRGB format's hardware encode would corrupt non-sRGB
            // signal (scRGB-linear / PQ / encoded extended-range sRGB) on
            // store.
            fc.color_spaces.contains(flag)
                && (resolved_transfer == DisplayTransfer::Srgb || !fc.format.is_srgb())
        }) {
            warn_once!(
                "This surface advertises no Auto-configurable formats; falling back to \
                {:?} in the {color_space:?} color space (resolved transfer: \
                {resolved_transfer:?}).",
                fc.format
            );
            return NegotiatedSurface {
                format: fc.format,
                color_space,
                resolved_transfer,
            };
        }
    }

    // Pre-existing behavior for a surface that is incompatible with the
    // adapter (both capability lists empty).
    panic!("No supported formats for surface");
}

// 2 is wgpu's default/what we've been using so far.
// 1 is the minimum, but may cause lower framerates due to the cpu waiting for the gpu to finish
// all work for the previous frame before starting work on the next frame, which then means the gpu
// has to wait for the cpu to finish to start on the next frame.
const DEFAULT_DESIRED_MAXIMUM_FRAME_LATENCY: u32 = 2;

/// Creates window surfaces.
pub fn create_surfaces(
    // By accessing a NonSend resource, we tell the scheduler to put this system on the main thread,
    // which is necessary for some OS's
    #[cfg(any(target_os = "macos", target_os = "ios"))] _marker: bevy_ecs::system::NonSendMarker,
    mut windows: ResMut<ExtractedWindows>,
    mut window_surfaces: ResMut<WindowSurfaces>,
    render_instance: Res<RenderInstance>,
    render_adapter: Res<RenderAdapter>,
    render_device: Res<RenderDevice>,
) {
    for window in windows.windows.values_mut() {
        if !window_surfaces.surfaces.contains_key(&window.entity) {
            let surface_target = SurfaceTargetUnsafe::RawHandle {
                raw_display_handle: Some(window.handle.get_display_handle()),
                raw_window_handle: window.handle.get_window_handle(),
            };
            // SAFETY: The window handles in ExtractedWindows will always be valid objects to create surfaces on
            let surface = unsafe {
                // NOTE: On some OSes this MUST be called from the main thread.
                // As of wgpu 0.15, only fallible if the given window is a HTML canvas and obtaining a WebGPU or WebGL2 context fails.
                render_instance
                    .create_surface_unsafe(surface_target)
                    .expect("Failed to create wgpu surface")
            };
            let caps = surface.get_capabilities(&render_adapter);
            let present_mode = present_mode(window, &caps);
            let negotiated = negotiate_surface_format(
                &caps.formats,
                &caps.format_capabilities,
                window.display_target.transfer,
                window.display_target.gamut,
            );

            // Non-sRGB 8-bit surface formats on the plain sRGB transfer are
            // rendered through an sRGB view so shaders always work in linear
            // space. HDR transfers never get a view, even when a last-resort
            // negotiation lands on an 8-bit format: the renderer's final
            // blit writes already-encoded signal values, and a hardware sRGB
            // encode on top would double-encode them. (Float/10-bit HDR
            // formats have no sRGB pair to begin with — `add_srgb_suffix`
            // is the identity.)
            let view_format = negotiated.format.add_srgb_suffix();
            let texture_view_format = (negotiated.resolved_transfer == DisplayTransfer::Srgb
                && view_format != negotiated.format)
                .then_some(view_format);
            let configuration = SurfaceConfiguration {
                format: negotiated.format,
                // The color space negotiated for the requested
                // `DisplayTarget::transfer` (`Auto` on the SDR path, letting
                // wgpu pick the color space). Every explicit pair comes from
                // `caps.format_capabilities`, so this configure cannot fail
                // wgpu's `UnsupportedColorSpace` validation.
                //
                // TODO: the wgpu surface color-space API does not (yet)
                // expose HDR10 mastering metadata (SMPTE ST 2086 display
                // primaries/luminance, CTA-861.3 MaxCLL/MaxFALL); drivers use
                // their own defaults. When it does, wire
                // `DisplayTarget::peak_luminance_nits` /
                // `min_luminance_nits` into it here at configure time.
                color_space: negotiated.color_space,
                width: window.physical_width,
                height: window.physical_height,
                usage: TextureUsages::RENDER_ATTACHMENT,
                present_mode,
                desired_maximum_frame_latency: window
                    .desired_maximum_frame_latency
                    .map(NonZero::<u32>::get)
                    .unwrap_or(DEFAULT_DESIRED_MAXIMUM_FRAME_LATENCY),
                alpha_mode: match window.alpha_mode {
                    CompositeAlphaMode::Auto => wgpu::CompositeAlphaMode::Auto,
                    CompositeAlphaMode::Opaque => wgpu::CompositeAlphaMode::Opaque,
                    CompositeAlphaMode::PreMultiplied => wgpu::CompositeAlphaMode::PreMultiplied,
                    CompositeAlphaMode::PostMultiplied => wgpu::CompositeAlphaMode::PostMultiplied,
                    CompositeAlphaMode::Inherit => wgpu::CompositeAlphaMode::Inherit,
                },
                view_formats: match texture_view_format {
                    Some(format) => vec![format],
                    None => vec![],
                },
            };

            render_device.configure_surface(&surface, &configuration);

            window_surfaces.surfaces.insert(
                window.entity,
                SurfaceData {
                    surface: WgpuWrapper::new(surface),
                    configuration,
                    texture_view_format,
                    resolved_transfer: negotiated.resolved_transfer,
                },
            );
        }
        let data = window_surfaces
            .surfaces
            .get_mut(&window.entity)
            .expect("surface was just created");

        if window.size_changed
            || window.present_mode_changed
            || window.display_target_transfer_changed
        {
            // normally this is dropped on present but we double check here to be safe as failure to
            // drop it will cause validation errors in wgpu
            drop(window.swap_chain_texture.take());
            #[cfg_attr(
                target_arch = "wasm32",
                expect(clippy::drop_non_drop, reason = "texture views are not drop on wasm")
            )]
            drop(window.swap_chain_texture_view.take());

            data.configuration.width = window.physical_width;
            data.configuration.height = window.physical_height;
            let caps = data.surface.get_capabilities(&render_adapter);
            data.configuration.present_mode = present_mode(window, &caps);
            if window.display_target_transfer_changed {
                // Re-run (format, color space) negotiation with the new
                // requested transfer. `cleanup_view_targets_for_resize` has
                // already invalidated the window's `ViewTarget`s this frame,
                // so pipelines specialized on the old output format are not
                // reused.
                data.apply_negotiated(negotiate_surface_format(
                    &caps.formats,
                    &caps.format_capabilities,
                    window.display_target.transfer,
                    window.display_target.gamut,
                ));
            } else if let Some(flag) = data.configuration.color_space.to_flag()
                && !caps.color_spaces(data.configuration.format).contains(flag)
            {
                // Defensive: explicit (non-`Auto`) color spaces can vanish
                // from the capability set at runtime — e.g. the user flips
                // the OS HDR toggle off, after which DX12 stops advertising
                // HDR10 for this output. Reconfiguring with the stored,
                // no-longer-advertised pair would fail wgpu validation
                // (`ConfigureSurfaceError::UnsupportedColorSpace`) and take
                // the renderer down, so renegotiate from the fresh
                // capabilities instead (typically degrading to SDR, with the
                // negotiation's own warnings). The new resolved transfer is
                // reported below; views key on it from this frame on.
                warn_once!(
                    "The configured surface color space ({:?}) is no longer supported \
                    for {:?} (did the OS HDR setting change?); renegotiating the \
                    swapchain from the current capabilities.",
                    data.configuration.color_space,
                    data.configuration.format
                );
                data.apply_negotiated(negotiate_surface_format(
                    &caps.formats,
                    &caps.format_capabilities,
                    window.display_target.transfer,
                    window.display_target.gamut,
                ));
            }
            render_device.configure_surface(&data.surface, &data.configuration);
        }

        // Report the transfer the configured surface can actually carry back
        // to the extracted window, where `prepare_view_display_targets` picks
        // it up to build each view's resolved `ViewDisplayTarget`.
        window.resolved_transfer = Some(data.resolved_transfer);

        window_surfaces.configured_windows.insert(window.entity);
    }
}

fn present_mode(
    window: &mut ExtractedWindow,
    caps: &wgpu::SurfaceCapabilities,
) -> wgpu::PresentMode {
    let present_mode = match window.present_mode {
        PresentMode::Fifo => wgpu::PresentMode::Fifo,
        PresentMode::FifoRelaxed => wgpu::PresentMode::FifoRelaxed,
        PresentMode::Mailbox => wgpu::PresentMode::Mailbox,
        PresentMode::Immediate => wgpu::PresentMode::Immediate,
        PresentMode::AutoVsync => wgpu::PresentMode::AutoVsync,
        PresentMode::AutoNoVsync => wgpu::PresentMode::AutoNoVsync,
    };
    let fallbacks = match present_mode {
        wgpu::PresentMode::AutoVsync => {
            &[wgpu::PresentMode::FifoRelaxed, wgpu::PresentMode::Fifo][..]
        }
        wgpu::PresentMode::AutoNoVsync => &[
            wgpu::PresentMode::Immediate,
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::Fifo,
        ][..],
        wgpu::PresentMode::Mailbox => &[
            wgpu::PresentMode::Mailbox,
            wgpu::PresentMode::Immediate,
            wgpu::PresentMode::Fifo,
        ][..],
        // Always end in FIFO to make sure it's always supported
        x => &[x, wgpu::PresentMode::Fifo][..],
    };
    let new_present_mode = fallbacks
        .iter()
        .copied()
        .find(|fallback| caps.present_modes.contains(fallback))
        .unwrap_or_else(|| {
            unreachable!(
                "Fallback system failed to choose present mode. \
                            This is a bug. Mode: {:?}, Options: {:?}",
                window.present_mode, &caps.present_modes
            );
        });
    if new_present_mode != present_mode && fallbacks.contains(&present_mode) {
        info!("PresentMode {present_mode:?} requested but not available. Falling back to {new_present_mode:?}");
    }
    new_present_mode
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fc(format: TextureFormat, color_spaces: SurfaceColorSpaces) -> SurfaceFormatCapabilities {
        SurfaceFormatCapabilities {
            format,
            color_spaces,
        }
    }

    /// [`negotiate_surface_format`] with the default Rec.709 gamut. The gamut
    /// only keys the `ExtendedSrgb` arm, so every other transfer's outcome is
    /// independent of it; tests that exercise the gamut call
    /// `negotiate_surface_format` directly.
    fn negotiate(
        auto_formats: &[TextureFormat],
        format_capabilities: &[SurfaceFormatCapabilities],
        requested_transfer: DisplayTransfer,
    ) -> NegotiatedSurface {
        negotiate_surface_format(
            auto_formats,
            format_capabilities,
            requested_transfer,
            DisplayGamut::Rec709,
        )
    }

    /// A Metal-like HDR-capable surface: every format also offers Display P3,
    /// `Rgba16Float` offers everything, and `Rgb10a2Unorm` adds HDR10/HLG.
    /// Order matters: the SDR path picks the first 8-bit sRGB format.
    fn metal_like() -> (Vec<TextureFormat>, Vec<SurfaceFormatCapabilities>) {
        (
            vec![
                TextureFormat::Bgra8UnormSrgb,
                TextureFormat::Bgra8Unorm,
                TextureFormat::Rgba16Float,
                TextureFormat::Rgb10a2Unorm,
            ],
            vec![
                fc(
                    TextureFormat::Bgra8UnormSrgb,
                    SurfaceColorSpaces::SRGB | SurfaceColorSpaces::DISPLAY_P3,
                ),
                fc(
                    TextureFormat::Bgra8Unorm,
                    SurfaceColorSpaces::SRGB | SurfaceColorSpaces::DISPLAY_P3,
                ),
                fc(
                    TextureFormat::Rgba16Float,
                    SurfaceColorSpaces::SRGB
                        | SurfaceColorSpaces::DISPLAY_P3
                        | SurfaceColorSpaces::EXTENDED_SRGB_LINEAR
                        | SurfaceColorSpaces::EXTENDED_SRGB
                        | SurfaceColorSpaces::EXTENDED_DISPLAY_P3
                        | SurfaceColorSpaces::HDR10
                        | SurfaceColorSpaces::HLG,
                ),
                fc(
                    TextureFormat::Rgb10a2Unorm,
                    SurfaceColorSpaces::SRGB
                        | SurfaceColorSpaces::DISPLAY_P3
                        | SurfaceColorSpaces::HDR10
                        | SurfaceColorSpaces::HLG,
                ),
            ],
        )
    }

    /// A browser-WebGPU-like surface on an HDR-capable display: the encoded
    /// extended-range sRGB / Display-P3 color spaces on `Rgba16Float` (the web
    /// HDR path), but NO `ExtendedSrgbLinear` (web cannot present a
    /// linear-transfer canvas) and NO HDR10.
    fn web_like() -> (Vec<TextureFormat>, Vec<SurfaceFormatCapabilities>) {
        (
            vec![TextureFormat::Bgra8UnormSrgb, TextureFormat::Rgba16Float],
            vec![
                fc(
                    TextureFormat::Bgra8UnormSrgb,
                    SurfaceColorSpaces::SRGB | SurfaceColorSpaces::DISPLAY_P3,
                ),
                fc(
                    TextureFormat::Rgba16Float,
                    SurfaceColorSpaces::SRGB
                        | SurfaceColorSpaces::DISPLAY_P3
                        | SurfaceColorSpaces::EXTENDED_SRGB
                        | SurfaceColorSpaces::EXTENDED_DISPLAY_P3,
                ),
            ],
        )
    }

    /// A Vulkan-like surface on an HDR-enabled display: `Rgb10a2Unorm` is
    /// HDR10-only (not `Auto`-configurable, so absent from `formats`).
    fn vulkan_hdr_like() -> (Vec<TextureFormat>, Vec<SurfaceFormatCapabilities>) {
        (
            vec![
                TextureFormat::Bgra8UnormSrgb,
                TextureFormat::Bgra8Unorm,
                TextureFormat::Rgba16Float,
            ],
            vec![
                fc(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
                fc(TextureFormat::Bgra8Unorm, SurfaceColorSpaces::SRGB),
                fc(
                    TextureFormat::Rgba16Float,
                    SurfaceColorSpaces::EXTENDED_SRGB_LINEAR,
                ),
                fc(
                    TextureFormat::Rgb10a2Unorm,
                    SurfaceColorSpaces::HDR10 | SurfaceColorSpaces::HLG,
                ),
            ],
        )
    }

    /// A surface with scRGB but no HDR10 (e.g. a backend/OS combination
    /// without PQ support).
    fn scrgb_only() -> (Vec<TextureFormat>, Vec<SurfaceFormatCapabilities>) {
        (
            vec![TextureFormat::Bgra8UnormSrgb, TextureFormat::Rgba16Float],
            vec![
                fc(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
                fc(
                    TextureFormat::Rgba16Float,
                    SurfaceColorSpaces::EXTENDED_SRGB_LINEAR,
                ),
            ],
        )
    }

    /// A surface without any HDR-capable color spaces (e.g. X11/GLES).
    fn sdr_only() -> (Vec<TextureFormat>, Vec<SurfaceFormatCapabilities>) {
        (
            vec![TextureFormat::Bgra8UnormSrgb, TextureFormat::Bgra8Unorm],
            vec![
                fc(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
                fc(TextureFormat::Bgra8Unorm, SurfaceColorSpaces::SRGB),
            ],
        )
    }

    const SDR_SELECTION: fn(TextureFormat) -> NegotiatedSurface = |format| NegotiatedSurface {
        format,
        color_space: SurfaceColorSpace::Auto,
        resolved_transfer: DisplayTransfer::Srgb,
    };

    #[test]
    fn srgb_default_selects_srgb_format_with_auto() {
        // The plain SDR default selects an sRGB format paired with `Auto`,
        // letting wgpu pick the color space.
        let (formats, caps) = metal_like();
        assert_eq!(
            negotiate(&formats, &caps, DisplayTransfer::Srgb),
            SDR_SELECTION(TextureFormat::Bgra8UnormSrgb)
        );
        let (formats, caps) = sdr_only();
        assert_eq!(
            negotiate(&formats, &caps, DisplayTransfer::Srgb),
            SDR_SELECTION(TextureFormat::Bgra8UnormSrgb)
        );
        // No sRGB format offered: fall back to the first supported format.
        assert_eq!(
            negotiate(
                &[TextureFormat::Bgra8Unorm, TextureFormat::Rgba16Float],
                &[],
                DisplayTransfer::Srgb
            ),
            SDR_SELECTION(TextureFormat::Bgra8Unorm)
        );
        // Rgba8UnormSrgb is picked when it is listed before Bgra8UnormSrgb.
        assert_eq!(
            negotiate(
                &[TextureFormat::Rgba8UnormSrgb, TextureFormat::Bgra8UnormSrgb],
                &[],
                DisplayTransfer::Srgb
            ),
            SDR_SELECTION(TextureFormat::Rgba8UnormSrgb)
        );
    }

    #[test]
    fn scrgb_picks_rgba16float_with_extended_srgb_linear() {
        let expected = NegotiatedSurface {
            format: TextureFormat::Rgba16Float,
            color_space: SurfaceColorSpace::ExtendedSrgbLinear,
            resolved_transfer: DisplayTransfer::ScRgbLinear,
        };
        let (formats, caps) = metal_like();
        assert_eq!(
            negotiate(&formats, &caps, DisplayTransfer::ScRgbLinear),
            expected
        );
        let (formats, caps) = vulkan_hdr_like();
        assert_eq!(
            negotiate(&formats, &caps, DisplayTransfer::ScRgbLinear),
            expected
        );
    }

    #[test]
    fn scrgb_requires_the_color_space_not_just_the_format() {
        // `Rgba16Float` is offered, but only in the sRGB color space:
        // presenting linear scRGB values into it would display incorrectly,
        // so the request must downgrade.
        let formats = vec![TextureFormat::Bgra8UnormSrgb, TextureFormat::Rgba16Float];
        let caps = vec![
            fc(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
            fc(TextureFormat::Rgba16Float, SurfaceColorSpaces::SRGB),
        ];
        assert_eq!(
            negotiate(&formats, &caps, DisplayTransfer::ScRgbLinear),
            SDR_SELECTION(TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn scrgb_downgrades_to_sdr_when_unavailable() {
        // Resolved transfer must report the downgrade so views take the
        // plain SDR path.
        let (formats, caps) = sdr_only();
        assert_eq!(
            negotiate(&formats, &caps, DisplayTransfer::ScRgbLinear),
            SDR_SELECTION(TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn pq_negotiates_hdr10_preferring_rgb10a2unorm() {
        let expected = NegotiatedSurface {
            format: TextureFormat::Rgb10a2Unorm,
            color_space: SurfaceColorSpace::Hdr10,
            resolved_transfer: DisplayTransfer::Pq,
        };
        // Vulkan-like: Rgb10a2Unorm is the only HDR10 format (and is not
        // Auto-configurable).
        let (formats, caps) = vulkan_hdr_like();
        assert_eq!(negotiate(&formats, &caps, DisplayTransfer::Pq), expected);
        // Metal-like: both Rgba16Float and Rgb10a2Unorm advertise HDR10;
        // Rgb10a2Unorm (PQ's native 10-bit container) is preferred even
        // though Rgba16Float is listed first.
        let (formats, caps) = metal_like();
        assert_eq!(negotiate(&formats, &caps, DisplayTransfer::Pq), expected);
    }

    #[test]
    fn pq_uses_rgba16float_when_it_is_the_only_hdr10_format() {
        let formats = vec![TextureFormat::Bgra8UnormSrgb, TextureFormat::Rgba16Float];
        let caps = vec![
            fc(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
            fc(
                TextureFormat::Rgba16Float,
                SurfaceColorSpaces::EXTENDED_SRGB_LINEAR | SurfaceColorSpaces::HDR10,
            ),
        ];
        assert_eq!(
            negotiate(&formats, &caps, DisplayTransfer::Pq),
            NegotiatedSurface {
                format: TextureFormat::Rgba16Float,
                color_space: SurfaceColorSpace::Hdr10,
                resolved_transfer: DisplayTransfer::Pq,
            }
        );
    }

    #[test]
    fn pq_takes_any_hdr10_format_as_a_last_resort() {
        // Weird combo: a driver advertising HDR10 on an 8-bit format only.
        let formats = vec![TextureFormat::Bgra8UnormSrgb, TextureFormat::Bgra8Unorm];
        let caps = vec![
            fc(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
            fc(
                TextureFormat::Bgra8Unorm,
                SurfaceColorSpaces::SRGB | SurfaceColorSpaces::HDR10,
            ),
        ];
        assert_eq!(
            negotiate(&formats, &caps, DisplayTransfer::Pq),
            NegotiatedSurface {
                format: TextureFormat::Bgra8Unorm,
                color_space: SurfaceColorSpace::Hdr10,
                resolved_transfer: DisplayTransfer::Pq,
            }
        );
    }

    #[test]
    fn pq_downgrades_through_scrgb_to_sdr() {
        // Downgrade chain: PQ → scRGB-linear when HDR10 is unavailable…
        let (formats, caps) = scrgb_only();
        assert_eq!(
            negotiate(&formats, &caps, DisplayTransfer::Pq),
            NegotiatedSurface {
                format: TextureFormat::Rgba16Float,
                color_space: SurfaceColorSpace::ExtendedSrgbLinear,
                resolved_transfer: DisplayTransfer::ScRgbLinear,
            }
        );
        // …and all the way to SDR sRGB when scRGB is unavailable too.
        let (formats, caps) = sdr_only();
        assert_eq!(
            negotiate(&formats, &caps, DisplayTransfer::Pq),
            SDR_SELECTION(TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn hlg_is_fulfilled_as_hdr10_pq() {
        // HLG is never negotiated as an HLG swapchain (the display pipeline
        // cannot produce a correct scene-referred HLG signal); it resolves to
        // PQ/HDR10 — even when the surface advertises the HLG color space —
        // and follows PQ's downgrade chain otherwise.
        let (formats, caps) = metal_like();
        assert_eq!(
            negotiate(&formats, &caps, DisplayTransfer::Hlg),
            NegotiatedSurface {
                format: TextureFormat::Rgb10a2Unorm,
                color_space: SurfaceColorSpace::Hdr10,
                resolved_transfer: DisplayTransfer::Pq,
            }
        );
        let (formats, caps) = scrgb_only();
        assert_eq!(
            negotiate(&formats, &caps, DisplayTransfer::Hlg),
            NegotiatedSurface {
                format: TextureFormat::Rgba16Float,
                color_space: SurfaceColorSpace::ExtendedSrgbLinear,
                resolved_transfer: DisplayTransfer::ScRgbLinear,
            }
        );
        let (formats, caps) = sdr_only();
        assert_eq!(
            negotiate(&formats, &caps, DisplayTransfer::Hlg),
            SDR_SELECTION(TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn empty_auto_formats_fall_back_to_an_explicit_color_space() {
        // Weird combo: a driver in OS HDR mode reporting formats only in
        // explicit-opt-in color spaces. Configuring with `Auto` would fail
        // validation, so the negotiation must pick an explicit pair (and
        // must not panic).
        let caps = vec![fc(TextureFormat::Rgb10a2Unorm, SurfaceColorSpaces::HDR10)];
        assert_eq!(
            negotiate(&[], &caps, DisplayTransfer::Srgb),
            NegotiatedSurface {
                format: TextureFormat::Rgb10a2Unorm,
                color_space: SurfaceColorSpace::Hdr10,
                resolved_transfer: DisplayTransfer::Pq,
            }
        );
        // An explicitly-advertised sRGB pair is preferred when present.
        let caps = vec![
            fc(TextureFormat::Rgb10a2Unorm, SurfaceColorSpaces::HDR10),
            fc(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
        ];
        assert_eq!(
            negotiate(&[], &caps, DisplayTransfer::Srgb),
            NegotiatedSurface {
                format: TextureFormat::Bgra8UnormSrgb,
                color_space: SurfaceColorSpace::Srgb,
                resolved_transfer: DisplayTransfer::Srgb,
            }
        );
    }

    #[test]
    fn extended_srgb_rec709_negotiates_extended_srgb() {
        let expected = NegotiatedSurface {
            format: TextureFormat::Rgba16Float,
            color_space: SurfaceColorSpace::ExtendedSrgb,
            resolved_transfer: DisplayTransfer::ExtendedSrgb,
        };
        // The web HDR path: an `Rgba16Float` `ExtendedSrgb` swapchain.
        let (formats, caps) = web_like();
        assert_eq!(
            negotiate_surface_format(
                &formats,
                &caps,
                DisplayTransfer::ExtendedSrgb,
                DisplayGamut::Rec709
            ),
            expected
        );
        // Also advertised on a Metal-like native surface.
        let (formats, caps) = metal_like();
        assert_eq!(
            negotiate_surface_format(
                &formats,
                &caps,
                DisplayTransfer::ExtendedSrgb,
                DisplayGamut::Rec709
            ),
            expected
        );
    }

    #[test]
    fn extended_srgb_displayp3_negotiates_extended_display_p3() {
        // A Display-P3 gamut request resolves to the wide-gamut HDR color
        // space; the resolved transfer is still `ExtendedSrgb` (the gamut rides
        // `DisplayTarget::gamut`, not the transfer).
        let expected = NegotiatedSurface {
            format: TextureFormat::Rgba16Float,
            color_space: SurfaceColorSpace::ExtendedDisplayP3,
            resolved_transfer: DisplayTransfer::ExtendedSrgb,
        };
        let (formats, caps) = web_like();
        assert_eq!(
            negotiate_surface_format(
                &formats,
                &caps,
                DisplayTransfer::ExtendedSrgb,
                DisplayGamut::DisplayP3
            ),
            expected
        );
    }

    #[test]
    fn extended_srgb_displayp3_without_p3_support_downgrades_straight_to_sdr() {
        // No cross-gamut downgrade: a Display-P3 request on a surface that
        // advertises only the Rec.709 `ExtendedSrgb` space degrades to SDR
        // rather than to that 709 surface (which would mismatch the encoder's
        // resolved P3 gamut).
        let formats = vec![TextureFormat::Bgra8UnormSrgb, TextureFormat::Rgba16Float];
        let caps = vec![
            fc(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
            fc(
                TextureFormat::Rgba16Float,
                SurfaceColorSpaces::EXTENDED_SRGB,
            ),
        ];
        assert_eq!(
            negotiate_surface_format(
                &formats,
                &caps,
                DisplayTransfer::ExtendedSrgb,
                DisplayGamut::DisplayP3
            ),
            SDR_SELECTION(TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn extended_srgb_without_support_downgrades_to_sdr() {
        let (formats, caps) = sdr_only();
        assert_eq!(
            negotiate_surface_format(
                &formats,
                &caps,
                DisplayTransfer::ExtendedSrgb,
                DisplayGamut::Rec709
            ),
            SDR_SELECTION(TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn scrgb_linear_does_not_fall_back_to_extended_srgb() {
        // scRGB-linear is native-only; on a web-like surface (which advertises
        // the encoded `ExtendedSrgb` but not `ExtendedSrgbLinear`) a
        // `ScRgbLinear` request degrades straight to SDR — there is no
        // cross-transfer auto-fallback to the encoded extended-sRGB transfer
        // (apps target the web HDR path by requesting `ExtendedSrgb`).
        let (formats, caps) = web_like();
        assert_eq!(
            negotiate(&formats, &caps, DisplayTransfer::ScRgbLinear),
            SDR_SELECTION(TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn pq_does_not_fall_back_to_extended_srgb() {
        // PQ's downgrade chain is unchanged (PQ → scRGB-linear → SDR); it never
        // resolves to the encoded extended-sRGB transfer. On a web-like surface
        // (no HDR10, no scRGB-linear) it lands on SDR.
        let (formats, caps) = web_like();
        assert_eq!(
            negotiate(&formats, &caps, DisplayTransfer::Pq),
            SDR_SELECTION(TextureFormat::Bgra8UnormSrgb)
        );
    }

    #[test]
    fn empty_auto_formats_fall_back_to_extended_srgb_spaces() {
        // A driver reporting only the encoded extended-sRGB space (no
        // Auto-configurable format): a non-P3 request drives the `ExtendedSrgb`
        // surface rather than panicking.
        let caps = vec![fc(
            TextureFormat::Rgba16Float,
            SurfaceColorSpaces::EXTENDED_SRGB,
        )];
        assert_eq!(
            negotiate_surface_format(&[], &caps, DisplayTransfer::Srgb, DisplayGamut::Rec709),
            NegotiatedSurface {
                format: TextureFormat::Rgba16Float,
                color_space: SurfaceColorSpace::ExtendedSrgb,
                resolved_transfer: DisplayTransfer::ExtendedSrgb,
            }
        );
        // The extended-P3 fallback is taken only for a Display-P3 request, so
        // the negotiated surface gamut matches what the encoder emits.
        let caps = vec![fc(
            TextureFormat::Rgba16Float,
            SurfaceColorSpaces::EXTENDED_DISPLAY_P3,
        )];
        assert_eq!(
            negotiate_surface_format(&[], &caps, DisplayTransfer::Srgb, DisplayGamut::DisplayP3),
            NegotiatedSurface {
                format: TextureFormat::Rgba16Float,
                color_space: SurfaceColorSpace::ExtendedDisplayP3,
                resolved_transfer: DisplayTransfer::ExtendedSrgb,
            }
        );
        // A non-P3 request must NOT take the extended-P3 fallback (it would
        // mismatch the encoder's Rec.709 gamut); with no other drivable space
        // it panics — exercised here by confirming the P3 entry is skipped via
        // a surface that also offers SRGB.
        let caps = vec![
            fc(
                TextureFormat::Rgba16Float,
                SurfaceColorSpaces::EXTENDED_DISPLAY_P3,
            ),
            fc(TextureFormat::Bgra8UnormSrgb, SurfaceColorSpaces::SRGB),
        ];
        assert_eq!(
            negotiate_surface_format(&[], &caps, DisplayTransfer::Srgb, DisplayGamut::Rec709),
            NegotiatedSurface {
                format: TextureFormat::Bgra8UnormSrgb,
                color_space: SurfaceColorSpace::Srgb,
                resolved_transfer: DisplayTransfer::Srgb,
            }
        );
    }
}
