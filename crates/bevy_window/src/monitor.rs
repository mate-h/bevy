use alloc::{string::String, vec::Vec};
use bevy_ecs::{component::Component, entity::Entity};
use bevy_math::{IVec2, UVec2};

#[cfg(all(feature = "serialize", feature = "bevy_reflect"))]
use bevy_reflect::{ReflectDeserialize, ReflectSerialize};
#[cfg(feature = "bevy_reflect")]
use {bevy_ecs::prelude::ReflectComponent, bevy_reflect::Reflect};

/// Represents an available monitor as reported by the user's operating system, which can be used
/// to query information about the display, such as its size, position, and video modes.
///
/// Each monitor corresponds to an entity and can be used to position a monitor using
/// [`MonitorSelection::Entity`](`crate::window::MonitorSelection::Entity`).
///
/// # Warning
///
/// This component is synchronized with `winit` through `bevy_winit`, but is effectively
/// read-only as `winit` does not support changing monitor properties.
///
/// # HDR capability metadata
///
/// `Monitor` currently carries **no luminance, gamut, or HDR-capability fields** (peak/min
/// luminance, EDID-derived primaries, current HDR enablement): `winit` 0.30 exposes none of
/// these on its `MonitorHandle`, and `wgpu` has no display-capability query either (the HDR
/// work under <https://github.com/gfx-rs/wgpu/issues/2920> covers surface color-space
/// *selection* — which Bevy's HDR output uses — but not monitor metadata). Such fields would
/// be purely additive here once an upstream source exists.
///
/// Until then, display calibration is described by the user-authoritative
/// [`DisplayTarget`](crate::DisplayTarget) component on each [`Window`](crate::Window),
/// populated from OS settings, user input, or an HGIG-style calibration flow (see the
/// `hdr_calibration` example). The [`WindowMonitorChanged`](crate::WindowMonitorChanged)
/// event signals when a window moves to a different monitor and recalibration may be
/// warranted.
#[derive(Component, Debug, Clone)]
#[require(HasWindows)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Component, Debug, Clone)
)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct Monitor {
    /// The name of the monitor
    pub name: Option<String>,
    /// The height of the monitor in physical pixels
    pub physical_height: u32,
    /// The width of the monitor in physical pixels
    pub physical_width: u32,
    /// The position of the monitor in physical pixels
    pub physical_position: IVec2,
    /// The refresh rate of the monitor in millihertz
    pub refresh_rate_millihertz: Option<u32>,
    /// The scale factor of the monitor
    pub scale_factor: f64,
    /// The video modes that the monitor supports
    pub video_modes: Vec<VideoMode>,
}

/// A marker component for the primary monitor
#[derive(Component, Debug, Clone)]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Component, Debug, Clone)
)]
pub struct PrimaryMonitor;

/// A relationship for all Windows on a specific Monitor.
#[derive(Component, Debug, Default)]
#[relationship_target(relationship = crate::window::OnMonitor, linked_spawn)]
pub struct HasWindows(Vec<Entity>);

impl Monitor {
    /// Returns the physical size of the monitor in pixels
    pub fn physical_size(&self) -> UVec2 {
        UVec2::new(self.physical_width, self.physical_height)
    }
}

/// Represents a video mode that a monitor supports
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "bevy_reflect", derive(Reflect), reflect(Debug, Clone))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct VideoMode {
    /// The resolution of the video mode
    pub physical_size: UVec2,
    /// The bit depth of the video mode
    pub bit_depth: u16,
    /// The refresh rate in millihertz
    pub refresh_rate_millihertz: u32,
}
