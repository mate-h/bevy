//! Near-field froxel volumetrics (Phase 1 of the unified volumetrics plan): a camera-aligned
//! grid of frustum voxels over the first [`VolumetricsSettings::max_distance`] metres holding the
//! air (Rayleigh, Mie, absorption from the atmosphere's medium) and an optional exponential
//! height fog, lit by the sun, integrated front to back and composited by `render_sky` in place
//! of the near part of the per-pixel haze march.
//!
//! Lineage: Wronski 2014 (AC4, 160x90x64, exponential near-weighted slices, 50-128 m),
//! Hillaire 2015 (Frostbite: material V-buffer, energy-conserving slice integration),
//! Bauer 2019 (RDR2: near froxels to <160 m, far raymarch "starts where voxels end", no temporal
//! blend on scattered light, dithered lookups). See `reports/Unified volumetrics plan for Bevy.md`.

use bevy_ecs::component::Component;
use bevy_math::{UVec2, UVec3, Vec3};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    extract_component::ExtractComponent,
    render_resource::{ShaderType, UniformBuffer},
    RenderApp,
};

/// Near-field froxel volumetrics for a camera with [`super::AtmosphereSettings`].
///
/// Phase 1: air plus an optional height fog, sun-lit (directional shadow cascades and the cloud
/// shadow map evaluated inline), no local lights, no ambient, no clouds in the grid. Runs only
/// for views that render a `VolumetricCloudLayer` without MSAA (the half-res haze prepass path).
#[derive(Clone, Component, ExtractComponent, Reflect)]
#[extract_app(RenderApp)]
#[reflect(Clone, Default)]
pub struct VolumetricsSettings {
    /// Whether the froxel grid is built and applied. Off, the previous per-pixel haze path is
    /// unchanged.
    pub enabled: bool,
    /// Screen pixels per froxel tile, at the cloud/haze render-size cap
    /// (`VolumetricCloudSettings::max_render_size`): 8 (FB15 slide 12 default; W14 160x90 at
    /// 720p; RDR2 160x88), 16 for a LOW budget (FB15 slide 36: 0.83 vs 2.95 ms on PS4).
    pub tile_px: u32,
    /// Depth slices: 64 (W14 p.25, FB15 slide 12, RDR2 slide 42, HDRP default).
    pub slices: u32,
    /// Far end of the grid (m): 128 (W14 p.25: 50-128 m; RDR2 slide 42: <160 m; design.md
    /// `D_f`). Beyond it the aerial-perspective LUT and the far haze march take over.
    pub max_distance: f32,
    /// Slice distribution, 0 = exponential (near precision) to 1 = uniform, as HDRP's
    /// "Slice Distribution Uniformity". The slice boundaries follow HDRP's logarithmic depth
    /// encoding (`VolumetricLighting.cs`, `ComputeLogarithmicDepthEncodingParams`, recalled
    /// from source and not re-verified this session):
    ///
    /// `c = max(2 - 2 * uniformity, 1e-3)` (1/m), `u(t) = log2(c t + 1) / log2(c d_max + 1)`,
    /// so slice `i` of `N` ends at `t_i = (2^((i / N) log2(c d_max + 1)) - 1) / c`.
    ///
    /// `c -> 0` is linear spacing, `c = 2` logarithmic from a 0.5 m scale. The form
    /// `t_i = mix(linear, near (d_max / near)^(i/N), uniformity)` the plan sketches is an
    /// inference (no primary source states its formula); the HDRP form is used because it has a
    /// closed-form inverse for the lookup and no free `near` constant. Default 0.75 (HDRP's
    /// `sliceDistributionUniformity` default).
    pub depth_uniformity: f32,
    /// Height fog extinction at `fog_height` (1/m); 0 = no fog (default). The fog is an
    /// exponential height profile `fog_density * exp(-fog_height_falloff * (h - fog_height))`
    /// (W14 p.37: "exponential height falloff"; UE Exponential Height Fog).
    pub fog_density: f32,
    /// Altitude above the planet surface (m) where the fog has `fog_density`.
    pub fog_height: f32,
    /// Fog density falloff with altitude (1/m).
    pub fog_height_falloff: f32,
    /// Single-scattering albedo of the fog (sigma_s / sigma_t), per channel.
    pub fog_albedo: Vec3,
    /// Henyey-Greenstein asymmetry of the fog's single lobe (FB15 slide 16: one HG lobe).
    /// Default 0.5, Bevy's `FogVolume::scattering_asymmetry` default.
    pub fog_anisotropy: f32,
}

impl Default for VolumetricsSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            tile_px: 8,
            slices: 64,
            max_distance: 128.0,
            depth_uniformity: 0.75,
            fog_density: 0.0,
            fog_height: 0.0,
            fog_height_falloff: 0.0,
            fog_albedo: Vec3::ONE,
            fog_anisotropy: 0.5,
        }
    }
}

impl VolumetricsSettings {
    /// Froxel grid size for a view: `ceil(fit / tile_px)` per axis, where `fit` is the render
    /// size scaled down (same aspect) to fit within `cap` (the cloud/haze render-size cap), times
    /// `slices`. 120x68x64 at a 960x540 cap, 160x90x64 at 1280x720.
    pub fn grid_size(&self, render_size: UVec2, cap: UVec2) -> UVec3 {
        let fit = fit_within(render_size.max(UVec2::ONE), cap.max(UVec2::ONE));
        let tile = self.tile_px.max(1);
        UVec3::new(
            fit.x.div_ceil(tile).max(1),
            fit.y.div_ceil(tile).max(1),
            self.slices.max(1),
        )
    }

    /// The GPU parameters for a grid of `grid` froxels.
    pub fn gpu_params(&self, grid: UVec3) -> GpuVolumetricsParams {
        // HDRP: c = 2 - 2 * uniformity, clamped away from 0 (linear limit).
        let c = (2.0 - 2.0 * self.depth_uniformity.clamp(0.0, 1.0)).max(1e-3);
        let d = self.max_distance.max(1.0);
        GpuVolumetricsParams {
            grid_size: grid,
            tile_px: self.tile_px.max(1),
            fog_albedo: self.fog_albedo.clamp(Vec3::ZERO, Vec3::ONE),
            fog_density: self.fog_density.max(0.0),
            max_distance: d,
            depth_c: c,
            depth_k: (c * d + 1.0).log2(),
            fog_height_falloff: self.fog_height_falloff.max(0.0),
            fog_height: self.fog_height,
            fog_anisotropy: self.fog_anisotropy.clamp(-0.99, 0.99),
            enabled: self.enabled as u32,
            _pad: 0,
        }
    }
}

/// `size` scaled down (same aspect) to fit within `cap`.
fn fit_within(size: UVec2, cap: UVec2) -> UVec2 {
    if size.x <= cap.x && size.y <= cap.y {
        return size;
    }
    let s = (cap.x as f64 / size.x as f64).min(cap.y as f64 / size.y as f64);
    UVec2::new(
        ((size.x as f64 * s).floor() as u32).max(1),
        ((size.y as f64 * s).floor() as u32).max(1),
    )
}

/// Per-view froxel parameters (`froxels.wesl` `VolumetricsParams`, group 0 binding 70).
#[derive(Clone, Copy, ShaderType)]
pub struct GpuVolumetricsParams {
    pub grid_size: UVec3,
    pub tile_px: u32,
    pub fog_albedo: Vec3,
    pub fog_density: f32,
    pub max_distance: f32,
    pub depth_c: f32,
    pub depth_k: f32,
    pub fog_height_falloff: f32,
    pub fog_height: f32,
    pub fog_anisotropy: f32,
    pub enabled: u32,
    pub _pad: u32,
}

/// Per-view uniform of [`GpuVolumetricsParams`] and the grid it was built for; present on views
/// whose froxel passes run this frame.
#[derive(Component)]
pub struct VolumetricsUniform {
    pub buffer: UniformBuffer<GpuVolumetricsParams>,
    pub grid: UVec3,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_from_render_size_cap() {
        let s = VolumetricsSettings::default();
        assert_eq!(
            s.grid_size(UVec2::new(1280, 720), UVec2::new(960, 540)),
            UVec3::new(120, 68, 64)
        );
        assert_eq!(
            s.grid_size(UVec2::new(1280, 720), UVec2::new(1280, 720)),
            UVec3::new(160, 90, 64)
        );
        assert_eq!(
            s.grid_size(UVec2::new(2560, 1440), UVec2::new(960, 540)),
            UVec3::new(120, 68, 64)
        );
    }

    #[test]
    fn slice_boundaries_span_the_grid() {
        let s = VolumetricsSettings::default();
        let p = s.gpu_params(UVec3::new(120, 68, 64));
        let t = |u: f32| ((u * p.depth_k).exp2() - 1.0) / p.depth_c;
        assert!(t(0.0).abs() < 1e-6);
        assert!((t(1.0) - 128.0).abs() < 1e-3);
        // Uniformity 0.75 (c = 0.5): slice 32 ends at (65^0.5 - 1) / 0.5 = 14.1 m.
        assert!((t(0.5) - 14.12).abs() < 0.05);
    }
}
