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

use bevy_camera::visibility::InheritedVisibility;
use bevy_color::ColorToComponents as _;
use bevy_ecs::{
    component::Component,
    resource::Resource,
    system::{Query, ResMut},
};
use bevy_light::FogVolume;
use bevy_math::{Mat4, UVec2, UVec3, Vec3, Vec4};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    extract_component::ExtractComponent,
    render_resource::{ShaderType, UniformBuffer},
    Extract, RenderApp,
};
use bevy_transform::components::GlobalTransform;

/// Near-field froxel volumetrics for a camera with [`super::AtmosphereSettings`].
///
/// Phase 2: air, an optional height fog and the scene's [`FogVolume`]s, sun-lit through a
/// temporally filtered shadow volume (directional shadow cascades x the cloud shadow map), no
/// local lights, no ambient, no clouds in the grid. Runs only for views that render a
/// `VolumetricCloudLayer` without MSAA (the half-res haze prepass path).
///
/// A camera with this component enabled and a [`bevy_light::VolumetricFog`] skips the per-pixel
/// `volumetric_fog` pass: the grid consumes its `FogVolume`s (`VolumetricLight` keeps its
/// meaning for point and spot lights; the atmosphere's directional lights always scatter).
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
    /// Weight of the current frame in the shadow volume's temporal filter (the history is
    /// reprojected through the previous view; RDR2 slide 43 filters the shadow volume "to resolve
    /// undersampling on high frequency shadow detail"). Default 0.05: FB15 slide 23 blends the
    /// current frame into the previous at 5%. 1 disables the filter.
    pub shadow_temporal_alpha: f32,
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
            shadow_temporal_alpha: 0.05,
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

    /// The GPU parameters for a grid of `grid` froxels, with no history (the temporal fields
    /// are filled by `prepare_volumetrics_uniforms` from the per-view state).
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
            prev_clip_from_world: Mat4::IDENTITY,
            prev_world_position: Vec3::ZERO,
            history_valid: 0,
            shadow_temporal_alpha: self.shadow_temporal_alpha.clamp(0.0, 1.0),
            _pad2: UVec3::ZERO,
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
    /// The previous frame's view-projection and camera position: the shadow volume's history is
    /// reprojected through them (world position -> previous clip -> previous grid uv, distance
    /// to the previous camera -> previous slice).
    pub prev_clip_from_world: Mat4,
    pub prev_world_position: Vec3,
    /// Whether the shadow history volume holds last frame's result for this grid (0 on the first
    /// frame, after a resize, or when the passes did not run last frame).
    pub history_valid: u32,
    pub shadow_temporal_alpha: f32,
    pub _pad2: UVec3,
}

/// Per-view uniform of [`GpuVolumetricsParams`] and the grid it was built for; present on views
/// whose froxel passes run this frame.
#[derive(Component)]
pub struct VolumetricsUniform {
    pub buffer: UniformBuffer<GpuVolumetricsParams>,
    pub grid: UVec3,
}

/// Most [`FogVolume`]s injected into the grid per frame (RDR2 slide 45: up to 32 fog volumes,
/// Z-binned; the grid tests every froxel against every volume, Z-binning is a Phase 5 item).
pub const MAX_FOG_VOLUMES: usize = 32;

/// A [`FogVolume`] for the material pass (`volumetrics.wesl` `FogVolume`): the unit cube
/// [-0.5, 0.5]^3 in its local space, with bevy_light's fields multiplied out the way the
/// per-pixel `volumetric_fog.wesl` applies them (sigma_t = density_factor x (absorption +
/// scattering); in-scatter = fog_color x light_tint x light_intensity x scattering x
/// density_factor x phase x light, so the two non-physical light scales fold into sigma_s).
/// `density_texture` and `density_texture_offset` are not applied yet (uniform density): one 3D
/// texture per volume needs a binding array the atmosphere's group 0 does not have.
#[derive(Clone, Copy, ShaderType)]
pub struct GpuFogVolume {
    pub local_from_world: Mat4,
    /// Scattering coefficient (1/m), per channel.
    pub sigma_s: Vec3,
    /// Extinction coefficient (1/m).
    pub sigma_t: f32,
    /// x: `scattering_asymmetry` (Henyey-Greenstein g); yzw reserved.
    pub params: Vec4,
}

impl GpuFogVolume {
    pub fn new(volume: &FogVolume, transform: &GlobalTransform) -> Self {
        let color = volume.fog_color.to_linear().to_vec3();
        let tint = volume.light_tint.to_linear().to_vec3();
        let density = volume.density_factor.max(0.0);
        Self {
            local_from_world: Mat4::from(transform.affine().inverse()),
            sigma_s: color * tint * volume.light_intensity * volume.scattering * density,
            sigma_t: density * (volume.absorption + volume.scattering),
            params: Vec4::new(volume.scattering_asymmetry.clamp(-0.99, 0.99), 0.0, 0.0, 0.0),
        }
    }
}

/// The frame's fog volumes (`volumetrics.wesl` `FogVolumes`, group 0 binding 81).
#[derive(Clone, ShaderType)]
pub struct GpuFogVolumes {
    pub count: u32,
    pub data: [GpuFogVolume; MAX_FOG_VOLUMES],
}

impl Default for GpuFogVolumes {
    fn default() -> Self {
        let empty = GpuFogVolume {
            local_from_world: Mat4::IDENTITY,
            sigma_s: Vec3::ZERO,
            sigma_t: 0.0,
            params: Vec4::ZERO,
        };
        Self {
            count: 0,
            data: [empty; MAX_FOG_VOLUMES],
        }
    }
}

/// The visible [`FogVolume`]s of the main world this frame, for the froxel grid (every view
/// sees the same list; the material pass box-tests each froxel against each).
#[derive(Resource, Default)]
pub struct ExtractedFogVolumes(pub GpuFogVolumes);

/// Extracts the visible [`FogVolume`]s (up to [`MAX_FOG_VOLUMES`]; `Visibility::Hidden` volumes
/// are skipped, as `volumetric_fog`'s extraction does).
pub fn extract_fog_volumes(
    mut extracted: ResMut<ExtractedFogVolumes>,
    volumes: Extract<Query<(&FogVolume, &GlobalTransform, Option<&InheritedVisibility>)>>,
) {
    let out = &mut extracted.0;
    out.count = 0;
    for (volume, transform, visibility) in &volumes {
        if !visibility.is_none_or(|v| v.get()) {
            continue;
        }
        if out.count as usize >= MAX_FOG_VOLUMES {
            bevy_log::warn_once!(
                "more than {MAX_FOG_VOLUMES} FogVolumes; the froxel grid injects the first {MAX_FOG_VOLUMES}"
            );
            break;
        }
        out.data[out.count as usize] = GpuFogVolume::new(volume, transform);
        out.count += 1;
    }
}

/// The uniform of [`ExtractedFogVolumes`], written by `prepare_volumetrics_uniforms`.
#[derive(Resource, Default)]
pub struct FogVolumesUniform(pub UniformBuffer<GpuFogVolumes>);

/// Per-view state of the froxel grid across frames (`node::PerViewLutState`): what the shadow
/// volume's temporal filter reprojects against.
#[derive(Default)]
pub struct VolumetricsViewState {
    /// Last frame's view-projection and camera position, when its passes ran.
    pub prev_view: Option<(Mat4, Vec3)>,
    /// The shadow history texture the passes last wrote (a resize makes a new texture).
    pub shadow_history: Option<bevy_render::render_resource::TextureId>,
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
