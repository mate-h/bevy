//! Near-field froxel grid (`volumetrics.rs`): the grid's depth mapping, the per-ray jitter and the
//! lookup of the integrated volume, shared by the froxel compute passes (`volumetrics.wesl`) and
//! the composite in `render_sky.wesl`.

import bevy_render::maths::PI;
import package::atmosphere::bindings::{view, atmosphere, lights, atmosphere_lut_sampler};
import package::atmosphere::functions::sample_transmittance_lut;
import package::render::utils::interleaved_gradient_noise;

@if(CLOUDS_ENABLED)
import package::atmosphere::{
    bindings::stbn_texture,
    functions::{sample_cloud_shadow_ground, ground_sky_irradiance_clouds},
};

/// `GpuVolumetricsParams` (`volumetrics.rs`).
struct VolumetricsParams {
    grid_size: vec3<u32>,
    tile_px: u32,
    fog_albedo: vec3<f32>,
    /// Fog extinction at `fog_height` (1/m).
    fog_density: f32,
    /// Far end of the grid (m).
    max_distance: f32,
    /// HDRP logarithmic depth encoding: c = 2 - 2 uniformity (1/m), k = log2(c d_max + 1).
    depth_c: f32,
    depth_k: f32,
    fog_height_falloff: f32,
    fog_height: f32,
    fog_anisotropy: f32,
    enabled: u32,
    _pad: u32,
    /// Last frame's view-projection and camera position (the shadow volume's reprojection).
    prev_clip_from_world: mat4x4<f32>,
    prev_world_position: vec3<f32>,
    /// Whether the shadow history holds last frame's volume for this grid.
    history_valid: u32,
    /// Weight of the current frame in the shadow volume's EMA (FB15 slide 23: 5%).
    shadow_temporal_alpha: f32,
    _pad2: vec3<u32>,
}
@group(0) @binding(70) var<uniform> volumetrics: VolumetricsParams;

/// Most `FogVolume`s injected per frame (`volumetrics.rs` `MAX_FOG_VOLUMES`; RDR2 slide 45).
const MAX_FOG_VOLUMES: u32 = 32u;

/// Integrated volume (in-scatter to the camera, pre-exposed, and transmittance), read by the apply.
@if(VOLUMETRICS_APPLY)
@group(0) @binding(76) var volumetrics_integrated: texture_3d<f32>;

/// Distance (m, along the ray) of the slice boundary at normalised depth `u` in [0, 1]:
/// t(u) = (2^(u k) - 1) / c (HDRP `DecodeLogarithmicDepthGeneralized`, near plane 0).
fn froxel_boundary_t(u: f32) -> f32 {
    return (exp2(u * volumetrics.depth_k) - 1.0) / volumetrics.depth_c;
}

/// Normalised depth of distance `t` (the inverse of `froxel_boundary_t`).
fn froxel_t_to_u(t: f32) -> f32 {
    return log2(volumetrics.depth_c * max(t, 0.0) + 1.0) / volumetrics.depth_k;
}

/// Near and far distance of slice `z`.
fn froxel_slice_bounds(z: u32) -> vec2<f32> {
    let n = f32(volumetrics.grid_size.z);
    return vec2(froxel_boundary_t(f32(z) / n), froxel_boundary_t(f32(z + 1u) / n));
}

/// World (atmosphere-space) ray direction through the centre of tile `xy`.
fn froxel_tile_uv(xy: vec2<u32>) -> vec2<f32> {
    return (vec2<f32>(xy) + 0.5) / vec2<f32>(volumetrics.grid_size.xy);
}

/// Blue-noise value in [0, 1) for screen position `px` (pixels or tiles) this frame: the STBN
/// layer `frame % layers` (the texture the atmosphere's marches jitter with), IGN when it is not
/// bound. Used once per ray so material and light samples are jittered in sync (FB15 slide 23).
fn froxel_noise(px: vec2<i32>) -> f32 {
    @if(CLOUDS_ENABLED) {
    let dims = textureDimensions(stbn_texture);
    if (all(dims > vec2(1u))) {
        let layer = i32(view.frame_count % u32(textureNumLayers(stbn_texture)));
        return textureLoad(stbn_texture, (px % vec2<i32>(dims) + vec2<i32>(dims)) % vec2<i32>(dims), layer, 0).r;
    }
    }
    return interleaved_gradient_noise(vec2<f32>(px), view.frame_count % 64u);
}

/// The jittered sample distance of froxel `(xy, z)`: the same offset `j` along the whole ray.
fn froxel_sample_t(z: u32, j: f32) -> f32 {
    let b = froxel_slice_bounds(z);
    return mix(b.x, b.y, j);
}

/// Height fog extinction (1/m) at radius `r` (W14 p.37 exponential height falloff).
fn froxel_fog_extinction(r: f32) -> f32 {
    let h = r - atmosphere.inner_radius - volumetrics.fog_height;
    return volumetrics.fog_density * exp(-volumetrics.fog_height_falloff * h);
}

/// (1 - exp(-x)) / x, the slab integral of FB16 Eq.17 per unit length, with its Taylor series
/// for small x: the direct form loses all precision when x (an air optical depth of 1e-6 over a
/// 0.1 m slice) is below f32's resolution of 1 - exp(-x).
fn one_minus_exp_over_x(x: f32) -> f32 {
    if (x < 1e-3) {
        return 1.0 - 0.5 * x + x * x / 6.0;
    }
    return (1.0 - exp(-x)) / x;
}

/// Integrated in-scatter (rgb, pre-exposed) and transmittance (a) from the camera to distance
/// `t` along the ray through `uv`, trilinear in the volume with a `dither` of +-0.5 slice (RDR2
/// slide 56: the dithered-lookup principle applied to the accumulation volume). Texel `z` holds
/// the integral to the far boundary of slice `z`, so slice coordinate `s` reads texel centre
/// `s - 0.5`; inside the first slice the integral is ramped linearly from zero, as the aerial
/// perspective LUT's lookup does.
@if(VOLUMETRICS_APPLY)
fn sample_froxel_integrated(uv: vec2<f32>, t: f32, dither: f32) -> vec4<f32> {
    let n = f32(volumetrics.grid_size.z);
    let s = max(froxel_t_to_u(min(t, volumetrics.max_distance)) * n + dither, 0.0);
    let fade = saturate(s);
    let w = (max(s, 1.0) - 0.5) / n;
    let v = textureSampleLevel(volumetrics_integrated, atmosphere_lut_sampler, vec3(uv, w), 0.0);
    return vec4(v.rgb * fade, mix(1.0, v.a, fade));
}

/// Radiance of the planet's surface at `p` (no mesh there): the sun through the transmittance
/// LUT and the cloud shadow map plus the cloud-occluded skylight, Lambert, as
/// `raymarch_atmosphere` adds it at the end of a ground-hitting ray (functions.wesl), but without
/// the ray's own air transmittance: the near froxels apply theirs, and the far air segment its
/// own. The far haze adds this instead of marching into the surface because the sphere
/// intersection from a point metres above a 6.4e6 m planet is below f32 precision (0.4 m of
/// radius), so shallow rays missed the ground and lost its radiance.
@if(CLOUDS_ENABLED)
fn froxel_planet_surface_radiance(p: vec3<f32>, pixel_coords: vec2<f32>) -> vec3<f32> {
    let n = normalize(p);
    var sun = vec3(0.0);
    for (var light_i = 0u; light_i < lights.n_directional_lights; light_i++) {
        let light = &lights.directional_lights[light_i];
        let light_dir = (*light).direction_to_light;
        let mu_light = dot(light_dir, n);
        sun += (*light).color.rgb * max(mu_light, 0.0)
            * sample_transmittance_lut(atmosphere.inner_radius, mu_light)
            * sample_cloud_shadow_ground(p, light_dir);
    }
    let e_sky = ground_sky_irradiance_clouds(p, pixel_coords);
    return atmosphere.ground_albedo / PI * (sun + e_sky);
}
