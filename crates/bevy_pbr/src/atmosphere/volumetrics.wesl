//! Near-field froxel passes (`volumetrics.rs`; plan: passes 1-4 of the unified volumetrics):
//! - `material`: per froxel, the medium at one jittered point on its ray: air (the atmosphere's
//!   medium LUTs), the height fog and the `FogVolume`s containing the point, into Material A
//!   (sigma_s.rgb, sigma_t) and Material B (phase g in .r, the rest reserved: FB15 slide 16
//!   emissive; RDR2 slide 44 emissive, ambient and water-droplet density).
//! - `shadow`: per froxel, the sun's visibility (cascade shadow x cloud shadow map), blended with
//!   the previous frame's volume reprojected through the previous view (RDR2 slide 43: the R16F
//!   shadow volume is "temporally filtered to resolve undersampling on high frequency shadow
//!   detail").
//! - `scattering`: per froxel, at the same point, the sun-lit single scattering (the shadow volume
//!   supplies the visibility) plus the air's multiple-scattering term, into the Scattering volume
//!   (L.rgb pre-exposed, sigma_t). No temporal blending (RDR2 slide 48).
//! - `integrate`: per tile, front to back over the slices with the energy-conserving slab
//!   integral (FB15 slide 29, FB16 Eq.17 / Listing 2), into the Integrated volume (in-scatter to
//!   the camera, transmittance).
//!
//! Extinction is one scalar per froxel (FB15 slide 16, "to have cheaper volumes"; RDR2 slide 28,
//! wavelength-independent transmittance): the mean of the air's RGB extinction plus the fog's.
//! Over 128 m of sea-level air the Rayleigh optical depth is ~7e-4, so its colour is below f16.
//!
//! Phase 2 hooks (not implemented): local lights.

import package::atmosphere::{
    bindings::{atmosphere, settings, view, lights, atmosphere_lut_sampler},
    bruneton_functions::ray_intersects_ground,
    shadows::{fetch_directional_shadow, directional_shadow_ray_t_max, atmosphere_to_world_position},
    functions::{
        sample_density_lut, sample_scattering_lut, sample_transmittance_lut,
        sample_multiscattering_lut, hg_phase, uv_to_ray_direction, get_view_position,
        clamp_to_surface, ndc_to_uv, ABSORPTION_DENSITY, SCATTERING_DENSITY,
    },
    froxels::{
        volumetrics, froxel_tile_uv, froxel_noise, froxel_sample_t, froxel_slice_bounds,
        froxel_fog_extinction, froxel_t_to_u, one_minus_exp_over_x, MAX_FOG_VOLUMES,
    },
};

@if(CLOUDS_ENABLED)
import package::atmosphere::{
    functions::sample_cloud_shadow_map,
    clouds::cloud_layer,
};

/// A `FogVolume` (bevy_light) for the grid (`volumetrics.rs` `GpuFogVolume`): the unit cube
/// [-0.5, 0.5]^3 in its local space, with its medium already multiplied out.
struct FogVolume {
    local_from_world: mat4x4<f32>,
    /// fog_color x light_tint x light_intensity x scattering x density_factor (1/m).
    sigma_s: vec3<f32>,
    /// density_factor x (absorption + scattering) (1/m).
    sigma_t: f32,
    /// x: scattering_asymmetry (HG g); yzw reserved.
    params: vec4<f32>,
}
struct FogVolumes {
    count: u32,
    data: array<FogVolume, MAX_FOG_VOLUMES>,
}

@if(VOLUMETRICS_MATERIAL)
@group(0) @binding(71) var material_a_out: texture_storage_3d<rgba16float, write>;
@if(VOLUMETRICS_MATERIAL)
@group(0) @binding(81) var<uniform> fog_volumes: FogVolumes;
@if(VOLUMETRICS_MATERIAL)
@group(0) @binding(82) var material_b_out: texture_storage_3d<rgba16float, write>;

@if(VOLUMETRICS_SHADOW)
@group(0) @binding(77) var shadow_history: texture_3d<f32>;
@if(VOLUMETRICS_SHADOW)
@group(0) @binding(78) var shadow_out: texture_storage_3d<rgba16float, write>;

@if(VOLUMETRICS_SCATTERING)
@group(0) @binding(72) var material_a: texture_3d<f32>;
@if(VOLUMETRICS_SCATTERING)
@group(0) @binding(73) var scattering_out: texture_storage_3d<rgba16float, write>;
@if(VOLUMETRICS_SCATTERING)
@group(0) @binding(79) var shadow_volume: texture_3d<f32>;
@if(VOLUMETRICS_SCATTERING)
@group(0) @binding(83) var material_b: texture_3d<f32>;

@if(VOLUMETRICS_INTEGRATE)
@group(0) @binding(74) var scattering_in: texture_3d<f32>;
@if(VOLUMETRICS_INTEGRATE)
@group(0) @binding(75) var integrated_out: texture_storage_3d<rgba16float, write>;

struct FroxelSample {
    /// Atmosphere-space position and the ray's direction.
    pos: vec3<f32>,
    dir: vec3<f32>,
    /// Distance along the ray (m).
    t: f32,
}

/// The jittered sample point of froxel `id`: the ray through the tile's centre, one blue-noise
/// offset per ray and frame shared by every slice (and by the material, shadow and scattering
/// passes: FB15 slide 23, "jittered in sync"). The grid knows no scene depth (a Hi-Z early-out is
/// Phase 5), so rays below the horizon keep going under the planet's surface; those froxels are
/// clamped to the surface (the transmittance LUT's parametrization is NaN below it, and the NaN
/// spread to every later slice of the tile).
fn froxel_sample(id: vec3<u32>) -> FroxelSample {
    let dir = uv_to_ray_direction(froxel_tile_uv(id.xy));
    let j = froxel_noise(vec2<i32>(id.xy));
    let t = froxel_sample_t(id.z, j);
    let pos = clamp_to_surface(atmosphere, get_view_position() + dir * t);
    return FroxelSample(pos, dir, t);
}

/// Pass 1: material voxelization. Sources add linearly except g, which is averaged weighted by
/// each source's (mean RGB) sigma_s (FB15 slide 16: properties are added, g is averaged); the
/// air's phase functions come from the medium LUT, so only the fog sources enter g. Each
/// `FogVolume` is tested by its local unit cube at the froxel's sample point; every volume is
/// visited per froxel (RDR2 slide 45 Z-bins them: Phase 5). A volume's density texture is not
/// sampled yet (uniform `density_factor`, see `GpuFogVolume`).
@if(VOLUMETRICS_MATERIAL)
@compute
@workgroup_size(8, 8, 1)
fn material(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= volumetrics.grid_size)) { return; }
    let s = froxel_sample(id);
    let r = length(s.pos);
    let air_scattering = sample_density_lut(r, SCATTERING_DENSITY);
    let air_absorption = sample_density_lut(r, ABSORPTION_DENSITY);
    let air_extinction = air_scattering + air_absorption;

    let height_fog_extinction = froxel_fog_extinction(r);
    var fog_sigma_s = volumetrics.fog_albedo * height_fog_extinction;
    var fog_sigma_t = height_fog_extinction;
    var g_weight = mean3(fog_sigma_s);
    var g_sum = volumetrics.fog_anisotropy * g_weight;

    let world = vec4(atmosphere_to_world_position(s.pos), 1.0);
    for (var i = 0u; i < min(fog_volumes.count, MAX_FOG_VOLUMES); i++) {
        let volume = &fog_volumes.data[i];
        let local = ((*volume).local_from_world * world).xyz;
        if (all(abs(local) <= vec3(0.5))) {
            fog_sigma_s += (*volume).sigma_s;
            fog_sigma_t += (*volume).sigma_t;
            let w = mean3((*volume).sigma_s);
            g_sum += (*volume).params.x * w;
            g_weight += w;
        }
    }
    let g = select(volumetrics.fog_anisotropy, g_sum / g_weight, g_weight > 0.0);

    let sigma_s = air_scattering + fog_sigma_s;
    let sigma_t = mean3(air_extinction) + fog_sigma_t;
    textureStore(material_a_out, id, vec4(sigma_s, sigma_t));
    textureStore(material_b_out, id, vec4(g, 0.0, 0.0, 0.0));
}

fn mean3(v: vec3<f32>) -> f32 {
    return (v.x + v.y + v.z) / 3.0;
}

/// A directional light's visibility at a froxel sample: the cloud shadow map (unless the sample
/// is above the clouds with the sun up) times the shadow cascades (within their range). Shared
/// by the shadow pass (light 0) and, for every other directional light, the scattering pass.
fn directional_visibility(light_i: u32, s: FroxelSample, r: f32, mesh_shadow_t_max: f32) -> f32 {
    let to_light = lights.directional_lights[light_i].direction_to_light;
    var visibility = 1.0;
    @if(CLOUDS_ENABLED) {
    let mu_light = dot(to_light, normalize(s.pos));
    let above_clouds = mu_light >= 0.0
        && r > max(cloud_layer.cloud_layer_end, settings.cloud_thin_shadow_max_radius);
    if (!above_clouds) {
        visibility *= sample_cloud_shadow_map(s.pos, to_light, false, true);
    }
    }
    @if(!ATMOSPHERE_NO_SHADOW_MAPS) {
    if (settings.shadows_enabled != 0u && s.t < mesh_shadow_t_max) {
        visibility *= fetch_directional_shadow(light_i, vec4(s.pos, 1.0), vec3(0.0));
    }
    }
    return visibility;
}

/// Pass 2: the shadow volume (RDR2 slide 43: cascade x cloud shadow map, R16F, temporally
/// filtered). The history is reprojected by world position through the previous frame's
/// view-projection and read at the previous ray's distance (one trilinear tap); samples that
/// left the previous grid (out of its frustum or beyond d_max), a resized grid and the first
/// frame take the current value alone, so a camera cut converges in `1 / shadow_temporal_alpha`
/// frames rather than smearing the old shadows. The volume holds the first directional light's
/// visibility only (one scalar, as RDR2's; the lights are sorted so a shadow-casting volumetric
/// light comes first); the scattering pass evaluates further suns inline.
@if(VOLUMETRICS_SHADOW)
@compute
@workgroup_size(8, 8, 1)
fn shadow(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= volumetrics.grid_size)) { return; }
    let s = froxel_sample(id);
    var visibility = 1.0;
    if (lights.n_directional_lights > 0u) {
        var mesh_shadow_t_max = -1.0;
        @if(!ATMOSPHERE_NO_SHADOW_MAPS) {
            mesh_shadow_t_max = directional_shadow_ray_t_max(get_view_position(), s.dir);
        }
        visibility = directional_visibility(0u, s, length(s.pos), mesh_shadow_t_max);
    }
    if (volumetrics.history_valid != 0u) {
        let world = atmosphere_to_world_position(s.pos);
        let clip = volumetrics.prev_clip_from_world * vec4(world, 1.0);
        if (clip.w > 0.0) {
            let uv = ndc_to_uv(clip.xy / clip.w);
            let u = froxel_t_to_u(distance(world, volumetrics.prev_world_position));
            if (all(uv >= vec2(0.0)) && all(uv <= vec2(1.0)) && u <= 1.0) {
                let history = textureSampleLevel(shadow_history, atmosphere_lut_sampler, vec3(uv, u), 0.0).r;
                visibility = mix(history, visibility, volumetrics.shadow_temporal_alpha);
            }
        }
    }
    textureStore(shadow_out, id, vec4(visibility, 0.0, 0.0, 0.0));
}

/// Pass 3: sun-lit scattering at the same point as the material pass. Per directional light:
///   L += color x [ T_sun(LUT) x !ground x V
///                  x (sum_i sigma_s_i p_i(theta) [medium LUT, Rayleigh + Mie] + sigma_s_fog p_HG(theta, g))
///                + psi_ms(LUT) x sigma_s_air x V ]
/// with V the shadow volume (one tap at the froxel) for the first light, inline for the others.
/// The multiple-scattering term is darkened by the sun's visibility: Phase 1 used the cloud
/// shadow alone (the term stands for light scattered twice or more from the whole sky, which a
/// nearby occluder of the sun does not remove); the volume holds the product, so the cascades
/// now darken it too, which over-darkens the MS term next to large occluders (inference; the
/// two factors are separable only by storing them in two channels).
/// The air's Rayleigh / Mie split per phase function comes from the medium LUT (a summed sigma_s
/// cannot give it); the fog's sigma_s is Material A's minus the air's and its g is Material B's,
/// so the volumes agree with what was voxelized.
@if(VOLUMETRICS_SCATTERING)
@compute
@workgroup_size(8, 8, 1)
fn scattering(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= volumetrics.grid_size)) { return; }
    let s = froxel_sample(id);
    let r = length(s.pos);
    let up = normalize(s.pos);
    let material = textureLoad(material_a, id, 0);
    let sigma_t = material.a;
    let g = textureLoad(material_b, id, 0).r;
    let air_scattering = sample_density_lut(r, SCATTERING_DENSITY);
    let fog_scattering = max(material.rgb - air_scattering, vec3(0.0));

    var mesh_shadow_t_max = -1.0;
    @if(!ATMOSPHERE_NO_SHADOW_MAPS) {
        mesh_shadow_t_max = directional_shadow_ray_t_max(get_view_position(), s.dir);
    }

    var l = vec3(0.0);
    for (var light_i = 0u; light_i < lights.n_directional_lights; light_i++) {
        let light = &lights.directional_lights[light_i];
        let to_light = (*light).direction_to_light;
        let mu_light = dot(to_light, up);
        // Same sign convention as `sample_local_inscattering`: cos(theta) = dot(L, ray_dir).
        let cos_theta = dot(to_light, s.dir);

        let sun_visibility = sample_transmittance_lut(r, mu_light) * f32(!ray_intersects_ground(r, mu_light));
        var visibility = 1.0;
        if (light_i == 0u) {
            visibility = textureLoad(shadow_volume, id, 0).r;
        } else {
            visibility = directional_visibility(light_i, s, r, mesh_shadow_t_max);
        }
        let phase_scattering = sample_scattering_lut(r, cos_theta) + fog_scattering * hg_phase(cos_theta, g);
        let single = sun_visibility * visibility * phase_scattering;
        let multi = sample_multiscattering_lut(r, mu_light) * air_scattering * visibility;
        l += (*light).color.rgb * (single + multi);
    }
    // Pre-exposed like the sky-view and aerial LUTs (f16 at twilight).
    textureStore(scattering_out, id, vec4(l * view.exposure, sigma_t));
}

/// Pass 4: slice scan, one thread per tile, front to back. Per slice of length ds with light L
/// and extinction sigma_t (FB16 Listing 2): T = exp(-sigma_t ds); S_int = (L - L T) / sigma_t;
/// acc.rgb += acc.a S_int; acc.a *= T. S_int is evaluated as L ds (1 - e^-x) / x, x = sigma_t ds,
/// with the series for small x (`one_minus_exp_over_x`), which is the same expression with its
/// 1e-7 floor replaced by an exact limit.
@if(VOLUMETRICS_INTEGRATE)
@compute
@workgroup_size(8, 8, 1)
fn integrate(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id.xy >= volumetrics.grid_size.xy)) { return; }
    var acc = vec4(0.0, 0.0, 0.0, 1.0);
    for (var z = 0u; z < volumetrics.grid_size.z; z++) {
        let s = textureLoad(scattering_in, vec3(id.xy, z), 0);
        let b = froxel_slice_bounds(z);
        let ds = b.y - b.x;
        let x = s.a * ds;
        let s_int = s.rgb * ds * one_minus_exp_over_x(x);
        acc = vec4(acc.rgb + acc.a * s_int, acc.a * exp(-x));
        textureStore(integrated_out, vec3(id.xy, z), acc);
    }
}
