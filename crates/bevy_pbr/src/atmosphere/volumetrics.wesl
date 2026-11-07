//! Near-field froxel passes (`volumetrics.rs`; plan: passes 1, 3, 4 of the unified volumetrics):
//! - `material`: per froxel, the medium at one jittered point on its ray: air (the atmosphere's
//!   medium LUTs) plus the height fog, into Material A (sigma_s.rgb, sigma_t).
//! - `scattering`: per froxel, at the same point, the sun-lit single scattering plus the air's
//!   multiple-scattering term, into the Scattering volume (L.rgb pre-exposed, sigma_t). No
//!   temporal blending (RDR2 slide 48).
//! - `integrate`: per tile, front to back over the slices with the energy-conserving slab
//!   integral (FB15 slide 29, FB16 Eq.17 / Listing 2), into the Integrated volume (in-scatter to
//!   the camera, transmittance).
//!
//! Extinction is one scalar per froxel (FB15 slide 16, "to have cheaper volumes"; RDR2 slide 28,
//! wavelength-independent transmittance): the mean of the air's RGB extinction plus the fog's.
//! Over 128 m of sea-level air the Rayleigh optical depth is ~7e-4, so its colour is below f16.
//!
//! Phase 2 hooks (not implemented): a Material B volume (emissive, phase g, ambient scale), a
//! temporally filtered Shadow volume (cascade x cloud shadow), `FogVolume` injection, local lights.

import package::atmosphere::{
    bindings::{atmosphere, settings, view, lights},
    bruneton_functions::ray_intersects_ground,
    shadows::{fetch_directional_shadow, directional_shadow_ray_t_max},
    functions::{
        sample_density_lut, sample_scattering_lut, sample_transmittance_lut,
        sample_multiscattering_lut, hg_phase, uv_to_ray_direction, get_view_position,
        clamp_to_surface, ABSORPTION_DENSITY, SCATTERING_DENSITY,
    },
    froxels::{
        volumetrics, froxel_tile_uv, froxel_noise, froxel_sample_t, froxel_slice_bounds,
        froxel_fog_extinction, one_minus_exp_over_x,
    },
};

@if(CLOUDS_ENABLED)
import package::atmosphere::{
    functions::sample_cloud_shadow_map,
    clouds::cloud_layer,
};

@if(VOLUMETRICS_MATERIAL)
@group(0) @binding(71) var material_a_out: texture_storage_3d<rgba16float, write>;

@if(VOLUMETRICS_SCATTERING)
@group(0) @binding(72) var material_a: texture_3d<f32>;
@if(VOLUMETRICS_SCATTERING)
@group(0) @binding(73) var scattering_out: texture_storage_3d<rgba16float, write>;

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
/// offset per ray and frame shared by every slice (and by the material and scattering passes).
/// The grid knows no scene depth (a Hi-Z early-out is Phase 5), so rays below the horizon keep
/// going under the planet's surface; those froxels are clamped to the surface (the transmittance
/// LUT's parametrization is NaN below it, and the NaN spread to every later slice of the tile).
fn froxel_sample(id: vec3<u32>) -> FroxelSample {
    let dir = uv_to_ray_direction(froxel_tile_uv(id.xy));
    let j = froxel_noise(vec2<i32>(id.xy));
    let t = froxel_sample_t(id.z, j);
    let pos = clamp_to_surface(atmosphere, get_view_position() + dir * t);
    return FroxelSample(pos, dir, t);
}

/// Pass 1: material voxelization.
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
    let fog_extinction = froxel_fog_extinction(r);
    let sigma_s = air_scattering + volumetrics.fog_albedo * fog_extinction;
    let sigma_t = (air_extinction.r + air_extinction.g + air_extinction.b) / 3.0 + fog_extinction;
    textureStore(material_a_out, id, vec4(sigma_s, sigma_t));
}

/// Pass 3: sun-lit scattering at the same point as the material pass. Per directional light:
///   L += color x [ T_sun(LUT) x !ground x cloud_shadow x cascade_shadow
///                  x (sum_i sigma_s_i p_i(theta) [medium LUT, Rayleigh + Mie] + sigma_s_fog p_HG(theta, g))
///                + psi_ms(LUT) x sigma_s_air x cloud_shadow ]
/// The multiple-scattering term is darkened by the cloud shadow (the audit flagged the per-pixel
/// haze leaving it unshadowed) but not by the mesh cascades: it stands for light scattered twice
/// or more from the whole sky, which a nearby occluder of the sun does not remove (inference).
/// The medium is re-evaluated here rather than read back from Material A, because the sun term
/// needs the Rayleigh / Mie split per phase function that a summed sigma_s cannot give; Material
/// A supplies the extinction so both volumes agree.
@if(VOLUMETRICS_SCATTERING)
@compute
@workgroup_size(8, 8, 1)
fn scattering(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= volumetrics.grid_size)) { return; }
    let s = froxel_sample(id);
    let r = length(s.pos);
    let up = normalize(s.pos);
    let sigma_t = textureLoad(material_a, id, 0).a;
    let air_scattering = sample_density_lut(r, SCATTERING_DENSITY);
    let fog_scattering = volumetrics.fog_albedo * froxel_fog_extinction(r);

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

        var sun_visibility = sample_transmittance_lut(r, mu_light) * f32(!ray_intersects_ground(r, mu_light));
        var cloud_shadow = 1.0;
        @if(CLOUDS_ENABLED) {
        let above_clouds = mu_light >= 0.0
            && r > max(cloud_layer.cloud_layer_end, settings.cloud_thin_shadow_max_radius);
        if (!above_clouds) {
            cloud_shadow = sample_cloud_shadow_map(s.pos, to_light, false, true);
        }
        }
        var cascade_shadow = 1.0;
        @if(!ATMOSPHERE_NO_SHADOW_MAPS) {
        if (settings.shadows_enabled != 0u && s.t < mesh_shadow_t_max) {
            cascade_shadow = fetch_directional_shadow(light_i, vec4(s.pos, 1.0), vec3(0.0));
        }
        }
        let phase_scattering = sample_scattering_lut(r, cos_theta)
            + fog_scattering * hg_phase(cos_theta, volumetrics.fog_anisotropy);
        let single = sun_visibility * cloud_shadow * cascade_shadow * phase_scattering;
        let multi = sample_multiscattering_lut(r, mu_light) * air_scattering * cloud_shadow;
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
