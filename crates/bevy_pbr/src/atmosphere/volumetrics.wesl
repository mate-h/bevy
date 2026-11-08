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
//!   supplies the visibility) plus the air's multiple-scattering term and the clustered point and
//!   spot lights with `VolumetricLight`, into the Scattering volume (L.rgb pre-exposed, sigma_t).
//!   No temporal blending (RDR2 slide 48).
//! - `integrate`: per tile, front to back over the slices with the energy-conserving slab
//!   integral (FB15 slide 29, FB16 Eq.17 / Listing 2), into the Integrated volume (in-scatter to
//!   the camera, transmittance).
//!
//! Extinction is one scalar per froxel (FB15 slide 16, "to have cheaper volumes"; RDR2 slide 28,
//! wavelength-independent transmittance): the mean of the air's RGB extinction plus the fog's.
//! Over 128 m of sea-level air the Rayleigh optical depth is ~7e-4, so its colour is below f16.
//!
//! Later phases: SH ambient and the transparent-mesh binding (Phase 3), clouds in the grid
//! (Phase 4), Z-binned fog volumes and a Hi-Z early-out (Phase 5).

import bevy_render::maths::orthonormalize;
import package::atmosphere::{
    bindings::{atmosphere, settings, view, lights, atmosphere_lut_sampler},
    bruneton_functions::ray_intersects_ground,
    shadows::{
        fetch_directional_shadow, directional_shadow_ray_t_max, atmosphere_to_world_position,
        sample_shadow_map_hardware, directional_shadow_textures_comparison_sampler,
    },
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
import package::render::mesh_view_types::{
    ClusteredLights, ClusterableObjectIndexLists, ClusterOffsetsAndCounts,
    POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT, POINT_LIGHT_FLAGS_VOLUMETRIC_BIT,
    POINT_LIGHT_FLAGS_SPOT_LIGHT_Y_NEGATIVE,
};

@if(CLOUDS_ENABLED)
import package::atmosphere::{
    functions::sample_cloud_shadow_map,
    clouds::cloud_layer,
};
@if(VOLUMETRICS_SCATTERING)
import bevy_render::maths::PI;
@if(VOLUMETRICS_SCATTERING)
import package::light_probe::environment_sh::{EnvironmentMapSh, sh9_basis, ENVIRONMENT_SH_MAX_EXCESS};
@if(VOLUMETRICS_SCATTERING)
import package::atmosphere::{
    functions::{cloud_column_map, sample_density_lut_mie_g_sigma},
    clouds::{cloud_column_map_origin, CLOUD_COLUMN_MAP_EXTENT_M},
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
// The view's clustered lights (the mesh pipeline's group-0 bindings 5-7, `cluster/mod.rs`) and
// the point-light shadow cubemaps; spot shadow maps live in the directional array (binding 16).
@if(VOLUMETRICS_SCATTERING)
@group(0) @binding(84) var<storage> clustered_lights: ClusteredLights;
@if(VOLUMETRICS_SCATTERING)
@group(0) @binding(85) var<storage> clusterable_object_index_lists: ClusterableObjectIndexLists;
@if(VOLUMETRICS_SCATTERING)
@group(0) @binding(86) var<storage> cluster_offsets_and_counts: ClusterOffsetsAndCounts;
@if(VOLUMETRICS_SCATTERING)
@group(0) @binding(87) var point_shadow_textures: texture_depth_cube_array;
// The view's sky harmonics (`light_probe/generate.rs` `EnvironmentMapShBuffers`: the atmosphere
// probe's cubemap projected onto L2 SH plus the over-bright texels as excess lights), or the
// zero-filled `SkyShFallbackBuffer` (`volumetrics.sky_sh_exposure` is then 0).
@if(VOLUMETRICS_SCATTERING)
@group(0) @binding(88) var<storage, read> sky_sh: EnvironmentMapSh;

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
/// Per clustered point / spot light with `VolumetricLight` (`POINT_LIGHT_FLAGS_VOLUMETRIC_BIT`):
///   L += color x falloff(d) x cone x shadow_map x (sum_i sigma_s_i p_i(theta) + sigma_s_fog p_HG(theta, g))
/// (plan: "local lights from the clustered light lists with their shadow maps"; RDR2 slide 38: a
/// single HG lobe). The light loop is skipped where sigma_s is zero (FB15 slide 20).
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

    if (any(material.rgb > vec3(0.0))) {
        l += local_lights_scattering(s, froxel_tile_uv(id.xy), r, fog_scattering, g);
        l += sky_ambient_scattering(s, r, up, air_scattering, fog_scattering, g);
    }
    // Pre-exposed like the sky-view and aerial LUTs (f16 at twilight).
    textureStore(scattering_out, id, vec4(l * view.exposure, sigma_t));
}

/// Sky ambient in-scatter toward the camera at a froxel: sigma_s x ∫ L_sky(ω) p(ω·d) dω over the
/// sphere, with L_sky the view's sky probe in spherical harmonics and d the ray direction (from
/// the camera through the froxel; light arriving from ω scatters toward the camera through the
/// angle cos θ = ω·d, the convention of the sun term). The integral is a Funk-Hecke convolution
/// (W14 p.52-55 rotate the zonal phase to the view vector and take the SH product; FB15 slide 19
/// uses the Lambert lobe in its place; arXiv 2405.09447 for the HG moments):
///
///   ∫ L(ω) p(ω·d) dω = Σ_l λ_l Σ_m L_lm Y_lm(d),   λ_l = 2π ∫_{-1}^{1} p(t) P_l(t) dt.
///
/// The buffer stores c_lm = (A_l / π) L_lm, the irradiance over π (`environment_sh.wesl`), with
/// the clamped-cosine moments A_l = (π, 2π/3, π/4), so the radiance harmonics are L_lm = c_lm x
/// (1, 3/2, 4). For Henyey-Greenstein, p(t) = (1 / 4π) Σ_l (2l + 1) g^l P_l(t) and
/// ∫ P_l P_l dt = 2 / (2l + 1), so λ_l = g^l: for g = 0 the result is L_00 Y_00 = L_00 x 0.282095,
/// the mean radiance over the sphere, and as g -> 1 (a delta toward d) it is L(d), the sky
/// radiance from ahead on the ray, which is what forward scattering sends to the camera.
///
/// Air: λ_0 = σ_s (every phase function integrates to 1); λ_1 = Σ g σ_s,Mie (the medium LUT's
/// stored first moment; Rayleigh's is 0); band 2 (0.1 σ_s,Rayleigh + Σ g² σ_s,Mie: Rayleigh's
/// second moment is 2π ∫ 3/(16π) (1 + t²) P_2 dt = 1/10) needs the Rayleigh/Mie split of σ_s
/// the LUT does not hold and is dropped (L1 for the air, Lumen's choice for fog; gap). Fog: HG
/// with Material B's g, all three bands. Each reconstruction is clamped at zero (L2 rings).
/// Excess lights (the sun's over-bright texels, radiance x solid angle) are directional lights
/// through the actual phase functions, as the sun term. The result is scaled by the probe's
/// exposure, `ambient_scale`, and the local cloud occlusion of the sky
/// (`sky_ambient_cloud_occlusion`).
@if(VOLUMETRICS_SCATTERING)
fn sky_ambient_scattering(s: FroxelSample, r: f32, up: vec3<f32>, air_scattering: vec3<f32>, fog_scattering: vec3<f32>, g: f32) -> vec3<f32> {
    if (volumetrics.sky_sh_exposure <= 0.0 || volumetrics.ambient_scale <= 0.0) {
        return vec3(0.0);
    }
    let inv_exposure = 1.0 / volumetrics.sky_sh_exposure;
    let y = sh9_basis(s.dir);
    let b0 = sky_sh.coefficients[0].rgb * y[0];
    var b1 = vec3(0.0);
    for (var i = 1u; i < 4u; i++) {
        b1 += sky_sh.coefficients[i].rgb * y[i];
    }
    b1 *= 1.5;
    var b2 = vec3(0.0);
    for (var i = 4u; i < 9u; i++) {
        b2 += sky_sh.coefficients[i].rgb * y[i];
    }
    b2 *= 4.0;
    let g_sigma_mie = sample_density_lut_mie_g_sigma(r);
    var l = max(air_scattering * b0 + g_sigma_mie * b1, vec3(0.0));
    l += fog_scattering * max(b0 + g * (b1 + g * b2), vec3(0.0));
    for (var i = 0u; i < min(sky_sh.excess_count, ENVIRONMENT_SH_MAX_EXCESS); i++) {
        let cos_theta = dot(sky_sh.excess_directions[i].xyz, s.dir);
        l += sky_sh.excess_radiances[i].rgb
            * (sample_scattering_lut(r, cos_theta) + fog_scattering * hg_phase(cos_theta, g));
    }
    l *= inv_exposure * volumetrics.ambient_scale;
    @if(CLOUD_COLUMN_MAP) {
        l *= sky_ambient_cloud_occlusion(s.pos, up, inv_exposure);
    }
    return l;
}

/// The local cloud occlusion of the sky ambient at a froxel: the ground-skylight column map's
/// cloud-occluded sky irradiance at the froxel's ground column (`cloud_shadow_map.wesl`
/// ground_sky, 64 cosine-weighted directions through the cloud field, pre-exposed) over the
/// probe's sky irradiance at the up normal (π x the harmonics' irradiance / π at `up`, plus the
/// excess lights), both for the upward-facing ground. The probe already sees the clouds from the
/// camera, so this is the ratio of the column's skylight to the camera's (RDR2 slide 36: the
/// froxels sample a 32x32 map of sky probes rather than one), 1 where the ratio is undefined
/// (no skylight, or outside the column map). One scalar (the RGB mean), as the shadow volume is.
@if(VOLUMETRICS_SCATTERING && CLOUD_COLUMN_MAP)
fn sky_ambient_cloud_occlusion(pos: vec3<f32>, up: vec3<f32>, inv_exposure: f32) -> f32 {
    let camera_xz = get_view_position().xz;
    let uv = (pos.xz - cloud_column_map_origin(camera_xz)) / (2.0 * CLOUD_COLUMN_MAP_EXTENT_M);
    if (any(uv < vec2(0.0)) || any(uv > vec2(1.0))) {
        return 1.0;
    }
    let e_column = textureSampleLevel(cloud_column_map, atmosphere_lut_sampler, uv, 0.0).gba / view.exposure;
    let y = sh9_basis(up);
    var e_probe = vec3(0.0);
    for (var i = 0u; i < 9u; i++) {
        e_probe += sky_sh.coefficients[i].rgb * y[i];
    }
    e_probe = max(e_probe, vec3(0.0));
    for (var i = 0u; i < min(sky_sh.excess_count, ENVIRONMENT_SH_MAX_EXCESS); i++) {
        e_probe += sky_sh.excess_radiances[i].rgb * max(dot(up, sky_sh.excess_directions[i].xyz), 0.0) / PI;
    }
    let e_probe_mean = mean3(e_probe) * PI * inv_exposure;
    if (e_probe_mean <= 0.0) {
        return 1.0;
    }
    return mean3(e_column) / e_probe_mean;
}

/// The scattering from the point and spot lights of the froxel's cluster (the mesh pipeline's
/// cluster lists, `clustered_forward.wesl`): sum over lights with `VolumetricLight` of
/// color x falloff x cone x shadow x phase-weighted sigma_s. Light positions are world space.
@if(VOLUMETRICS_SCATTERING)
fn local_lights_scattering(s: FroxelSample, uv: vec2<f32>, r: f32, fog_scattering: vec3<f32>, g: f32) -> vec3<f32> {
    let world = atmosphere_to_world_position(s.pos);
    let view_z = (view.view_from_world * vec4(world, 1.0)).z;
    let ranges = cluster_offsets_and_counts.data[froxel_cluster_index(uv, view_z)][0];
    let point_offset = ranges.x;
    let spot_offset = point_offset + ranges.y;
    let rect_offset = spot_offset + ranges.z;

    var l = vec3(0.0);
    for (var i = point_offset; i < rect_offset; i++) {
        let light_id = clusterable_object_index_lists.data[i];
        let light = &clustered_lights.data[light_id];
        if (((*light).flags & POINT_LIGHT_FLAGS_VOLUMETRIC_BIT) == 0u) {
            continue;
        }
        let to_light = (*light).position_radius.xyz - world;
        let distance_square = dot(to_light, to_light);
        // Filament's smooth range falloff (`getDistanceAttenuation`, pbr_lighting.wesl).
        let factor = distance_square * (*light).color_inverse_square_range.w;
        let smooth_factor = saturate(1.0 - factor * factor);
        var attenuation = smooth_factor * smooth_factor / max(distance_square, 0.0001);
        if (attenuation <= 0.0) {
            continue;
        }
        if (i < spot_offset) {
            if (((*light).flags & POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u) {
                attenuation *= fetch_point_shadow_hardware(light_id, world);
            }
        } else {
            let spot_dir = spot_light_direction(light_id);
            // Filament's cone (`spot_scale` / `spot_offset` precomputed in `light_custom_data.zw`).
            let cd = dot(-spot_dir, normalize(to_light));
            let cone = saturate(cd * (*light).light_custom_data.z + (*light).light_custom_data.w);
            attenuation *= cone * cone;
            if (((*light).flags & POINT_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u) {
                attenuation *= fetch_spot_shadow_hardware(light_id, world, spot_dir);
            }
        }
        let cos_theta = dot(normalize(to_light), s.dir);
        let phase_scattering = sample_scattering_lut(r, cos_theta) + fog_scattering * hg_phase(cos_theta, g);
        l += (*light).color_inverse_square_range.rgb * attenuation * phase_scattering;
    }
    return l;
}

/// The cluster of a froxel sample (`clustered_forward.wesl` `view_fragment_cluster_index` /
/// `view_z_to_z_slice`, with the tile's view uv in place of the fragment's viewport position).
@if(VOLUMETRICS_SCATTERING)
fn froxel_cluster_index(uv: vec2<f32>, view_z: f32) -> u32 {
    let dims = lights.cluster_dimensions;
    let xy = min(vec2<u32>(floor(uv * vec2<f32>(dims.xy))), dims.xy - 1u);
    let is_orthographic = view.clip_from_view[3].w == 1.0;
    var z_slice = 0u;
    if (is_orthographic) {
        z_slice = u32(floor((view_z - lights.cluster_factors.z) * lights.cluster_factors.w));
    } else {
        z_slice = u32(log(-view_z) * lights.cluster_factors.z - lights.cluster_factors.w + 1.0);
    }
    z_slice = min(z_slice, dims.z - 1u);
    return min((xy.y * dims.x + xy.x) * dims.z + z_slice, dims.w - 1u);
}

/// Hardware 2x2 comparison tap of a point light's shadow cubemap at a world position
/// (`render/shadows.wesl` `fetch_point_shadow` without the surface normal: the medium has none).
@if(VOLUMETRICS_SCATTERING)
fn fetch_point_shadow_hardware(light_id: u32, world: vec3<f32>) -> f32 {
    let light = &clustered_lights.data[light_id];
    let surface_to_light = (*light).position_radius.xyz - world;
    let depth_offset = (*light).shadow_depth_bias * normalize(surface_to_light);
    // The shadow maps align with the axes and the frustum planes are at 45 degrees, so the
    // light-space depth is the largest absolute axis.
    let frag_ls = (world + depth_offset) - (*light).position_radius.xyz;
    let abs_position_ls = abs(frag_ls);
    let major_axis_magnitude = max(abs_position_ls.x, max(abs_position_ls.y, abs_position_ls.z));
    // projection * vec4(0, 0, -major_axis_magnitude, 1), depth terms only.
    let zw = -major_axis_magnitude * (*light).light_custom_data.xy + (*light).light_custom_data.zw;
    let depth = zw.x / zw.y;
    // Cubemaps assume a left-handed coordinate space: flip z.
    return textureSampleCompareLevel(
        point_shadow_textures,
        directional_shadow_textures_comparison_sampler,
        frag_ls * vec3(1.0, 1.0, -1.0),
        i32(light_id),
        depth,
    );
}

/// A spot light's direction from its packed (x, z) and the y-sign flag.
@if(VOLUMETRICS_SCATTERING)
fn spot_light_direction(light_id: u32) -> vec3<f32> {
    let light = &clustered_lights.data[light_id];
    var spot_dir = vec3((*light).light_custom_data.x, 0.0, (*light).light_custom_data.y);
    spot_dir.y = sqrt(max(0.0, 1.0 - spot_dir.x * spot_dir.x - spot_dir.z * spot_dir.z));
    if (((*light).flags & POINT_LIGHT_FLAGS_SPOT_LIGHT_Y_NEGATIVE) != 0u) {
        spot_dir.y = -spot_dir.y;
    }
    return spot_dir;
}

/// Hardware 2x2 comparison tap of a spot light's shadow map (in the directional array at
/// `spot_light_shadowmap_offset + light_id`) at a world position (`render/shadows.wesl`
/// `fetch_spot_shadow` without the surface normal).
@if(VOLUMETRICS_SCATTERING)
fn fetch_spot_shadow_hardware(light_id: u32, world: vec3<f32>, spot_dir: vec3<f32>) -> f32 {
    let light = &clustered_lights.data[light_id];
    let surface_to_light = (*light).position_radius.xyz - world;
    // The light's view matrix: z is the reverse of its forward (-spot_dir).
    let fwd = -spot_dir;
    let offset_position = -surface_to_light + (*light).shadow_depth_bias * normalize(surface_to_light);
    let light_inv_rot = orthonormalize(fwd);
    // A pure rotation: its inverse is the transpose, applied by post-multiplying.
    let projected_position = offset_position * light_inv_rot;
    let f_div_minus_z = 1.0 / ((*light).spot_light_tan_angle * -projected_position.z);
    let shadow_xy_ndc = projected_position.xy * f_div_minus_z;
    let shadow_uv = shadow_xy_ndc * vec2(0.5, -0.5) + vec2(0.5, 0.5);
    let depth = (*light).shadow_map_near_z / -projected_position.z;
    return sample_shadow_map_hardware(shadow_uv, depth, i32(light_id) + lights.spot_light_shadowmap_offset);
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
