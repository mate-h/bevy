#define_import_path bevy_pbr::atmosphere::functions

#import bevy_render::maths::{PI, HALF_PI, PI_2, fast_acos, fast_acos_4, fast_atan2}

#import bevy_pbr::atmosphere::{
    types::Atmosphere,
    bruneton_functions::{
        transmittance_lut_r_mu_to_uv, transmittance_lut_uv_to_r_mu, 
        ray_intersects_ground, distance_to_top_atmosphere_boundary, 
        distance_to_bottom_atmosphere_boundary
    },
}

#import bevy_pbr::mesh_view_types::Lights

// NOTE FOR CONVENTIONS: 
// r:
//   radius, or distance from planet center 
//
// altitude:
//   distance from planet **surface**
//
// mu:
//   cosine of the zenith angle of a ray with
//   respect to the planet normal
//
// atmosphere space:
//   abbreviated as "as" (contrast with vs, cs, ws), this space is similar
//   to view space, but with the camera positioned horizontally on the planet
//   surface, so the horizon is a horizontal line centered vertically in the
//   frame. This enables the non-linear latitude parametrization the paper uses 
//   to concentrate detail near the horizon 


// LUT UV PARAMATERIZATIONS
fn multiscattering_lut_r_mu_to_uv(settings: CoreLutSettings, planet: Planet, r: f32, mu: f32) -> vec2<f32> {
    let u = 0.5 + 0.5 * mu;
    let v = saturate((r - planet.bottom_radius) / planet.space_altitude); //TODO
    return vec2(u, v);
}

fn multiscattering_lut_uv_to_r_mu(uv: vec2<f32>) -> vec2<f32> {
    let r = mix(atmosphere.bottom_radius, atmosphere.top_radius, uv.y);
    let mu = uv.x * 2 - 1;
    return vec2(r, mu);
}

fn sky_view_lut_r_mu_azimuth_to_uv(planet: Planet, r: f32, mu: f32, azimuth: f32) -> vec2<f32> {
    let u = (azimuth * FRAC_2_PI) + 0.5;

    // Horizon parameters
    let v_horizon = sqrt(r * r - planet.bottom_radius_sq);
    let cos_beta = v_horizon / r;
    // Using fast_acos_4 for better precision at small angles
    // to avoid artifacts at the horizon
    let beta = fast_acos_4(cos_beta);
    let horizon_zenith = PI - beta;
    let view_zenith = fast_acos_4(mu);

    // Apply non-linear transformation to compress more texels 
    // near the horizon where high-frequency details matter most
    // l is latitude in [-π/2, π/2] and v is texture coordinate in [0,1]
    let l = view_zenith - horizon_zenith;
    let abs_l = abs(l);

    let v = 0.5 + 0.5 * sign(l) * sqrt(abs_l / HALF_PI);

    return vec2(u, v);
}

fn sky_view_lut_uv_to_zenith_azimuth(planet: Planet, r: f32, uv: vec2<f32>) -> vec2<f32> {
    let adj_uv = vec2(uv.x, 1.0 - uv.y);
    let azimuth = (adj_uv.x - 0.5) * PI_2;

    // Horizon parameters
    let v_horizon = sqrt(r * r - planet.bottom_radius_sq);
    let cos_beta = v_horizon / r;
    // Using fast_acos_4 for better precision at small angles
    // to avoid artifacts at the horizon
    let beta = fast_acos_4(cos_beta);
    let horizon_zenith = PI - beta;

    // Inverse of horizon-detail mapping to recover original latitude from texture coordinate
    let t = abs(2.0 * (adj_uv.y - 0.5));
    let l = sign(adj_uv.y - 0.5) * HALF_PI * t * t;

    return vec2(horizon_zenith - l, azimuth);
}

// LUT SAMPLING

fn sample_transmittance_lut(lut: texture_2d<f32>, smp: sampler, r: f32, mu: f32) -> vec3<f32> {
    let uv = transmittance_lut_r_mu_to_uv(r, mu);
    return textureSampleLevel(lut, smp, uv, 0.0).rgb;
}

fn sample_multiscattering_lut(lut: texture_2d<f32>, smp: sampler, r: f32, mu: f32) -> vec3<f32> {
    let uv = multiscattering_lut_r_mu_to_uv(r, mu);
    return textureSampleLevel(lut, smp, uv, 0.0).rgb;
}

fn sample_sky_view_lut(lut: texture_2d<f32>, smp: sampler, r: f32, ray_dir_as: vec3<f32>) -> vec3<f32> {
    let mu = ray_dir_as.y;
    let azimuth = fast_atan2(ray_dir_as.x, -ray_dir_as.z);
    let uv = sky_view_lut_r_mu_azimuth_to_uv(r, mu, azimuth);
    return textureSampleLevel(lut, smp, uv, 0.0).rgb;
}

fn ndc_to_camera_dist(ndc: vec3<f32>) -> f32 {
    let view_pos = view.view_from_clip * vec4(ndc, 1.0);
    let t = length(view_pos.xyz / view_pos.w) * settings.scene_units_to_m;
    return t;
}

// RGB channels: total inscattered light along the camera ray to the current sample.
// A channel: average transmittance across all wavelengths to the current sample.
fn sample_aerial_view_lut(settings: AuxLutSettings, lut: texture_3d<f32>, smp: sampler, pos_ndc: vec3<f32>) -> vec4<f32> {
    let view_pos = view.view_from_clip * vec4(pos_ndc, 1.0); //TODO: use transform fns to get dist to camera
    let dist = length(view_pos.xyz / view_pos.w);
    let t_max = settings.aerial_view_lut_max_distance;
    let num_slices = f32(settings.aerial_view_lut_size.z);
    // Offset the W coordinate by -0.5 over the max distance in order to 
    // align sampling position with slice boundaries, since each texel 
    // stores the integral over its entire slice
    let uvw = vec3(uv, saturate(dist / t_max - 0.5 / num_slices));
    let sample = textureSampleLevel(lut, smp, uvw, 0.0);
    // Treat the first slice specially since there is 0 scattering at the camera
    let delta_slice = t_max / num_slices;
    let fade = saturate(dist / delta_slice);
    // Recover the values from log space
    return exp(sample.rgb) * fade;
}


// TRANSFORM UTILITIES

fn max_atmosphere_distance(r: f32, mu: f32) -> f32 {
    let t_top = distance_to_top_atmosphere_boundary(r, mu);
    let t_bottom = distance_to_bottom_atmosphere_boundary(r, mu);
    let hits = ray_intersects_ground(r, mu);
    return select(t_top, t_bottom, hits);
}

// We assume the `up` vector at the view position is the y axis, since the world is locally flat/level.
// t = distance along view ray in atmosphere space
// NOTE: this means that if your world is actually spherical, this will be wrong.
fn get_local_up(r: f32, t: f32, ray_dir: vec3<f32>) -> vec3<f32> {
    return normalize(vec3(0.0, r, 0.0) + t * ray_dir);
}

// Given a ray starting at radius r, with mu = cos(zenith angle),
// and a t = distance along the ray, gives the new radius at point t
fn get_local_r(r: f32, mu: f32, t: f32) -> f32 {
    return sqrt(t * t + 2.0 * r * mu * t + r * r);
}

/// Converts a direction in world space to atmosphere space
fn direction_world_to_atmosphere(tf: AtmosphereTransforms, dir_ws: vec3<f32>) -> vec3<f32> {
    let dir_as = tf.atmosphere_from_world * vec4(dir_ws, 0.0);
    return dir_as.xyz;
}

/// Converts a direction in atmosphere space to world space
fn direction_atmosphere_to_world(tf: AtmosphereTransforms, dir_as: vec3<f32>) -> vec3<f32> {
    let dir_ws = tf.world_from_atmosphere * vec4(dir_as, 0.0);
    return dir_ws.xyz;
}

fn zenith_azimuth_to_ray_dir_vs(zenith: f32, azimuth: f32) -> vec3<f32> {
    let sin_zenith = sin(zenith);
    let mu = cos(zenith);
    let sin_azimuth = sin(azimuth);
    let cos_azimuth = cos(azimuth);
    return vec3(sin_azimuth * sin_zenith, mu, -cos_azimuth * sin_zenith);
}
