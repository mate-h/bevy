#define_import_path bevy_pbr::atmosphere::functions

#import bevy_render::maths::{PI, HALF_PI, PI_2, FRAC_4_PI, FRAC_3_16_PI, fast_acos, fast_acos_4, fast_atan2}

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
fn multiscattering_lut_r_mu_to_uv(planet: Planet, r: f32, mu: f32) -> vec2<f32> {
    let u = 0.5 + 0.5 * mu;
    let v = saturate((r - planet.bottom_radius) / planet.space_altitude);
    return vec2(u, v);
}

fn multiscattering_lut_uv_to_r_mu(planet: Planet, uv: vec2<f32>) -> vec2<f32> {
    let r = mix(planet.bottom_radius, planet.top_radius, uv.y);
    let mu = uv.x * 2 - 1;
    return vec2(r, mu);
}

// LUT SAMPLING

fn sample_transmittance_lut(planet: Planet, lut: texture_2d<f32>, smp: sampler, r: f32, mu: f32) -> vec3<f32> {
    let uv = transmittance_lut_r_mu_to_uv(planet, r, mu);
    return textureSampleLevel(lut, smp, uv, 0.0).rgb;
}

fn sample_multiscattering_lut(planet: Planet, lut: texture_2d<f32>, smp: sampler, r: f32, mu: f32) -> vec3<f32> {
    let uv = multiscattering_lut_r_mu_to_uv(planet, r, mu);
    return textureSampleLevel(plut, smp, uv, 0.0).rgb;
}


// PHASE FUNCTIONS 

// -(L . V) == (L . -V). -V here is our ray direction, which points away from the view 
// instead of towards it (which would be the *view direction*, V)

// evaluates the rayleigh phase function, which describes the likelihood
// of a rayleigh scattering event scattering light from the light direction towards the view
fn rayleigh_phase(neg_LdotV: f32) -> f32 {
    return FRAC_3_16_PI * (neg_LdotV * neg_LdotV + 1);
}

// evaluates the henyey-greenstein phase function, which describes the likelihood
// of a mie scattering event scattering light from the light direction towards the view
fn henyey_greenstein_phase(neg_LdotV: f32, g: f32) -> f32 {
    let denom = 1.0 + g * g - 2.0 * g * neg_LdotV;
    return FRAC_4_PI * (1.0 - g * g) / (denom * sqrt(denom));
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

fn zenith_azimuth_to_ray_dir_vs(zenith: f32, azimuth: f32) -> vec3<f32> {
    let sin_zenith = sin(zenith);
    let mu = cos(zenith);
    let sin_azimuth = sin(azimuth);
    let cos_azimuth = cos(azimuth);
    return vec3(sin_azimuth * sin_zenith, mu, -cos_azimuth * sin_zenith);
}
