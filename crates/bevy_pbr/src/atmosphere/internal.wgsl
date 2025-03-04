// This file contains utility functions for atmospherics that are either dependent on specific bindings,
// or shouldn't be exposed in the public api. 

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

#import bevy_render::maths::{PI_2, FRAC_2_PI, FRAC_PI}

// BINDINGS

// core bindings
@group(0) @binding(0) var<storage> atmosphere: Atmosphere;
@group(0) @binding(1) var atmosphere_sampler: sampler;

// input for lut-based
@group(0) @binding(3) var<uniform> view: View;
@group(0) @binding(4) var<uniform> lights: Lights;
@group(0) @binding(1) var<uniform> core_settings: CoreSettings;
@group(0) @binding(5) var<uniform> lut_based_settings: LutBasedSettings;

// luts
@group(0) @binding(6) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(7) var multiscattering_lut: texture_2d<f32>;
@group(0) @binding(8) var sky_view_lut: texture_2d<f32>;
@group(0) @binding(9) var aerial_view_lut: texture_2d<f32>;


// During raymarching, each segment is sampled at a single point. This constant determines
// where in the segment that sample is taken (0.0 = start, 0.5 = middle, 1.0 = end).
// We use 0.3 to sample closer to the start of each segment, which better approximates
// the exponential falloff of atmospheric density.
const MIDPOINT_RATIO: f32 = 0.3;

// LUT PARAMETRIZATION
fn sky_view_lut_r_mu_azimuth_to_uv(r: f32, mu: f32, azimuth: f32) -> vec2<f32> {
    let u = (azimuth * FRAC_2_PI) + 0.5;

    // Horizon parameters
    let v_horizon = sqrt(r * r - atmosphere.planet.bottom_radius_sq);
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

fn sky_view_lut_uv_to_zenith_azimuth(r: f32, uv: vec2<f32>) -> vec2<f32> {
    let adj_uv = vec2(uv.x, 1.0 - uv.y);
    let azimuth = (adj_uv.x - 0.5) * PI_2;

    // Horizon parameters
    let v_horizon = sqrt(r * r - atmosphere.planet.bottom_radius_sq);
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

fn sample_sky_view_lut(r: f32, ray_dir_as: vec3<f32>) -> vec3<f32> {
    let mu = ray_dir_as.y;
    let azimuth = fast_atan2(ray_dir_as.x, -ray_dir_as.z);
    let uv = sky_view_lut_r_mu_azimuth_to_uv(r, mu, azimuth);
    return textureSampleLevel(sky_view_lut, atmosphere_sampler, uv, 0.0).rgb;
}

// RGB channels: total inscattered light along the camera ray to the current sample.
// A channel: average transmittance across all wavelengths to the current sample.
fn sample_aerial_view_lut(pos_ndc: vec3<f32>) -> vec4<f32> {
    let view_pos = view.view_from_clip * vec4(pos_ndc, 1.0); //TODO: use transform fns to get dist to camera
    let dist = length(view_pos.xyz / view_pos.w);
    let t_max = lut_based_settings.aerial_view_lut_max_distance;
    let num_slices = f32(lut_based_settings.aerial_view_lut_size.z);
    // Offset the W coordinate by -0.5 over the max distance in order to 
    // align sampling position with slice boundaries, since each texel 
    // stores the integral over its entire slice
    let uvw = vec3(uv, saturate(dist / t_max - 0.5 / num_slices));
    let sample = textureSampleLevel(aerial_view_lut, atmosphere_sampler, uvw, 0.0);
    // Treat the first slice specially since there is 0 scattering at the camera
    let delta_slice = t_max / num_slices;
    let fade = saturate(dist / delta_slice);
    // Recover the values from log space
    return exp(sample.rgb) * fade;
}

// ATMOSPHERE SAMPLING

struct Medium {
    /// units: m^-1
    rayleigh_scattering: vec3<f32>,

    /// units: m^-1
    mie_scattering: f32,

    /// the sum of scattering and absorption. Since the phase function doesn't
    /// matter for this, we combine rayleigh and mie extinction to a single 
    //  value.
    //
    /// units: m^-1
    extinction: vec3<f32>
}

/// Samples the atmosphere medium at a given radius and returns the optical density of each scattering component
fn sample_medium(r: f32) -> Medium {
    let altitude = clamp(r, atmosphere.planet.bottom_radius, atmosphere.planet.top_radius) - atmosphere.planet.bottom_radius;

    // atmosphere values at altitude
    let mie_density = exp(-atmosphere.profile.mie_density_exp_scale * altitude);
    let rayleigh_density = exp(-atmosphere.rayleigh_density_exp_scale * altitude);
    var ozone_density: f32 = max(0.0, 1.0 - (abs(altitude - atmosphere.ozone_layer_altitude) / (atmosphere.ozone_layer_width * 0.5)));

    let mie_scattering = mie_density * atmosphere.mie_scattering;
    let mie_absorption = mie_density * atmosphere.mie_absorption;
    let mie_extinction = mie_scattering + mie_absorption;

    let rayleigh_scattering = rayleigh_density * atmosphere.rayleigh_scattering;
    // no rayleigh absorption
    // rayleigh extinction is the sum of scattering and absorption

    // ozone doesn't contribute to scattering
    let ozone_absorption = ozone_density * atmosphere.ozone_absorption;

    var sample: AtmosphereSample;
    sample.rayleigh_scattering = rayleigh_scattering;
    sample.mie_scattering = mie_scattering;
    sample.extinction = rayleigh_scattering + mie_extinction + ozone_absorption;

    return sample;
}

/// evaluates L_scat, equation 3 in the paper, which gives the total single-order scattering towards the view at a single point
fn L_scattering(medium: Medium, ray_dir_ws: vec3<f32>, local_r: f32, local_up: vec3<f32>) -> vec3<f32> {
    var inscattering = vec3(0.0);
    for (var light_i: u32 = 0u; light_i < lights.n_directional_lights; light_i++) {
        let light = &lights.directional_lights[light_i];

        let mu_light = dot((*light).direction_to_light, local_up);

        // -(L . V) == (L . -V). -V here is our ray direction, which points away from the view
        // instead of towards it (as is the convention for V)
        let neg_LdotV = dot((*light).direction_to_light, ray_dir_ws);

        // Phase functions give the proportion of light
        // scattered towards the camera for each scattering type
        let rayleigh_phase = rayleigh(neg_LdotV);
        let mie_phase = henyey_greenstein(neg_LdotV);
        let scattering_coeff = medium.rayleigh_scattering * rayleigh_phase + medium.mie_scattering * mie_phase;

        let transmittance_to_light = sample_transmittance_lut(local_r, mu_light);
        let shadow_factor = transmittance_to_light * f32(!ray_intersects_ground(local_r, mu_light));

        // Transmittance from scattering event to light source
        let scattering_factor = shadow_factor * scattering_coeff;

        // Additive factor from the multiscattering LUT
        let psi_ms = sample_multiscattering_lut(local_r, mu_light);
        let multiscattering_factor = psi_ms * (medium.rayleigh_scattering + medium.mie_scattering);

        inscattering += (*light).color.rgb * (scattering_factor + multiscattering_factor);
    }
    return inscattering * view.exposure;
}

const SUN_ANGULAR_SIZE: f32 = 0.0174533; // angular diameter of sun in radians

// evaluates the luminance from 
fn L_sun(transmittance: vec3<f32>, r: f32, ray_dir_ws: vec3<f32>) -> vec3<f32> {
    let mu_view = ray_dir_ws.y;
    let shadow_factor = f32(!ray_intersects_ground(atmosphere.planet, r, mu_view));
    var sun_illuminance = vec3(0.0);
    for (var light_i: u32 = 0u; light_i < lights.n_directional_lights; light_i++) {
        let light = &lights.directional_lights[light_i];
        let neg_LdotV = dot((*light).direction_to_light, ray_dir_ws);
        let angle_to_sun = fast_acos(neg_LdotV);
        let pixel_size = fwidth(angle_to_sun);
        let factor = smoothstep(0.0, -pixel_size * ROOT_2, angle_to_sun - SUN_ANGULAR_SIZE * 0.5);
        let sun_solid_angle = (SUN_ANGULAR_SIZE * SUN_ANGULAR_SIZE) * 4.0 * FRAC_PI;
        sun_illuminance += ((*light).color.rgb / sun_solid_angle) * factor * shadow_factor;
    }
    return sun_illuminance * transmittance * view.exposure;
}


// MISC TRANSFORMS

fn view_radius() -> f32 {
    return view.position.y + atmosphere.planet.bottom_radius;
}

// Modified from skybox.wgsl. For this pass we don't need to apply a separate sky transform or consider camera viewport.
// w component is the cosine of the view direction with the view forward vector, to correct step distance at the edges of the viewport
fn uv_to_ray_dir_ws(uv: vec2<f32>) -> vec4<f32> {
    // Using world positions of the fragment and camera to calculate a ray direction
    // breaks down at large translations. This code only needs to know the ray direction.
    // The ray direction is along the direction from the camera to the fragment position.
    // In view space, the camera is at the origin, so the view space ray direction is
    // along the direction of the fragment position - (0,0,0) which is just the
    // fragment position.
    // Use the position on the near clipping plane to avoid -inf world position
    // because the far plane of an infinite reverse projection is at infinity.
    let near_clip_pos = view.view_from_clip * vec4(
        uv_to_ndc(uv),
        1.0,
        1.0,
    );

    let ray_dir_vs = near_clip_pos.xyz / near_clip_pos.w;
    // Transforming the view space ray direction by the inverse view matrix, transforms the
    // direction to world space. Note that the w element is set to 0.0, as this is a
    // vector direction, not a position, That causes the matrix multiplication to ignore
    // the translations from the view matrix.
    let ray_dir_ws = (view.world_from_view * vec4(ray_dir_vs, 0.0)).xyz;

    return vec4(normalize(ray_dir_ws), -ray_dir_ws.z);
}

// Convert uv [0.0 .. 1.0] coordinate to ndc space xy [-1.0 .. 1.0]
fn uv_to_ndc(uv: vec2<f32>) -> vec2<f32> {
    return uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
}

/// Convert ndc space xy coordinate [-1.0 .. 1.0] to uv [0.0 .. 1.0]
fn ndc_to_uv(ndc: vec2<f32>) -> vec2<f32> {
    return ndc * vec2(0.5, -0.5) + vec2(0.5);
}

fn ndc_to_camera_dist(ndc: vec3<f32>) -> f32 {
    let view_pos = view.view_from_clip * vec4(ndc, 1.0);
    let t = length(view_pos.xyz / view_pos.w) * settings.scene_units_to_m;
    return t;
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
