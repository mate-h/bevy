#define_import_path bevy_pbr::atmosphere::lighting

// Soft-sun atmosphere transmittance for mesh-view shaders (PBR, volumetric fog).
// LUT bindings come from `mesh_view_bindings`, not `atmosphere::bindings`.

#import bevy_pbr::atmosphere::bruneton_functions::transmittance_lut_r_mu_to_uv
#import bevy_pbr::atmosphere::functions::{
    atmosphere_path_toward_light,
    atmosphere_chord_lut_sample,
    atmosphere_path_is_sample,
    visible_sun_ratio_for_sphere,
    resolve_path_transmittance,
}
#import bevy_pbr::mesh_view_bindings::{
    atmosphere,
    atmosphere_transmittance_texture,
    atmosphere_transmittance_sampler,
    globals,
}
#import bevy_pbr::utils::interleaved_gradient_noise
#import bevy_render::maths::PI

const ATMOSPHERE_SUN_DISK_SAMPLES: u32 = 4u;

fn sample_transmittance_lut(r: f32, mu: f32) -> vec3<f32> {
    let uv = transmittance_lut_r_mu_to_uv(atmosphere, r, mu);
    return textureSampleLevel(
        atmosphere_transmittance_texture,
        atmosphere_transmittance_sampler,
        uv,
        0.0,
    ).rgb;
}

fn sun_disk_sample_direction(
    L: vec3<f32>,
    half_angle: f32,
    i: u32,
    n: u32,
    rotation: f32,
) -> vec3<f32> {
    let helper = select(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), abs(L.y) < 0.999);
    let tangent = normalize(cross(helper, L));
    let bitangent = cross(L, tangent);
    let golden_angle = 2.399963229728653;
    let disk0 = vec2(cos(f32(i) * golden_angle), sin(f32(i) * golden_angle))
        * sqrt((f32(i) + 0.5) / f32(n));
    let c = cos(rotation);
    let s = sin(rotation);
    let disk = vec2(c * disk0.x - s * disk0.y, s * disk0.x + c * disk0.y);
    let cos_theta = mix(cos(half_angle), 1.0, dot(disk, disk));
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let phi = atan2(disk.y, disk.x);
    return normalize(
        tangent * (sin_theta * cos(phi)) + bitangent * (sin_theta * sin(phi)) + L * cos_theta
    );
}

fn mean_shell_transmittance(
    position_as: vec3<f32>,
    L: vec3<f32>,
    half_angle: f32,
    r: f32,
    mu: f32,
    frag_coord: vec2<f32>,
) -> vec3<f32> {
#ifdef TEMPORAL_JITTER
    let frame = globals.frame_count;
#else
    let frame = 0u;
#endif
    let rotation = 2.0 * PI * interleaved_gradient_noise(frag_coord, frame);

    var shell_acc = vec3(0.0);
    var shell_count = 0.0;
    let n = ATMOSPHERE_SUN_DISK_SAMPLES;
    for (var i = 0u; i < n; i++) {
        let dir = sun_disk_sample_direction(L, half_angle, i, n, rotation);
        let path = atmosphere_path_toward_light(atmosphere, position_as, dir);
        if atmosphere_path_is_sample(path) {
            shell_acc += sample_transmittance_lut(path.sample_r, path.sample_mu);
            shell_count += 1.0;
        }
    }
    if shell_count > 0.0 {
        return shell_acc / shell_count;
    }

    // No shell hits at this pixel; use the sun-center chord.
    let b = r * sqrt(max(1.0 - mu * mu, 0.0));
    let chord = atmosphere_chord_lut_sample(atmosphere, b);
    return sample_transmittance_lut(chord.x, chord.y);
}

/// Soft-sun transmittance from a world-space point toward a world-space light direction.
fn atmosphere_directional_light_transmittance(
    position_ws: vec3<f32>,
    direction_to_light_ws: vec3<f32>,
    sun_angular_size: f32,
    frag_coord: vec2<f32>,
) -> vec3<f32> {
    let position_as = (atmosphere.world_to_atmosphere * vec4(position_ws, 1.0)).xyz;
    let L = normalize((atmosphere.world_to_atmosphere * vec4(direction_to_light_ws, 0.0)).xyz);
    let half_angle = 0.5 * sun_angular_size;
    if half_angle <= 1e-5 {
        let path = atmosphere_path_toward_light(atmosphere, position_as, L);
        return resolve_path_transmittance(
            path,
            sample_transmittance_lut(path.sample_r, path.sample_mu),
        );
    }

    let r = length(position_as);
    let mu = clamp(dot(L, position_as / max(r, 1e-6)), -1.0, 1.0);
    let f_clear = visible_sun_ratio_for_sphere(r, mu, sun_angular_size, atmosphere.outer_radius);
    let f_shell = max(
        visible_sun_ratio_for_sphere(r, mu, sun_angular_size, atmosphere.inner_radius) - f_clear,
        0.0,
    );
    if f_shell <= 0.0 {
        return vec3(f_clear);
    }

    return vec3(f_clear) + mean_shell_transmittance(position_as, L, half_angle, r, mu, frag_coord) * f_shell;
}
