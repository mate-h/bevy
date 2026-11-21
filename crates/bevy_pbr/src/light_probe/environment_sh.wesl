// The spherical harmonic irradiance of a generated environment map, as the
// `project_irradiance_sh` pass of `environment_filter.wesl` writes it, once per
// probe per frame, into the probe's `EnvironmentMapShBuffer`.
//
// Import this module to read that buffer from another pass (the Rust side binds
// the `Buffer` of `EnvironmentMapShBuffer`, or the one `EnvironmentMapShBuffers`
// maps the probe's image id to, with `storage_buffer_read_only::<EnvironmentMapSh>`):
//
//     import package::light_probe::environment_sh::{
//         EnvironmentMapSh, excess_light_irradiance, sh9_irradiance,
//     };
//     @group(G) @binding(B) var<storage, read> sky_sh: EnvironmentMapSh;
//
//     // Irradiance over π at the normal `n`: the harmonics, then the excess lights
//     // (the loop is the caller's, so it can be skipped where the sun is handled
//     // separately, or convolved with a phase function instead of the cosine).
//     var diffuse = sh9_irradiance(sky_sh.coefficients, n);
//     for (var i = 0u; i < sky_sh.excess_count; i++) {
//         diffuse += excess_light_irradiance(
//             sky_sh.excess_directions[i].xyz, sky_sh.excess_radiances[i].rgb, n);
//     }
//
// The buffer may be bound as `var<uniform>` as well: every field is 16-byte aligned.
// For an isotropic medium, `coefficients[0].rgb * 0.282095` is the mean radiance
// over the sphere (the L0 term), and each excess light adds `radiance / (4π)`.

import bevy_render::maths::PI;

/// At most this many over-bright texels are stored as excess lights; any beyond
/// are projected onto the harmonics in full.
const ENVIRONMENT_SH_MAX_EXCESS: u32 = 64u;

/// A probe's diffuse lighting: 9 RGB spherical harmonic coefficients (L2) plus
/// up to `ENVIRONMENT_SH_MAX_EXCESS` point lights for the excess of texels that
/// were too bright to project without ringing (such as the sun).
///
/// Layout (std430 and std140 agree, 2208 bytes in total):
///
/// | offset | size | field               | contents                                        |
/// |-------:|-----:|---------------------|-------------------------------------------------|
/// |      0 |  144 | `coefficients`      | 9 × (R, G, B, 0); band order `sh9_basis`        |
/// |    144 |    4 | `excess_count`      | lights in use, `≤ ENVIRONMENT_SH_MAX_EXCESS`    |
/// |    148 |   12 | (padding)           |                                                 |
/// |    160 | 1024 | `excess_directions` | 64 × (x, y, z, 0), unit vectors toward the light |
/// |   1184 | 1024 | `excess_radiances`  | 64 × (R, G, B, 0), radiance × solid angle       |
///
/// The coefficients are already convolved with the clamped cosine and divided
/// by π (band scales 1, 2/3, 1/4), so `dot(coefficients, sh9_basis(n))` is the
/// irradiance at the normal `n` over π: what a Lambertian surface of albedo 1
/// reflects. Each excess light adds `radiance · max(n · direction, 0) / π`.
struct EnvironmentMapSh {
    coefficients: array<vec4f, 9>,
    excess_count: u32,
    excess_directions: array<vec4f, ENVIRONMENT_SH_MAX_EXCESS>,
    excess_radiances: array<vec4f, ENVIRONMENT_SH_MAX_EXCESS>,
}

/// The real spherical harmonic basis up to the second band at the unit
/// direction `d`, in the order Y00, Y1-1, Y10, Y11, Y2-2, Y2-1, Y20, Y21, Y22.
fn sh9_basis(d: vec3f) -> array<f32, 9> {
    return array<f32, 9>(
        0.282095,
        0.488603 * d.y,
        0.488603 * d.z,
        0.488603 * d.x,
        1.092548 * d.x * d.y,
        1.092548 * d.y * d.z,
        0.315392 * (3.0 * d.z * d.z - 1.0),
        1.092548 * d.x * d.z,
        0.546274 * (d.x * d.x - d.y * d.y),
    );
}

/// The irradiance over π at the unit normal `normal` from the harmonics
/// `coefficients` of an `EnvironmentMapSh`, clamped at zero since two bands can
/// ring below it. Excess lights are not included: see `excess_light_irradiance`.
fn sh9_irradiance(coefficients: array<vec4f, 9>, normal: vec3f) -> vec3f {
    let basis = sh9_basis(normal);
    var irradiance = vec3f(0.0);
    for (var i = 0u; i < 9u; i++) {
        irradiance += coefficients[i].rgb * basis[i];
    }
    return max(irradiance, vec3f(0.0));
}

/// The irradiance over π at the unit normal `normal` from one excess light of an
/// `EnvironmentMapSh`: `radiance` (times solid angle) arriving from the unit
/// `direction`, as a clamped-cosine point light.
fn excess_light_irradiance(direction: vec3f, radiance: vec3f, normal: vec3f) -> vec3f {
    return radiance * max(dot(normal, direction), 0.0) / PI;
}
