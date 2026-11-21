import package::atmosphere::{
    bindings::atmosphere,
    functions::{
        sample_transmittance_lut, sample_multiscattering_lut, sample_density_lut,
        sample_scattering_lut, ground_irradiance_lut_u_to_mu, GROUND_IRRADIANCE_LUT_SIZE,
        MIN_EXTINCTION, ABSORPTION_DENSITY, SCATTERING_DENSITY,
    },
    bruneton_functions::{distance_to_top_atmosphere_boundary, ray_intersects_ground},
};

import bevy_render::maths::PI;

@group(0) @binding(13) var ground_irradiance_lut_out: texture_storage_2d<rgba16float, write>;

/// Cosine-weighted sky directions and march steps per direction. The LUT is rebuilt only with
/// the transmittance and multiscattering LUTs, so this can be far finer than a per-texel march.
const DIRS: u32 = 128u;
const STEPS: u32 = 24u;

/// Clear-sky irradiance on the ground per unit sun illuminance: the Lambert-weighted integral of
/// the sky radiance over the upper hemisphere, with the sun at zenith cosine mu. Single scattering
/// from the transmittance LUT, higher orders from the multiscattering LUT, as in the sky-view LUT
/// (no cloud or mesh shadows). By symmetry it depends on mu alone.
@compute
@workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) idx: vec3<u32>) {
    if (idx.x >= GROUND_IRRADIANCE_LUT_SIZE) {
        return;
    }
    let mu_sun = ground_irradiance_lut_u_to_mu(f32(idx.x) / f32(GROUND_IRRADIANCE_LUT_SIZE - 1u));
    let r = atmosphere.inner_radius;
    let p = vec3(0.0, r, 0.0);
    let light_dir = vec3(sqrt(max(0.0, 1.0 - mu_sun * mu_sun)), mu_sun, 0.0);

    var sum = vec3(0.0);
    for (var k = 0u; k < DIRS; k++) {
        // Cosine-weighted (pdf cos / pi), golden-angle spiral.
        let u = (f32(k) + 0.5) / f32(DIRS);
        let phi = f32(k) * 2.399963;
        let st = sqrt(u);
        let ct = sqrt(1.0 - u);
        let d = vec3(st * cos(phi), ct, st * sin(phi));
        let t_end = distance_to_top_atmosphere_boundary(atmosphere, r, ct);
        let neg_LdotV = dot(light_dir, d);
        var tr = vec3(1.0);
        var l = vec3(0.0);
        for (var i = 0u; i < STEPS; i++) {
            // Quadratic spacing: dense near the ground, where the haze and Mie layers are.
            let ua = f32(i) / f32(STEPS);
            let ub = f32(i + 1u) / f32(STEPS);
            let ta = t_end * ua * ua;
            let tb = t_end * ub * ub;
            let q = p + d * (0.5 * (ta + tb));
            let rq = length(q);
            let mu_light = dot(light_dir, q / rq);
            let scattering = sample_density_lut(rq, SCATTERING_DENSITY);
            let extinction = sample_density_lut(rq, ABSORPTION_DENSITY) + scattering;
            let st_i = exp(-extinction * (tb - ta));
            let sun = sample_transmittance_lut(rq, mu_light) * f32(!ray_intersects_ground(rq, mu_light));
            let ins = sun * sample_scattering_lut(rq, neg_LdotV)
                + sample_multiscattering_lut(rq, mu_light) * scattering;
            l += tr * (ins - ins * st_i) / max(extinction, MIN_EXTINCTION);
            tr *= st_i;
        }
        sum += l;
    }
    textureStore(ground_irradiance_lut_out, vec2(idx.x, 0u), vec4(PI * sum / f32(DIRS), 1.0));
}
