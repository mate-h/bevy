import package::light_probe::{light_probe_iterator_new, light_probe_iterator_next};
import package::render::mesh_view_bindings as bindings;
import package::render::mesh_view_bindings::light_probes;
import package::render::mesh_view_types::{
    LIGHT_PROBE_FLAG_AFFECTS_LIGHTMAPPED_MESH_DIFFUSE, LIGHT_PROBE_FLAG_PARALLAX_CORRECT,
    LIGHT_PROBE_FLAG_MANSON_SLOAN_ENVIRONMENT_IBL
};
import package::render::pbr_lighting::{D_GGX, V_SmithGGXCorrelated, sample_visible_ggx, ggx_vndf_pdf, F_Schlick_vec, LightingInput, LayerLightingInput, LAYER_BASE, LAYER_CLEARCOAT, Fd_Burley, derive_lighting_input, directional_disk_diffuse_cosine, directional_disk_specular};
import package::render::clustered_forward::ClusterableObjectIndexRanges;
import bevy_render::maths::quat_rotate;

/// The maximum representable value in a 32-bit floating point number.
const FLOAT_MAX: f32 = 3.40282347e+38;

struct EnvironmentMapLight {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
};

struct EnvironmentMapRadiances {
    irradiance: vec3<f32>,
    radiance: vec3<f32>,
    fresnel_radiance: vec3<f32>,
}

// https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling
// Experimental shade-time FIS. The reflection_probes harness enables this only
// when it binds an ordinary box-filtered radiance pyramid, never a GGX chain.
const SHADE_TIME_FIS: bool = false;
const FIS_SAMPLES: u32 = 64u;

// Experimental view-dependent split sum. At grazing views the reflected GGX lobe
// keeps its width in the plane of incidence but narrows by about N⋅V across it,
// while a prefiltered mip is round. This reads the mip that matches the narrow
// width and adds taps along the plane of incidence to restore the wide one. The
// reflection_probes harness enables it with `IBL_FILTER=stretch`.
const STRETCHED_SPLIT_SUM: bool = false;
const STRETCH_TAPS: u32 = 5u;
/// Angular standard deviation of the reflected lobe, per unit of roughness α.
const STRETCH_LOBE_SCALE: f32 = 1.0;

// Experimental fitted view-dependent lookups. The reflected GGX lobe at
// (roughness, N⋅V) is fitted offline with one or two reads of the round
// prefiltered map, each rotated in the plane of N and R by its own angle and read
// at its own roughness (`kernel_table.py` in the reflection_probes notes). The
// reflection_probes harness enables it with `IBL_LOBE=fitted1|fitted2`. The
// table is fitted to the lobe's response to spherical-cap lights of 5° to 90°,
// each size weighted equally so that light the round kernels collect from below
// the horizon counts; rows are perceptual roughness 0.2, 0.3, …, 1, columns N⋅V
// 1, 0.8, 0.6, 0.45, 0.3, 0.2, 0.1, 0.05.
const FITTED_LOBE_TAPS: u32 = 0u;
// Removes each fitted tap's light from below the surface horizon and renormalizes
// by the part of its kernel that is left (`horizon_clipped_radiance`). Needs a
// table fitted with clipped kernels (`HORIZON_CLIP=1 kernel_table.py`).
const FITTED_LOBE_HORIZON_CLIP: bool = false;
// The same clip for the plain split-sum lookup along the dominant direction
// (`horizon_clipped_split_sum_radiance`): one more read, and no table.
const SPLIT_SUM_HORIZON_CLIP: bool = false;
/// Per cell, roughness-major: `(angle₀, roughness₀, angle₁, roughness₁)` of the
/// two-tap fit, then `(weight₀, angle, roughness, 0)` with the one-tap fit. Angles
/// rotate R toward N, in radians.
const FITTED_LOBE_TABLE: array<vec4<f32>, 144> = array(
    vec4(-0.0192, 0.8475, 0.0000, 0.2018), vec4(0.0137, 0.0000, 0.2071, 0.0),
    vec4(-0.0154, 0.1819, 0.0546, 0.1899), vec4(0.7474, 0.0006, 0.1909, 0.0),
    vec4(-0.0328, 0.1745, 0.0413, 0.1569), vec4(0.5445, 0.0015, 0.1816, 0.0),
    vec4(-0.0266, 0.1345, 0.0557, 0.1868), vec4(0.5345, 0.0091, 0.1765, 0.0),
    vec4(-0.0277, 0.1479, 0.0775, 0.1204), vec4(0.6339, 0.0006, 0.1539, 0.0),
    vec4(0.0105, 0.1457, 0.1834, 0.1926), vec4(0.8854, 0.0191, 0.1644, 0.0),
    vec4(0.0113, 0.0309, 0.1123, 0.1690), vec4(0.3568, 0.0609, 0.1804, 0.0),
    vec4(0.0965, 0.1224, 0.3874, 0.3179), vec4(0.7176, 0.1388, 0.2195, 0.0),
    vec4(-0.0006, 0.9576, 0.0000, 0.3020), vec4(0.0384, 0.0000, 0.3093, 0.0),
    vec4(-0.0158, 0.2695, 0.1660, 0.3472), vec4(0.8225, 0.0045, 0.2893, 0.0),
    vec4(-0.0464, 0.2346, 0.1391, 0.2777), vec4(0.6444, 0.0032, 0.2732, 0.0),
    vec4(-0.0458, 0.2021, 0.1453, 0.2683), vec4(0.5614, 0.0329, 0.2616, 0.0),
    vec4(-0.0260, 0.1934, 0.1884, 0.2443), vec4(0.5848, 0.0101, 0.2222, 0.0),
    vec4(0.0223, 0.1875, 0.2849, 0.2851), vec4(0.6701, 0.0617, 0.2242, 0.0),
    vec4(0.0790, 0.1722, 0.3827, 0.3303), vec4(0.6164, 0.1352, 0.2415, 0.0),
    vec4(0.1286, 0.1706, 0.4529, 0.3609), vec4(0.5534, 0.1923, 0.2522, 0.0),
    vec4(-0.0000, 1.0000, 0.0000, 0.4028), vec4(0.0954, -0.0001, 0.4206, 0.0),
    vec4(-0.0270, 0.3682, 0.2491, 0.4364), vec4(0.7834, 0.0089, 0.3929, 0.0),
    vec4(-0.0650, 0.3054, 0.1921, 0.3820), vec4(0.5231, 0.0216, 0.3647, 0.0),
    vec4(-0.0887, 0.2376, 0.1856, 0.3396), vec4(0.3887, 0.0551, 0.3447, 0.0),
    vec4(-0.0387, 0.2049, 0.2090, 0.3373), vec4(0.2806, 0.0736, 0.3308, 0.0),
    vec4(0.0488, 0.2071, 0.3302, 0.3635), vec4(0.3811, 0.1123, 0.2980, 0.0),
    vec4(0.1667, 0.2177, 0.5567, 0.4311), vec4(0.4972, 0.1980, 0.3023, 0.0),
    vec4(0.2206, 0.2168, 0.6584, 0.4678), vec4(0.4838, 0.2511, 0.3161, 0.0),
    vec4(-0.0000, 0.5001, 0.0000, 1.0000), vec4(0.7995, -0.0002, 0.5432, 0.0),
    vec4(-0.0153, 0.4684, 0.5413, 0.6056), vec4(0.8181, 0.0289, 0.5031, 0.0),
    vec4(-0.0547, 0.3837, 0.3377, 0.4972), vec4(0.5180, 0.0716, 0.4654, 0.0),
    vec4(0.0248, 0.3600, 0.4835, 0.4930), vec4(0.5819, 0.1248, 0.4424, 0.0),
    vec4(0.0332, 0.2434, 0.4284, 0.4633), vec4(0.3377, 0.2059, 0.4315, 0.0),
    vec4(0.1218, 0.2344, 0.5555, 0.4858), vec4(0.3616, 0.2808, 0.4344, 0.0),
    vec4(0.2143, 0.2270, 0.7038, 0.5239), vec4(0.3571, 0.3812, 0.4499, 0.0),
    vec4(0.2622, 0.2240, 0.7880, 0.5509), vec4(0.3462, 0.4437, 0.4651, 0.0),
    vec4(-0.0000, 0.5857, 0.0000, 1.0000), vec4(0.6217, -0.0002, 0.6775, 0.0),
    vec4(-0.0108, 0.5506, 0.6354, 0.7765), vec4(0.7080, 0.0887, 0.6220, 0.0),
    vec4(0.0280, 0.4751, 0.6754, 0.6791), vec4(0.5906, 0.1755, 0.5807, 0.0),
    vec4(0.0923, 0.4314, 0.7504, 0.6554), vec4(0.5312, 0.2607, 0.5609, 0.0),
    vec4(0.1902, 0.4042, 0.8965, 0.6625), vec4(0.5033, 0.3713, 0.5546, 0.0),
    vec4(0.2669, 0.3931, 1.0239, 0.6827), vec4(0.4862, 0.4637, 0.5612, 0.0),
    vec4(0.3512, 0.3864, 1.1779, 0.7223), vec4(0.4653, 0.5752, 0.5805, 0.0),
    vec4(0.3978, 0.3863, 1.2763, 0.7582), vec4(0.4563, 0.6395, 0.5976, 0.0),
    vec4(-0.0000, 0.6393, 0.0000, 1.0000), vec4(0.3454, -0.0004, 0.8198, 0.0),
    vec4(0.0076, 0.6136, 0.6116, 0.9459), vec4(0.5224, 0.2091, 0.7591, 0.0),
    vec4(0.0973, 0.5493, 0.8485, 0.8666), vec4(0.4977, 0.3484, 0.7227, 0.0),
    vec4(0.1906, 0.5151, 1.0106, 0.8466), vec4(0.4805, 0.4647, 0.7099, 0.0),
    vec4(0.2971, 0.4893, 1.1791, 0.8524), vec4(0.4585, 0.5991, 0.7119, 0.0),
    vec4(0.3740, 0.4763, 1.3015, 0.8746), vec4(0.4405, 0.7025, 0.7247, 0.0),
    vec4(0.4547, 0.4675, 1.4412, 0.9247), vec4(0.4206, 0.8203, 0.7527, 0.0),
    vec4(0.4956, 0.4658, 1.5240, 0.9751), vec4(0.4101, 0.8867, 0.7778, 0.0),
    vec4(-0.0002, 0.5419, 0.0000, 1.0000), vec4(0.0400, -0.0003, 0.9639, 0.0),
    vec4(-0.0301, 0.6525, 0.6060, 1.0000), vec4(0.2963, 0.3910, 1.0000, 0.0),
    vec4(0.1375, 0.6051, 0.8879, 1.0000), vec4(0.3280, 0.6019, 1.0000, 0.0),
    vec4(0.2501, 0.5780, 1.0747, 1.0000), vec4(0.3322, 0.7519, 1.0000, 0.0),
    vec4(0.3555, 0.5560, 1.2590, 1.0000), vec4(0.3291, 0.9034, 1.0000, 0.0),
    vec4(0.4216, 0.5445, 1.3869, 1.0000), vec4(0.3257, 1.0065, 1.0000, 0.0),
    vec4(0.4837, 0.5394, 1.5329, 1.0000), vec4(0.3283, 1.1107, 1.0000, 0.0),
    vec4(0.5140, 0.5437, 1.6272, 1.0000), vec4(0.3388, 1.1618, 1.0000, 0.0),
    vec4(-1.1039, 1.0000, 0.0422, 1.0000), vec4(0.0570, -0.0000, 1.0000, 0.0),
    vec4(-0.1829, 1.0000, 0.6109, 1.0000), vec4(0.1341, 0.5226, 1.0000, 0.0),
    vec4(0.0016, 0.6484, 0.9402, 1.0000), vec4(0.1704, 0.7708, 1.0000, 0.0),
    vec4(0.1478, 0.6326, 1.1354, 1.0000), vec4(0.1911, 0.9333, 1.0000, 0.0),
    vec4(0.2740, 0.6229, 1.3290, 1.0000), vec4(0.2124, 1.0886, 1.0000, 0.0),
    vec4(0.3521, 0.6223, 1.4673, 1.0000), vec4(0.2321, 1.1903, 1.0000, 0.0),
    vec4(0.4326, 0.6334, 1.6339, 1.0000), vec4(0.2671, 1.2903, 1.0000, 0.0),
    vec4(0.4845, 0.6524, 1.7536, 1.0000), vec4(0.3039, 1.3378, 1.0000, 0.0),
    vec4(-0.3351, 1.0000, 0.2543, 1.0000), vec4(0.4366, -0.0000, 1.0000, 0.0),
    vec4(0.5562, 1.0000, 1.7738, 1.0000), vec4(0.8898, 0.6435, 1.0000, 0.0),
    vec4(0.8065, 1.0000, 2.0659, 0.7258), vec4(0.8784, 0.9273, 1.0000, 0.0),
    vec4(-0.0309, 0.7318, 1.2486, 1.0000), vec4(0.1420, 1.1040, 1.0000, 0.0),
    vec4(1.0835, 1.0000, 2.3934, 0.7420), vec4(0.8274, 1.2661, 1.0000, 0.0),
    vec4(1.1420, 1.0000, 2.4818, 0.7558), vec4(0.7945, 1.3694, 1.0000, 0.0),
    vec4(1.1527, 1.0000, 2.5402, 0.7891), vec4(0.7348, 1.4705, 1.0000, 0.0),
    vec4(0.8573, 1.0000, 2.1581, 1.0000), vec4(0.4926, 1.5207, 1.0000, 0.0),
);

fn fitted_lobe_cell(i: u32, j: u32) -> array<vec4<f32>, 2> {
    let index = 2u * (i * 8u + j);
    return array(FITTED_LOBE_TABLE[index], FITTED_LOBE_TABLE[index + 1u]);
}

// A round prefiltered read along the unit direction `d`, with the part of its
// kernel below the horizon taken out when `clip` is set. That part's weight is
// `½(1 − N⋅d)^p(r)` and its centroid lies `β(r, N⋅d)` below the horizon, in the
// plane of N and d, both fitted to the Karis kernel (`horizon_analytic.py`); the
// light there is read from the same mip.
fn horizon_clipped_radiance(
    map: texture_cube<f32>,
    N: vec3<f32>,
    d: vec3<f32>,
    roughness: f32,
    level: f32,
    clip: bool,
    world_position: vec3<f32>,
    light_from_world: mat4x4<f32>,
    bounds: vec3<f32>,
    parallax_correct: bool,
    rotation: vec4<f32>,
) -> vec3<f32> {
    let direction = compute_cubemap_sample_dir(world_position, d, light_from_world, bounds,
        parallax_correct, rotation);
    let radiance = textureSampleLevel(map, bindings::environment_map_sampler, direction, level).rgb;
    if !clip {
        return radiance;
    }
    let NdotD = dot(N, d);
    let r = max(roughness, 0.05);
    let p = 1.0 + 0.4507 * (pow(r, -2.1412) - 1.0);
    let tail = 0.5 * pow(1.0 - abs(NdotD), p);
    let below = select(1.0 - tail, tail, NdotD >= 0.0);
    let beta = max(0.643 * r - 0.0403, 0.0) * (1.0 - clamp(3.0816 * r - 1.7164, 0.0, 0.75) * NdotD);
    // Toward d along the horizon; when d is N the below part is empty anyway.
    let side_raw = d - NdotD * N;
    let side = side_raw * inverseSqrt(max(dot(side_raw, side_raw), 1e-12));
    let below_direction = compute_cubemap_sample_dir(world_position,
        -sin(beta) * N + cos(beta) * side, light_from_world, bounds, parallax_correct, rotation);
    let below_radiance =
        textureSampleLevel(map, bindings::environment_map_sampler, below_direction, level).rgb;
    return max(radiance - below * below_radiance, vec3(0.0)) / max(1.0 - below, 0.05);
}

fn horizon_clipped_split_sum_radiance(
    map: texture_cube<f32>,
    input: LayerLightingInput,
    max_specular_mip: f32,
    world_position: vec3<f32>,
    light_from_world: mat4x4<f32>,
    bounds: vec3<f32>,
    parallax_correct: bool,
    rotation: vec4<f32>,
) -> vec3<f32> {
    let dominant = normalize(radiance_sample_direction_ggx_split_sum(input.N, input.R, input.roughness));
    return horizon_clipped_radiance(map, input.N, dominant, input.perceptual_roughness,
        specular_radiance_level(input.perceptual_roughness, max_specular_mip), true,
        world_position, light_from_world, bounds, parallax_correct, rotation);
}

fn fitted_lobe_radiance(
    map: texture_cube<f32>,
    input: LayerLightingInput,
    max_specular_mip: f32,
    world_position: vec3<f32>,
    light_from_world: mat4x4<f32>,
    bounds: vec3<f32>,
    parallax_correct: bool,
    rotation: vec4<f32>,
) -> vec3<f32> {
    let N = input.N;
    let R = input.R;
    let perceptual = input.perceptual_roughness;
    let NdotV = clamp(input.NdotV, 0.05, 1.0);

    // Bilinear weights over the table; below its first row, blend toward the
    // plain lookup at zero roughness.
    let row = clamp((perceptual - 0.2) / 0.1, 0.0, 8.0);
    let i0 = min(u32(row), 7u);
    let fi = row - f32(i0);
    // The N⋅V column and the values bounding it, from literals: indexing a constant
    // array of the columns per pixel tripled the one-tap lookup's extra cost.
    var j0 = 0u;
    var upper = 1.0;
    var lower = 0.8;
    if NdotV < 0.8 { j0 = 1u; upper = 0.8; lower = 0.6; }
    if NdotV < 0.6 { j0 = 2u; upper = 0.6; lower = 0.45; }
    if NdotV < 0.45 { j0 = 3u; upper = 0.45; lower = 0.3; }
    if NdotV < 0.3 { j0 = 4u; upper = 0.3; lower = 0.2; }
    if NdotV < 0.2 { j0 = 5u; upper = 0.2; lower = 0.1; }
    if NdotV < 0.1 { j0 = 6u; upper = 0.1; lower = 0.05; }
    let fj = (upper - NdotV) / (upper - lower);
    let c00 = fitted_lobe_cell(i0, j0);
    let c01 = fitted_lobe_cell(i0, j0 + 1u);
    let c10 = fitted_lobe_cell(i0 + 1u, j0);
    let c11 = fitted_lobe_cell(i0 + 1u, j0 + 1u);
    let blend = clamp(perceptual / 0.2, 0.0, 1.0);
    let identity = array(vec4(0.0, perceptual, 0.0, perceptual), vec4(1.0, 0.0, perceptual, 0.0));
    var cell: array<vec4<f32>, 2>;
    for (var k = 0u; k < 2u; k++) {
        let fitted = mix(mix(c00[k], c01[k], fj), mix(c10[k], c11[k], fj), fi);
        cell[k] = mix(identity[k], fitted, blend);
    }

    // A tap's angle turns R toward N in the plane of incidence: P is the unit vector
    // there perpendicular to R on N's side, so a tap is `cos a R + sin a P`.
    let toward_N = N - dot(N, R) * R;
    let P = toward_N * inverseSqrt(max(dot(toward_N, toward_N), 1e-12));

    var angles = vec2(cell[1].y, 0.0);
    var roughnesses = vec2(cell[1].z, 0.0);
    var weights = vec2(1.0, 0.0);
    if FITTED_LOBE_TAPS == 2u {
        angles = cell[0].xz;
        roughnesses = cell[0].yw;
        weights = vec2(cell[1].x, 1.0 - cell[1].x);
    }
    var radiance = vec3(0.0);
    for (var k = 0u; k < FITTED_LOBE_TAPS; k++) {
        let roughness = saturate(roughnesses[k]);
        let tap = cos(angles[k]) * R + sin(angles[k]) * P;
        radiance += weights[k] * horizon_clipped_radiance(map, N, tap, roughness,
            specular_radiance_level(roughness, max_specular_mip), FITTED_LOBE_HORIZON_CLIP,
            world_position, light_from_world, bounds, parallax_correct, rotation);
    }
    return radiance;
}

fn stretched_split_sum_radiance(
    map: texture_cube<f32>,
    input: LayerLightingInput,
    max_specular_mip: f32,
    world_position: vec3<f32>,
    light_from_world: mat4x4<f32>,
    bounds: vec3<f32>,
    parallax_correct: bool,
    rotation: vec4<f32>,
) -> vec3<f32> {
    let N = input.N;
    let alpha = input.roughness;
    let NdotV = clamp(input.NdotV, 1e-4, 1.0);
    let center = radiance_sample_direction_ggx_split_sum(N, input.R, alpha);

    // In the plane of N and the lookup direction, perpendicular to the lookup.
    let toward_normal = N - dot(N, center) * center;
    let toward_normal_length_squared = dot(toward_normal, toward_normal);

    let sigma_wide = STRETCH_LOBE_SCALE * alpha;
    let sigma_narrow = sigma_wide * NdotV;
    let spread = sqrt(max(sigma_wide * sigma_wide - sigma_narrow * sigma_narrow, 0.0));
    // The mip must also blur at least half the tap spacing, or the taps band.
    // Taps span ±2 standard deviations with Gaussian weights.
    let spacing = 4.0 * spread / f32(STRETCH_TAPS - 1u);
    let level_alpha = max(sigma_narrow, 0.5 * spacing) / STRETCH_LOBE_SCALE;
    let level = clamp(sqrt(level_alpha) * max_specular_mip, 0.0, max_specular_mip);

    if toward_normal_length_squared < 1e-8 || spread < 1e-4 {
        let direction = compute_cubemap_sample_dir(
            world_position, center, light_from_world, bounds, parallax_correct, rotation);
        return textureSampleLevel(map, bindings::environment_map_sampler, direction,
            specular_radiance_level(input.perceptual_roughness, max_specular_mip)).rgb;
    }
    let tangent = toward_normal * inverseSqrt(toward_normal_length_squared);

    var sum = vec3(0.0);
    var weight_sum = 0.0;
    for (var i = 0u; i < STRETCH_TAPS; i++) {
        let u = (f32(i) / f32(STRETCH_TAPS - 1u) * 2.0 - 1.0) * 2.0 * spread;
        let weight = exp(-0.5 * u * u / max(spread * spread, 1e-12));
        let direction = compute_cubemap_sample_dir(world_position, normalize(center + u * tangent),
            light_from_world, bounds, parallax_correct, rotation);
        sum += weight * textureSampleLevel(map, bindings::environment_map_sampler, direction, level).rgb;
        weight_sum += weight;
    }
    return sum / weight_sum;
}

// Integrate both Schlick coefficients so Fresnel remains inside the integral.
// Bounded visible-normal sampling and its matching reflected-direction PDF are
// shared with the renderer. Below-horizon samples contribute zero, not a
// renormalized weight. The correlated Smith BRDF matches the pathtracer.
fn filtered_importance_radiance(
    map: texture_cube<f32>,
    input: LayerLightingInput,
    world_position: vec3<f32>,
    light_from_world: mat4x4<f32>,
    bounds: vec3<f32>,
    parallax_correct: bool,
    rotation: vec4<f32>,
) -> mat2x3<f32> {
    let N = input.N;
    let V = normalize(2.0 * input.NdotV * N - input.R);
    let NoV = max(dot(N, V), 0.0001);
    let alpha = max(input.roughness, 0.001);
    let local_V = vec3(sqrt(max(0.0, 1.0 - NoV * NoV)), 0.0, NoV);
    let size = f32(textureDimensions(map, 0).x);
    let texel_solid_angle = 12.56637061436 / (6.0 * size * size);
    var a = vec3(0.0);
    var b = vec3(0.0);
    for (var i = 0u; i < FIS_SAMPLES; i++) {
        let xi = vec2((f32(i) + 0.5) / f32(FIS_SAMPLES),
            f32(reverseBits(i)) * 2.3283064365386963e-10);
        let L = sample_visible_ggx(xi, alpha, N, V);
        let NoL = dot(N, L);
        if NoL <= 0.0 { continue; }
        let H = normalize(V + L);
        let NoH = max(dot(N, H), 0.0);
        let pdf = max(ggx_vndf_pdf(local_V, NoH, alpha), 1e-8);
        let sample_solid_angle = 1.0 / (f32(FIS_SAMPLES) * pdf);
        let lod = clamp(0.5 * log2(sample_solid_angle / texel_solid_angle),
            0.0, f32(textureNumLevels(map) - 1u));
        let direction = compute_cubemap_sample_dir(world_position, L,
            light_from_world, bounds, parallax_correct, rotation);
        let Li = textureSampleLevel(map, bindings::environment_map_sampler, direction, lod).rgb;
        let weight = D_GGX(alpha, NoH) * V_SmithGGXCorrelated(alpha, NoV, NoL) * NoL / pdf;
        let Fc = pow(1.0 - saturate(dot(V, H)), 5.0);
        a += Li * (weight * (1.0 - Fc));
        b += Li * (weight * Fc);
    }
    return mat2x3(a / f32(FIS_SAMPLES), b / f32(FIS_SAMPLES));
}

/// Split-sum weight with the same multiple-scattering compensation as
/// `specular_multiscatter` and the pathtracer. Energy missing from the single
/// scatter is scaled back onto this lobe.
///
/// That energy used to be added as `FmsEms * irradiance` (Kulla–Conty). Irradiance
/// is a cosine lobe, so a small HDR source grew a halo a GGX integral does not have.
fn compensated_split_sum(F0: vec3<f32>, F_ab: vec2<f32>, specular_occlusion: f32) -> vec3<f32> {
    let ess = max(F_ab.x + F_ab.y, 1e-4);
    let single_scatter = (F0 * F_ab.x + F_ab.y) * specular_occlusion;
    let compensation = 1.0 + F0 * (1.0 / ess - 1.0);
    return single_scatter * compensation;
}

/// Computes the direction at which to sample the reflection probe.
///
/// * `light_from_world` is the matrix that transforms world space into light
///   probe space (a 1×1×1 cube centered on the origin).
///
/// * `parallax_correction_bounds` is the half-extents of the simulated
///   reflection boundaries used for parallax correction, in light probe space.
///   It's ignored if the `parallax_correct` parameter is false.
///
/// * `parallax_correct` is true if parallax correction is to be applied and
///   false otherwise.
fn compute_cubemap_sample_dir(
    world_ray_origin: vec3<f32>,
    world_ray_direction: vec3<f32>,
    light_from_world: mat4x4<f32>,
    parallax_correction_bounds: vec3<f32>,
    parallax_correct: bool,
    view_rotation: vec4<f32>,
) -> vec3<f32> {
    var sample_dir: vec3<f32>;

    // If we're supposed to parallax correct, then intersect with the light cube.
    if (parallax_correct) {
        // Compute the direction of the ray bouncing off the surface, in light
        // probe space.
        // Recall that light probe space is a 1×1×1 cube centered at the origin.
        let ray_origin = (light_from_world * vec4(world_ray_origin, 1.0)).xyz;
        let ray_direction = (light_from_world * vec4(world_ray_direction, 0.0)).xyz;

        // Solve for the intersection of that ray with each side of the cube.
        // Since our light probe is a 1×1×1 cube centered at the origin in light
        // probe space, the faces of the cube are at X = ±0.5, Y = ±0.5, and Z =
        // ±0.5.
        var t0 = (-parallax_correction_bounds - ray_origin) / ray_direction;
        var t1 = (parallax_correction_bounds - ray_origin) / ray_direction;

        // We're shooting the rays forward, so we need to rule out negative time
        // values. So, if t is negative, make it a large value so that we won't
        // choose it below.
        // We would use infinity here but WGSL forbids it:
        // https://github.com/gfx-rs/wgpu/issues/5515
        t0 = select(vec3(FLOAT_MAX), t0, t0 >= vec3(0.0));
        t1 = select(vec3(FLOAT_MAX), t1, t1 >= vec3(0.0));

        // Choose the minimum valid time value to find the intersection of the
        // first cube face.
        let t_min = min(t0, t1);
        let t = min(min(t_min.x, t_min.y), t_min.z);

        // Compute the sample direction. (It doesn't have to be normalized.)
        sample_dir = ray_origin + ray_direction * t;
    } else {
        // We treat the reflection as infinitely far away in the non-parallax
        // case, so the ray origin is irrelevant.
        sample_dir = (light_from_world * vec4(world_ray_direction, 0.0)).xyz;
    }

    // Rotating the world space ray direction by the environment map light view rotation,
    // it is equivalent to rotating the environment cubemap itself.
    sample_dir = quat_rotate(view_rotation, sample_dir);

    // Cubemaps are left-handed, so we negate the Z coordinate.
    sample_dir.z = -sample_dir.z;
    return sample_dir;
}

// Mip level for a prefiltered specular cubemap. Level `i` stores perceptual
// roughness `i / maxMip` (Karis 2013; the same linear schedule the Manson–Sloan
// filter writes). α = perceptual² is used only inside the GGX convolution.
fn specular_radiance_level(perceptual_roughness: f32, max_specular_mip: f32) -> f32 {
    return perceptual_roughness * max_specular_mip;
}

// GGX split-sum dominant reflection direction (Frostbite Listing 22).
// Used for both prefiltered and MS-generated specular maps — both store radiance
// indexed by direction, so the sample direction must match.
fn radiance_sample_direction_ggx_split_sum(N: vec3<f32>, R: vec3<f32>, mip_roughness: f32) -> vec3<f32> {
    let smoothness = saturate(1.0 - mip_roughness);
    let lerp_factor = smoothness * (sqrt(smoothness) + mip_roughness);
    return mix(N, R, lerp_factor);
}

// Define two versions of this function, one for the case in which there are
// multiple light probes and one for the case in which only the view light probe
// is present.

@if(MULTIPLE_LIGHT_PROBES_IN_ARRAY)

fn compute_radiances(
    input: LayerLightingInput,
    clusterable_object_index_ranges: ptr<function, ClusterableObjectIndexRanges>,
    world_position: vec3<f32>,
    found_diffuse_indirect: bool,
) -> EnvironmentMapRadiances {
    // Unpack.
    let N = input.N;
    let R = input.R;
    let perceptual_roughness = input.perceptual_roughness;
    let roughness = input.roughness;

    var radiances: EnvironmentMapRadiances;
    radiances.fresnel_radiance = vec3(0.0);

    // Find all reflection probes that contain the fragment. We're going to
    // accumulate all the radiance and irradiance from them in a weighted sum.
    var iterator = light_probe_iterator_new(
        world_position,
        /*is_irradiance_volume=*/ false,
        clusterable_object_index_ranges,
    );

    var total_weight = 0.0;
    var fis_a = vec3(0.0);
    radiances.irradiance = vec3(0.0);
    radiances.radiance = vec3(0.0);

    while (true) {
        var query_result = light_probe_iterator_next(&iterator);
        var view_rotation = vec4<f32>(0.0, 0.0, 0.0, 1.0);

        // If we reached the end of the light probe list, and we didn't find
        // enough reflection probes to reach a weight of 1.0, use the view
        // environment map if applicable. This allows for e.g. nice transitions
        // between the interior of a building and the outdoor environment map.
        if (query_result.texture_index < 0 && total_weight < 0.9999) {
            query_result.texture_index = light_probes.view_cubemap_index;
            query_result.intensity = light_probes.intensity_for_view;
            view_rotation = light_probes.view_rotation;
            query_result.light_from_world = mat4x4(
                vec4(1.0, 0.0, 0.0, 0.0),
                vec4(0.0, 1.0, 0.0, 0.0),
                vec4(0.0, 0.0, 1.0, 0.0),
                vec4(0.0, 0.0, 0.0, 1.0)
            );
            query_result.parallax_correction_bounds = vec3(0.0);
            query_result.flags = 0u;
            if light_probes.view_environment_map_affects_lightmapped_mesh_diffuse != 0u {
                query_result.flags |= LIGHT_PROBE_FLAG_AFFECTS_LIGHTMAPPED_MESH_DIFFUSE;
            }
            if light_probes.view_manson_sloan_environment_ibl != 0u {
                query_result.flags |= LIGHT_PROBE_FLAG_MANSON_SLOAN_ENVIRONMENT_IBL;
            }
            query_result.weight = 1.0 - total_weight;
        }

        // If we reached the end, we're done.
        if (query_result.texture_index < 0) {
            break;
        }

        // Split-sum approximation for image based lighting: https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
        let max_specular_mip = f32(textureNumLevels(
            bindings::specular_environment_maps[query_result.texture_index]) - 1u);
        // Both glTF-IBL-Sampler (prefiltered) and Esoterica/MS-generated maps encode mip `i`
        // as perceptual roughness `i/(n-1)` (α = perceptual² used only inside the GGX filter).
        let radiance_level = specular_radiance_level(perceptual_roughness, max_specular_mip);

        // If we're lightmapped, and we shouldn't accumulate diffuse light from the
        // environment map, note that.
        var enable_diffuse = !found_diffuse_indirect;
        @if(LIGHTMAP)
        enable_diffuse = enable_diffuse &&
            (query_result.flags & LIGHT_PROBE_FLAG_AFFECTS_LIGHTMAPPED_MESH_DIFFUSE) != 0u;

        let parallax_correct = (query_result.flags & LIGHT_PROBE_FLAG_PARALLAX_CORRECT) != 0u;

        if (enable_diffuse) {
            let irradiance_sample_dir = compute_cubemap_sample_dir(
                world_position,
                N,
                query_result.light_from_world,
                query_result.parallax_correction_bounds,
                parallax_correct,
                view_rotation,
            );
            radiances.irradiance += textureSampleLevel(
                bindings::diffuse_environment_maps[query_result.texture_index],
                bindings::environment_map_sampler,
                irradiance_sample_dir,
                0.0).rgb * query_result.intensity * query_result.weight;
        }

        var radiance_sample_dir =
            radiance_sample_direction_ggx_split_sum(N, R, roughness);
        radiance_sample_dir = compute_cubemap_sample_dir(
            world_position,
            radiance_sample_dir,
            query_result.light_from_world,
            query_result.parallax_correction_bounds,
            parallax_correct,
            view_rotation,
        );
        // The fitted or stretched lookups replace the round one, so it isn't read.
        var specular: vec3<f32>;
        if FITTED_LOBE_TAPS > 0u {
            specular = fitted_lobe_radiance(
                bindings::specular_environment_maps[query_result.texture_index], input,
                max_specular_mip, world_position, query_result.light_from_world,
                query_result.parallax_correction_bounds, parallax_correct, view_rotation);
        } else if SPLIT_SUM_HORIZON_CLIP {
            specular = horizon_clipped_split_sum_radiance(
                bindings::specular_environment_maps[query_result.texture_index], input,
                max_specular_mip, world_position, query_result.light_from_world,
                query_result.parallax_correction_bounds, parallax_correct, view_rotation);
        } else if STRETCHED_SPLIT_SUM {
            specular = stretched_split_sum_radiance(
                bindings::specular_environment_maps[query_result.texture_index], input,
                max_specular_mip, world_position, query_result.light_from_world,
                query_result.parallax_correction_bounds, parallax_correct, view_rotation);
        } else {
            specular = textureSampleLevel(
                bindings::specular_environment_maps[query_result.texture_index],
                bindings::environment_map_sampler,
                radiance_sample_dir,
                radiance_level).rgb;
        }
        radiances.radiance += specular * query_result.intensity * query_result.weight;

        if SHADE_TIME_FIS {
            let filtered = filtered_importance_radiance(
                bindings::specular_environment_maps[query_result.texture_index], input,
                world_position, query_result.light_from_world,
                query_result.parallax_correction_bounds, parallax_correct, view_rotation);
            // Replace this probe's split-sum contribution below by accumulating
            // the independent Schlick integrals in a separate sum.
            fis_a += filtered[0] * query_result.intensity * query_result.weight;
            radiances.fresnel_radiance += filtered[1] * query_result.intensity * query_result.weight;
        }

        total_weight += query_result.weight;
    }

    if SHADE_TIME_FIS { radiances.radiance = fis_a; }
    if (total_weight != 0.0) {
        radiances.fresnel_radiance /= total_weight;
        radiances.irradiance /= total_weight;
        radiances.radiance /= total_weight;
    }

    return radiances;
}

@else   // MULTIPLE_LIGHT_PROBES_IN_ARRAY

fn compute_radiances(
    input: LayerLightingInput,
    clusterable_object_index_ranges: ptr<function, ClusterableObjectIndexRanges>,
    world_position: vec3<f32>,
    found_diffuse_indirect: bool,
) -> EnvironmentMapRadiances {
    // Unpack.
    let N = input.N;
    let R = input.R;
    let perceptual_roughness = input.perceptual_roughness;
    let roughness = input.roughness;

    var radiances: EnvironmentMapRadiances;
    radiances.fresnel_radiance = vec3(0.0);

    // If we have no light probe, bail.
    if (light_probes.view_cubemap_index < 0) {
        radiances.irradiance = vec3(0.0);
        radiances.radiance = vec3(0.0);
        return radiances;
    }

    // Split-sum approximation for image based lighting: https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
    // Technically we could use textureNumLevels(specular_environment_map) - 1 here, but we use a uniform
    // because textureNumLevels() does not work on WebGL2.
    // Same mip index as the multi-probe path: perceptual roughness times the last mip.
    let max_specular_mip = f32(light_probes.smallest_specular_mip_level_for_view);
    let radiance_level = specular_radiance_level(perceptual_roughness, max_specular_mip);

    let intensity = light_probes.intensity_for_view;

    // If we're lightmapped, and we shouldn't accumulate diffuse light from the
    // environment map, note that.
    var enable_diffuse = !found_diffuse_indirect;
    @if(LIGHTMAP)
    enable_diffuse = enable_diffuse &&
        light_probes.view_environment_map_affects_lightmapped_mesh_diffuse;

    if (enable_diffuse) {
        var irradiance_sample_dir = N;
        // Rotating the world space ray direction by the environment map light view rotation,
        // it is equivalent to rotating the environment cubemap itself.
        irradiance_sample_dir = quat_rotate(light_probes.view_rotation, irradiance_sample_dir);
        // Cube maps are left-handed so we negate the z coordinate.
        irradiance_sample_dir.z = -irradiance_sample_dir.z;
        radiances.irradiance = textureSampleLevel(
            bindings::diffuse_environment_map,
            bindings::environment_map_sampler,
            irradiance_sample_dir,
            0.0).rgb * intensity;
    }

    var radiance_sample_dir =
        radiance_sample_direction_ggx_split_sum(N, R, roughness);
    // Rotating the world space ray direction by the environment map light view rotation,
    // it is equivalent to rotating the environment cubemap itself.
    radiance_sample_dir = quat_rotate(light_probes.view_rotation, radiance_sample_dir);
    // Cube maps are left-handed so we negate the z coordinate.
    radiance_sample_dir.z = -radiance_sample_dir.z;
    radiances.radiance = textureSampleLevel(
        bindings::specular_environment_map,
        bindings::environment_map_sampler,
        radiance_sample_dir,
        radiance_level).rgb * intensity;

    if SPLIT_SUM_HORIZON_CLIP {
        radiances.radiance = horizon_clipped_split_sum_radiance(bindings::specular_environment_map,
            input, max_specular_mip, world_position, mat4x4<f32>(
                vec4(1.0, 0.0, 0.0, 0.0), vec4(0.0, 1.0, 0.0, 0.0),
                vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0)),
            vec3(0.0), false, light_probes.view_rotation) * intensity;
    }

    if STRETCHED_SPLIT_SUM {
        radiances.radiance = stretched_split_sum_radiance(bindings::specular_environment_map,
            input, max_specular_mip, world_position, mat4x4<f32>(
                vec4(1.0, 0.0, 0.0, 0.0), vec4(0.0, 1.0, 0.0, 0.0),
                vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0)),
            vec3(0.0), false, light_probes.view_rotation) * intensity;
    }

    if FITTED_LOBE_TAPS > 0u {
        radiances.radiance = fitted_lobe_radiance(bindings::specular_environment_map,
            input, max_specular_mip, world_position, mat4x4<f32>(
                vec4(1.0, 0.0, 0.0, 0.0), vec4(0.0, 1.0, 0.0, 0.0),
                vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0)),
            vec3(0.0), false, light_probes.view_rotation) * intensity;
    }

    if SHADE_TIME_FIS {
        let filtered = filtered_importance_radiance(bindings::specular_environment_map,
            input, world_position, mat4x4<f32>(
                vec4(1.0, 0.0, 0.0, 0.0), vec4(0.0, 1.0, 0.0, 0.0),
                vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0)),
            vec3(0.0), false, light_probes.view_rotation);
        radiances.radiance = filtered[0] * intensity;
        radiances.fresnel_radiance = filtered[1] * intensity;
    }
    return radiances;
}

/// Disk lights extracted from the environment maps
/// (`EnvironmentMapSourceOutput::Shading`), weighted like the maps they came
/// from. Their directions are in each probe's space, so they are taken back to
/// world space through the probe's rotation; parallax correction does not
/// apply, since the sources are distant.
@if(MULTIPLE_LIGHT_PROBES_IN_ARRAY && AREA_LIGHT_LUTS)
fn environment_map_sources_light(
    input: ptr<function, LightingInput>,
    clusterable_object_index_ranges: ptr<function, ClusterableObjectIndexRanges>,
    enable_diffuse: bool,
) -> vec3<f32> {
    let N = (*input).layers[LAYER_BASE].N;
    let V = (*input).V;
    var iterator = light_probe_iterator_new(
        (*input).P,
        /*is_irradiance_volume=*/ false,
        clusterable_object_index_ranges,
    );
    var total_weight = 0.0;
    var light = vec3(0.0);
    while (true) {
        var query_result = light_probe_iterator_next(&iterator);
        var view_rotation = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        var light_from_world = query_result.light_from_world;
        if (query_result.texture_index < 0 && total_weight < 0.9999) {
            query_result.texture_index = light_probes.view_cubemap_index;
            query_result.intensity = light_probes.intensity_for_view;
            view_rotation = light_probes.view_rotation;
            light_from_world = mat4x4(
                vec4(1.0, 0.0, 0.0, 0.0),
                vec4(0.0, 1.0, 0.0, 0.0),
                vec4(0.0, 0.0, 1.0, 0.0),
                vec4(0.0, 0.0, 0.0, 1.0)
            );
            query_result.weight = 1.0 - total_weight;
        }
        if (query_result.texture_index < 0) {
            break;
        }
        total_weight += query_result.weight;

        let probe_sources = &bindings::environment_map_sources[query_result.texture_index];
        let count = min((*probe_sources).count, 4u);
        // `compute_cubemap_sample_dir` maps world to probe space as
        // `rotate(view_rotation, light_from_world * d)`; invert that.
        let world_from_light = transpose(mat3x3(
            normalize(light_from_world[0].xyz),
            normalize(light_from_world[1].xyz),
            normalize(light_from_world[2].xyz),
        ));
        let inverse_view_rotation = vec4(-view_rotation.xyz, view_rotation.w);
        for (var i = 0u; i < count; i++) {
            let source = (*probe_sources).sources[i];
            let L = normalize(world_from_light * quat_rotate(inverse_view_rotation, source.direction));
            let radius = 0.5 * source.angular_size;
            let cos_r = cos(radius);
            let sin_r = max(sin(radius), 1e-4);
            let illuminance = source.illuminance * query_result.intensity * query_result.weight;
            var color = directional_disk_specular(input, LAYER_BASE, L, cos_r, sin_r);
            if enable_diffuse {
                var derived = derive_lighting_input(N, V, L);
                color += (*input).diffuse_color * Fd_Burley(input, &derived)
                    * directional_disk_diffuse_cosine(dot(N, L), sin_r);
            }
            light += color * illuminance;
        }
    }
    if total_weight != 0.0 {
        light /= total_weight;
    }
    return light;
}

@if(STANDARD_MATERIAL_CLEARCOAT)

/// Adds the environment map light from the clearcoat layer to that of the base
/// layer.
fn environment_map_light_clearcoat(
    out: ptr<function, EnvironmentMapLight>,
    input: ptr<function, LightingInput>,
    clusterable_object_index_ranges: ptr<function, ClusterableObjectIndexRanges>,
    found_diffuse_indirect: bool,
) {
    // Unpack.
    let world_position = (*input).P;
    let clearcoat_NdotV = (*input).layers[LAYER_CLEARCOAT].NdotV;
    let clearcoat_strength = (*input).clearcoat_strength;

    // Calculate the Fresnel term `Fc` for the clearcoat layer.
    // 0.04 is a hardcoded value for F0 from the Filament spec.
    let clearcoat_F0 = vec3<f32>(0.04);
    let Fc = F_Schlick_vec(clearcoat_F0, 1.0, clearcoat_NdotV) * clearcoat_strength;
    let inv_Fc = 1.0 - Fc;

    let clearcoat_radiances = compute_radiances(
        (*input).layers[LAYER_CLEARCOAT],
        clusterable_object_index_ranges,
        world_position,
        found_diffuse_indirect,
    );

    // Composite the clearcoat layer on top of the existing one.
    // These formulas are from Filament:
    // <https://google.github.io/filament/Filament.md.html#lighting/imagebasedlights/clearcoat>
    (*out).diffuse *= inv_Fc;
    if SHADE_TIME_FIS {
        (*out).specular = (*out).specular * inv_Fc * inv_Fc
            + (clearcoat_F0 * clearcoat_radiances.radiance + clearcoat_radiances.fresnel_radiance) * clearcoat_strength;
        return;
    }
    (*out).specular = (*out).specular * inv_Fc * inv_Fc + clearcoat_radiances.radiance * Fc;
}

fn environment_map_light(
    input: ptr<function, LightingInput>,
    clusterable_object_index_ranges: ptr<function, ClusterableObjectIndexRanges>,
    found_diffuse_indirect: bool,
) -> EnvironmentMapLight {
    // Unpack.
    let diffuse_color = (*input).diffuse_color;
    let metallic = (*input).metallic;
    let F_ab = (*input).F_ab;
    let F0_dielectric = (*input).F0_dielectric;
    let F0_metallic = (*input).F0_metallic;
    let world_position = (*input).P;

    var out: EnvironmentMapLight;

    let radiances = compute_radiances(
        (*input).layers[LAYER_BASE],
        clusterable_object_index_ranges,
        world_position,
        found_diffuse_indirect,
    );

    var sources_light = vec3(0.0);
    @if(MULTIPLE_LIGHT_PROBES_IN_ARRAY && AREA_LIGHT_LUTS)
    sources_light = environment_map_sources_light(input, clusterable_object_index_ranges, !found_diffuse_indirect);

    // A residual can be black once every source is extracted.
    if (all(radiances.irradiance == vec3(0.0)) && all(radiances.radiance == vec3(0.0))) {
        out.diffuse = vec3(0.0);
        out.specular = sources_light;
        return out;
    }

    // No real world material has specular values under 0.02, so we use this range as a
    // "pre-baked specular occlusion" that extinguishes the fresnel term, for artistic control.
    // See: https://google.github.io/filament/Filament.md.html#specularocclusion
    let F0_surface = mix(F0_dielectric, F0_metallic, metallic);
    let specular_occlusion = saturate(dot(F0_surface, vec3(50.0 * 0.33)));

    // Compensation is nonlinear in F0, so dielectric and metallic are evaluated
    // separately and then mixed, matching `lobe_reflectances` in the pathtracer.
    let specular_dielectric = compensated_split_sum(F0_dielectric, F_ab, specular_occlusion);
    let specular_metallic = compensated_split_sum(F0_metallic, F_ab, specular_occlusion);
    out.specular = mix(specular_dielectric, specular_metallic, metallic) * radiances.radiance;

    if SHADE_TIME_FIS {
        let compensation_dielectric = 1.0 + F0_dielectric * (1.0 / max(F_ab.x + F_ab.y, 1e-4) - 1.0);
        let compensation_metallic = 1.0 + F0_metallic * (1.0 / max(F_ab.x + F_ab.y, 1e-4) - 1.0);
        out.specular = mix(
            (F0_dielectric * radiances.radiance + radiances.fresnel_radiance) * compensation_dielectric,
            (F0_metallic * radiances.radiance + radiances.fresnel_radiance) * compensation_metallic,
            metallic) * specular_occlusion;
    }

    // `diffuse_color` already includes `(1 - metallic)`. Removing the compensated
    // dielectric specular albedo keeps a constant environment equal to the base color.
    if (!found_diffuse_indirect) {
        out.diffuse = diffuse_color * saturate(vec3(1.0) - specular_dielectric) * radiances.irradiance;
    } else {
        out.diffuse = vec3(0.0);
    }

    // Extracted sources: full BRDF, so they go straight into the specular term,
    // below the clearcoat like the rest of the environment.
    out.specular += sources_light;

    @if(STANDARD_MATERIAL_CLEARCOAT)
    environment_map_light_clearcoat(
        &out,
        input,
        clusterable_object_index_ranges,
        found_diffuse_indirect,
    );

    return out;
}
