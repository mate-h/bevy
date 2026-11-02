import package::render::utils::{sample_cosine_hemisphere, dir_to_cube_uv, sample_cube_dir};
import bevy_render::maths::PI;

struct FilteringConstants {
    mip_level: f32,
    sample_count: u32,
    perceptual_roughness: f32,
    lod_resolution_bias: f32,
    output_size: vec2u,
    noise_size_bits: vec2u,
}

// EsotericaFilterFitter const_8 cubemap table: 7 * 5 * 1 * 6 float4s.
const MS_POLY_VEC4S: u32 = 210u;
const NUM_MS_TAPS_DIV_4: u32 = 2u;
const MS_NUM_INDICES: u32 = 6u;

@group(0) @binding(0) var input_texture: texture_2d_array<f32>;
@group(0) @binding(1) var input_sampler: sampler;
@group(0) @binding(2) var output_texture: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(3) var<uniform> constants: FilteringConstants;
@group(0) @binding(4) var blue_noise_texture: texture_2d_array<f32>;
@group(0) @binding(5) var<storage, read> ms_poly_table: array<vec4f, MS_POLY_VEC4S>;

/// Bevy samples environment cubemaps left-handed (`dir.z = -dir.z` at runtime).
/// Filtering must use the same mapping so generated maps match prefiltered assets.
fn world_to_cubemap_dir(dir: vec3f) -> vec3f {
    return vec3f(dir.x, dir.y, -dir.z);
}

/// Sample an environment map with a specific LOD (world-space direction).
fn sample_environment(world_dir: vec3f, level: f32) -> vec4f {
    let cube_uv = dir_to_cube_uv(world_to_cubemap_dir(world_dir));
    return textureSampleLevel(input_texture, input_sampler, cube_uv.uv, cube_uv.face, level);
}

/// World-space direction represented by this output texel.
/// Runtime looks up `world_to_cubemap_dir(world)`, so this texel must store F(world).
fn output_texel_world_dir(uv: vec2f, face: u32) -> vec3f {
    return world_to_cubemap_dir(sample_cube_dir(uv, face));
}

fn dir_axis(v: vec3f, ax: u32) -> f32 {
    switch ax {
        case 0u: { return v.x; }
        case 1u: { return v.y; }
        default: { return v.z; }
    }
}

// Unnormalized face direction for this cubemap texel (texture / hardware space).
fn cube_face_dir_unorm(uv: vec2f, face: u32) -> vec3f {
    let uvc = 2.0 * uv - 1.0;
    switch face {
        case 0u: { return vec3f(1.0, -uvc.y, -uvc.x); }
        case 1u: { return vec3f(-1.0, -uvc.y, uvc.x); }
        case 2u: { return vec3f(uvc.x, 1.0, uvc.y); }
        case 3u: { return vec3f(uvc.x, -1.0, -uvc.y); }
        case 4u: { return vec3f(uvc.x, -uvc.y, 1.0); }
        case 5u: { return vec3f(-uvc.x, -uvc.y, -1.0); }
        default: { return vec3f(1.0, 0.0, 0.0); }
    }
}

// const_8 layout: float4[level][band][index]
fn ms_poly(level: u32, band: u32, ix: u32) -> vec4f {
    return ms_poly_table[(level * 5u + band) * MS_NUM_INDICES + ix];
}

// Interpolate coefficients between adjacent table rows (linear perceptual roughness).
fn ms_poly_lerp(table_t: f32, band: u32, ix: u32) -> vec4f {
    let lo = u32(clamp(floor(table_t), 0.0, 6.0));
    let hi = u32(clamp(ceil(table_t), 0.0, 6.0));
    let w = table_t - f32(lo);
    return mix(ms_poly(lo, band, ix), ms_poly(hi, band, ix), w);
}

@compute
@workgroup_size(8, 8, 1)
fn generate_radiance_map(@builtin(global_invocation_id) global_id: vec3u) {
    let size = max(constants.output_size, vec2u(1u));
    let inv_size = 1.0 / vec2f(size);

    let coords = vec2u(global_id.xy);
    let face = global_id.z;

    if (any(coords >= size)) {
        return;
    }

    let uv = (vec2f(coords) + 0.5) * inv_size;
    let normal = output_texel_world_dir(uv, face);
    let perceptual_roughness = constants.perceptual_roughness;

    if (perceptual_roughness < 0.001) {
        let radiance = sample_environment(normal, 0.0).rgb;
        textureStore(output_texture, coords, face, vec4f(radiance, 1.0));
        return;
    }

    let table_t = clamp(perceptual_roughness * 6.0, 0.0, 6.0);
    let max_lod = f32(textureNumLevels(input_texture) - 1u);

    // MS frames are built in world space (Z-flipped from hardware face dirs).
    let dir = world_to_cubemap_dir(cube_face_dir_unorm(uv, face));
    let ad = abs(dir);

    var color_rgb = vec3f(0.0);
    var color_w = 0.0;

    for (var axis = 0u; axis < 3u; axis++) {
        let other_axis0 = 1u - (axis & 1u) - (axis >> 1u);
        let other_axis1 = 2u - (axis >> 1u);
        let a0 = dir_axis(ad, other_axis0);
        let a1 = dir_axis(ad, other_axis1);
        var frameweight = (max(a0, a1) - 0.75) / 0.25;
        if (frameweight <= 0.0) {
            continue;
        }

        var upv = vec3f(0.0);
        switch axis {
            case 0u: { upv = vec3f(1.0, 0.0, 0.0); }
            case 1u: { upv = vec3f(0.0, 1.0, 0.0); }
            default: { upv = vec3f(0.0, 0.0, 1.0); }
        }

        let frame_z = normalize(dir);
        var frame_x = cross(upv, frame_z);
        let x_len = length(frame_x);
        if (x_len < 1e-20) {
            continue;
        }
        frame_x = frame_x / x_len;
        let frame_y = cross(frame_z, frame_x);

        for (var i_super = 0u; i_super < NUM_MS_TAPS_DIV_4; i_super++) {
            let coeff_ix = NUM_MS_TAPS_DIV_4 * axis + i_super;

            let c0d = ms_poly_lerp(table_t, 0u, coeff_ix);
            let c1d = ms_poly_lerp(table_t, 1u, coeff_ix);
            let c2d = ms_poly_lerp(table_t, 2u, coeff_ix);
            let c_l = ms_poly_lerp(table_t, 3u, coeff_ix);
            let c_w = ms_poly_lerp(table_t, 4u, coeff_ix);

            for (var i_sub = 0u; i_sub < 4u; i_sub++) {
                var sample_dir = frame_x * c0d[i_sub] + frame_y * c1d[i_sub] + frame_z * c2d[i_sub];
                var sample_level = c_l[i_sub];
                var sample_weight = c_w[i_sub] * frameweight;

                let ax = max(abs(sample_dir.x), max(abs(sample_dir.y), abs(sample_dir.z)));
                sample_dir = sample_dir / ax;
                sample_level += 0.75 * log2(dot(sample_dir, sample_dir));
                sample_level = clamp(sample_level + constants.lod_resolution_bias, 0.0, max_lod);

                let radiance_s = sample_environment(sample_dir, sample_level).rgb;
                color_rgb += radiance_s * sample_weight;
                color_w += sample_weight;
            }
        }
    }

    var radiance = vec3f(0.0);
    if (color_w > 1e-20) {
        radiance = color_rgb / color_w;
    } else {
        radiance = sample_environment(normal, max_lod).rgb;
    }

    textureStore(output_texture, coords, face, vec4f(max(radiance, vec3f(0.0)), 1.0));
}

/// Filtered-importance-sampling LOD (Krivánek–Colbert, Karis footprint).
/// `0.5 * log2(sample_solid_angle / texel_solid_angle)`, the same convention as
/// shade-time FIS and the offline prefilter, so HDR peaks don't firefly.
fn irradiance_sample_lod(pdf: f32, src_width: f32, sample_count: f32, max_lod: f32) -> f32 {
    let sample_solid_angle = 1.0 / (sample_count * max(pdf, 1e-6));
    let texel_solid_angle = 4.0 * PI / (6.0 * src_width * src_width);
    let lod = 0.5 * log2(sample_solid_angle / texel_solid_angle);
    return clamp(lod, 0.0, max_lod);
}

/// Irradiance from a 9-coefficient spherical harmonic projection (Ramamoorthi &
/// Hanrahan 2001) instead of Monte Carlo samples: deterministic, and exact for
/// the cosine lobe up to the truncation of the lighting to two bands.
const IRRADIANCE_SH: bool = true;
/// Face size of the level projected onto spherical harmonics.
const IRRADIANCE_SH_SOURCE_SIZE: u32 = 32u;
/// Two bands ring around a small, very bright source such as a sun (±20% on a
/// sphere under a beach HDRI). Texels brighter than this many times the map's mean
/// luminance are clamped to it, keeping their color, before the projection, and
/// their excess is added back per output texel as exact clamped-cosine point
/// lights. 0 projects every texel.
const IRRADIANCE_SH_EXCESS_SCALE: f32 = 16.0;
/// At most this many excess texels are kept as point lights; any beyond go into the
/// projection with the rest.
const IRRADIANCE_SH_MAX_EXCESS: u32 = 64u;

var<workgroup> sh_partial: array<vec3f, 64>;
var<workgroup> excess_count: atomic<u32>;
var<workgroup> excess_direction: array<vec3f, 64>;
var<workgroup> excess_radiance: array<vec3f, 64>;

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

fn texel_corner(x: f32, y: f32) -> f32 {
    return atan2(x * y, sqrt(1.0 + x * x + y * y));
}

/// Sums `value` over the workgroup; every invocation gets the total.
fn workgroup_sum(local_index: u32, value: vec3f) -> vec3f {
    sh_partial[local_index] = value;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if local_index < stride {
            sh_partial[local_index] += sh_partial[local_index + stride];
        }
        workgroupBarrier();
    }
    let total = sh_partial[0];
    workgroupBarrier();
    return total;
}

/// Texel `t` of the projected level, as its direction and its radiance times its
/// solid angle.
fn sh_source_texel(t: u32, size: u32, level: u32) -> array<vec3f, 2> {
    let face = t / (size * size);
    let rem = t - face * size * size;
    let coords = vec2u(rem % size, rem / size);
    let inv = 2.0 / f32(size);
    let lo = vec2f(coords) * inv - 1.0;
    let hi = lo + inv;
    let solid_angle = texel_corner(hi.x, hi.y) - texel_corner(lo.x, hi.y)
        - texel_corner(hi.x, lo.y) + texel_corner(lo.x, lo.y);
    let uv = (vec2f(coords) + 0.5) / f32(size);
    return array(output_texel_world_dir(uv, face),
        textureLoad(input_texture, coords, face, level).rgb * solid_angle);
}

/// Projects the environment onto 9 SH coefficients, cosine-convolved and divided
/// by π, so `dot(coefficients, Y(n))` is the white Lambertian radiance at `n`,
/// leaving out the excess of texels above `IRRADIANCE_SH_EXCESS_SCALE`, which it
/// stores in `excess_direction` and `excess_radiance` (radiance times solid angle).
/// Every invocation of the workgroup gets the full result.
fn project_irradiance_sh(local_index: u32) -> array<vec3f, 9> {
    let base_size = textureDimensions(input_texture).x;
    let size = min(IRRADIANCE_SH_SOURCE_SIZE, base_size);
    let level = u32(round(log2(f32(base_size) / f32(size))));
    let texels = 6u * size * size;
    let luma = vec3f(0.2126, 0.7152, 0.0722);

    if local_index == 0u {
        atomicStore(&excess_count, 0u);
    }
    workgroupBarrier();
    // The clamp is on luminance times solid angle. The mean radiance comes from the
    // chain's 1×1 level, whose faces are box averages: close enough for a threshold.
    var clamp_luminance = 3.4e38;
    if IRRADIANCE_SH_EXCESS_SCALE > 0.0 {
        let last = textureNumLevels(input_texture) - 1u;
        var mean = 0.0;
        for (var face = 0u; face < 6u; face++) {
            mean += dot(textureLoad(input_texture, vec2u(0u), face, last).rgb, luma) / 6.0;
        }
        clamp_luminance = IRRADIANCE_SH_EXCESS_SCALE * mean * (4.0 * PI / f32(texels));
    }

    var partial: array<vec3f, 9>;
    for (var t = local_index; t < texels; t += 64u) {
        let texel = sh_source_texel(t, size, level);
        var radiance = texel[1];
        let luminance = dot(radiance, luma);
        if luminance > clamp_luminance {
            let clamped = radiance * (clamp_luminance / luminance);
            let slot = atomicAdd(&excess_count, 1u);
            if slot < IRRADIANCE_SH_MAX_EXCESS {
                excess_direction[slot] = texel[0];
                excess_radiance[slot] = radiance - clamped;
                radiance = clamped;
            }
        }
        let basis = sh9_basis(texel[0]);
        for (var i = 0u; i < 9u; i++) {
            partial[i] += radiance * basis[i];
        }
    }

    // Cosine lobe convolution over π: 1, 2/3 and 1/4 for bands 0, 1 and 2.
    let band_scale = array<f32, 9>(1.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.25, 0.25, 0.25, 0.25, 0.25);
    var result: array<vec3f, 9>;
    for (var i = 0u; i < 9u; i++) {
        result[i] = workgroup_sum(local_index, partial[i]) * band_scale[i];
    }
    return result;
}

@compute
@workgroup_size(8, 8, 1)
fn generate_irradiance_map(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    if IRRADIANCE_SH {
        // Every workgroup projects the same coefficients, then writes its tile.
        let coefficients = project_irradiance_sh(local_index);
        let size = max(constants.output_size, vec2u(1u));
        if any(global_id.xy >= size) {
            return;
        }
        let uv = (vec2f(global_id.xy) + 0.5) / vec2f(size);
        let normal = output_texel_world_dir(uv, global_id.z);
        let basis = sh9_basis(normal);
        var irradiance = vec3f(0.0);
        for (var i = 0u; i < 9u; i++) {
            irradiance += coefficients[i] * basis[i];
        }
        // Two bands can still ring below zero around what the clamp leaves.
        irradiance = max(irradiance, vec3f(0.0));
        let excess = min(atomicLoad(&excess_count), IRRADIANCE_SH_MAX_EXCESS);
        for (var i = 0u; i < excess; i++) {
            irradiance += excess_radiance[i] * (max(dot(normal, excess_direction[i]), 0.0) / PI);
        }
        textureStore(output_texture, global_id.xy, global_id.z, vec4f(irradiance, 1.0));
        return;
    }
    generate_irradiance_map_sampled(global_id);
}

fn generate_irradiance_map_sampled(global_id: vec3u) {
    let size = max(constants.output_size, vec2u(1u));
    let inv_size = 1.0 / vec2f(size);

    let coords = vec2u(global_id.xy);
    let face = global_id.z;

    if (any(coords >= size)) {
        return;
    }

    let uv = (vec2f(coords) + 0.5) * inv_size;
    let normal = output_texel_world_dir(uv, face);
    let src_width = f32(textureDimensions(input_texture).x);
    let max_lod = f32(textureNumLevels(input_texture) - 1u);
    let sample_count = f32(max(constants.sample_count, 1u));

    var irradiance = vec3f(0.0);

    for (var i = 0u; i < constants.sample_count; i++) {
        var rng: u32 = (coords.x * 2131358057u) ^ (coords.y * 3416869721u) ^ (face * 1199786941u) ^ (i * 566200673u);

        let sample_dir = sample_cosine_hemisphere(normal, &rng);
        // Cosine-weighted hemisphere PDF: cosθ / π.
        let cos_theta = max(dot(normal, sample_dir), 0.0);
        let pdf = cos_theta / PI;
        let lod = irradiance_sample_lod(pdf, src_width, sample_count, max_lod);

        let sample_color = sample_environment(sample_dir, lod).rgb;

        irradiance += sample_color;
    }

    irradiance = irradiance / sample_count;

    textureStore(output_texture, coords, face, vec4f(irradiance, 1.0));
}
