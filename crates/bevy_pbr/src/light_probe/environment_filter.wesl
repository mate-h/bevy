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

/// Filtered-importance-sampling LOD (Krivánek–Colbert / glTF-IBL-Sampler).
/// Matches offline Lambertian prefilters so HDR peaks don't firefly into a bright wrap.
fn irradiance_sample_lod(pdf: f32, src_width: f32, sample_count: f32, max_lod: f32) -> f32 {
    let safe_pdf = max(pdf, 1e-6);
    let lod = 0.5 * log2(6.0 * src_width * src_width / (sample_count * safe_pdf));
    return clamp(lod, 0.0, max_lod);
}

@compute
@workgroup_size(8, 8, 1)
fn generate_irradiance_map(@builtin(global_invocation_id) global_id: vec3u) {
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
