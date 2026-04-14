#import bevy_render::maths::PI
#import bevy_pbr::utils::{dir_to_cube_uv, sample_cube_dir, rand_vec2f}

struct FilteringConstants {
    mip_level: f32,
    sample_count: u32,
    perceptual_roughness: f32,
    lod_resolution_bias: f32,
    output_size: vec2u,
    noise_size_bits: vec2u,
}

/// Activision `coeffs_quad_32`: 7 × 5 × 3 × 24 vec4<f32>.
const MS_POLY_VEC4S: u32 = 2520u;
const NUM_MS_TAPS_DIV_4: u32 = 8u;

@group(0) @binding(0) var input_texture: texture_2d_array<f32>;
@group(0) @binding(1) var input_sampler: sampler;
@group(0) @binding(2) var output_texture: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(3) var<uniform> constants: FilteringConstants;
@group(0) @binding(4) var blue_noise_texture: texture_2d_array<f32>;
@group(0) @binding(5) var<storage, read> ms_poly_table: array<vec4f, MS_POLY_VEC4S>;

fn sample_environment(dir: vec3f, level: f32) -> vec4f {
    let cube_uv = dir_to_cube_uv(dir);
    return textureSampleLevel(input_texture, input_sampler, cube_uv.uv, cube_uv.face, level);
}

fn dir_axis(v: vec3f, ax: u32) -> f32 {
    switch ax {
        case 0u: { return v.x; }
        case 1u: { return v.y; }
        default: { return v.z; }
    }
}

// Unnormalized direction for this texel — same layout as `sample_cube_dir` (bevy_pbr::utils).
// Activision reference used +uvc.y on ±X; Bevy uses −uvc.y, so we follow Bevy or mip0 (mirror)
// and rougher mips disagree on +X/−X (vertical flip).
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

fn ms_poly(level: u32, band: u32, term: u32, ix: u32) -> vec4f {
    let stride = 24u;
    let b = level * 5u + band;
    let t = b * 3u + term;
    return ms_poly_table[t * stride + ix];
}

// Linearly interpolate polynomial coeffs between adjacent table rows. The Activision table has
// only **7** roughness rows (128²…1²); hard `round(perceptual * 6)` jumps the kernel centroid and
// reads as horizontal/vertical drift on cube faces when roughness changes. See Manson & Sloan
// EGSR 2015 §5–6 (spatially varying polynomials; one row per mip in the reference optimizer).
fn ms_poly_lerp(table_t: f32, band: u32, term: u32, ix: u32) -> vec4f {
    let lo = u32(clamp(floor(table_t), 0.0, 6.0));
    let hi = u32(clamp(ceil(table_t), 0.0, 6.0));
    let w = table_t - f32(lo);
    let a = ms_poly(lo, band, term, ix);
    let b = ms_poly(hi, band, term, ix);
    return mix(a, b, w);
}

@compute @workgroup_size(8, 8, 1)
fn generate_radiance_map(@builtin(global_invocation_id) global_id: vec3u) {
    // Guard: (0,0) extent → inf UVs → NaN directions → black; encase/uniform bugs or stale binds.
    let size = max(constants.output_size, vec2u(1u));
    let inv_size = 1.0 / vec2f(size);

    let coords = vec2u(global_id.xy);
    let face = global_id.z;

    if (any(coords >= size)) {
        return;
    }

    let uv = (vec2f(coords) + 0.5) * inv_size;
    let normal = sample_cube_dir(uv, face);
    let perceptual_roughness = constants.perceptual_roughness;

    // Match `SpecularEnvironmentIntegration::MansonSloan` shading: LOD uses perceptual roughness
    // (not GGX alpha). Don't use `perceptualRoughnessToRoughness` here or the Filament clamp
    // (min 0.089) collapses several low mips into the same near-mirror path.
    if (perceptual_roughness < 0.001) {
        let radiance = sample_environment(normal, 0.0).rgb;
        textureStore(output_texture, coords, face, vec4f(radiance, 1.0));
        return;
    }

    // Seven table rows → continuous index in [0, 6] for coefficient interpolation.
    let table_t = clamp(perceptual_roughness * 6.0, 0.0, 6.0);
    let max_lod = f32(textureNumLevels(input_texture) - 1u);

    let dir = cube_face_dir_unorm(uv, face);
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

        var nx = dir_axis(dir, other_axis0);
        var ny = dir_axis(dir, other_axis1);
        let nz = dir_axis(ad, axis);
        let nmax_xy = max(abs(ny), abs(nx));
        nx = nx / nmax_xy;
        ny = ny / nmax_xy;

        var theta: f32;
        if (ny < nx) {
            if (ny <= -0.999) {
                theta = nx;
            } else {
                theta = ny;
            }
        } else {
            if (ny >= 0.999) {
                theta = -nx;
            } else {
                theta = -ny;
            }
        }

        var phi: f32;
        if (nz <= -0.999) {
            phi = -nmax_xy;
        } else if (nz >= 0.999) {
            phi = nmax_xy;
        } else {
            phi = nz;
        }

        let theta2 = theta * theta;
        let phi2 = phi * phi;

        for (var i_super = 0u; i_super < NUM_MS_TAPS_DIV_4; i_super++) {
            let coeff_ix = NUM_MS_TAPS_DIV_4 * axis + i_super;

            var c0d: array<vec4f, 3u>;
            var c1d: array<vec4f, 3u>;
            var c2d: array<vec4f, 3u>;
            var c_l: array<vec4f, 3u>;
            var c_w: array<vec4f, 3u>;
            for (var ic = 0u; ic < 3u; ic++) {
                c0d[ic] = ms_poly_lerp(table_t, 0u, ic, coeff_ix);
                c1d[ic] = ms_poly_lerp(table_t, 1u, ic, coeff_ix);
                c2d[ic] = ms_poly_lerp(table_t, 2u, ic, coeff_ix);
                c_l[ic] = ms_poly_lerp(table_t, 3u, ic, coeff_ix);
                c_w[ic] = ms_poly_lerp(table_t, 4u, ic, coeff_ix);
            }

            for (var i_sub = 0u; i_sub < 4u; i_sub++) {
                let d0 = c0d[0][i_sub] + c0d[1][i_sub] * theta2 + c0d[2][i_sub] * phi2;
                let d1 = c1d[0][i_sub] + c1d[1][i_sub] * theta2 + c1d[2][i_sub] * phi2;
                let d2 = c2d[0][i_sub] + c2d[1][i_sub] * theta2 + c2d[2][i_sub] * phi2;
                var sample_dir = frame_x * d0 + frame_y * d1 + frame_z * d2;
                var sample_level = c_l[0][i_sub] + c_l[1][i_sub] * theta2 + c_l[2][i_sub] * phi2;
                var sample_weight = c_w[0][i_sub] + c_w[1][i_sub] * theta2 + c_w[2][i_sub] * phi2;
                sample_weight *= frameweight;

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

#import bevy_pbr::utils::sample_cosine_hemisphere

@compute @workgroup_size(8, 8, 1)
fn generate_irradiance_map(@builtin(global_invocation_id) global_id: vec3u) {
    let size = max(constants.output_size, vec2u(1u));
    let inv_size = 1.0 / vec2f(size);

    let coords = vec2u(global_id.xy);
    let face = global_id.z;

    if (any(coords >= size)) {
        return;
    }

    let uv = (vec2f(coords) + 0.5) * inv_size;
    let normal = sample_cube_dir(uv, face);

    var irradiance = vec3f(0.0);

    for (var i = 0u; i < constants.sample_count; i++) {
        var rng: u32 = (coords.x * 2131358057u) ^ (coords.y * 3416869721u) ^ (face * 1199786941u) ^ (i * 566200673u);

        let sample_dir = sample_cosine_hemisphere(normal, &rng);

        let sample_color = sample_environment(sample_dir, 0.0).rgb;

        irradiance += sample_color;
    }

    irradiance = irradiance / f32(constants.sample_count);

    textureStore(output_texture, coords, face, vec4f(irradiance, 1.0));
}
