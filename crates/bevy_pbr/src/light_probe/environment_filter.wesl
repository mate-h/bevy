import package::render::utils::{sample_cosine_hemisphere, sample_cube_dir};

struct FilteringConstants {
    perceptual_roughness: f32,
    lod_bias: f32,
}

// The filter table: 7 roughness levels × 5 parameters (tap direction xyz,
// source level, weight) × 6 float4s (3 axes × 8 taps / 4).
const TABLE_LEVELS: u32 = 7u;
const TABLE_PARAMETERS: u32 = 5u;
const TABLE_INDICES: u32 = 6u;
const TAPS_PER_AXIS_DIV_4: u32 = 2u;

// The source as a cubemap, so filtering crosses face edges.
@group(0) @binding(5) var input_cube: texture_cube<f32>;
@group(0) @binding(1) var input_sampler: sampler;
@group(0) @binding(2) var output_texture: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(3) var<uniform> constants: FilteringConstants;
@group(0) @binding(4) var<storage, read> filter_table: array<vec4f, 210>;

/// Sample an environment map with a specific LOD
fn sample_environment(dir: vec3f, level: f32) -> vec4f {
    return textureSampleLevel(input_cube, input_sampler, dir, level);
}

/// The table's frames are right-handed, while cubemaps are left-handed.
fn flip_handedness(dir: vec3f) -> vec3f {
    return vec3f(dir.x, dir.y, -dir.z);
}

/// Table coefficients, interpolated between the two nearest roughness levels.
fn filter_coefficients(level: f32, parameter: u32, index: u32) -> vec4f {
    let lo = u32(floor(level));
    let hi = min(lo + 1u, TABLE_LEVELS - 1u);
    let a = filter_table[(lo * TABLE_PARAMETERS + parameter) * TABLE_INDICES + index];
    let b = filter_table[(hi * TABLE_PARAMETERS + parameter) * TABLE_INDICES + index];
    return mix(a, b, fract(level));
}

/// Convolves the source with the GGX lobe of `constants.perceptual_roughness`
/// with the fitted filter of Manson and Sloan, "Fast Filtering of Reflection
/// Probes" (2016): 8 taps in each of the 3 axial frames, each tap a bilinear,
/// trilinear-filtered read of the source mip chain.
@compute
@workgroup_size(8, 8, 1)
fn generate_radiance_map(@builtin(global_invocation_id) global_id: vec3u) {
    let size = textureDimensions(output_texture).xy;
    let coords = global_id.xy;
    let face = global_id.z;
    if (any(coords >= size)) {
        return;
    }

    let uv = (vec2f(coords) + 0.5) / vec2f(size);
    let normal = sample_cube_dir(uv, face);

    if (constants.perceptual_roughness == 0.0) {
        textureStore(output_texture, coords, face, vec4f(sample_environment(normal, 0.0).rgb, 1.0));
        return;
    }

    let level = constants.perceptual_roughness * f32(TABLE_LEVELS - 1u);
    let max_lod = f32(textureNumLevels(input_cube) - 1u);
    let dir = flip_handedness(normal);
    // On the cube, so that the other two coordinates are the face position.
    let cube_dir = abs(dir) / max(abs(dir.x), max(abs(dir.y), abs(dir.z)));

    var color = vec3f(0.0);
    var total_weight = 0.0;
    for (var axis = 0u; axis < 3u; axis++) {
        // Each frame fades out toward the edges of the faces it is centered on.
        let other_a = cube_dir[(axis + 1u) % 3u];
        let other_b = cube_dir[(axis + 2u) % 3u];
        let frame_weight = (max(other_a, other_b) - 0.75) / 0.25;
        if (frame_weight <= 0.0) {
            continue;
        }

        var up = vec3f(0.0);
        up[axis] = 1.0;
        let frame_x = normalize(cross(up, dir));
        let frame_y = cross(dir, frame_x);

        for (var i = 0u; i < TAPS_PER_AXIS_DIV_4; i++) {
            let index = TAPS_PER_AXIS_DIV_4 * axis + i;
            let tap_x = filter_coefficients(level, 0u, index);
            let tap_y = filter_coefficients(level, 1u, index);
            let tap_z = filter_coefficients(level, 2u, index);
            let tap_level = filter_coefficients(level, 3u, index);
            let tap_weight = filter_coefficients(level, 4u, index);

            for (var j = 0u; j < 4u; j++) {
                var tap = frame_x * tap_x[j] + frame_y * tap_y[j] + dir * tap_z[j];
                // Project onto the cube, and widen the footprint where the cube
                // texels are smaller.
                tap /= max(abs(tap.x), max(abs(tap.y), abs(tap.z)));
                let lod = clamp(
                    tap_level[j] + 0.75 * log2(dot(tap, tap)) + constants.lod_bias,
                    0.0,
                    max_lod,
                );
                let weight = tap_weight[j] * frame_weight;
                color += sample_environment(flip_handedness(tap), lod).rgb * weight;
                total_weight += weight;
            }
        }
    }

    textureStore(output_texture, coords, face, vec4f(max(color / total_weight, vec3f(0.0)), 1.0));
}

@compute
@workgroup_size(8, 8, 1)
fn generate_irradiance_map(@builtin(global_invocation_id) global_id: vec3u) {
    let size = textureDimensions(output_texture).xy;
    let invSize = 1.0 / vec2f(size);

    let coords = vec2u(global_id.xy);
    let face = global_id.z;

    if (any(coords >= size)) {
        return;
    }

    // Convert texture coordinates to direction vector
    let uv = (vec2f(coords) + 0.5) * invSize;
    let normal = sample_cube_dir(uv, face);

    var irradiance = vec3f(0.0);

    // 32 phi, 32 theta = 1024 samples total
    let sample_count = 1024u;

    // Use uniform sampling on a hemisphere
    for (var i = 0u; i < sample_count; i++) {
        // Build a deterministic RNG seed for this pixel / sample
        // 4 randomly chosen 32-bit primes
        var rng: u32 = (coords.x * 2131358057u) ^ (coords.y * 3416869721u) ^ (face * 1199786941u) ^ (i * 566200673u);

        // Sample a direction from the upper hemisphere around the normal
        var sample_dir = sample_cosine_hemisphere(normal, &rng);

        // Sample environment with level 0 (no mip)
        var sample_color = sample_environment(sample_dir, 0.0).rgb;

        // Accumulate the contribution
        irradiance += sample_color;
    }

    // Normalize by number of samples (cosine-weighted sampling already accounts for PDF)
    irradiance = irradiance / f32(sample_count);

    // Write result to output texture
    textureStore(output_texture, coords, face, vec4f(irradiance, 1.0));
}
