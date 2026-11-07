import bevy_render::maths::PI;
import package::render::utils::sample_cube_dir;

struct FilteringConstants {
    perceptual_roughness: f32,
    lod_bias: f32,
    // The first cubemap face this dispatch filters (`EnvironmentMapFilterSlicing`).
    face_offset: u32,
}

// The filter table: 7 roughness levels × 5 parameters (tap direction xyz,
// source level, weight) × 6 float4s (3 axes × 8 taps / 4).
const TABLE_LEVELS: u32 = 7u;
const TABLE_PARAMETERS: u32 = 5u;
const TABLE_INDICES: u32 = 6u;
const TAPS_PER_AXIS_DIV_4: u32 = 2u;

@group(0) @binding(0) var input_texture: texture_2d_array<f32>;
// The same source as a cubemap, so filtering crosses face edges.
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
    let face = global_id.z + constants.face_offset;
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

// Irradiance is projected from the mip with faces of this size, or the source
// if it is smaller.
const IRRADIANCE_SOURCE_SIZE: u32 = 32u;
// Two SH bands ring around a small, very bright source such as the sun. So texels
// brighter than this many times the mean luminance are clamped before the
// projection, and their excess is added back exactly as point lights.
const IRRADIANCE_EXCESS_SCALE: f32 = 16.0;
// At most this many texels are clamped; any beyond are projected in full.
const IRRADIANCE_MAX_EXCESS: u32 = 64u;

var<workgroup> sh_partial_sums: array<vec3f, 64>;
var<workgroup> excess_count: atomic<u32>;
var<workgroup> excess_directions: array<vec3f, IRRADIANCE_MAX_EXCESS>;
var<workgroup> excess_radiances: array<vec3f, IRRADIANCE_MAX_EXCESS>;

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

/// The solid angle of the part of a cube face between its center and the
/// position `(x, y)` on it, in [-1, 1].
fn face_corner_solid_angle(x: f32, y: f32) -> f32 {
    return atan2(x * y, sqrt(1.0 + x * x + y * y));
}

/// Sums `value` over the workgroup; every invocation gets the total.
fn workgroup_sum(local_index: u32, value: vec3f) -> vec3f {
    sh_partial_sums[local_index] = value;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if (local_index < stride) {
            sh_partial_sums[local_index] += sh_partial_sums[local_index + stride];
        }
        workgroupBarrier();
    }
    let total = sh_partial_sums[0];
    workgroupBarrier();
    return total;
}

/// Projects the source onto 9 spherical harmonic coefficients (Ramamoorthi and
/// Hanrahan 2001), convolved with the clamped cosine and divided by π, so that
/// the irradiance over π at the normal `n` is their dot product with the basis
/// at `n`. The excess of texels brighter than `IRRADIANCE_EXCESS_SCALE` times
/// the mean is left out and stored in `excess_directions` and
/// `excess_radiances`, as radiance times solid angle.
fn project_irradiance(local_index: u32) -> array<vec3f, 9> {
    let base_size = textureDimensions(input_texture).x;
    let size = min(IRRADIANCE_SOURCE_SIZE, base_size);
    let level = firstLeadingBit(base_size / size);
    let texel_count = 6u * size * size;
    let luminance_weights = vec3f(0.2126, 0.7152, 0.0722);

    if (local_index == 0u) {
        atomicStore(&excess_count, 0u);
    }
    workgroupBarrier();

    // The mean comes from the 1×1 mip, which is close enough for a threshold.
    let last_level = textureNumLevels(input_texture) - 1u;
    var mean_luminance = 0.0;
    for (var face = 0u; face < 6u; face++) {
        let radiance = textureLoad(input_texture, vec2u(0u), face, last_level).rgb;
        mean_luminance += dot(radiance, luminance_weights) / 6.0;
    }
    let max_luminance = IRRADIANCE_EXCESS_SCALE * mean_luminance * 4.0 * PI / f32(texel_count);

    var partial_sums: array<vec3f, 9>;
    for (var texel = local_index; texel < texel_count; texel += 64u) {
        let face = texel / (size * size);
        let coords = vec2u(texel % size, (texel / size) % size);
        let lo = 2.0 * vec2f(coords) / f32(size) - 1.0;
        let hi = 2.0 * vec2f(coords + 1u) / f32(size) - 1.0;
        let solid_angle = face_corner_solid_angle(hi.x, hi.y) - face_corner_solid_angle(lo.x, hi.y)
            - face_corner_solid_angle(hi.x, lo.y) + face_corner_solid_angle(lo.x, lo.y);
        let direction = sample_cube_dir((vec2f(coords) + 0.5) / f32(size), face);

        // Radiance times solid angle, and its luminance.
        var radiance = textureLoad(input_texture, coords, face, level).rgb * solid_angle;
        let luminance = dot(radiance, luminance_weights);
        if (luminance > max_luminance) {
            let slot = atomicAdd(&excess_count, 1u);
            if (slot < IRRADIANCE_MAX_EXCESS) {
                let clamped = radiance * (max_luminance / luminance);
                excess_directions[slot] = direction;
                excess_radiances[slot] = radiance - clamped;
                radiance = clamped;
            }
        }

        let basis = sh9_basis(direction);
        for (var i = 0u; i < 9u; i++) {
            partial_sums[i] += radiance * basis[i];
        }
    }

    // The clamped cosine convolved with each band, over π.
    let band_scales = array<f32, 9>(1.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.25, 0.25, 0.25, 0.25, 0.25);
    var coefficients: array<vec3f, 9>;
    for (var i = 0u; i < 9u; i++) {
        coefficients[i] = workgroup_sum(local_index, partial_sums[i]) * band_scales[i];
    }
    return coefficients;
}

/// Computes the diffuse map from a projection of the source onto spherical
/// harmonics, which every workgroup computes in full before writing its texels.
@compute
@workgroup_size(8, 8, 1)
fn generate_irradiance_map(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let coefficients = project_irradiance(local_index);

    let size = textureDimensions(output_texture).xy;
    let coords = global_id.xy;
    let face = global_id.z + constants.face_offset;
    if (any(coords >= size)) {
        return;
    }

    let normal = sample_cube_dir((vec2f(coords) + 0.5) / vec2f(size), face);
    let basis = sh9_basis(normal);
    var irradiance = vec3f(0.0);
    for (var i = 0u; i < 9u; i++) {
        irradiance += coefficients[i] * basis[i];
    }
    // Two bands can ring below zero around what the clamp leaves.
    irradiance = max(irradiance, vec3f(0.0));

    let excess_texel_count = min(atomicLoad(&excess_count), IRRADIANCE_MAX_EXCESS);
    for (var i = 0u; i < excess_texel_count; i++) {
        irradiance += excess_radiances[i] * max(dot(normal, excess_directions[i]), 0.0) / PI;
    }

    textureStore(output_texture, coords, face, vec4f(irradiance, 1.0));
}
