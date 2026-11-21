//! Downsamples one mip of a cubemap into the next with the quadratic B-spline
//! filter that the Manson–Sloan filter table is fitted for: four bilinear reads,
//! three quarters of a source texel from the destination texel's center along
//! each axis.

@group(0) @binding(0) var source: texture_2d_array<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var destination: texture_storage_2d_array<rgba16float, write>;

@compute
@workgroup_size(8, 8, 1)
fn downsample(@builtin(global_invocation_id) global_id: vec3u) {
    let size = textureDimensions(destination).xy;
    let coords = global_id.xy;
    let face = global_id.z;
    if (any(coords >= size)) {
        return;
    }

    let offsets = array(vec2f(-0.75, -0.75), vec2f(0.75, -0.75), vec2f(-0.75, 0.75), vec2f(0.75, 0.75));
    let center = 2.0 * vec2f(coords) + 1.0;
    let source_size = 2.0 * vec2f(size);

    var uvs: array<vec2f, 4>;
    var weights: array<f32, 4>;
    var total_weight = 0.0;
    for (var i = 0u; i < 4u; i++) {
        uvs[i] = (center + offsets[i]) / source_size;
        // `|d|³` of the unnormalized direction `d` through the tap.
        let face_position = 2.0 * uvs[i] - 1.0;
        let length_squared = 1.0 + dot(face_position, face_position);
        weights[i] = length_squared * sqrt(length_squared);
        total_weight += weights[i];
    }

    var color = vec4f(0.0);
    for (var i = 0u; i < 4u; i++) {
        let weight = 0.5 * weights[i] / total_weight + 0.125;
        color += textureSampleLevel(source, source_sampler, uvs[i], face, 0.0) * weight;
    }
    textureStore(destination, coords, face, color);
}
