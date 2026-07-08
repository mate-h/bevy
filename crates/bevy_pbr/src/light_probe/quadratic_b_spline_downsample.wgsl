// Manson–Sloan cubemap downsample.

struct QuadraticDownsampleUniforms {
    src_mip_level: u32,
    dst_mip_level: u32,
    dst_size: u32,
    _pad: u32,
}

@group(0) @binding(0) var src_texture: texture_2d_array<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(2) var<uniform> uniforms: QuadraticDownsampleUniforms;

fn unorm_dir_from_corner_params(face: u32, u: f32, v: f32) -> vec3f {
    switch face {
        case 0u: { return vec3f(1.0, v, -u); }
        case 1u: { return vec3f(-1.0, v, u); }
        case 2u: { return vec3f(u, 1.0, -v); }
        case 3u: { return vec3f(u, -1.0, v); }
        case 4u: { return vec3f(u, v, 1.0); }
        default: { return vec3f(-u, v, -1.0); }
    }
}

// Cube-map Jacobian weight for an unnormalized direction.
fn jacobian_weight_unorm(p_unnorm: vec3f) -> f32 {
    let l2 = dot(p_unnorm, p_unnorm);
    return l2 * sqrt(l2);
}

// Map reference corner parameters to Bevy face UV.
fn corner_params_to_face_uv01(u: f32, v: f32) -> vec2f {
    let uvc_x = u;
    let uvc_y = -v;
    return vec2f(uvc_x + 1.0, uvc_y + 1.0) * 0.5;
}

fn bilinear_load_face(src_sz: u32, face: u32, uv01: vec2f) -> vec4f {
    let max_f = f32(src_sz - 1u);
    let pf = uv01 * vec2f(f32(src_sz)) - vec2f(0.5);
    let p0 = vec2<i32>(floor(pf));
    let t = pf - vec2f(p0);
    let p1 = p0 + vec2<i32>(1);
    let x0 = u32(clamp(f32(p0.x), 0.0, max_f));
    let y0 = u32(clamp(f32(p0.y), 0.0, max_f));
    let x1 = u32(clamp(f32(p1.x), 0.0, max_f));
    let y1 = u32(clamp(f32(p1.y), 0.0, max_f));
    let c00 = textureLoad(src_texture, vec2u(x0, y0), face, 0u);
    let c10 = textureLoad(src_texture, vec2u(x1, y0), face, 0u);
    let c01 = textureLoad(src_texture, vec2u(x0, y1), face, 0u);
    let c11 = textureLoad(src_texture, vec2u(x1, y1), face, 0u);
    let c0 = mix(c00, c10, t.x);
    let c1 = mix(c01, c11, t.x);
    return mix(c0, c1, t.y);
}

@compute @workgroup_size(8, 8, 1)
fn downsample_quadratic_mip(@builtin(global_invocation_id) global_id: vec3u) {
    let dst_size = uniforms.dst_size;
    let face = global_id.z;
    let coord = global_id.xy;
    if (coord.x >= dst_size || coord.y >= dst_size) {
        return;
    }

    let src_size = dst_size * 2u;
    let inv_lo = 1.0 / f32(dst_size);

    // Sample four Jacobian-weighted corners.
    let u0 = (f32(coord.x) * 2.0 + 1.0 - 0.75) * inv_lo - 1.0;
    let u1 = (f32(coord.x) * 2.0 + 1.0 + 0.75) * inv_lo - 1.0;
    let v0 = (f32(coord.y) * 2.0 + 1.0 - 0.75) * (-inv_lo) + 1.0;
    let v1 = (f32(coord.y) * 2.0 + 1.0 + 0.75) * (-inv_lo) + 1.0;

    var w0 = jacobian_weight_unorm(unorm_dir_from_corner_params(face, u0, v0));
    var w1 = jacobian_weight_unorm(unorm_dir_from_corner_params(face, u1, v0));
    var w2 = jacobian_weight_unorm(unorm_dir_from_corner_params(face, u0, v1));
    var w3 = jacobian_weight_unorm(unorm_dir_from_corner_params(face, u1, v1));

    let wsum = 0.5 / (w0 + w1 + w2 + w3);
    w0 = w0 * wsum + 0.125;
    w1 = w1 * wsum + 0.125;
    w2 = w2 * wsum + 0.125;
    w3 = w3 * wsum + 0.125;

    let uv00 = corner_params_to_face_uv01(u0, v0);
    let uv10 = corner_params_to_face_uv01(u1, v0);
    let uv01 = corner_params_to_face_uv01(u0, v1);
    let uv11 = corner_params_to_face_uv01(u1, v1);

    var color = bilinear_load_face(src_size, face, uv00) * w0;
    color += bilinear_load_face(src_size, face, uv10) * w1;
    color += bilinear_load_face(src_size, face, uv01) * w2;
    color += bilinear_load_face(src_size, face, uv11) * w3;

    textureStore(dst_texture, coord, face, color);
}
