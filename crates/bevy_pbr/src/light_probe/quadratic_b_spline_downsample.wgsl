// Quadratic B-spline + Jacobian-weighted mip generation (Manson & Sloan EGSR 2015 §4).
// 16-tap tensor product (weights 1/8, 3/8, 3/8, 1/8 per axis); Jacobian uses the unnormalized
// cube direction for each source texel. All samples stay within the same cube face (cf. Fig. 2).

struct QuadraticDownsampleUniforms {
    src_mip_level: u32,
    dst_mip_level: u32,
    dst_size: u32,
    _pad: u32,
}

@group(0) @binding(0) var src_texture: texture_2d_array<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(2) var<uniform> uniforms: QuadraticDownsampleUniforms;

// Point on cube before normalization (−1..1 on the three components; one is ±1 on each face).
fn dir_on_cube_from_uv_face(face: u32, uv: vec2f) -> vec3f {
    let uvc = 2.0 * uv - 1.0;
    switch (face) {
        case 0u: { return vec3f(1.0, -uvc.y, -uvc.x); }
        case 1u: { return vec3f(-1.0, -uvc.y, uvc.x); }
        case 2u: { return vec3f(uvc.x, 1.0, uvc.y); }
        case 3u: { return vec3f(uvc.x, -1.0, -uvc.y); }
        case 4u: { return vec3f(uvc.x, -uvc.y, 1.0); }
        case 5u: { return vec3f(-uvc.x, -uvc.y, -1.0); }
        default: { return vec3f(1.0, 0.0, 0.0); }
    }
}

fn jacobian_weight(p_unnorm: vec3f) -> f32 {
    let l2 = dot(p_unnorm, p_unnorm);
    let J = 1.0 / max(pow(l2, 1.5), 1e-6);
    return 0.5 * (1.0 + J);
}

@compute @workgroup_size(8, 8, 1)
fn downsample_quadratic_mip(@builtin(global_invocation_id) global_id: vec3u) {
    let dst_size = uniforms.dst_size;
    let face = global_id.z;
    let coord = global_id.xy;
    if (coord.x >= dst_size || coord.y >= dst_size) {
        return;
    }

    // `src_texture` is bound as a single-mip view at the parent mip `uniforms.src_mip_level`.
    // Mip level in `textureLoad` is *relative to that view* → always 0 (not the absolute mip).
    let src_size = dst_size * 2u;
    let base = coord * 2u;
    let max_c = src_size - 1u;

    let w = array<f32, 4>(0.125, 0.375, 0.375, 0.125);
    var color = vec4f(0.0);
    var denom = 0.0;

    for (var j = 0u; j < 4u; j++) {
        for (var i = 0u; i < 4u; i++) {
            let sc = base + vec2u(i, j);
            // The 4×4 B-spline footprint extends past the face for edge output texels; clamp so
            // textureLoad stays in-bounds (OOB loads are zero and wipe small mips).
            let sc_clamped = vec2u(min(sc.x, max_c), min(sc.y, max_c));
            let uv = (vec2f(sc_clamped) + vec2f(0.5)) / f32(src_size);
            let p_u = dir_on_cube_from_uv_face(face, uv);
            let jw = jacobian_weight(p_u);
            let wt = w[i] * w[j] * jw;
            color += textureLoad(src_texture, sc_clamped, face, 0u) * wt;
            denom += wt;
        }
    }

    if (denom > 0.0) {
        color /= denom;
    }

    textureStore(dst_texture, coord, face, color);
}
