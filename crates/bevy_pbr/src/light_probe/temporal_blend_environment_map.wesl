struct TemporalBlendUniforms {
    alpha: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var scratch_texture: texture_2d_array<f32>;
@group(0) @binding(1) var history_texture: texture_2d_array<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(3) var<uniform> uniforms: TemporalBlendUniforms;

@compute @workgroup_size(8, 8, 1)
fn temporal_blend_specular(@builtin(global_invocation_id) global_id: vec3u) {
    let size = textureDimensions(output_texture).xy;
    let coords = vec2u(global_id.xy);
    let face = global_id.z;
    if (any(coords >= size)) {
        return;
    }

    let a = clamp(uniforms.alpha, 0.0, 1.0);
    let s = textureLoad(scratch_texture, coords, face, 0u);
    let h = textureLoad(history_texture, coords, face, 0u);
    let out = vec4f(mix(h.rgb, s.rgb, a), 1.0);
    textureStore(output_texture, coords, face, out);
}
