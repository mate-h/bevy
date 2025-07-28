#import bevy_render::maths::{PI}
#import bevy_pbr::{
    atmosphere::{
        types::{Atmosphere, AtmosphereSettings},
        bindings::{atmosphere, settings, probe_transform_buffer},
        functions::{max_atmosphere_distance, raymarch_atmosphere},
    },
    utils::sample_cube_dir
}

@group(0) @binding(13) var output: texture_storage_2d_array<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(output);
    let slice_index = global_id.z;
    
    if (global_id.x >= dimensions.x || global_id.y >= dimensions.y || slice_index >= 6u) {
        return;
    }
    
    // Calculate normalized UV coordinates for this pixel
    let uv = vec2<f32>(
        (f32(global_id.x) + 0.5) / f32(dimensions.x),
        (f32(global_id.y) + 0.5) / f32(dimensions.y)
    );

    var world_pos = probe_transform_buffer[3].xyz;

    // offset by the origin point of the atmosphere scene
    world_pos += atmosphere.origin;

    let r = length(world_pos);

    let ray_dir_ws = sample_cube_dir(uv, slice_index);
    let up = normalize(world_pos);
    let mu = dot(ray_dir_ws, up);
    let raymarch_steps = 16.0;
    let t_max = max_atmosphere_distance(r, mu);
    let sample_count = mix(1.0, raymarch_steps, clamp(t_max * 0.01, 0.0, 1.0));
    let result = raymarch_atmosphere(world_pos, ray_dir_ws, t_max, sample_count, uv, false, true, false);
    let inscattering = result.inscattering;
    let color = vec4<f32>(inscattering, 1.0);

    textureStore(output, vec2<i32>(global_id.xy), i32(slice_index), color);
}