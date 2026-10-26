// Weather window (`.clouds-handoff/notes/weather_layers_design.md`, A2): the weather around one
// view's camera, composed each frame from the tiled weather map and the physical defaults, where
// world-anchored sources (influences, land/sea, terrain) will be added. Window texel j holds tile
// texel origin + j of the weather map, in the map's own (rotated, advected) frame, so inside the
// window the clouds sample exactly what the tiled map gives (`clouds.wesl` sample_weather).
// - out_art (A): the tiled map's channels (coverage, type, density scale, mesoscale fraction).
// - out_physical (B): cloud-fraction scale, air mass (the map's type until a source sets it),
//   droplet number N_d (cm^-3), cloud-base offset (m). Not read by the clouds yet.

@group(0) @binding(0) var weather_tile: texture_2d<f32>;
@group(0) @binding(1) var out_art: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var out_physical: texture_storage_2d<rgba16float, write>;

struct WeatherWindowParams {
    // Window texel 0 in weather-map texels (integer valued), `clouds.wesl` weather_window_origin.
    origin: vec2<f32>,
    droplet_number: f32,
    _pad: f32,
}
@group(0) @binding(3) var<uniform> params: WeatherWindowParams;

@compute @workgroup_size(8, 8, 1)
fn compose(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(out_art);
    if (any(gid.xy >= size)) {
        return;
    }
    let dims = vec2<i32>(textureDimensions(weather_tile));
    let t = ((vec2<i32>(params.origin) + vec2<i32>(gid.xy)) % dims + dims) % dims;
    let art = textureLoad(weather_tile, t, 0);
    textureStore(out_art, vec2<i32>(gid.xy), art);
    textureStore(out_physical, vec2<i32>(gid.xy), vec4(1.0, art.g, params.droplet_number, 0.0));
}
