import package::atmosphere::{
    bindings::{atmosphere, view, settings, cloud_shadow_map, atmosphere_lut_sampler},
    bruneton_functions::{sample_transmittance_lut_segment, ray_intersects_ground},
    functions::{
        sample_transmittance_lut,
        sample_sky_view_lut, direction_world_to_atmosphere,
        uv_to_ray_direction, uv_to_ndc, sample_aerial_view_lut,
        sample_sun_radiance, ndc_to_camera_dist, raymarch_atmosphere,
        get_view_position, max_atmosphere_distance
    },
};

import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput;



@if(DUAL_SOURCE_BLENDING)
enable dual_source_blending;

@if(MULTISAMPLED)
@group(0) @binding(13) var depth_texture: texture_depth_multisampled_2d;
@else
@group(0) @binding(13) var depth_texture: texture_depth_2d;

// Output of the low-res `clouds_raymarch` pass: rgb = pre-exposed cloud luminance, a = cloud
// transmittance; and the transmittance-weighted cloud distance (m).
@if(CLOUDS_ENABLED)
@group(0) @binding(26) var cloud_texture: texture_2d<f32>;
@if(CLOUDS_ENABLED)
@group(0) @binding(27) var cloud_depth_texture: texture_2d<f32>;

// Half-resolution atmosphere (haze) from the `haze` prepass: rgb = pre-exposure in-scattering;
// rgb = transmittance, a = ray distance (km) the haze was integrated to. In `main` this frame's
// haze; in the `haze` prepass the previous frame's (its temporal history).
@if(SKY_HAZE_HALF_RES)
@group(0) @binding(50) var haze_inscattering: texture_2d<f32>;
@if(SKY_HAZE_HALF_RES)
@group(0) @binding(51) var haze_transmittance: texture_2d<f32>;

// Shared with clouds_reconstruct (resources.rs `CloudsTemporalParams`).
struct CloudsTemporalParams {
    prev_clip_from_world: mat4x4<f32>,
    cloud_motion: vec3<f32>,
    history_valid: u32,
    cirrus_motion: vec3<f32>,
    new_sample_weight: f32,
    cirrus_min_radius: f32,
    history_exposure_scale: f32,
}
@if(SKY_HAZE_HALF_RES)
@group(0) @binding(52) var<uniform> haze_temporal: CloudsTemporalParams;

// Weight of this frame's march in the haze history (exponential average over ~1 / w frames). The
// march is jittered per pixel and frame (blue noise), so averaging stands in for samples.
const HAZE_NEW_SAMPLE_WEIGHT: f32 = 0.2;
// History is rejected where its ray length differs by more than this fraction (disocclusion).
const HAZE_DEPTH_TOLERANCE: f32 = 0.1;

// Distance used for sky pixels' haze lookups and as the depth key (km): the ray's length to the
// ground or the top of the atmosphere.
fn sky_ray_length(r: f32, mu: f32) -> f32 {
    return max_atmosphere_distance(r, mu);
}

const HAZE_SAMPLE_SCALE: u32 = 1u;

struct HazeOutput {
    @location(0) inscattering: vec4<f32>,
    @location(1) transmittance: vec4<f32>,
}

// This frame's checkerboard march (the `haze` pass), read by `haze_resolve`: texel (i, y) holds
// haze texel (2i + ((y + frame) & 1), y).
@if(SKY_HAZE_RESOLVE)
@group(0) @binding(53) var haze_sparse_inscattering: texture_2d<f32>;
@if(SKY_HAZE_RESOLVE)
@group(0) @binding(54) var haze_sparse_transmittance: texture_2d<f32>;

// Weight of a neighbour estimate in the history of a texel not marched this frame: keeps it
// responsive between its own marches without spreading the neighbours' noise into it.
const HAZE_FILL_WEIGHT: f32 = 0.1;

// Column of the haze texel row `y` marched this frame at checkerboard index `i`.
fn haze_marched_x(i: i32, y: i32) -> i32 {
    return 2 * i + ((y + i32(view.frame_count & 1u)) & 1);
}

struct HazeTexelRay {
    uv: vec2<f32>,
    // Farthest depth of the texel's full-res footprint (0 = sky).
    depth: f32,
    // Ray length (m): to that depth, or through the atmosphere for the sky.
    t: f32,
}

/// The ray of haze texel `texel` (of a `dims` target): through the texel's centre (2h + 1
/// full-res pixels), where the history and the bilinear upsample in `main` expect its sample,
/// with the block's farthest depth (the background, as the cloud pass). Tracing through the
/// farthest pixel itself put the sample a quarter texel off the centre (bottom right when all
/// four depths tie, e.g. sky): every frame's reprojection then read its history shifted by that
/// quarter texel, a recursive one-directional blur even with a still camera.
fn haze_texel_ray(texel: vec2<f32>, dims: vec2<f32>) -> HazeTexelRay {
    // The haze targets cover the camera's viewport; the depth texture covers the whole render
    // target, so its texels are offset by the viewport origin.
    let full = max(vec2<i32>(view.viewport.zw), vec2(1));
    let scale = vec2<f32>(full) / dims;
    let base = vec2<i32>(floor(texel * scale));
    let footprint = vec2<i32>(ceil((texel + 1.0) * scale)) - base;
    var depth = 1.0;
    for (var y = 0; y < footprint.y; y++) {
        for (var x = 0; x < footprint.x; x++) {
            let p = vec2<i32>(view.viewport.xy) + min(base + vec2(x, y), full - 1);
            let d = textureLoad(depth_texture, p, 0);
            if (d <= depth) {
                depth = d;
            }
        }
    }
    let uv = (texel + 0.5) / dims;
    let world_pos = get_view_position();
    let ray_dir_ws = uv_to_ray_direction(uv);
    var t = sky_ray_length(length(world_pos), dot(ray_dir_ws, normalize(world_pos)));
    if depth != 0.0 {
        t = ndc_to_camera_dist(vec3(uv_to_ndc(uv), depth));
    }
    return HazeTexelRay(uv, depth, t);
}

/// Haze march, half of the haze texels per frame (a checkerboard alternating each frame,
/// compacted to a half-width target so every SIMD group marches): the per-pixel atmosphere
/// raymarch (or LUT lookup) at one full-res texel of each 2x2 block, plus its ray length as the
/// upsampling depth key. `haze_resolve` merges it with the history. The march is the dominant
/// cost of render_sky at full resolution (16 shadowed samples per pixel); the haze is smooth
/// except at silhouettes, which the depth-aware upsample in `main` handles.
@if(SKY_HAZE_HALF_RES)
@fragment
fn haze(in: FullscreenVertexOutput) -> HazeOutput {
    // The haze targets are half the viewport or smaller (the cloud buffers' size cap,
    // `VolumetricCloudSettings::max_render_size`); the history bound here has the full haze size.
    let dims = vec2<f32>(max(textureDimensions(haze_inscattering), vec2(1u)));
    let row = i32(in.position.y);
    let texel = vec2(f32(haze_marched_x(i32(in.position.x), row)), floor(in.position.y));
    let ray = haze_texel_ray(texel, dims);
    let uv = ray.uv;
    let depth = ray.depth;
    let t = ray.t;
    let samples = settings.sky_max_samples * HAZE_SAMPLE_SCALE;
    // The march's `uv` only drives its jitter noise (uv x full-res viewport = noise pixel). At
    // half resolution that read every other texel of the blue-noise tile, which loses its
    // blue-noise property (a visible hatch); use this pass's own contiguous pixel lattice.
    let noise_uv = in.position.xy / view.viewport.zw;
    let ray_dir_ws = uv_to_ray_direction(uv);
    let world_pos = get_view_position();
    let r = length(world_pos);
    let mu = dot(ray_dir_ws, normalize(world_pos));
    var inscattering: vec3<f32>;
    var transmittance: vec3<f32>;
    if depth == 0.0 {
        transmittance = sample_transmittance_lut(r, mu);
        inscattering = sample_sky_view_lut(r, direction_world_to_atmosphere(ray_dir_ws));
        if settings.rendering_method == 1u {
            let result = raymarch_atmosphere(world_pos, ray_dir_ws, t, samples, noise_uv, true);
            inscattering = result.inscattering;
            transmittance = result.transmittance;
        }
    } else {
        inscattering = sample_aerial_view_lut(uv, t);
        transmittance = sample_transmittance_lut_segment(r, mu, t);
        if settings.rendering_method == 1u {
            let result = raymarch_atmosphere(world_pos, ray_dir_ws, t, samples, noise_uv, false);
            inscattering = result.inscattering;
            transmittance = result.transmittance;
        }
    }
    return HazeOutput(vec4(inscattering, 0.0), vec4(transmittance, t * 0.001));
}

/// Merges this frame's checkerboard march with the haze history at every haze texel.
/// - Marched texels blend their new sample into the history.
/// - The others keep their history, nudged toward a depth-weighted average of their four
///   marched neighbours (left, right, above, below).
/// - Where the history is rejected (disocclusion, off screen), texels take their new sample or
///   that average.
/// The history is clamped to the range of this frame's nearby samples at the same depth (the
/// texel's own and its four nearest marched neighbours), as the cloud reconstruction does: with
/// the light turning 0.3 deg per frame the haze lagged the march by tens of frames (25 dB from
/// the unaccumulated march at the end of a sweep), the sky staying lit after sunset.
@if(SKY_HAZE_RESOLVE)
@fragment
fn haze_resolve(in: FullscreenVertexOutput) -> HazeOutput {
    let dims_u = max(textureDimensions(haze_inscattering), vec2(1u));
    let dims = vec2<f32>(dims_u);
    let sparse_dims = vec2<i32>(textureDimensions(haze_sparse_inscattering));
    let p = vec2<i32>(in.position.xy);
    let ray = haze_texel_ray(floor(in.position.xy), dims);
    let t_km = ray.t * 0.001;
    let marched = haze_marched_x(p.x >> 1u, p.y) == p.x;

    // The nearest marched texels: for a marched texel, its four diagonals (the other row parity
    // and the other column); otherwise left, right (this row's parity) and above, below (the
    // other row parity, this column).
    var l = vec3(0.0);
    var tr = vec3(0.0);
    var w_sum = 0.0;
    var best = 1e30;
    var best_l = vec3(0.0);
    var best_tr = vec3(1.0);
    var l_min = vec3(1e30);
    var l_max = vec3(0.0);
    var t_min = vec3(1.0);
    var t_max = vec3(0.0);
    for (var k = 0; k < 5; k++) {
        var o: vec2<i32>;
        if (k == 4) {
            o = vec2(0);
        } else if (marched) {
            o = vec2(select(-1, 1, (k & 1) == 1), select(-1, 1, k >= 2));
        } else {
            o = select(select(vec2(0, 1), vec2(0, -1), k == 2), select(vec2(1, 0), vec2(-1, 0), k == 0), k < 2);
        }
        let n = clamp(p + o, vec2(0), vec2<i32>(dims_u) - 1);
        // Clamped at the border, a neighbour may land on a texel not marched this frame.
        if (haze_marched_x(n.x >> 1u, n.y) != n.x) {
            continue;
        }
        let q = clamp(vec2(n.x >> 1u, n.y), vec2(0), sparse_dims - 1);
        let tap_l = textureLoad(haze_sparse_inscattering, q, 0).rgb;
        let tap = textureLoad(haze_sparse_transmittance, q, 0);
        let rel = abs(tap.a - t_km) / max(t_km, 1e-3);
        if (k < 4 && !marched) {
            let w = exp(-rel / 0.1);
            l += w * tap_l;
            tr += w * tap.rgb;
            w_sum += w;
            if (rel < best) {
                best = rel;
                best_l = tap_l;
                best_tr = tap.rgb;
            }
        }
        if (k == 4) {
            l = tap_l;
            tr = tap.rgb;
        }
        if (rel < HAZE_DEPTH_TOLERANCE) {
            l_min = min(l_min, tap_l);
            l_max = max(l_max, tap_l);
            t_min = min(t_min, tap.rgb);
            t_max = max(t_max, tap.rgb);
        }
    }
    var inscattering = l;
    var transmittance = tr;
    if (!marched) {
        inscattering = select(best_l, l / max(w_sum, 1e-6), w_sum >= 1e-3);
        transmittance = select(best_tr, tr / max(w_sum, 1e-6), w_sum >= 1e-3);
    }

    // Temporal accumulation: reproject the previous frame's haze at the point this ray's haze
    // ends (scene point, or the ray's far end for the sky; rotation dominates for those) and
    // blend, unless the history's ray length disagrees (disocclusion) or it is off screen.
    var use_history = haze_temporal.history_valid != 0u;
    @if(ATM_DBG_NO_HAZE_HISTORY) { use_history = false; }
    if (use_history) {
        let world_pos = get_view_position();
        let ray_dir_ws = uv_to_ray_direction(ray.uv);
        let p_world = (atmosphere.atmosphere_to_world * vec4(world_pos + ray_dir_ws * ray.t, 1.0)).xyz;
        let prev_clip = haze_temporal.prev_clip_from_world * vec4(p_world, 1.0);
        let prev_uv = (prev_clip.xy / prev_clip.w) * vec2(0.5, -0.5) + vec2(0.5);
        if (prev_clip.w > 0.0 && all(prev_uv >= vec2(0.0)) && all(prev_uv <= vec2(1.0))) {
            var hist_l = textureSampleLevel(haze_inscattering, atmosphere_lut_sampler, prev_uv, 0.0).rgb;
            let hist_t = textureSampleLevel(haze_transmittance, atmosphere_lut_sampler, prev_uv, 0.0);
            if (abs(hist_t.a - t_km) < HAZE_DEPTH_TOLERANCE * max(t_km, 1e-3)) {
                var hist_tr = hist_t.rgb;
                @if(!ATM_DBG_NO_HAZE_CLAMP) {
                if (all(l_max >= l_min)) {
                    hist_l = clamp(hist_l, l_min, l_max);
                    hist_tr = clamp(hist_tr, t_min, t_max);
                }
                }
                let w = select(HAZE_FILL_WEIGHT, HAZE_NEW_SAMPLE_WEIGHT, marched);
                inscattering = mix(hist_l, inscattering, w);
                transmittance = mix(hist_tr, transmittance, w);
            }
        }
    }
    return HazeOutput(vec4(inscattering, 0.0), vec4(transmittance, t_km));
}

/// Joint-bilateral upsample of the half-res haze: bilinear weights times a depth-similarity
/// weight on the ray length (relative tolerance 10 %), falling back to the closest-depth tap
/// when no neighbour matches (thin features).
@if(SKY_HAZE_HALF_RES)
fn upsample_haze(uv: vec2<f32>, t: f32) -> array<vec3<f32>, 2> {
    let dims_u = textureDimensions(haze_inscattering);
    let coord = uv * vec2<f32>(dims_u) - 0.5;
    let base = vec2<i32>(floor(coord));
    let f = coord - floor(coord);
    let t_km = t * 0.001;
    var l = vec3(0.0);
    var tr = vec3(0.0);
    var w_sum = 0.0;
    var best = 1e30;
    var best_l = vec3(0.0);
    var best_tr = vec3(1.0);
    for (var k = 0; k < 4; k++) {
        let o = vec2(k & 1, k >> 1);
        let p = clamp(base + o, vec2(0), vec2<i32>(dims_u) - 1);
        let tap_l = textureLoad(haze_inscattering, p, 0).rgb;
        let tap = textureLoad(haze_transmittance, p, 0);
        let rel = abs(tap.a - t_km) / max(t_km, 1e-3);
        let bw = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
        let w = bw * exp(-rel / 0.1);
        l += w * tap_l;
        tr += w * tap.rgb;
        w_sum += w;
        if (rel < best) {
            best = rel;
            best_l = tap_l;
            best_tr = tap.rgb;
        }
    }
    if (w_sum < 1e-3) {
        return array(best_l, best_tr);
    }
    return array(l / w_sum, tr / w_sum);
}

struct RenderSkyOutput {
    @if(DUAL_SOURCE_BLENDING)
    @location(0) @blend_src(0) inscattering: vec4<f32>,
    @if(DUAL_SOURCE_BLENDING)
    @location(0) @blend_src(1) transmittance: vec4<f32>,
    @else
    @location(0) inscattering: vec4<f32>,
}


@fragment
fn main(in: FullscreenVertexOutput) -> RenderSkyOutput {
    let depth = textureLoad(depth_texture, vec2<i32>(in.position.xy), 0);

    let ray_dir_ws = uv_to_ray_direction(in.uv);
    let world_pos = get_view_position();
    let r = length(world_pos);
    let up = normalize(world_pos);
    let mu = dot(ray_dir_ws, up);
    let max_samples = settings.sky_max_samples;
    let should_raymarch = settings.rendering_method == 1u;

    var transmittance: vec3<f32>;
    var inscattering: vec3<f32>;

    let sun_radiance = sample_sun_radiance(ray_dir_ws);

    @if(SKY_HAZE_HALF_RES) {
    var t_pixel = sky_ray_length(r, mu);
    if depth != 0.0 {
        t_pixel = ndc_to_camera_dist(vec3(uv_to_ndc(in.uv), depth));
    }
    let haze = upsample_haze(in.uv, t_pixel);
    inscattering = haze[0];
    transmittance = haze[1];
    if depth == 0.0 {
        inscattering += sun_radiance * transmittance;
        if ray_intersects_ground(r, mu) {
            transmittance = vec3(0.0);
        }
    }
    }
    @if(!SKY_HAZE_HALF_RES) {
    if depth == 0.0 {
        let ray_dir_as = direction_world_to_atmosphere(ray_dir_ws);
        transmittance = sample_transmittance_lut(r, mu);
        inscattering = sample_sky_view_lut(r, ray_dir_as);
        if should_raymarch {
            let t_max = max_atmosphere_distance(r, mu);
            let result = raymarch_atmosphere(world_pos, ray_dir_ws, t_max, max_samples, in.uv, true);
            inscattering = result.inscattering;
            transmittance = result.transmittance;
        }
        inscattering += sun_radiance * transmittance;
        // The planet surface is opaque: its radiance is already in `inscattering`, so nothing
        // behind it (clear color, skybox) may show through via the transmittance blend.
        if ray_intersects_ground(r, mu) {
            transmittance = vec3(0.0);
        }
    } else {
        let t = ndc_to_camera_dist(vec3(uv_to_ndc(in.uv), depth));
        inscattering = sample_aerial_view_lut(in.uv, t);
        transmittance = sample_transmittance_lut_segment(r, mu, t);
        if should_raymarch {
            let result = raymarch_atmosphere(world_pos, ray_dir_ws, t, max_samples, in.uv, false);
            inscattering = result.inscattering;
            transmittance = result.transmittance;
        }
    }
    }

    // exposure compensation
    inscattering *= view.exposure;

    @if(CLOUDS_ENABLED) {
    // Composite the low-res cloud pass with a depth-aware bilinear upsample (roadmap 1e, first
    // part): each of the 4 bilinear taps only contributes if its cloud lies in front of this
    // full-res pixel's scene depth, then the weights are renormalized. With the cloud at
    // weighted distance d, transmittance T_c and luminance L_c:
    //   L = L_behind * T_c + (1 - T_c) * L_ap(0..d) + T_ap(0..d) * L_c
    // which assumes the atmosphere in-scattering behind the cloud is ~ the full-ray value.
    var scene_t = 1e30;
    if depth != 0.0 {
        scene_t = ndc_to_camera_dist(vec3(uv_to_ndc(in.uv), depth));
    }
    let cloud_dims_u = textureDimensions(cloud_texture);
    let cloud_dims = vec2<f32>(cloud_dims_u);
    let cloud_coord = in.uv * cloud_dims - 0.5;
    let cloud_base = vec2<i32>(floor(cloud_coord));
    let cloud_frac = cloud_coord - floor(cloud_coord);
    var clouds = vec4(0.0);
    var cloud_t = 0.0;
    var cloud_w_sum = 0.0;
    for (var tap = 0; tap < 4; tap++) {
        let o = vec2(tap & 1, tap >> 1);
        let p = clamp(cloud_base + o, vec2(0), vec2<i32>(cloud_dims_u) - 1);
        let bw = select(1.0 - cloud_frac.x, cloud_frac.x, o.x == 1)
            * select(1.0 - cloud_frac.y, cloud_frac.y, o.y == 1);
        let c = textureLoad(cloud_texture, p, 0);
        let d = textureLoad(cloud_depth_texture, p, 0).r;
        // Cloud taps behind the scene (upsampling across a silhouette) count as clear sky.
        var valid = d < scene_t;
        @if(SKY_HAZE_HALF_RES) {
        // A tap whose march stopped at this pixel's surface holds only cloud in front of it,
        // whatever its mean depth: the haze texel at the same position (the same size and
        // footprint as the cloud texel) stores that bound, the farthest scene depth of the
        // texel. Where a cloud wraps around an object its mean depth lies at the surface, and
        // the depth test alone kept or dropped whole low-res texels (blocks and pinholes of
        // cloud on a tower through the layer). Past a silhouette the bound is the background's
        // and the depth test decides.
        let bound = textureLoad(haze_transmittance, p, 0).a * 1000.0;
        valid = valid || bound <= scene_t * (1.0 + HAZE_DEPTH_TOLERANCE);
        }
        let c_eff = select(vec4(0.0, 0.0, 0.0, 1.0), c, valid);
        clouds += c_eff * bw;
        let opacity = 1.0 - c_eff.a;
        cloud_t += d * bw * opacity;
        cloud_w_sum += bw * opacity;
    }
    cloud_t = cloud_t / max(cloud_w_sum, 1e-6);
    // Reference path tracer: the cloud texture holds the whole path-traced in-scattering (air +
    // clouds, sun only) and the camera-ray transmittance; replace the LUT sky/aerial perspective.
    @if(CLOUD_REFERENCE) {
    let ref_c = textureSampleLevel(cloud_texture, atmosphere_lut_sampler, in.uv, 0.0);
    var sun_term = vec3(0.0);
    // ref_c.a is the camera ray's transmittance through air and clouds (ratio tracking); the
    // transmittance LUT would count the air twice (audit 35).
    if depth == 0.0 { sun_term = sun_radiance * ref_c.a * view.exposure; }
    @if(DUAL_SOURCE_BLENDING)
    return RenderSkyOutput(vec4(ref_c.rgb + sun_term, 0.0), vec4(vec3(ref_c.a), 1.0));
    @if(!DUAL_SOURCE_BLENDING)
    return RenderSkyOutput(vec4(ref_c.rgb + sun_term, ref_c.a));
    }
    // Taps behind this pixel's surface were dropped above; the mean depth of those kept may
    // still lie a little behind it (cloud cut off at the surface), which is no reason to drop
    // them all (single-pixel holes in cloud in front of geometry).
    if clouds.a < 0.999 && cloud_t > 0.0 {
        // The cloud pass already applied the (cloud-shadowed) aerial perspective in front of
        // the cloud: clouds.rgb = T_ap * L_c + (1 - T_c) * L_ap.
        let t_c = clouds.a;
        inscattering = inscattering * t_c + clouds.rgb;
        transmittance *= t_c;
    }
    }
    
    @if(DUAL_SOURCE_BLENDING)
    return RenderSkyOutput(vec4(inscattering, 0.0), vec4(transmittance, 1.0));
    @if(!DUAL_SOURCE_BLENDING)
    let mean_transmittance = (transmittance.r + transmittance.g + transmittance.b) / 3.0;
    @if(!DUAL_SOURCE_BLENDING)
    return RenderSkyOutput(vec4(inscattering, mean_transmittance));
    
}
