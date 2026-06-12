// Volumetric cloud modeling.
//
// The density model follows Schneider, "The Real-Time Volumetric Cloudscapes of
// Horizon Zero Dawn" (SIGGRAPH 2015) and the NUBIS family of techniques
// ("Nubis Cubed", Advances in Real-Time Rendering, SIGGRAPH 2023):
//
// 1. A 2D macro map (coverage + vertical profile types) is combined with analytic
//    bottom/top height profiles into a *dimensional profile*.
// 2. A 3D Perlin-Worley texture forms the base cloud shapes via value erosion:
//    density = remap(shape_noise, 1 - dimensional_profile, 1, 0, 1).
// 3. A 3D Worley texture, distorted by 2D curl noise, erodes the cloud edges
//    (wispy at the base, billowy at the top).
#define_import_path bevy_pbr::atmosphere::clouds

#import bevy_render::maths::ray_sphere_intersect

struct CloudLayer {
    cloud_layer_start: f32,
    cloud_layer_end: f32,
    cloud_density: f32,
    cloud_absorption: f32,
    cloud_scattering: f32,
    coverage: f32,
    noise_scale: f32,
    shape_noise_scale: f32,
    detail_noise_scale: f32,
    detail_strength: f32,
    curl_strength: f32,
    phase_g_forward: f32,
    phase_g_backward: f32,
    phase_blend: f32,
    multiscatter_scatter: f32,
    multiscatter_extinction: f32,
    multiscatter_phase: f32,
    multiscatter_octaves: u32,
    powder_strength: f32,
    ambient_strength: f32,
    noise_offset: vec3<f32>,
    wind_velocity: vec3<f32>,
}

@group(0) @binding(14) var<uniform> cloud_layer: CloudLayer;
// 2D macro map:
// - R: coverage (macro placement)
// - G: bottom type (vertical profile shaping)
// - B: top type (vertical profile shaping)
// - A: small-scale coverage variation
@group(0) @binding(15) var noise_texture_2d: texture_2d<f32>;
@group(0) @binding(16) var noise_sampler_3d: sampler;
// 3D shape noise:
// - R: Perlin-Worley
// - G/B/A: Worley FBM at 1x/2x/4x frequency
@group(0) @binding(18) var perlin_worley_noise_3d: texture_3d<f32>;
// 3D detail/erosion noise: Worley FBM at 1x/2x/4x frequency in R/G/B.
@group(0) @binding(20) var detail_noise_3d: texture_3d<f32>;
// 2D curl noise (signed RGB) used to distort detail sampling.
@group(0) @binding(21) var curl_noise_2d: texture_2d<f32>;

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let l2 = dot(v, v);
    if (l2 <= 1e-12) {
        return vec3(0.0, 1.0, 0.0);
    }
    return v * inverseSqrt(l2);
}

fn remap(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    let t = clamp((x - a) / max(1e-6, b - a), 0.0, 1.0);
    return mix(c, d, t);
}

// 30 degree rotation applied to XZ noise lookups to break up axis-aligned
// stretching/repetition artifacts.
const NOISE_ROT: mat2x2<f32> = mat2x2<f32>(
    0.8660254, -0.5,
    0.5, 0.8660254
);

/// Sample the packed 2D macro map (coverage + profile types) over world XZ.
fn sample_cloud_macro_map(world_pos: vec3<f32>) -> vec4<f32> {
    let xz = NOISE_ROT * (world_pos.xz + cloud_layer.noise_offset.xz);
    let uv = xz / cloud_layer.noise_scale;
    return textureSampleLevel(noise_texture_2d, noise_sampler_3d, uv, 0.0);
}

/// Sample the 3D Perlin-Worley shape noise with full 3D world coordinates.
fn sample_shape_noise(pos: vec3<f32>) -> vec4<f32> {
    let uvw = (pos + cloud_layer.noise_offset) / cloud_layer.shape_noise_scale;
    return textureSampleLevel(perlin_worley_noise_3d, noise_sampler_3d, uvw, 0.0);
}

/// Sample the 3D Worley detail noise with full 3D world coordinates.
/// Detail scrolls slightly faster than the base shape so erosion churns over time.
fn sample_detail_noise(pos: vec3<f32>) -> vec4<f32> {
    let uvw = (pos + cloud_layer.noise_offset * 1.35) / cloud_layer.detail_noise_scale;
    return textureSampleLevel(detail_noise_3d, noise_sampler_3d, uvw, 0.0);
}

/// Sample the signed curl noise field over world XZ.
fn sample_curl_noise(world_pos: vec3<f32>) -> vec3<f32> {
    // Tile the curl field a bit above the detail frequency so the distortion
    // varies across a single detail tile.
    let uv = (world_pos.xz + cloud_layer.noise_offset.xz) / (cloud_layer.detail_noise_scale * 4.0);
    return textureSampleLevel(curl_noise_2d, noise_sampler_3d, uv, 0.0).rgb;
}

// --- Vertical profile (NUBIS-style) ---
//
// In Nubis/UE the "top type" and "bottom type" are used to sample 2D profile lookup textures:
// - x axis: type
// - y axis: height in layer
//
// We don't have those LUT textures bound yet, so we approximate them with analytic curves that are:
// - distinct for top vs bottom
// - smoothly varying with type
//
// If/when we add real LUT textures, these become simple texture samples.
fn cloud_bottom_profile(h: f32, bottom_type: f32) -> f32 {
    // Bottom profile: controls how quickly density builds from the base.
    // Type 0: thin base / slow build.
    // Type 1: thick base / fast build.
    let knee = mix(0.35, 0.08, bottom_type);
    let x = smoothstep(0.0, knee, h);
    let exp = mix(2.4, 0.75, bottom_type);
    return pow(x, exp);
}

fn cloud_top_profile(h: f32, top_type: f32) -> f32 {
    // Top profile: controls how quickly density fades near the cap.
    // Type 0: hard cap (more anvil-ish)
    // Type 1: soft, billowy fade
    let t = 1.0 - h;
    let knee = mix(0.10, 0.45, top_type);
    let x = smoothstep(0.0, knee, t);
    let exp = mix(0.9, 3.2, top_type);
    return pow(x, exp);
}

/// Result of sampling the cloud field at a point.
struct CloudSample {
    /// Shape density in [0, 1] (multiply by `cloud_layer.cloud_density` for medium density).
    density: f32,
    /// Dimensional profile in [0, 1]: coverage x vertical profile, before noise shaping.
    /// Used by the lighting model (NUBIS-style ambient scattering).
    profile: f32,
    /// Normalized height within the cloud layer in [0, 1].
    height: f32,
}

/// Wind shear: skew sample positions along the wind direction with height, so cloud
/// tops trail their bases slightly.
const WIND_SHEAR_M: f32 = 500.0;

/// Sample the full cloud field at a point.
fn sample_cloud(r: f32, world_pos: vec3<f32>) -> CloudSample {
    var out: CloudSample;
    out.density = 0.0;
    out.profile = 0.0;
    out.height = 0.0;

    // Check if we're within the cloud layer.
    if (r < cloud_layer.cloud_layer_start || r > cloud_layer.cloud_layer_end) {
        return out;
    }

    let layer_thickness = cloud_layer.cloud_layer_end - cloud_layer.cloud_layer_start;
    let h = clamp((r - cloud_layer.cloud_layer_start) / max(1.0, layer_thickness), 0.0, 1.0);
    out.height = h;

    // 1. Dimensional profile: coverage x vertical profile.
    let macro_map = sample_cloud_macro_map(world_pos);

    // Global coverage control: 0 clears the sky, 1 fills it. The macro FBM
    // concentrates around 0.5, so treat `coverage` as a sliding threshold and
    // rebuild a high-contrast signal above it: cloud interiors saturate to full
    // coverage while edges ramp off softly.
    let coverage_threshold = 1.0 - cloud_layer.coverage;
    var coverage = remap(macro_map.r, coverage_threshold, coverage_threshold + 0.15, 0.0, 1.0);
    // Small-scale coverage variation breaks up the macro structure.
    coverage = saturate(coverage + (macro_map.a - 0.5) * 0.2);
    if (coverage <= 0.0) {
        return out;
    }

    let bottom_profile = cloud_bottom_profile(h, macro_map.g);
    let top_profile = cloud_top_profile(h, macro_map.b);
    let vertical_profile = saturate(bottom_profile * top_profile);

    let dimensional_profile = vertical_profile * coverage;
    out.profile = dimensional_profile;
    if (dimensional_profile <= 0.0) {
        return out;
    }

    // Height-based wind shear applied to the 3D noise lookups.
    let wind_dir = cloud_layer.wind_velocity;
    let wind_speed2 = dot(wind_dir, wind_dir);
    var shaped_pos = world_pos;
    if (wind_speed2 > 1e-6) {
        shaped_pos += (wind_dir * inverseSqrt(wind_speed2)) * (h * WIND_SHEAR_M);
    }

    // 2. Base shape: Perlin-Worley remapped by Worley FBM octaves (GPU Pro 7),
    //    then value-eroded by the dimensional profile (NUBIS):
    //    density = saturate((noise - (1 - profile)) / profile)
    let pw = sample_shape_noise(shaped_pos);
    let shape_fbm = pw.g * 0.625 + pw.b * 0.25 + pw.a * 0.125;
    let shape = remap(pw.r, shape_fbm - 1.0, 1.0, 0.0, 1.0);
    var density = remap(shape, 1.0 - dimensional_profile, 1.0, 0.0, 1.0);

    // 3. Detail erosion at cloud edges, distorted by curl noise (HZD).
    if (density > 0.0 && cloud_layer.detail_strength > 0.0) {
        // Curl distortion is strongest near the base for wispy, turbulent bottoms.
        let curl = sample_curl_noise(world_pos);
        let detail_pos = shaped_pos + curl * (cloud_layer.curl_strength * (1.0 - h));

        let dn = sample_detail_noise(detail_pos);
        let detail_fbm = dn.r * 0.625 + dn.g * 0.25 + dn.b * 0.125;

        // Wispy shapes (direct Worley) at the base transition to billowy shapes
        // (inverted Worley) at the top.
        let detail_mod = mix(detail_fbm, 1.0 - detail_fbm, saturate(h * 5.0));

        // Erode most aggressively at the edges (low density) and leave cloud
        // interiors intact.
        let erosion = detail_mod * cloud_layer.detail_strength * (1.0 - density);
        density = remap(density, erosion, 1.0, 0.0, 1.0);
    }

    out.density = density;
    return out;
}

/// Get cloud scattering coefficient per unit density
fn get_cloud_scattering_coeff() -> f32 {
    return cloud_layer.cloud_scattering;
}

/// Get cloud absorption coefficient per unit density
fn get_cloud_absorption_coeff() -> f32 {
    return cloud_layer.cloud_absorption;
}

/// Cloud *medium density* used for extinction / scattering integration.
/// This is the normalized shape density scaled by `cloud_layer.cloud_density`.
fn get_cloud_medium_density(r: f32, world_pos: vec3<f32>) -> f32 {
    return sample_cloud(r, world_pos).density * cloud_layer.cloud_density;
}

/// Returns (density, grad_mag) for the cloud field.
/// - density is the normalized shape density in [0,1]
/// - grad_mag is an estimate of |∇density| in 1/m, useful for adaptive stepping.
fn sample_cloud_field_density_and_grad(r: f32, world_pos: vec3<f32>) -> vec2<f32> {
    // Outside cloud layer => empty and flat field.
    if (r < cloud_layer.cloud_layer_start || r > cloud_layer.cloud_layer_end) {
        return vec2(0.0, 0.0);
    }

    // Approximate gradient magnitude with finite differences in the local tangent
    // plane (cheap-ish and good enough for adaptive substepping decisions).
    let d0 = sample_cloud(r, world_pos).density;
    const EPS_M: f32 = 500.0;
    // Sample along local tangent directions to avoid directional bias / skew.
    let up = safe_normalize(world_pos);
    let east = safe_normalize(vec3(-up.z, 0.0, up.x));
    let north = safe_normalize(cross(up, east));
    let dx = sample_cloud(r, world_pos + east * EPS_M).density;
    let dz = sample_cloud(r, world_pos + north * EPS_M).density;
    // |∇density| ≈ sqrt((dd/dx)^2 + (dd/dz)^2)
    let ddx = abs(dx - d0) / EPS_M;
    let ddz = abs(dz - d0) / EPS_M;
    let grad_mag = sqrt(ddx * ddx + ddz * ddz);
    return vec2(d0, grad_mag);
}

/// Returns (start_t, end_t, valid, first_exit_t) for the span of the ray that can contain clouds.
/// - start_t is the first entry into the cloud shell, end_t the *last* exit (meters along ray_dir).
/// - valid is 1.0 if the span exists, 0.0 otherwise.
/// - first_exit_t is the first exit from the shell after start_t: either dipping below the
///   bottom sphere or leaving through the top sphere. For rays that graze under the bottom of
///   the layer and re-enter it beyond the dip, [start_t, end_t] covers both cloud segments
///   (the gap in between has zero cloud density), while [start_t, first_exit_t] covers only
///   the first one.
///
/// Spanning to the last exit keeps the interval *continuous* as a ray crosses the bottom-shell
/// tangent direction. Cutting the span at the first bottom-sphere entry (as Unreal-style single
/// interval selection does) deletes all clouds beyond the dip and produces a hard band parallel
/// to the horizon when viewed from inside or above the layer.
fn cloud_layer_segment(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> vec4<f32> {
    // IMPORTANT: do the ray/sphere intersection math in kilometers to improve numerical stability.
    // In meters, `r` is ~6.3e6 and for grazing angles the discriminant can become ill-conditioned,
    // producing "banded" invalid intersections as the sun elevation changes.
    // Doing the math in km keeps magnitudes ~6.3e3 and greatly reduces cancellation.
    const M_TO_KM: f32 = 0.001;
    const KM_TO_M: f32 = 1000.0;

    let r_km = length(ray_origin) * M_TO_KM;
    let up = normalize(ray_origin);
    let mu = dot(ray_dir, up);

    let bottom_radius_km = cloud_layer.cloud_layer_start * M_TO_KM;
    let top_radius_km = cloud_layer.cloud_layer_end * M_TO_KM;

    let t_top = ray_sphere_intersect(r_km, mu, top_radius_km);
    // The top sphere (which contains the whole shell) is missed or fully behind the origin.
    if (t_top.y <= 0.0) {
        return vec4(0.0);
    }
    let t_bottom = ray_sphere_intersect(r_km, mu, bottom_radius_km);

    // Entry into the shell, by origin position:
    // - above the layer: when crossing the top sphere
    // - below the layer (inside the bottom sphere): when leaving the bottom sphere
    // - inside the layer: immediately
    var t_enter: f32;
    if (r_km >= top_radius_km) {
        t_enter = t_top.x;
    } else if (r_km <= bottom_radius_km) {
        t_enter = t_bottom.y;
    } else {
        t_enter = 0.0;
    }
    t_enter = max(0.0, t_enter);

    // The last exit is always through the top sphere.
    let t_exit = t_top.y;

    // First exit after entry: dip into the bottom sphere if it lies ahead, else the top exit.
    var t_first_exit = t_exit;
    if (t_bottom.x > t_enter) {
        t_first_exit = t_bottom.x;
    }

    let valid = t_exit > t_enter;
    return vec4(
        t_enter * KM_TO_M,
        t_exit * KM_TO_M,
        select(0.0, 1.0, valid),
        t_first_exit * KM_TO_M,
    );
}
