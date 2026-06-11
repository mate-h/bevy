// Tileable 2D cloud-map noise generator.
//
// `main` writes the packed cloud macro map (sampled over world XZ):
// - R: coverage (macro placement)
// - G: bottom type (vertical profile shaping)
// - B: top type (vertical profile shaping)
// - A: small-scale coverage variation
//
// `curl_main` writes a tileable curl-noise texture (RGB, signed) used to distort
// cloud detail sampling at cloud edges (Schneider, "The Real-Time Volumetric
// Cloudscapes of Horizon Zero Dawn", SIGGRAPH 2015).

@group(0) @binding(13) var noise_texture_out: texture_storage_2d<rgba16float, write>;

// Parameters for FBM noise generation
struct FbmParams {
    octaves: u32,
    frequency: f32,
    amplitude: f32,
    lacunarity: f32,
    persistence: f32,
}

@group(0) @binding(14) var<uniform> fbm_params: FbmParams;

fn hash_2d(p: vec2<f32>, salt: f32) -> f32 {
    let p3 = fract(vec3(p.x, p.y, p.x + salt) * 0.1031);
    let s = dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * (p3.z + s));
}

// Tileable value noise in [-1, 1]: the lattice wraps at `period`, so `uv` in [0, 1)
// tiles seamlessly when sampled with AddressMode::Repeat.
fn value_noise_2d(uv: vec2<f32>, period: f32, salt: f32) -> f32 {
    let p = uv * period;
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let i00 = (i + vec2(0.0, 0.0)) % vec2(period);
    let i10 = (i + vec2(1.0, 0.0)) % vec2(period);
    let i01 = (i + vec2(0.0, 1.0)) % vec2(period);
    let i11 = (i + vec2(1.0, 1.0)) % vec2(period);

    let a = hash_2d(i00, salt);
    let b = hash_2d(i10, salt);
    let c = hash_2d(i01, salt);
    let d = hash_2d(i11, salt);

    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 2.0 - 1.0;
}

// Tileable FBM in [-1, 1].
fn fbm_2d(uv: vec2<f32>, base_period: f32, octaves: u32, persistence: f32, salt: f32) -> f32 {
    var period = base_period;
    var amplitude = 1.0;
    var sum = 0.0;
    var norm = 0.0;
    for (var i = 0u; i < octaves; i++) {
        sum += amplitude * value_noise_2d(uv, period, salt + f32(i) * 17.0);
        norm += amplitude;
        period *= 2.0;
        amplitude *= persistence;
    }
    return sum / max(norm, 1e-5);
}

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let texture_size = textureDimensions(noise_texture_out);

    if (gid.x >= texture_size.x || gid.y >= texture_size.y) {
        return;
    }

    // Normalized coordinates [0, 1)
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(texture_size);

    let base_period = max(1.0, round(fbm_params.frequency * 3.0));

    // R: coverage (macro placement). Many octaves so the 512x512 map carries
    // detail beyond the macro structure.
    let coverage = clamp(
        fbm_2d(uv, base_period, max(fbm_params.octaves, 7u), fbm_params.persistence, 0.0) * 0.5 + 0.5,
        0.0,
        1.0,
    );

    // G/B: "type" controls used to shape the vertical profile (NUBIS-style top/bottom types).
    // Low frequency so types vary slowly over the world.
    let bottom_type = clamp(fbm_2d(uv, 2.0, 2u, 0.5, 37.0) * 0.5 + 0.5, 0.0, 1.0);
    let top_type = clamp(fbm_2d(uv, 2.0, 2u, 0.5, 71.0) * 0.5 + 0.5, 0.0, 1.0);

    // A: higher-frequency coverage variation.
    let detail = clamp(
        fbm_2d(uv, base_period * 8.0, 5u, 0.5, 13.0) * 0.5 + 0.5,
        0.0,
        1.0,
    );

    textureStore(
        noise_texture_out,
        vec2<i32>(gid.xy),
        vec4(coverage, bottom_type, top_type, detail),
    );
}

// Curl noise (Bridson 2007): a divergence-free vector field derived from scalar
// potentials, used to swirl/distort detail-noise sample positions so eroded cloud
// edges look wispy and turbulent instead of static.
//
// XY components come from one potential (true 2D curl); Z comes from a second,
// independent potential so vertical distortion is decorrelated.
@compute
@workgroup_size(8, 8, 1)
fn curl_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let texture_size = textureDimensions(noise_texture_out);

    if (gid.x >= texture_size.x || gid.y >= texture_size.y) {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(texture_size);

    const POTENTIAL_PERIOD: f32 = 4.0;
    const POTENTIAL_OCTAVES: u32 = 4u;
    let e = 1.0 / f32(texture_size.x);

    // Potential 1 partial derivatives (central differences, wrapping is seamless
    // because the underlying noise tiles).
    let p1_x0 = fbm_2d(uv - vec2(e, 0.0), POTENTIAL_PERIOD, POTENTIAL_OCTAVES, 0.5, 101.0);
    let p1_x1 = fbm_2d(uv + vec2(e, 0.0), POTENTIAL_PERIOD, POTENTIAL_OCTAVES, 0.5, 101.0);
    let p1_y0 = fbm_2d(uv - vec2(0.0, e), POTENTIAL_PERIOD, POTENTIAL_OCTAVES, 0.5, 101.0);
    let p1_y1 = fbm_2d(uv + vec2(0.0, e), POTENTIAL_PERIOD, POTENTIAL_OCTAVES, 0.5, 101.0);

    // Potential 2, only the x derivative is needed for the z component.
    let p2_x0 = fbm_2d(uv - vec2(e, 0.0), POTENTIAL_PERIOD, POTENTIAL_OCTAVES, 0.5, 211.0);
    let p2_x1 = fbm_2d(uv + vec2(e, 0.0), POTENTIAL_PERIOD, POTENTIAL_OCTAVES, 0.5, 211.0);

    let dp1_dx = (p1_x1 - p1_x0) / (2.0 * e);
    let dp1_dy = (p1_y1 - p1_y0) / (2.0 * e);
    let dp2_dx = (p2_x1 - p2_x0) / (2.0 * e);

    // 2D curl of potential 1 in XY, independent component in Z.
    var curl = vec3(dp1_dy, -dp1_dx, dp2_dx);
    // Derivative magnitudes are O(period); normalize to roughly [-1, 1].
    curl = clamp(curl * (0.5 / POTENTIAL_PERIOD), vec3(-1.0), vec3(1.0));

    textureStore(noise_texture_out, vec2<i32>(gid.xy), vec4(curl, 0.0));
}
