// Distribution of the cloud moisture fluctuation, one row per shape-noise mip after the cloud-type
// rows of the cloud profile texture. The fluctuation is multi-scale: the standardized shape noise
// (`clouds.wesl` sample_cumulus_shape_lod: low-frequency Perlin-Worley eroded by Worley fBm) plus
// the standardized billow noise (detail texture B, low-frequency alligator) weighted by
// `params.billow_weight`, the Kolmogorov ratio of their scales (`clouds.wesl`
// cloud_fluctuation_weight). Per row, texel i:
// - R: the value the fluctuation exceeds with probability 1 - i / (width - 1) (quantile);
// - G: the distribution F(f) at f = FLUCT_MIN + i / (width - 1) * (FLUCT_MAX - FLUCT_MIN);
// - B, A: texel 0 the shape noise's mean and standard deviation, texel 1 the billow noise's.
// The statistical cloud scheme thresholds the fluctuation at probability p (Sommeria & Deardorff
// 1977), so p is the cloud fraction at every level of detail. Measured over the whole tiles.

@group(0) @binding(0) var noise: texture_3d<f32>;
@group(0) @binding(1) var<storage, read_write> bins: array<atomic<u32>, 6144>;
@group(0) @binding(2) var profile_lut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var detail: texture_3d<f32>;

struct QuantileParams {
    billow_weight: f32,
    // Mip of the detail texture sampled alongside shape mip m: m + detail_mip_offset.
    detail_mip_offset: f32,
    _pad0: f32,
    _pad1: f32,
}
@group(0) @binding(4) var<uniform> params: QuantileParams;
@group(0) @binding(5) var<storage, read_write> stats: array<f32, 32>;

// After the cloud-type rows (`cloud_profile.rs` CLOUD_PROFILE_TYPE_RES).
const FIRST_QUANTILE_ROW: u32 = 64u;
const MIPS: u32 = 8u;
const BINS: u32 = 256u;
// Bin blocks: shape noise, billow noise (both on [0, 1]), fluctuation (on [FLUCT_MIN, FLUCT_MAX]).
const SHAPE_BLOCK: u32 = 0u;
const BILLOW_BLOCK: u32 = MIPS * BINS;
const FLUCT_BLOCK: u32 = 2u * MIPS * BINS;
// Must match `clouds.wesl`.
const FLUCT_MIN: f32 = -6.0;
const FLUCT_MAX: f32 = 6.0;

// Must match clouds.wesl: remap and the shape-noise composite.
fn remap(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    let t = clamp((x - a) / max(1e-6, b - a), 0.0, 1.0);
    return mix(c, d, t);
}

fn shape_noise(pw: vec4<f32>) -> f32 {
    let wfbm = pw.y * 0.625 + pw.z * 0.25 + pw.w * 0.125;
    return saturate(remap(pw.x, wfbm - 1.0, 1.0, 0.0, 1.0));
}

fn levels() -> u32 {
    return min(textureNumLevels(noise), MIPS);
}

fn detail_level(m: u32) -> u32 {
    return min(u32(round(f32(m) + params.detail_mip_offset)), textureNumLevels(detail) - 1u);
}

fn bin_of(v: f32, lo: f32, hi: f32) -> u32 {
    return min(u32(saturate((v - lo) / (hi - lo)) * f32(BINS)), BINS - 1u);
}

@compute @workgroup_size(4, 4, 4)
fn histogram_marginals(@builtin(global_invocation_id) gid: vec3<u32>) {
    for (var m = 0u; m < levels(); m++) {
        if (all(gid < textureDimensions(noise, m))) {
            let s = shape_noise(textureLoad(noise, vec3<i32>(gid), i32(m)));
            atomicAdd(&bins[SHAPE_BLOCK + m * BINS + bin_of(s, 0.0, 1.0)], 1u);
        }
        let md = detail_level(m);
        if (all(gid < textureDimensions(detail, md))) {
            let d = textureLoad(detail, vec3<i32>(gid), i32(md)).b;
            atomicAdd(&bins[BILLOW_BLOCK + m * BINS + bin_of(d, 0.0, 1.0)], 1u);
        }
    }
}

// Mean and standard deviation of a [0, 1] histogram block.
fn moments(block: u32) -> vec2<f32> {
    var n = 0.0;
    var s1 = 0.0;
    var s2 = 0.0;
    for (var b = 0u; b < BINS; b++) {
        let c = f32(atomicLoad(&bins[block + b]));
        let v = (f32(b) + 0.5) / f32(BINS);
        n += c;
        s1 += c * v;
        s2 += c * v * v;
    }
    let mean = s1 / max(n, 1.0);
    return vec2(mean, sqrt(max(s2 / max(n, 1.0) - mean * mean, 1e-8)));
}

@compute @workgroup_size(8)
fn marginal_stats(@builtin(local_invocation_index) m: u32) {
    if (m >= levels()) {
        return;
    }
    let s = moments(SHAPE_BLOCK + m * BINS);
    let d = moments(BILLOW_BLOCK + m * BINS);
    stats[m * 4u + 0u] = s.x;
    stats[m * 4u + 1u] = s.y;
    stats[m * 4u + 2u] = d.x;
    stats[m * 4u + 3u] = d.y;
}

@compute @workgroup_size(4, 4, 4)
fn histogram_fluctuation(@builtin(global_invocation_id) gid: vec3<u32>) {
    for (var m = 0u; m < levels(); m++) {
        if (!all(gid < textureDimensions(noise, m))) {
            continue;
        }
        let md = detail_level(m);
        // The billow texture at an independent texel of its own level (the two noises are
        // independent fields; the shader samples them at unrelated tilings).
        let dd = textureDimensions(detail, md);
        let s = shape_noise(textureLoad(noise, vec3<i32>(gid), i32(m)));
        let d = textureLoad(detail, vec3<i32>((gid * 7u + vec3(3u, 5u, 11u)) % dd), i32(md)).b;
        let f = (s - stats[m * 4u]) / stats[m * 4u + 1u]
            + params.billow_weight * (d - stats[m * 4u + 2u]) / stats[m * 4u + 3u];
        atomicAdd(&bins[FLUCT_BLOCK + m * BINS + bin_of(f, FLUCT_MIN, FLUCT_MAX)], 1u);
    }
}

// One invocation per mip.
@compute @workgroup_size(8)
fn quantiles(@builtin(local_invocation_index) m: u32) {
    if (m >= levels()) {
        return;
    }
    let base = FLUCT_BLOCK + m * BINS;
    var total = 0u;
    for (var b = 0u; b < BINS; b++) {
        total += atomicLoad(&bins[base + b]);
    }
    let width = textureDimensions(profile_lut).x;
    let bin_width = (FLUCT_MAX - FLUCT_MIN) / f32(BINS);
    var cumulative = 0u;
    var b = 0u;
    for (var i = 0u; i < width; i++) {
        let u = f32(i) / f32(width - 1u);
        // R: quantile at probability u.
        let rank = u * f32(total);
        while (b < BINS - 1u && f32(cumulative + atomicLoad(&bins[base + b])) < rank) {
            cumulative += atomicLoad(&bins[base + b]);
            b++;
        }
        let frac = saturate((rank - f32(cumulative)) / f32(max(atomicLoad(&bins[base + b]), 1u)));
        let quantile = FLUCT_MIN + (f32(b) + frac) * bin_width;
        // G: the distribution at f = FLUCT_MIN + u (FLUCT_MAX - FLUCT_MIN).
        let x = u * f32(BINS);
        let xb = min(u32(x), BINS - 1u);
        var below = 0u;
        for (var k = 0u; k < xb; k++) {
            below += atomicLoad(&bins[base + k]);
        }
        let cdf = (f32(below) + (x - f32(xb)) * f32(atomicLoad(&bins[base + xb]))) / f32(max(total, 1u));
        var moments = vec2(0.0);
        if (i == 0u) {
            moments = vec2(stats[m * 4u], stats[m * 4u + 1u]);
        } else if (i == 1u) {
            moments = vec2(stats[m * 4u + 2u], stats[m * 4u + 3u]);
        }
        textureStore(profile_lut, vec2<i32>(i32(i), i32(FIRST_QUANTILE_ROW + m)),
            vec4(quantile, saturate(cdf), moments));
    }
}
