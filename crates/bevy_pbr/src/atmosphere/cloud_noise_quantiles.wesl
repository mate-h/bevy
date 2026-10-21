// Cloud-scheme tables in the rows of the cloud profile texture after the cloud-type rows
// (`clouds.wesl` cloud_thermal_excess):
// - one row per shape-noise mip, texel i at v = i / (width - 1): R the distribution F(v) of the
//   shape noise (`clouds.wesl` sample_cumulus_shape_lod: low-frequency Perlin-Worley eroded by
//   Worley fBm), G and B those of the low- and high-frequency billow noise (detail texture B and
//   A) at the mip sampled alongside, so the shader maps all three to standard normal variables
//   exactly, g = Phi^-1(F(v)). Measured over the whole tiles.
// - one row of the thermal cloud scheme (`clouds.wesl` cloud_thermal_excess), texel i at cloud
//   fraction p = (i / (width - 1))^2: R the threshold t that Y = g + r1 max(z1, 0) +
//   r2 max(z2, 0) exceeds with probability p (g, z1, z2 independent standard normal: the
//   Gaussianized shape and billow noises; r1, r2 the bubble weights), G the mean of Y - t over the cloud, B 1 / E[1 - exp(-(Y - t) / G)] over
//   the cloud (the liquid-water magnitude after the up-res compression). See `scheme`.

@group(0) @binding(0) var noise: texture_3d<f32>;
@group(0) @binding(1) var<storage, read_write> bins: array<atomic<u32>, 6144>;
@group(0) @binding(2) var profile_lut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var detail: texture_3d<f32>;

struct QuantileParams {
    // Mip of the detail texture sampled alongside shape mip m: m + detail_mip_offset.
    detail_mip_offset: f32,
    // Weight r1 of the low-frequency thermal bubbles (`clouds.wesl` cloud_bubble_weight).
    bubble_weight: f32,
    _pad1: f32,
    _pad2: f32,
}
@group(0) @binding(4) var<uniform> params: QuantileParams;

// After the cloud-type rows (`cloud_profile.rs` CLOUD_PROFILE_TYPE_RES).
const FIRST_DISTRIBUTION_ROW: u32 = 64u;
const MIPS: u32 = 8u;
const SCHEME_ROW: u32 = FIRST_DISTRIBUTION_ROW + MIPS;
const BINS: u32 = 256u;
// Bin blocks: shape noise, low- and high-frequency billow noise (all on [0, 1]).
const SHAPE_BLOCK: u32 = 0u;
const BILLOW_BLOCK: u32 = MIPS * BINS;
const BILLOW_HI_BLOCK: u32 = 2u * MIPS * BINS;
// Second bubble octave weight relative to the first: the high-frequency billows (detail A, 5
// cells per tile) against the low (B, 3 cells), (3 / 5)^(2/3). Must match `clouds.wesl`
// CLOUD_BUBBLE_OCTAVE.
const BUBBLE_OCTAVE: f32 = 0.71137866;

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

fn bin_of(v: f32) -> u32 {
    return min(u32(saturate(v) * f32(BINS)), BINS - 1u);
}

@compute @workgroup_size(4, 4, 4)
fn histogram_marginals(@builtin(global_invocation_id) gid: vec3<u32>) {
    for (var m = 0u; m < levels(); m++) {
        if (all(gid < textureDimensions(noise, m))) {
            let s = shape_noise(textureLoad(noise, vec3<i32>(gid), i32(m)));
            atomicAdd(&bins[SHAPE_BLOCK + m * BINS + bin_of(s)], 1u);
        }
        let md = detail_level(m);
        if (all(gid < textureDimensions(detail, md))) {
            let d = textureLoad(detail, vec3<i32>(gid), i32(md));
            atomicAdd(&bins[BILLOW_BLOCK + m * BINS + bin_of(d.b)], 1u);
            atomicAdd(&bins[BILLOW_HI_BLOCK + m * BINS + bin_of(d.a)], 1u);
        }
    }
}

// Distribution of a [0, 1] histogram block at v (piecewise linear within a bin).
fn cdf(block: u32, v: f32) -> f32 {
    var total = 0u;
    for (var b = 0u; b < BINS; b++) {
        total += atomicLoad(&bins[block + b]);
    }
    let x = saturate(v) * f32(BINS);
    let xb = min(u32(x), BINS - 1u);
    var below = 0u;
    for (var k = 0u; k < xb; k++) {
        below += atomicLoad(&bins[block + k]);
    }
    return saturate((f32(below) + (x - f32(xb)) * f32(atomicLoad(&bins[block + xb]))) / f32(max(total, 1u)));
}

// One invocation per texel of every mip row.
@compute @workgroup_size(64)
fn distributions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = textureDimensions(profile_lut).x;
    let i = gid.x % width;
    let m = gid.x / width;
    if (m >= levels()) {
        return;
    }
    let v = f32(i) / f32(width - 1u);
    textureStore(profile_lut, vec2<i32>(i32(i), i32(FIRST_DISTRIBUTION_ROW + m)),
        vec4(cdf(SHAPE_BLOCK + m * BINS, v), cdf(BILLOW_BLOCK + m * BINS, v),
            cdf(BILLOW_HI_BLOCK + m * BINS, v), 0.0));
}

fn phi(x: f32) -> f32 {
    return 0.39894228 * exp(-0.5 * x * x);
}

// Mills ratio R(x) = Q(x) / phi(x), Abramowitz & Stegun 7.1.26 (as clouds.wesl mills_ratio).
fn mills_ratio(x: f32) -> f32 {
    let z = abs(x) * 0.70710678;
    let k = 1.0 / (1.0 + 0.3275911 * z);
    let poly = k * (0.254829592 + k * (-0.284496736 + k * (1.421413741 + k * (-1.453152027 + k * 1.061405429))));
    let r_abs = 0.5 * poly * 2.50662827;
    if (x >= 0.0) {
        return r_abs;
    }
    let ph = phi(x);
    return (1.0 - r_abs * ph) / ph;
}

// Upper tail Q(x) = P(N > x).
fn upper(x: f32) -> f32 {
    if (x >= 0.0) {
        return mills_ratio(x) * phi(x);
    }
    return 1.0 - mills_ratio(-x) * phi(x);
}

const BUBBLE_STEPS: u32 = 32u;
const BUBBLE_RANGE: f32 = 6.0;

fn simpson_weight(j: u32) -> f32 {
    let h = BUBBLE_RANGE / f32(BUBBLE_STEPS);
    return select(select(2.0, 4.0, (j & 1u) == 1u), 1.0, j == 0u || j == BUBBLE_STEPS) * h / 3.0;
}

// For Y = g + r1 max(z1, 0) + r2 max(z2, 0) (g, z1, z2 standard normal, independent) and a
// threshold t: (P(Y > t), E[(Y - t)+], E[(1 - exp(-k (Y - t)))+]). A quarter of the mass carries
// no bubble (Y = g), a quarter each one bubble, a quarter both; given the bubbles, Y ~ N(r1 z1 +
// r2 z2, 1), integrated over z > 0 by Simpson's rule (32 x 32 steps: < 1e-3 from Monte Carlo).
fn thermal_moments(t: f32, k: f32) -> vec3<f32> {
    let r1 = params.bubble_weight;
    let r2 = r1 * BUBBLE_OCTAVE;
    let h = BUBBLE_RANGE / f32(BUBBLE_STEPS);
    var out = 0.25 * level_moments(0.0, t, k);
    for (var i = 0u; i <= BUBBLE_STEPS; i++) {
        let z1 = f32(i) * h;
        let w1 = simpson_weight(i) * phi(z1);
        out += 0.5 * w1 * (level_moments(r1 * z1, t, k) + level_moments(r2 * z1, t, k));
        for (var j = 0u; j <= BUBBLE_STEPS; j++) {
            let z2 = f32(j) * h;
            out += w1 * simpson_weight(j) * phi(z2) * level_moments(r1 * z1 + r2 * z2, t, k);
        }
    }
    return out;
}

// For Y ~ N(mu, 1): (P(Y > t), E[(Y - t)+], E[(1 - exp(-k (Y - t)))+]).
fn level_moments(mu: f32, t: f32, k: f32) -> vec3<f32> {
    let d = mu - t;
    let p_above = upper(-d);
    let excess = phi(d) + d * p_above;
    // E[exp(-k Y') 1{Y' > 0}], Y' = Y - t ~ N(d, 1): exp(-k d + k^2 / 2) Phi(d - k); written with
    // the Mills ratio where Phi(d - k) is a lower tail (no overflow).
    let z = d - k;
    var damped: f32;
    if (z < 0.0) {
        damped = mills_ratio(-z) * phi(d);
    } else {
        damped = exp(-k * d + 0.5 * k * k) * (1.0 - mills_ratio(z) * phi(z));
    }
    return vec3(p_above, excess, p_above - damped);
}

// One invocation per scheme texel.
@compute @workgroup_size(64)
fn scheme(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = textureDimensions(profile_lut).x;
    let i = gid.x;
    if (i >= width) {
        return;
    }
    let u = f32(i) / f32(width - 1u);
    let p = clamp(u * u, 1e-5, 1.0 - 1e-5);
    // Threshold: bisection on P(Y > t) = p over the support of Y.
    var lo = -8.0;
    var hi = 8.0 + params.bubble_weight * (1.0 + BUBBLE_OCTAVE) * BUBBLE_RANGE;
    for (var it = 0u; it < 48u; it++) {
        let mid = 0.5 * (lo + hi);
        if (thermal_moments(mid, 0.0).x > p) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = 0.5 * (lo + hi);
    let mean_excess = max(thermal_moments(t, 0.0).y / p, 1e-4);
    let compressed = thermal_moments(t, 1.0 / mean_excess).z / p;
    textureStore(profile_lut, vec2<i32>(i32(i), i32(SCHEME_ROW)),
        vec4(t, mean_excess, 1.0 / max(compressed, 1e-3), 1.0));
}
