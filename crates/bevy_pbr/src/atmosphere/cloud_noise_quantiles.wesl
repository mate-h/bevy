// Quantiles of the shape noise (`clouds.wesl` sample_cumulus_shape_lod: the low-frequency
// Perlin-Worley eroded by Worley fBm), one row per noise mip after the cloud-type rows of the
// cloud profile texture: texel i holds the value the noise exceeds with probability 1 - i / 63.
// The erosion threshold at a dimensional profile p is then the quantile at 1 - p of the sampled
// mip, so p is the cloud fraction at every level of detail (statistical cloud scheme, Sommeria &
// Deardorff 1977). Measured over the whole tile.

@group(0) @binding(0) var noise: texture_3d<f32>;
@group(0) @binding(1) var<storage, read_write> bins: array<atomic<u32>, 2048>;
@group(0) @binding(2) var profile_lut: texture_storage_2d<rgba16float, write>;

// After the cloud-type rows (`cloud_profile.rs` CLOUD_PROFILE_TYPE_RES).
const FIRST_QUANTILE_ROW: u32 = 64u;
const MIPS: u32 = 8u;
const BINS: u32 = 256u;

// Must match clouds.wesl: remap and the shape-noise composite.
fn remap(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    let t = clamp((x - a) / max(1e-6, b - a), 0.0, 1.0);
    return mix(c, d, t);
}

fn shape_noise(pw: vec4<f32>) -> f32 {
    let wfbm = pw.y * 0.625 + pw.z * 0.25 + pw.w * 0.125;
    return saturate(remap(pw.x, wfbm - 1.0, 1.0, 0.0, 1.0));
}

@compute @workgroup_size(4, 4, 4)
fn histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
    let levels = min(textureNumLevels(noise), MIPS);
    for (var m = 0u; m < levels; m++) {
        if (any(gid >= textureDimensions(noise, m))) {
            return;
        }
        let v = shape_noise(textureLoad(noise, vec3<i32>(gid), i32(m)));
        atomicAdd(&bins[m * BINS + min(u32(v * f32(BINS)), BINS - 1u)], 1u);
    }
}

// One invocation per mip.
@compute @workgroup_size(8)
fn quantiles(@builtin(local_invocation_index) m: u32) {
    if (m >= min(textureNumLevels(noise), MIPS)) {
        return;
    }
    let base = m * BINS;
    var total = 0u;
    for (var b = 0u; b < BINS; b++) {
        total += atomicLoad(&bins[base + b]);
    }
    // One quantile per texel of the row: texel i at probability i / (width - 1).
    let quantiles = textureDimensions(profile_lut).x;
    var cumulative = 0u;
    var b = 0u;
    for (var i = 0u; i < quantiles; i++) {
        let rank = f32(i) / f32(quantiles - 1u) * f32(total);
        // Advance to the bin holding the rank; interpolate within it.
        while (b < BINS - 1u && f32(cumulative + atomicLoad(&bins[base + b])) < rank) {
            cumulative += atomicLoad(&bins[base + b]);
            b++;
        }
        let in_bin = f32(max(atomicLoad(&bins[base + b]), 1u));
        let frac = saturate((rank - f32(cumulative)) / in_bin);
        let value = (f32(b) + frac) / f32(BINS);
        textureStore(profile_lut, vec2<i32>(i32(i), i32(FIRST_QUANTILE_ROW + m)), vec4(value, 0.0, 0.0, 1.0));
    }
}
