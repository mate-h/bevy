// Quantiles of the shape noise (`clouds.wesl` sample_cumulus_shape_lod: the low-frequency
// Perlin-Worley eroded by Worley fBm), written into the last row of the cloud profile texture:
// texel i holds the value the noise exceeds with probability 1 - i / 63. The erosion threshold
// at a dimensional profile p is then the quantile at 1 - p, so p is the cloud fraction
// (statistical cloud scheme, Sommeria & Deardorff 1977). Measured at mip 0 over the whole tile.

@group(0) @binding(0) var noise: texture_3d<f32>;
@group(0) @binding(1) var<storage, read_write> bins: array<atomic<u32>, 256>;
@group(0) @binding(2) var profile_lut: texture_storage_2d<rgba16float, write>;

const QUANTILE_ROW: i32 = 32;
const QUANTILES: u32 = 64u;

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
    let size = textureDimensions(noise);
    if (any(gid >= size)) {
        return;
    }
    let v = shape_noise(textureLoad(noise, vec3<i32>(gid), 0));
    atomicAdd(&bins[min(u32(v * 256.0), 255u)], 1u);
}

@compute @workgroup_size(1)
fn quantiles() {
    var total = 0u;
    for (var b = 0u; b < 256u; b++) {
        total += atomicLoad(&bins[b]);
    }
    var cumulative = 0u;
    var b = 0u;
    for (var i = 0u; i < QUANTILES; i++) {
        let rank = f32(i) / f32(QUANTILES - 1u) * f32(total);
        // Advance to the bin holding the target rank; interpolate within it.
        while (b < 255u && f32(cumulative + atomicLoad(&bins[b])) < rank) {
            cumulative += atomicLoad(&bins[b]);
            b++;
        }
        let in_bin = f32(max(atomicLoad(&bins[b]), 1u));
        let frac = saturate((rank - f32(cumulative)) / in_bin);
        let value = (f32(b) + frac) / 256.0;
        textureStore(profile_lut, vec2<i32>(i32(i), QUANTILE_ROW), vec4(value, 0.0, 0.0, 1.0));
    }
}
