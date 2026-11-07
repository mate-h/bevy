// Lighting field of the voxel cloud (`VoxelCloud`): the sky's cloud field (`cloud_field.wesl`) has
// ~190 m cells over 24 km, coarser than a voxel cloud (the Nubis pack's are ~0.3-0.8 km), so its
// escape probability and diffuse sunlight came out as a few blurred blocks. This field covers the
// voxel cloud's box in the box's own frame, at one cell per 2 x 2 x 2 data texels (~15.6 m for the
// pack), with the same physics as the sky field:
//   extinction: the cloud extinction per cell, from the density every cloud pass reads
//               (`cloud_column_at`, the data at the cell's mip: prefiltered to the cell);
//   escape:     escape probability of multiply scattered light over 14 directions, and the
//               optical depth up and down (the ambient's columns);
//   diffusion_source / sor_red / sor_black / diffusion_publish: the delta-Eddington diffusion
//               solve for the diffuse sunlight, warm-started every frame.
// The march reads it inside the box (`clouds_raymarch.wesl`). Paths end at the box's sides: the
// sky's clouds outside it reach the cells only through the cloud shadow map (the sun's beam).

import package::atmosphere::bindings::{atmosphere, view, lights, settings, cloud_shadow_map};
import package::atmosphere::functions::CLOUD_MIE_MEAN_COS;
@if(CLOUDS_ENABLED)
import package::atmosphere::functions::{cloud_shadow_frame, cloud_shadow_transmittance, cloud_thin_shadow_weight};
import package::atmosphere::clouds::{
    cloud_layer,
    cloud_column_at_lod,
    cloud_coarse_from_column,
    cloud_shape_from_column,
    get_cloud_scattering_coeff,
    get_cloud_absorption_coeff,
    cloud_shadow_xy_to_uv,
    VOXEL_DETAIL_TILE_M,
};

// extinction: (sigma (1/m), 0, 0, 0).
@group(0) @binding(66) var out_ext: texture_storage_3d<rgba16float, write>;
// The other steps read it.
@group(0) @binding(67) var in_ext: texture_3d<f32>;
// (P_esc, tau up, tau down, 0).
@group(0) @binding(68) var out_esc: texture_storage_3d<rgba16float, write>;
@group(0) @binding(69) var diff_src: texture_storage_3d<r32float, read_write>;
@group(0) @binding(70) var diff_u: texture_storage_3d<r32float, read_write>;
// (U per unit sun irradiance, 0, 0, sigma (1/m)), as the sky field's `cloud_field_diff`.
@group(0) @binding(71) var out_diff: texture_storage_3d<rgba16float, write>;

const SIGMA_MIN: f32 = 1e-4;
const ESC_STEPS: u32 = 12u;
const ESC_STEP_GROWTH: f32 = 1.35;
const SUN_STEPS: u32 = 16u;
const OMEGA: f32 = 1.85;
const DIRS: array<vec3<f32>, 14> = array(
    vec3(1.0, 0.0, 0.0), vec3(-1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0),
    vec3(0.0, -1.0, 0.0), vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, -1.0),
    vec3(0.57735, 0.57735, 0.57735), vec3(-0.57735, 0.57735, 0.57735),
    vec3(0.57735, -0.57735, 0.57735), vec3(-0.57735, -0.57735, 0.57735),
    vec3(0.57735, 0.57735, -0.57735), vec3(-0.57735, 0.57735, -0.57735),
    vec3(0.57735, -0.57735, -0.57735), vec3(-0.57735, -0.57735, -0.57735),
);

fn dims() -> vec3<u32> {
    return cloud_layer.voxel_field_dims;
}

fn in_grid(c: vec3<i32>) -> bool {
    return all(c >= vec3(0)) && all(c < vec3<i32>(dims()));
}

/// Edges of a cell (m) along the box's axes.
fn cell_m() -> vec3<f32> {
    return cloud_layer.voxel_box_size / vec3<f32>(dims());
}

/// Atmosphere-space centre of a cell.
fn cell_centre(c: vec3<u32>) -> vec3<f32> {
    let uvw = (vec3<f32>(c) + 0.5) / vec3<f32>(dims());
    return (cloud_layer.voxel_atmosphere_from_uvw * vec4(uvw, 1.0)).xyz;
}

/// An atmosphere-space direction in cells per metre (the box may be rotated and scaled).
fn grid_dir(dir: vec3<f32>) -> vec3<f32> {
    let m = cloud_layer.voxel_uvw_from_atmosphere;
    return (mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz) * dir) * vec3<f32>(dims());
}

/// Optical depth from a cell centre along an atmosphere-space direction until the box's side:
/// steps growing from the smallest cell edge. Returns (optical depth, distance marched).
fn march(c: vec3<u32>, dir: vec3<f32>, steps: u32) -> vec2<f32> {
    let g = grid_dir(dir);
    let centre = vec3<f32>(c) + 0.5;
    let res = vec3<f32>(dims());
    var t = 0.0;
    var dt = min(min(cell_m().x, cell_m().y), cell_m().z);
    var tau = 0.0;
    for (var s = 0u; s < steps; s++) {
        let q = centre + g * (t + 0.5 * dt);
        if (any(q < vec3(0.0)) || any(q >= res)) {
            break;
        }
        tau += textureLoad(in_ext, vec3<i32>(q), 0).r * dt;
        t += dt;
        dt *= ESC_STEP_GROWTH;
    }
    return vec2(tau, t);
}

/// Distance mip of the cloud density for a cell: the detail noise averaged over the cell (the
/// voxel up-res reads it at mip + log2(4000 m * 2.2e-3 * 128 / tile), `cloud_shape_from_column`).
fn density_mip() -> f32 {
    let cell = min(min(cell_m().x, cell_m().y), cell_m().z);
    let detail_mip = log2(max(cell, 1e-3) / (VOXEL_DETAIL_TILE_M / 128.0));
    return max(detail_mip - log2(4000.0 * 2.2e-3 * 128.0 / VOXEL_DETAIL_TILE_M), 0.0);
}

@compute @workgroup_size(4, 4, 4)
fn extinction(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= dims())) {
        return;
    }
    let p = cell_centre(id);
    let r = length(p);
    var sigma = 0.0;
    if (r >= cloud_layer.cloud_layer_start && r <= cloud_layer.cloud_layer_end) {
        let column = cloud_column_at_lod(r, p, cloud_layer.voxel_field_lod);
        if (cloud_coarse_from_column(column) > 0.0) {
            sigma = cloud_shape_from_column(column, density_mip()) * cloud_layer.cloud_density
                * (get_cloud_scattering_coeff() + get_cloud_absorption_coeff());
        }
    }
    textureStore(out_ext, id, vec4(sigma, 0.0, 0.0, 0.0));
}

/// As the sky field's escape (`cloud_field.wesl`): P_esc = (1/4pi) int exp(-tau'(w)) dw with the
/// similarity-reduced optical depth, over the 14 cube directions; and the optical depth straight
/// up and down (radially) for the ambient.
@compute @workgroup_size(4, 4, 4)
fn escape(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= dims())) {
        return;
    }
    let k = 1.0 - CLOUD_MIE_MEAN_COS;
    var p_esc = 0.0;
    for (var d = 0u; d < 14u; d++) {
        p_esc += exp(-k * march(id, DIRS[d], ESC_STEPS).x);
    }
    let up = normalize(cell_centre(id));
    let tau_up = march(id, up, ESC_STEPS).x;
    let tau_down = march(id, -up, ESC_STEPS).x;
    textureStore(out_esc, id, vec4(p_esc / 14.0, tau_up, tau_down, 0.0));
}

fn diff_g() -> vec2<f32> {
    let g = CLOUD_MIE_MEAN_COS;
    return vec2(1.0 - g * g, g / (1.0 + g));
}

/// Optical depth toward the light past `p` from the cloud shadow map (the sky's clouds and thin
/// layers beyond the box), one texel (`cloud_field.wesl` diffusion_sun_tau_beyond).
@if(CLOUDS_ENABLED)
fn sun_tau_beyond(p: vec3<f32>, l: vec3<f32>) -> f32 {
    let frame = cloud_shadow_frame(l);
    let extent = settings.cloud_shadow_map_extent;
    let rel = p - frame.anchor;
    let xy = vec2(dot(rel, frame.bx), dot(rel, frame.by));
    if (!frame.mapped || extent <= 0.0 || any(abs(xy) > vec2(extent))) {
        return 0.0;
    }
    let d_sample = dot(rel, normalize(-l)) + 2.0 * extent;
    let size = vec2<i32>(textureDimensions(cloud_shadow_map));
    let texel = clamp(vec2<i32>(cloud_shadow_xy_to_uv(xy, extent) * vec2<f32>(size)), vec2(0), size - 1);
    let q = textureLoad(cloud_shadow_map, texel, 0);
    let t = cloud_shadow_transmittance(q.rgb, d_sample) * exp(-cloud_thin_shadow_weight(p) * q.a);
    return -log(max(t, 1e-6));
}

/// Source of the diffusion: (1 + g') sigma' T'_sun per cell, the sun's optical depth marched
/// through the field to the box's side and the shadow map beyond.
@if(CLOUDS_ENABLED)
@compute @workgroup_size(4, 4, 4)
fn diffusion_source(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= dims())) {
        return;
    }
    let sigma = textureLoad(in_ext, id, 0).r;
    let gg = diff_g();
    var src = 0.0;
    if (sigma > SIGMA_MIN && lights.n_directional_lights > 0u) {
        let l = lights.directional_lights[0].direction_to_light;
        let m = march(id, l, SUN_STEPS);
        let tau = m.x + sun_tau_beyond(cell_centre(id) + l * m.y, l);
        src = (1.0 + gg.y) * gg.x * sigma * exp(-gg.x * tau);
    }
    textureStore(diff_src, id, vec4(src));
}

/// One colour of a red-black SOR sweep of -div(D grad U) = source, D = 1 / (3 sigma'), with the
/// Marshak condition toward clear cells and the box's sides (`cloud_field.wesl` sor), over cells
/// of edges `cell_m()`.
fn sor(half_id: vec3<u32>, parity: u32) {
    let id = vec3(2u * half_id.x + ((half_id.y + half_id.z + parity) & 1u), half_id.y, half_id.z);
    if (any(id >= dims())) {
        return;
    }
    let c = vec3<i32>(id);
    let sigma = textureLoad(in_ext, c, 0).r;
    if (sigma <= SIGMA_MIN) {
        textureStore(diff_u, c, vec4(0.0));
        return;
    }
    let gg = diff_g();
    let d = 1.0 / (3.0 * gg.x * sigma);
    let hm = cell_m();
    var sum = 0.0;
    var diag = 0.0;
    for (var k = 0; k < 6; k++) {
        let axis = k / 2;
        var o = vec3(0);
        o[axis] = select(-1, 1, (k & 1) == 1);
        let h = hm[axis];
        let n = c + o;
        var dn = 0.0;
        if (in_grid(n)) {
            let sn = textureLoad(in_ext, n, 0).r;
            if (sn > SIGMA_MIN) {
                dn = 1.0 / (3.0 * gg.x * sn);
            }
        }
        if (dn > 0.0) {
            let cond = 2.0 * d * dn / (d + dn) / (h * h);
            sum += cond * textureLoad(diff_u, n).r;
            diag += cond;
        } else {
            diag += 1.0 / (h * (h / (2.0 * d) + 2.0));
        }
    }
    let u = textureLoad(diff_u, c).r;
    let new_u = (textureLoad(diff_src, c).r + sum) / max(diag, 1e-30);
    textureStore(diff_u, c, vec4(max(mix(u, new_u, OMEGA), 0.0)));
}

@compute @workgroup_size(4, 4, 4)
fn sor_red(@builtin(global_invocation_id) id: vec3<u32>) {
    sor(id, 0u);
}

@compute @workgroup_size(4, 4, 4)
fn sor_black(@builtin(global_invocation_id) id: vec3<u32>) {
    sor(id, 1u);
}

/// Copies the solution into the filterable volume the march samples; clear cells next to cloud
/// take the mean of their cloud neighbours (trilinear filtering against a stored 0 halved the
/// diffuse light at every surface, `cloud_field.wesl` diffusion_publish).
@compute @workgroup_size(4, 4, 4)
fn diffusion_publish(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id >= dims())) {
        return;
    }
    let c = vec3<i32>(id);
    var sigma = textureLoad(in_ext, c, 0).r;
    var u = textureLoad(diff_u, c).r;
    if (sigma <= SIGMA_MIN) {
        var sum = 0.0;
        var sum_sigma = 0.0;
        var n = 0.0;
        for (var k = 0; k < 6; k++) {
            var o = vec3(0);
            o[k / 2] = select(-1, 1, (k & 1) == 1);
            let q = c + o;
            if (in_grid(q)) {
                let sq = textureLoad(in_ext, q, 0).r;
                if (sq > SIGMA_MIN) {
                    sum += textureLoad(diff_u, q).r;
                    sum_sigma += sq;
                    n += 1.0;
                }
            }
        }
        u = select(0.0, sum / max(n, 1.0), n > 0.0);
        sigma = select(0.0, sum_sigma / max(n, 1.0), n > 0.0);
    }
    textureStore(out_diff, c, vec4(u, 0.0, 0.0, sigma));
}
