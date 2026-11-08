// Per-sample lighting of the cumulus medium (sun: light march or shadow map, multiple scattering
// from the escape series and the cloud field's diffusion solve; sky and ground ambient), shared by
// the cloud pass (`clouds_raymarch.wesl`) and the atmosphere environment probe (`functions.wesl`,
// ATMOSPHERE_ENV_PROBE), so the probe's clouds (IBL, reflections, the froxel ambient's sky
// harmonics) are lit like the visible ones. Reads the cloud field's escape (48) and diffusion
// (59, and 68 with CLOUD_SKY_SH_BOUNDARY) volumes and the cloud shadow map with its mips.

import bevy_render::maths::PI;
// The cloud field's sky and ground light (cloud_field.wesl diffusion solve), the ambient where it
// has a value.
@if(CLOUD_SKY_SH_BOUNDARY)
import package::atmosphere::bindings::cloud_field_diff_amb;
import package::atmosphere::{
    bindings::{atmosphere, settings, view, lights, atmosphere_lut_sampler, cloud_field_sdf, cloud_shadow_map},
    bruneton_functions::ray_intersects_ground,
    functions::{
        sample_transmittance_lut, sample_sky_view_lut, sample_cloud_shadow_map, cloud_shadow_frame,
        cloud_shadow_ground_uv, cloud_shadow_optical_depth, cloud_far_transmittance,
        cloud_mie_phase, CLOUD_MIE_MEAN_COS, GOLDEN_RATIO_FRAC,
    },
    clouds::{
        get_cloud_medium_density_lod, cloud_layer, cloud_field_uvw, cloud_field_voxel_m,
        CLOUD_FIELD_RES_Z, cloud_shadow_lod, CLOUD_SHADOW_WARP_A,
    },
};

// Escape probability of multiply scattered light (r) and the vertical optical depths to the
// layer top (g) and bottom (b) (cloud_field.wesl `escape`; cell from cloud_field_sdf).
@group(0) @binding(48) var cloud_field_esc: texture_3d<f32>;
// Diffuse sunlight of the cloud field per unit sun irradiance, U = 4 pi J_diff / E (r), and its
// build cell (gb) (cloud_field.wesl diffusion solve).
@group(0) @binding(59) var cloud_field_diff: texture_3d<f32>;

/// The cloud field's diffusion solution at p (0 outside the grid).
fn cloud_field_diffuse(p: vec3<f32>) -> f32 {
    return cloud_field_diffuse_sample(p, textureLoad(cloud_field_diff, vec3(0), 0).gb).r;
}

/// The published volume at p with the solution's build cell: (U, build cell, extinction); 0
/// outside the grid.
fn cloud_field_diffuse_sample(p: vec3<f32>, cell: vec2<f32>) -> vec4<f32> {
    let uvw = cloud_field_uvw(p, cell);
    if (any(uvw.xy < vec2(0.0)) || any(uvw.xy > vec2(1.0))) {
        return vec4(0.0);
    }
    return textureSampleLevel(cloud_field_diff, atmosphere_lut_sampler, saturate(uvw), 0.0);
}

/// The diffusion solution's in-scatter toward the camera, per unit sun irradiance in units of
/// U (the cloud pass's multiple-scattering source is E (.) / (4 pi) per unit sigma_s), and U.
/// The Eddington diffuse radiance is I(w) = (U + 3 F.w) / (4 pi) with the diffuse flux
/// F = -D grad U, D = 1 / (3 sigma') (cloud_field.wesl); in-scattered by the droplet phase
/// function (first moment g) it gives the source (U + 3 g F.w) / (4 pi) for light leaving along
/// w = -ray_dir (Joseph, Wiscombe & Weinman 1976, eq. 3-6: the first-moment term). U grows inward
/// from a cloud's surface (Marshak: outward flux U / 2), so faces seen from outside send more than
/// U / (4 pi). F.w from a central difference of U along the ray over one cell. Only the flux of
/// the diffusion term: the beam's own first moment (g' T'_sun along the sun) is not representable
/// by this linear expansion with g ~ 0.88 (negative radiance near sunlit faces) and stays in the
/// forward-peak beam term. IPRT C2 cube vs MYSTIC on this grid, all 9 views
/// (.clouds-handoff/validation/iprt_c2/c2_grid.py): 0.79 -> 0.89, 0.79-1.06 per view.
fn cloud_field_diffuse_toward(p: vec3<f32>, ray_dir: vec3<f32>) -> vec2<f32> {
    let cell = textureLoad(cloud_field_diff, vec3(0), 0).gb;
    let c = cloud_field_diffuse_sample(p, cell);
    let up = normalize(p);
    let v = dot(ray_dir, up);
    let hz = (cloud_layer.cloud_layer_end - cloud_layer.cloud_layer_start) / f32(CLOUD_FIELD_RES_Z);
    let eps = 0.5 / length(vec2(sqrt(max(1.0 - v * v, 0.0)) / cloud_field_voxel_m(), v / hz));
    let u_far = cloud_field_diffuse_sample(p + ray_dir * eps, cell).r;
    let u_near = cloud_field_diffuse_sample(p - ray_dir * eps, cell).r;
    let g = CLOUD_MIE_MEAN_COS;
    let sigma_p = (1.0 - g * g) * max(c.a, 1e-6);
    let du = (u_far - u_near) / (2.0 * eps);
    // Flux-limited diffusion (Levermore & Pomraning 1981, ApJ 248): F = -(lambda(R) / sigma')
    // grad U, R = |grad U| / (sigma' U), lambda = (coth R - 1/R) / R: 1/3 (Fick, D above) where
    // the field varies slowly, and |F| -> U (free streaming) where it varies within a mean free
    // path, as at the edge of the published volume (the extinction filtered toward clear cells;
    // with D alone, 15-60x fireflies in the hero, backlit and low-sun views). R from the
    // derivative along the ray. C2 cube: 0.90 -> 0.89 (c2_grid.py LIMIT=1).
    let r = abs(du) / (sigma_p * max(c.r, 1e-9));
    let r_safe = max(r, 1e-2);
    let lambda = select((1.0 / tanh(r_safe) - 1.0 / r_safe) / r_safe, 1.0 / 3.0 - r * r / 45.0, r < 1e-2);
    // -3 g F.w with w = -ray_dir: + 3 g (lambda / sigma') dU/dt along the ray.
    return vec2(max(c.r + 3.0 * g * lambda / sigma_p * du, 0.0), c.r);
}
/// Second exponential integral E2(x) = int_1^inf exp(-x t) / t^2 dt (the hemispheric escape of a
/// slab of optical depth x). Rational fit exp(-x) (x^2 + a x + b) / (x^3 + c x^2 + d x + b),
/// E2(0) = 1, max relative error 0.64 % on [0, 40] (least squares against scipy.special.expn).
fn exp_integral_e2(x: f32) -> f32 {
    return exp(-x) * (x * x + 1.6706291 * x + 0.13822509)
        / (x * x * x + 3.6251676 * x * x + 2.2005648 * x + 0.13822509);
}

/// (P_esc, tau_up, tau_down). Outside the grid: the plane-parallel P_esc from the coarse march
/// toward the layer top and bottom (cloud_far_transmittance).
fn cloud_escape_and_column(p: vec3<f32>) -> vec3<f32> {
    let uvw = cloud_field_uvw(p, textureLoad(cloud_field_sdf, vec3(0), 0).ba);
    var outside = any(uvw.xy < vec2(0.0)) || any(uvw.xy > vec2(1.0));
    @if(ATM_DBG_ESC_FALLBACK) { outside = true; }
    if (outside) {
        // Plane-parallel limit of the same escape integral (1/4pi) int exp(-tau'(w)) dw over the
        // two hemispheres, from the vertical columns: P_esc = (E2(tau'_up) + E2(tau'_down)) / 2,
        // tau' = (1 - <cos>) tau (similarity). The former constant (a slab surface's 0.5)
        // under-lit every cloud beyond the field grid (seen from far away or from space).
        let n = normalize(p);
        let tau_up = -log(max(cloud_far_transmittance(p, n, true), 1e-6));
        let tau_down = -log(max(cloud_far_transmittance(p, -n, true), 1e-6));
        let k = 1.0 - CLOUD_MIE_MEAN_COS;
        let p_esc = 0.5 * (exp_integral_e2(k * tau_up) + exp_integral_e2(k * tau_down));
        return vec3(p_esc, tau_up, tau_down);
    }
    return textureSampleLevel(cloud_field_esc, atmosphere_lut_sampler, saturate(uvw), 0.0).rgb;
}

// Short light taps toward the sun for near-surface self-shadowing detail (Nubis3 2:09:52: "the
// first two samples are taken as before"; the cloud shadow map covers the long range).
// 12 taps growing x1.4 from 20 m (~2.8 km) with a gentler LOD ramp: the old 6 doubling taps
// (1.26 km) under-estimated sun optical depth (single scatter +25-40% vs CLOUD_REFERENCE).
const LIGHT_TAP_STEP_M: f32 = 20.0;
const LIGHT_TAP_GROWTH: f32 = 1.4;
const LIGHT_TAP_LOD: f32 = 0.25;
const LIGHT_LONG_TAP_M: f32 = 1500.0;

fn cloud_phase(cos_theta: f32) -> f32 {
    // Droplet Mie phase (Jendersie & d'Eon 2023 HG + Draine fit, d = 20 um; functions.wesl).
    return cloud_mie_phase(cos_theta);
}

/// Flux-equivalent sky radiance E_sky / pi at the camera radius: the boundary value of the sky
/// side of the ambient two-stream (Marshak boundary condition: incoming hemispheric flux) and the
/// sky part of the Lambertian ground. Cosine-weighted quadrature over the sky-view LUT (atmosphere
/// space): mu = sqrt(1 - u) for stratified u, golden-angle azimuths, so the plain mean of the
/// samples is E / pi.
const CLOUD_SKY_IRR_DIRS: u32 = 16u;
fn sample_sky_irradiance(r: f32) -> vec3<f32> {
    var sum = vec3(0.0);
    for (var i = 0u; i < CLOUD_SKY_IRR_DIRS; i++) {
        let u = (f32(i) + 0.5) / f32(CLOUD_SKY_IRR_DIRS);
        let mu = sqrt(1.0 - u);
        let s = sqrt(u);
        let a = f32(i) * 2.399963;
        sum += sample_sky_view_lut(r, vec3(s * sin(a), mu, -s * cos(a)));
    }
    return sum / f32(CLOUD_SKY_IRR_DIRS);
}

/// Isotropic ambient in-scatter radiance (per unit sigma_s): the mean intensity of the diffuse
/// field driven by the sky above (`sky_rad`, mean sky radiance) and the ground below
/// (`ground_rad`, Lambertian bounce of sun + sky); Nubis Cubed's sky-ward ambient attenuation
/// (talk 135:16) in physical form. A conservative medium leaves a uniform isotropic field
/// unchanged, so the in-cloud gradient comes from the sky/ground contrast: the diffusion
/// (two-stream) solution of a slab in similarity-reduced depth tau' = (1 - g) tau, with the
/// extrapolation length z0 = 2/3,
///   J = L_sky + (L_ground - L_sky) (tau'_up + z0) / (tau'_up + tau'_down + 2 z0).
/// No cloud: (L_sky + L_ground) / 2. vs MC (validation/sky/amb_slab_mc.py): within 3 %.
const CLOUD_DIFFUSION_Z0: f32 = 2.0 / 3.0;
fn cloud_ambient_term(sky_rad: vec3<f32>, ground_rad: vec3<f32>, tau_up: f32, tau_down: f32) -> vec3<f32> {
    let k = 1.0 - CLOUD_MIE_MEAN_COS;
    let w = (k * tau_up + CLOUD_DIFFUSION_Z0) / (k * (tau_up + tau_down) + 2.0 * CLOUD_DIFFUSION_Z0);
    return mix(sky_rad, ground_rad, w);
}

/// The cloud field's published sky and ground light at p (`cloud_field.wesl` diffusion_publish):
/// rgb = 4 pi J pre-exposed times a, a = the interpolated weight of the cells that hold a value
/// (so rgb / a interpolates over those cells only); 0 outside the grid.
@if(CLOUD_SKY_SH_BOUNDARY)
fn cloud_field_ambient_sample(p: vec3<f32>, cell: vec2<f32>) -> vec4<f32> {
    let uvw = cloud_field_uvw(p, cell);
    if (any(uvw.xy < vec2(0.0)) || any(uvw.xy > vec2(1.0))) {
        return vec4(0.0);
    }
    return textureSampleLevel(cloud_field_diff_amb, atmosphere_lut_sampler, saturate(uvw), 0.0);
}

/// The sky and ground light's in-scatter toward the camera from the cloud field's diffusion
/// solve (CLOUD_SKY_SH_BOUNDARY): rgb = 4 pi J + 3 g F.w in absolute radiance (the ambient
/// mean intensity is rgb / (4 pi), in place of `cloud_ambient_term`), a = 1 where the solve has a
/// value, 0 where it has none (beyond the grid, or farther than a cell from the solve's cloud
/// cells: the slab term stands in there). The flux term is `cloud_field_diffuse_toward`'s, per
/// channel, with the same Levermore-Pomraning limiter: the sky's diffuse field obeys the same
/// P1 radiance I = (U + 3 F.w) / (4 pi), and like the sun's it grows inward from the Marshak
/// faces, so faces seen from outside send more than U / (4 pi).
@if(CLOUD_SKY_SH_BOUNDARY)
fn cloud_field_ambient_toward(p: vec3<f32>, ray_dir: vec3<f32>) -> vec4<f32> {
    let cell = textureLoad(cloud_field_diff, vec3(0), 0).gb;
    let s = cloud_field_ambient_sample(p, cell);
    if (s.a <= 0.0) {
        return vec4(0.0);
    }
    let u = s.rgb / s.a;
    let c = cloud_field_diffuse_sample(p, cell);
    let up = normalize(p);
    let v = dot(ray_dir, up);
    let hz = (cloud_layer.cloud_layer_end - cloud_layer.cloud_layer_start) / f32(CLOUD_FIELD_RES_Z);
    let eps = 0.5 / length(vec2(sqrt(max(1.0 - v * v, 0.0)) / cloud_field_voxel_m(), v / hz));
    let s_far = cloud_field_ambient_sample(p + ray_dir * eps, cell);
    let s_near = cloud_field_ambient_sample(p - ray_dir * eps, cell);
    let u_far = select(u, s_far.rgb / max(s_far.a, 1e-6), s_far.a > 0.0);
    let u_near = select(u, s_near.rgb / max(s_near.a, 1e-6), s_near.a > 0.0);
    let g = CLOUD_MIE_MEAN_COS;
    let sigma_p = (1.0 - g * g) * max(c.a, 1e-6);
    let du = (u_far - u_near) / (2.0 * eps);
    let r = abs(du) / (sigma_p * max(u, vec3(1e-9)));
    let r_safe = max(r, vec3(1e-2));
    let lambda = select((1.0 / tanh(r_safe) - 1.0 / r_safe) / r_safe, vec3(1.0 / 3.0) - r * r / 45.0, r < vec3(1e-2));
    return vec4(max(u + 3.0 * g * lambda / sigma_p * du, vec3(0.0)) / view.exposure, 1.0);
}

/// Diffuse (multiply scattered) sunlight in a conservative cloud slab lit from above at cosine
/// mu0, per unit sun irradiance: M = 4 pi J_diff / E at optical depth tau_up below the top of a
/// slab of optical depth tau_up + tau_down (the shader's multiple-scattering source is
/// E M / (4 pi) per unit sigma_s). Delta-Eddington (Joseph, Wiscombe & Weinman 1976): forward
/// peak g^2 left in the direct beam, scaled depths tau' = (1 - g^2) tau, scaled asymmetry
/// g' = g / (1 + g); with U = 4 pi J_diff, U'' = -3 E (1 + g') exp(-tau'/mu0) and diffuse flux
/// F = -U'/3 + g' mu0 E exp(-tau'/mu0), so U = A + B tau' + c exp(-tau'/mu0),
/// c = -3 (1 + g') mu0^2, with A, B from Marshak boundaries (no diffuse light entering: U/4 + F/2
/// = 0 at the top, U/4 - F/2 = 0 at the bottom). Against a slab Monte Carlo with the droplet phase
/// function (.clouds-handoff/validation/sky/ms_vertical_slab.py), max(this, the escape series):
/// 1.06-1.23 x the MC (5-95 %: 0.84-1.39) at optical depths 20-60 and sun cosines 0.3-1, where the
/// series alone gives 0.00-1.0 (~0 on the shaded side); thin slabs (2-5) stay under (0.4-0.8).
fn cloud_diffuse_sun(tau_up: f32, tau_down: f32, mu0: f32) -> f32 {
    let g = CLOUD_MIE_MEAN_COS;
    let s = 1.0 - g * g;
    let gs = g / (1.0 + g);
    let tp = max(tau_up, 0.0) * s;
    let t_total = (max(tau_up, 0.0) + max(tau_down, 0.0)) * s;
    let c = -3.0 * (1.0 + gs) * mu0 * mu0;
    let k = c / (3.0 * mu0) + gs * mu0;
    let e_t = exp(-t_total / mu0);
    let r1 = -(0.25 * c + 0.5 * k);
    let r2 = -(0.25 * c - 0.5 * k) * e_t;
    let det = 0.0625 * t_total + 1.0 / 12.0;
    let a = (r1 * (0.25 * t_total + 1.0 / 6.0) + r2 / 6.0) / det;
    let b = 0.25 * (r2 - r1) / det;
    return max(a + b * tp + c * exp(-tp / mu0), 0.0);
}

/// Mean sun visibility of the ground a point at `p` sees below it: the ground radiance enters the
/// ambient as a mean intensity (uniform in solid angle, i.e. uniform in mu = cos(nadir angle)),
/// so it averages the ground over kilometres. Sampling only the point straight below projected
/// the crisp cloud-shadow pattern on the ground back onto the cloud bases, offset by
/// H tan(sun zenith) (it lined up only with the sun overhead). Quadrature: midpoint nodes
/// mu = 1/4, 3/4 x 8 azimuths (staggered); ground point at horizontal distance
/// H sqrt(1 - mu^2) / mu. The ground's sun visibility is a sharp pattern, and the nodes sweep
/// across it as the height changes: with 8 nodes each crossing stepped the average by 1/8 (16 nodes: 1/16),
/// which showed as fixed horizontal bands on the shadowed sides of clouds (ambient-lit).
///
/// Area averages of the cloud shadow map's mip chain (`cloud_shadow_mips.wesl`, transmittance
/// averaged per 2x2) replace the 16 point taps, which with their off-map branches were the
/// largest inlined code in the cloud pass after the far-field march. The mean over mu of the ring
/// averages, integrated by parts, is a mean of disc averages D(rho) around the point below p:
///   V = int D(rho) w(rho) drho,  w = 1.5 H rho^3 / (H^2 + rho^2)^(5/2),
/// whose distribution function is F = 1 - 1.5 mu + 0.5 mu^3 with mu = H / sqrt(H^2 + rho^2).
/// Two equal-weight nodes at its quartile midpoints (F = 1/4, 3/4: mu = 0.5579, 0.1683) are the
/// discs rho = 1.49 H and 5.86 H. Each disc is read at the mip whose texels cover the square of
/// equal area, seen along the light (foreshortened by mu_sun). The mips filter over the whole
/// disc, so nothing steps across the shadow pattern as the height changes.
/// `ATM_DBG_GROUND_VIS_TAPS`: the 16 point taps. `ATM_DBG_GROUND_VIS_DENSE`: 512 taps (reference).
/// `ATM_DBG_GROUND_VIS_ONE`: no cloud shadow (ablation).
const GROUND_VIS_ZONE_RADII: vec2<f32> = vec2(1.4877, 5.8587);
fn ground_sun_visibility(p: vec3<f32>, l: vec3<f32>) -> f32 {
    @if(ATM_DBG_GROUND_VIS_ONE) { return 1.0; }
    @if(ATM_DBG_GROUND_VIS_TAPS) { return ground_sun_visibility_taps(p, l); }
    @if(ATM_DBG_GROUND_VIS_DENSE) { return ground_sun_visibility_dense(p, l); }
    let r_g = atmosphere.inner_radius;
    let up = normalize(p);
    let height = max(length(p) - r_g, 0.0);
    let p_g = up * r_g;
    let frame = cloud_shadow_frame(l);
    let uv = cloud_shadow_ground_uv(frame, p_g);
    if (uv.x < 0.0) {
        return cloud_far_transmittance(p_g, l, true);
    }
    let extent = settings.cloud_shadow_map_extent;
    let size = f32(settings.cloud_shadow_map_size.x);
    let levels = textureNumLevels(cloud_shadow_map);
    // Side of the square with the area of a disc of radius 1 projected along the light, in units
    // of the warped map's texel area: cloud_shadow_lod divides by the tangential texel size, and
    // the radial one is (a + 3 (1 - a) rho^2) / (a + (1 - a) rho^2) times larger (up to 2.75 x at
    // the rim), so selecting by the tangential size alone averaged over too large an area.
    let rho_uv2 = dot(2.0 * uv - 1.0, 2.0 * uv - 1.0);
    let radial = (CLOUD_SHADOW_WARP_A + 3.0 * (1.0 - CLOUD_SHADOW_WARP_A) * rho_uv2)
        / (CLOUD_SHADOW_WARP_A + (1.0 - CLOUD_SHADOW_WARP_A) * rho_uv2);
    let side = sqrt(PI * max(abs(dot(l, up)), 0.05) / radial);
    var vis = 0.0;
    for (var i = 0u; i < 2u; i++) {
        let lod = cloud_shadow_lod(side * GROUND_VIS_ZONE_RADII[i] * height, uv, extent, size, levels);
        let m = u32(lod);
        var t = cloud_shadow_ground_bilinear(uv, m);
        if (m + 1u < levels) {
            t = mix(t, cloud_shadow_ground_bilinear(uv, m + 1u), lod - f32(m));
        }
        vis += 0.5 * t;
    }
    return vis;
}

/// Bilinear filter of the per-texel sun transmittance of the ground at mip `mip` (the texels'
/// optical depths are averaged as transmittance, so they are interpolated as transmittance).
fn cloud_shadow_ground_bilinear(uv: vec2<f32>, mip: u32) -> f32 {
    let size = vec2<i32>(textureDimensions(cloud_shadow_map, mip));
    let st = uv * vec2<f32>(size) - 0.5;
    let i0 = vec2<i32>(floor(st));
    let f = st - floor(st);
    var t: array<f32, 4>;
    for (var k = 0; k < 4; k++) {
        let c = clamp(i0 + vec2(k & 1, k >> 1), vec2(0), size - 1);
        let d = textureLoad(cloud_shadow_map, c, mip);
        t[k] = cloud_shadow_optical_depth(d.rgb) * exp(-d.a);
    }
    return mix(mix(t[0], t[1], f.x), mix(t[2], t[3], f.x), f.y);
}

/// Validation (`ATM_DBG_GROUND_VIS_DENSE`): the mean over mu with 16 midpoint nodes x 32
/// azimuths, the reference for the 2-zone mip lookup and the 16 taps.
fn ground_sun_visibility_dense(p: vec3<f32>, l: vec3<f32>) -> f32 {
    let r_g = atmosphere.inner_radius;
    let up = normalize(p);
    let height = max(length(p) - r_g, 0.0);
    let t1 = normalize(cross(up, select(vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), abs(up.x) > 0.9)));
    let t2 = cross(up, t1);
    var vis = 0.0;
    var n_off_map = 0.0;
    let frame = cloud_shadow_frame(l);
    for (var i = 0u; i < 16u; i++) {
        let mu = (f32(i) + 0.5) / 16.0;
        let rho = height * sqrt(1.0 - mu * mu) / mu;
        for (var k = 0u; k < 32u; k++) {
            let a = (f32(k) + 0.5 * f32(i & 1u)) * (6.2831853 / 32.0);
            let p_g = normalize(up * r_g + (t1 * cos(a) + t2 * sin(a)) * rho) * r_g;
            let uv = cloud_shadow_ground_uv(frame, p_g);
            if (uv.x < 0.0) {
                n_off_map += 1.0;
            } else {
                let d = textureSampleLevel(cloud_shadow_map, atmosphere_lut_sampler, uv, 0.0);
                vis += cloud_shadow_optical_depth(d.rgb) * exp(-d.a);
            }
        }
    }
    if (n_off_map > 0.0) {
        vis += n_off_map * cloud_far_transmittance(up * r_g, l, true);
    }
    return vis / 512.0;
}

fn ground_sun_visibility_taps(p: vec3<f32>, l: vec3<f32>) -> f32 {
    let r_g = atmosphere.inner_radius;
    let up = normalize(p);
    let height = max(length(p) - r_g, 0.0);
    let t1 = normalize(cross(up, select(vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), abs(up.x) > 0.9)));
    let t2 = cross(up, t1);
    var vis = 0.0;
    var n_off_map = 0.0;
    let frame = cloud_shadow_frame(l);
    for (var i = 0u; i < 2u; i++) {
        let mu = (f32(i) + 0.5) / 2.0;
        let rho = height * sqrt(1.0 - mu * mu) / mu;
        for (var k = 0u; k < 8u; k++) {
            let a = (f32(k) + 0.5 * f32(i)) * 0.78539816;
            let p_g = normalize(up * r_g + (t1 * cos(a) + t2 * sin(a)) * rho) * r_g;
            // Shadow map, optical depth only (the ground is below every cloud), one bilinear tap.
            let uv = cloud_shadow_ground_uv(frame, p_g);
            if (uv.x < 0.0) {
                n_off_map += 1.0;
            } else {
                let d = textureSampleLevel(cloud_shadow_map, atmosphere_lut_sampler, uv, 0.0);
                vis += cloud_shadow_optical_depth(d.rgb) * exp(-d.a);
            }
        }
    }
    // Ground points off the map (clouds near its 64 km edge) share one far-field march from the
    // ground below p. Marching per point inlined a full density loop into these 16 taps, inside
    // the light loop inside the view march: the cloud pass spilled 160 bytes of registers
    // (Metal compiler, Apple M5), 0 without the ambient path.
    if (n_off_map > 0.0) {
        vis += n_off_map * cloud_far_transmittance(up * r_g, l, true);
    }
    return vis / 16.0;
}

/// Lighting quality of `cloud_sample_lighting`'s sun term.
/// FULL: the local light march (`settings.cloud_light_taps` jittered taps + one long tap) through
/// the fine density, the cloud shadow map beyond it (the cloud pass).
/// SHADOW_MAP: the cloud shadow map at the sample (the environment probe).
const CLOUD_LIGHT_FULL: u32 = 0u;
const CLOUD_LIGHT_SHADOW_MAP: u32 = 1u;

struct CloudSampleLighting {
    /// In-scattered radiance toward the camera per unit sigma_s at the sample (sun: direct +
    /// multiple scattering, for every directional light; sky and ground ambient), absolute (not
    /// pre-exposed). The luminance gains sigma_s x (segment transmittance integral) x this.
    radiance: vec3<f32>,
    /// ATM_DBG_OUT_SUN_OD / ATM_DBG_OUT_SUN_TAU: the first light's values; -1 without lights.
    dbg: vec4<f32>,
}

/// The cloud medium's in-scattered radiance per unit sigma_s at `p` toward the camera along
/// `ray_dir` (Nubis Cubed structure, Schneider, Advances 2023: "Direct = Transmittance * Phase +
/// Multiple Scattering * Secondary Phase", with physically derived terms; the ambient as the
/// two-stream slab or the field's diffusion solve). Shared by the cloud pass and the atmosphere
/// environment probe so the probe's clouds are lit like the visible ones (RDR2, Bauer 2019,
/// slide 70: the sky probe "needs to match" the visible sky).
/// - `tau_view`: the view ray's optical depth inside the current cloud up to `p` (the view half
///   of the local escape estimate).
/// - `mip`: the density LOD of the sample (the light taps' base LOD; FULL quality only).
/// - `sigma_s_unit`, `sigma_t_unit`: the medium's coefficients per unit density.
/// - `sky_irr`: E_sky / pi at the camera (`sample_sky_irradiance`, once per ray).
/// - `tap_jitter`, `lit_i`: the ray's light-tap jitter and the index of this lit sample on the
///   ray (FULL quality only; the taps' phases).
/// - `quality`: `CLOUD_LIGHT_FULL` or `CLOUD_LIGHT_SHADOW_MAP`.
/// - `density`: the medium density at `p` (debug outputs only).
fn cloud_sample_lighting(
    p: vec3<f32>,
    ray_dir: vec3<f32>,
    tau_view: f32,
    mip: f32,
    sigma_s_unit: f32,
    sigma_t_unit: f32,
    sky_irr: vec3<f32>,
    tap_jitter: f32,
    lit_i: u32,
    quality: u32,
    density: f32,
) -> CloudSampleLighting {
    let r_p = length(p);
    let up = p / r_p;
    let esc_col = cloud_escape_and_column(p);
    var dbg = vec4(-1.0);

    var direct_sum = vec3(0.0);
    for (var light_i: u32 = 0u; light_i < lights.n_directional_lights; light_i++) {
        let light = &lights.directional_lights[light_i];
        let l = (*light).direction_to_light;
        let mu_light = dot(l, up);
        let t_atm = sample_transmittance_lut(r_p, mu_light)
            * f32(!ray_intersects_ground(r_p, mu_light));
        // Local light march through the real (fine) density (Nubis/HZD cone march): 6 taps with
        // doubling step (~1.3 km) + one long tap, so billows self-shadow. The cloud shadow map
        // (coarse density, 4x4 time-sliced, temporally filtered) is only used for the
        // occlusion *beyond* the local march; using it at `p` replaced all per-billow detail
        // with smooth weather-map blobs once it converged.
        var light_od = 0.0;
        var tap_t = 0.0;
        var t_far = 1.0;
        if (quality == CLOUD_LIGHT_FULL) {
            var tap_dt = LIGHT_TAP_STEP_M;
            var n_taps = settings.cloud_light_taps;
            var tap_growth = LIGHT_TAP_GROWTH;
            var tap_lod = LIGHT_TAP_LOD;
            // Validation: a dense uniform march toward the light (256 x 30 m at the sample's
            // mip) to test the light taps' quadrature against the reference.
            @if(ATM_DBG_LIGHT_BRUTE) {
                n_taps = 256u;
                tap_dt = 30.0;
                tap_growth = 1.0;
                tap_lod = 0.0;
            }
            // Jittered taps: tap k covers [t_k, t_k + dt_k) and samples it at
            // t_k + xi_k dt_k with xi_k uniform in [0, 1), a one-sample Monte Carlo
            // estimate of the interval's optical depth: E[density(t_k + xi_k dt_k) dt_k]
            // is the interval's exact integral, the same expected value as the midpoint
            // rule (xi = 0.5) had, but the midpoint rule aliased every billow thinner than
            // dt_k (and every LOD step) into self-shadow contours fixed in space that
            // rotated with the sun, which the temporal reconstruction cannot average.
            // Precedent: HZD15 samples its 6 light taps in a cone "to smooth the banding
            // we would normally get with 6 samples"; FB16 §5.5.2 jitters its progressive
            // shadow samples per frame and relies on the temporal integration.
            // xi_k = fract(xi + k phi) (golden-ratio / Kronecker sequence, see
            // `GOLDEN_RATIO_FRAC`): the taps' phases are spread, not independent draws
            // that could all land at one end. xi is the ray's second STBN channel (g),
            // decorrelated from the view-step jitter (r), advanced by the lit-sample
            // index so the lit samples of one ray do not share phases either.
            let xi = fract(tap_jitter + f32(lit_i * (n_taps + 1u)) * GOLDEN_RATIO_FRAC);
            for (var k = 0u; k < n_taps; k++) {
                let xi_k = fract(xi + f32(k) * GOLDEN_RATIO_FRAC);
                let pk = p + l * (tap_t + xi_k * tap_dt);
                let dk = get_cloud_medium_density_lod(length(pk), pk, mip + tap_lod * f32(k));
                light_od += dk * tap_dt;
                tap_t += tap_dt;
                tap_dt *= tap_growth;
            }
            // The long tap is interval n_taps of the same sequence.
            let xi_far = fract(xi + f32(n_taps) * GOLDEN_RATIO_FRAC);
            let p_far = p + l * (tap_t + LIGHT_LONG_TAP_M * xi_far);
            let d_far = get_cloud_medium_density_lod(length(p_far), p_far, mip + 3.0);
            light_od += d_far * LIGHT_LONG_TAP_M;
            let p_beyond = p + l * (tap_t + LIGHT_LONG_TAP_M);
            t_far = sample_cloud_shadow_map(p_beyond, l, false, false);
        } else {
            // CLOUD_LIGHT_SHADOW_MAP: the cloud shadow map at the sample itself (coarse
            // density, 1 texel per 4x4 tile per frame, temporally filtered; `cloud_shadow_map.wesl`),
            // the occlusion the full quality reads only beyond its local taps. No per-billow
            // self-shadow detail (the map's 512 texels over 128 km are 250 m): for the
            // environment probe, whose 128 x 128 faces resolve ~1 deg, well below it.
            t_far = sample_cloud_shadow_map(p, l, false, false);
        }
        let light_optical_depth = light_od * sigma_t_unit;
        @if(ATM_DBG_NO_FAR_SHADOW) { t_far = 1.0; }
        let t_local = exp(-light_optical_depth);
        @if(ATM_DBG_OUT_SUN_OD) {
            if (light_i == 0u) { dbg = vec4(light_optical_depth, cloud_field_diffuse(p), esc_col.x, density); }
        }
        // ATM_DBG_OUT_SUN_TAU: (sun optical depth, local march + map beyond; the map alone
        // at p; the field's diffusion solution; density) at the first in-cloud sample.
        @if(ATM_DBG_OUT_SUN_TAU) {
            if (light_i == 0u) {
                dbg = vec4(light_optical_depth - log(max(t_far, 1e-6)), -log(max(sample_cloud_shadow_map(p, l, false, false), 1e-6)), cloud_field_diffuse(p), density);
            }
        }
        let t_cloud = t_far * t_local;
        let cos_theta = dot(l, ray_dir);
        // Nubis Cubed structure (Schneider, Advances 2023, "Direct = Transmittance * Phase +
        // Multiple Scattering * Secondary Phase") with physically derived terms. Multiple
        // scattering (orders >= 2), isotropic: E_sun * exp(-(1 - g) tau_sun) * F_ms / 4pi.
        //  - exp(-(1 - g) tau_sun): diffuse sunlight under similarity scaling (Joseph,
        //    Wiscombe & Weinman 1976), tau_sun = local march + shadow-map optical depth.
        //  - F_ms = f_ms / (1 - f_ms), f_ms = albedo (1 - P_esc): Hillaire 2020's transfer
        //    series with the escape probability on the reduced optical depth (cloud field).
        // vs MC (validation/sky/ms_models_vs_mc.py, tau <= 5): median error -29..+25 %.
        var direct = t_local * t_far * cloud_phase(cos_theta);
        @if(!CLOUD_TERM_S1) {
            let tau_sun = light_optical_depth - log(max(t_far, 1e-6));
            let albedo = sigma_s_unit / max(sigma_t_unit, 1e-9);
            // P_esc of a finite cloud is positive; the escape volume stores it in
            // RGBA16F, whose smallest normal (6.1e-5) is the floor kept here, so a thick
            // conservative cloud (zero absorption) cannot make 1 - f_ms zero (audit 22).
            // Escape probability: the cloud field's (375 m cells; validated for clouds
            // larger than that) or, where smaller, a local estimate from the fine density
            // along the two directions the march already knows, toward the sun and back
            // along the view ray: P_esc ~ mean of exp(-(1 - g) tau) (similarity-reduced,
            // as the field's). Small cumuli are much smaller than a field cell, which
            // diluted their extinction there: P_esc ~ 1, no multiple scattering, and
            // their sunlit sides rendered gray although tau ~ 10-20 across them.
            let p_esc_local = 0.5 * (exp(-(1.0 - CLOUD_MIE_MEAN_COS) * tau_sun)
                + exp(-(1.0 - CLOUD_MIE_MEAN_COS) * tau_view));
            var p_esc = min(esc_col.x, p_esc_local);
            @if(ATM_DBG_ESC_ZERO) { p_esc = 0.0; }
            @if(ATM_DBG_ESC_FIELD_ONLY) { p_esc = esc_col.x; }
            let f_ms = albedo * (1.0 - max(p_esc, 6.1035e-5));
            var ms_vol = exp(-(1.0 - CLOUD_MIE_MEAN_COS) * tau_sun) * f_ms / (1.0 - f_ms);
            // Thick cloud: the escape series decays as exp(-(1 - g) tau_sun) and leaves
            // the shaded side dark (MC: ~0 below a reduced depth of a few), while the
            // diffuse light of a conservative medium falls off linearly through it. The
            // delta-Eddington diffusion solution of the vertical slab (the field's
            // columns to the layer top and bottom), lit by the sunlight that reaches the
            // column (the far shadow map), takes over where it is larger
            // (ATM_DBG_MS_DIFFUSION).
            // The cloud field's 3D diffusion solve (cloud_field.wesl): the diffuse light of
            // the whole cloud around p, lit by the first directional light. The escape
            // series stays where it is larger (thin cloud, beyond the grid).
            var field_on = light_i == 0u;
            @if(ATM_DBG_MS_SERIES_ONLY) { field_on = false; }
            if (field_on) {
                // The solve is delta-Eddington (Joseph, Wiscombe & Weinman 1976): its
                // diffuse field leaves out the light scattered into the forward peak
                // (a fraction g^2 of every scattering), which stays in the delta-scaled
                // direct beam T' = exp(-(1 - g^2) tau_sun) and is not in exp(-tau_sun).
                // That beam, T' - T, is in-scattered with the full phase function like
                // the direct sunlight. IPRT C2 cube vs MYSTIC (Emde et al. 2019), all 9
                // views: 0.67 -> 0.80 of the radiance, lowest view 0.39 -> 0.71
                // (.clouds-handoff/validation/iprt_c2/c2_flux.py). The escape series (thin
                // cloud) was fitted to all orders >= 2, the forward-peak beam included.
                var u_diff = cloud_field_diffuse_toward(p, ray_dir);
                @if(ATM_DBG_NO_DIFF_FLUX) { u_diff.x = u_diff.y; }
                let t_delta = exp(-(1.0 - CLOUD_MIE_MEAN_COS * CLOUD_MIE_MEAN_COS) * tau_sun);
                let beam_fwd = max(t_delta - t_cloud, 0.0);
                @if(!ATM_DBG_NO_FWD_BEAM) {
                    if (u_diff.y + beam_fwd > ms_vol) {
                        ms_vol = u_diff.x;
                        direct += beam_fwd * cloud_phase(cos_theta);
                    }
                }
                @if(ATM_DBG_NO_FWD_BEAM) { ms_vol = max(ms_vol, u_diff.x); }
            }
            // Experiment: the vertical-slab solution (lights only the layer top).
            @if(ATM_DBG_MS_DIFFUSION) {
                if (mu_light > 0.0) {
                    ms_vol = max(ms_vol, t_far * cloud_diffuse_sun(esc_col.y, esc_col.z, max(mu_light, 0.05)));
                }
            }
            direct += ms_vol * (0.25 / PI);
        }
        direct_sum += (*light).color.rgb * t_atm * direct;
    }
    // Sunlit ground below this sample (Lambert, albedo): the first directional light's
    // irradiance through the atmosphere times the mean cloud-shadow visibility of the
    // ground it sees. After the light loop, not in it: its 16 shadow taps kept their state
    // live through the multiple-scattering work (register spills, Apple M5).
    var ground_rad = sky_irr * atmosphere.ground_albedo;
    if (lights.n_directional_lights > 0u) {
        let light0 = &lights.directional_lights[0];
        let l0 = (*light0).direction_to_light;
        let mu_g = dot(l0, up);
        let e_sun = (*light0).color.rgb * sample_transmittance_lut(atmosphere.inner_radius, mu_g)
            * max(mu_g, 0.0) * ground_sun_visibility(p, l0);
        ground_rad = atmosphere.ground_albedo * (e_sun / PI + sky_irr);
    }
    var ambient = cloud_ambient_term(sky_irr, ground_rad, esc_col.y, esc_col.z);
    // The cloud field's diffusion solve where it has a value (CLOUD_SKY_SH_BOUNDARY):
    // there the sky and ground light enter once, through the solve's Marshak
    // boundary (cloud_field.wesl), and the slab term above stands in only beyond it.
    @if(CLOUD_SKY_SH_BOUNDARY) {
        let amb_field = cloud_field_ambient_toward(p, ray_dir);
        if (amb_field.a > 0.0) {
            ambient = amb_field.rgb * (0.25 / PI);
        }
    }
    // Validation term toggles (CLOUD_TERMS=s1|sun|amb, see resources.rs).
    @if(CLOUD_TERM_NO_AMB) { ambient = vec3(0.0); }
    @if(CLOUD_TERM_NO_SUN) { direct_sum = vec3(0.0); }
    var out: CloudSampleLighting;
    out.radiance = direct_sum + ambient;
    out.dbg = dbg;
    return out;
}
