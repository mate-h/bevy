//! The froxel grid's parameters and the lookup of its integrated volume, with no bindings of
//! their own: shared by the atmosphere's passes (`froxels.wesl`, which binds the params at group 0
//! binding 70 and the volume at 76) and the mesh pipeline (`mesh_view_bindings.wesl` bindings 34
//! and 35, `pbr_functions.wesl` `apply_froxel_volumetrics` for alpha-blended meshes).

/// `GpuVolumetricsParams` (`volumetrics.rs`).
struct VolumetricsParams {
    grid_size: vec3<u32>,
    tile_px: u32,
    fog_albedo: vec3<f32>,
    /// Fog extinction at `fog_height` (1/m).
    fog_density: f32,
    /// Far end of the grid (m).
    max_distance: f32,
    /// HDRP logarithmic depth encoding: c = 2 - 2 uniformity (1/m), k = log2(c d_max + 1).
    depth_c: f32,
    depth_k: f32,
    fog_height_falloff: f32,
    fog_height: f32,
    fog_anisotropy: f32,
    enabled: u32,
    /// Scale of the sky ambient in-scatter (`VolumetricsSettings::ambient_scale`).
    ambient_scale: f32,
    /// Last frame's view-projection and camera position (the shadow volume's reprojection).
    prev_clip_from_world: mat4x4<f32>,
    prev_world_position: vec3<f32>,
    /// Whether the shadow history holds last frame's volume for this grid.
    history_valid: u32,
    /// Weight of the current frame in the shadow volume's EMA (FB15 slide 23: 5%).
    shadow_temporal_alpha: f32,
    /// The exposure the sky harmonics (binding 88) are stored with; 0 = no sky probe.
    sky_sh_exposure: f32,
    _pad2: vec2<u32>,
}

/// Normalised depth of distance `t` along a ray: u(t) = log2(c t + 1) / k (the inverse of HDRP's
/// `DecodeLogarithmicDepthGeneralized`, near plane 0).
fn froxel_params_t_to_u(p: VolumetricsParams, t: f32) -> f32 {
    return log2(p.depth_c * max(t, 0.0) + 1.0) / p.depth_k;
}

/// Where to read the integrated volume for distance `t` with a `dither` of +-0.5 slice (RDR2
/// slide 56: the dithered-lookup principle applied to the accumulation volume): `.x` the texture's
/// w coordinate, `.y` the fade of the first slice. Texel `z` holds the integral to the far
/// boundary of slice `z`, so slice coordinate `s` reads texel centre `s - 0.5`; inside the first
/// slice the integral is ramped linearly from zero, as the aerial perspective LUT's lookup does.
fn froxel_integrated_coord(p: VolumetricsParams, t: f32, dither: f32) -> vec2<f32> {
    let n = f32(p.grid_size.z);
    let s = max(froxel_params_t_to_u(p, min(t, p.max_distance)) * n + dither, 0.0);
    return vec2((max(s, 1.0) - 0.5) / n, saturate(s));
}

/// The integrated in-scatter (rgb, pre-exposed) and transmittance (a) from a trilinear sample `v`
/// of the volume at `froxel_integrated_coord`'s `.x`, faded by its `.y`.
fn froxel_integrated_fade(v: vec4<f32>, fade: f32) -> vec4<f32> {
    return vec4(v.rgb * fade, mix(1.0, v.a, fade));
}
