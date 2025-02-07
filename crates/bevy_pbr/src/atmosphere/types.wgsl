#define_import_path bevy_pbr::atmosphere::types

struct ControlPoint {
    altitude_density: vec4<f32>,
}

struct DensityProfile {
    profile_type: u32,
    scale: f32,
    scale_high: f32,
    split_altitude: f32,
    control_points: array<ControlPoint, 6>,
    num_points: u32,
}

struct Scatterer {
    density_profile: DensityProfile,
    asymmetry: f32,
    scattering: vec3<f32>,
    absorption: vec3<f32>,
}

struct Atmosphere {
    // Planet properties
    bottom_radius: f32,
    top_radius: f32,
    ground_albedo: vec3<f32>,

    // Array of scatterers (fixed size 3)
    scatterers: array<Scatterer, 3>,
}

struct AtmosphereSettings {
    transmittance_lut_size: vec2<u32>,
    multiscattering_lut_size: vec2<u32>,
    sky_view_lut_size: vec2<u32>,
    aerial_view_lut_size: vec3<u32>,
    transmittance_lut_samples: u32,
    multiscattering_lut_dirs: u32,
    multiscattering_lut_samples: u32,
    sky_view_lut_samples: u32,
    aerial_view_lut_samples: u32,
    aerial_view_lut_max_distance: f32,
    scene_units_to_m: f32,
}


// "Atmosphere space" is just the view position with y=0 and oriented horizontally,
// so the horizon stays a horizontal line in our luts
struct AtmosphereTransforms {
    world_from_atmosphere: mat4x4<f32>,
    atmosphere_from_world: mat4x4<f32>,
}
