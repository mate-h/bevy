#define_import_path bevy_pbr::atmosphere::bindings

#import bevy_render::view::View;

#import bevy_pbr::{
    mesh_view_types::Lights,
    atmosphere::types::Atmosphere,
}

@group(0) @binding(0) var<storage> atmosphere: Atmosphere;
@group(0) @binding(1) var<uniform> view: View;
@group(0) @binding(2) var<uniform> lights: Lights;
@group(0) @binding(3) var atmo_sampler: sampler;

@group(0) @binding(4) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(5) var multiscattering_lut: texture_2d<f32>;
