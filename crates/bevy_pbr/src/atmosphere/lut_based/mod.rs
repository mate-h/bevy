use bevy_app::{App, Plugin};
use bevy_asset::load_internal_asset;
use bevy_core_pipeline::core_3d::{graph::Node3d, Camera3d};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::With,
    resource::Resource,
    system::{Commands, Query, Res, ResMut}, world::{FromWorld, World},
};
use bevy_math::{Mat4, UVec2, UVec3, Vec3};
use bevy_reflect::Reflect;
use bevy_render::{
    render_resource::{BindGroup, BindGroupLayout, DynamicUniformBuffer, Extent3d, Sampler, Shader, ShaderType, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages},
    renderer::{RenderDevice, RenderQueue},
    texture::{CachedTexture, TextureCache},
    view::ExtractedView,
};

use super::{Atmosphere, AtmosphericScattering};

mod node;

pub mod shaders {
    use bevy_asset::{weak_handle, Handle};
    use bevy_render::render_resource::Shader;

    pub const SKY_VIEW_LUT: Handle<Shader> = weak_handle!("f87e007a-bf4b-4f99-9ef0-ac21d369f0e5");
    pub const AERIAL_VIEW_LUT: Handle<Shader> =
        weak_handle!("a3daf030-4b64-49ae-a6a7-354489597cbe");
    pub const RESOLVE: Handle<Shader> = weak_handle!("09422f46-d0f7-41c1-be24-121c17d6e834");
}

pub struct LutBasedAtmospherePlugin;

impl Plugin for LutBasedAtmospherePlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, shaders::SKY_VIEW_LUT, "sky_view_lut.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, shaders::AERIAL_VIEW_LUT, "aerial_view_lut.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, shaders::RESOLVE, "resolve.wgsl", Shader::from_wgsl);

        app.register_type::<Settings>()
        .init_resource::<Layout>()
        .init_resource::<Pipelines>()

    }

    fn finish(&self, app: &mut App) {
        app   
        .add_systems(Render, (prepare_luts.in_set(RenderSet::PrepareAssets, )))
        .add_render_graph_node<node::LutsNode>(Core3d, node::LutsLabel)
        .add_render_graph_edges(Core3d, (node::LutsLabel, node::ResolveLabel))
        .add_render_graph_node<node::ResolveNode>(Core3d, node::ResolveLabel))
        .add_render_graph_edges(Core3d, (Node3d::EndMainPass, node::ResolveLabel, Node3d::Tonemapping));   
    }
}

/// This component controls the resolution of the atmosphere LUTs, and
/// how many samples are used when computing them.
///
/// The transmittance LUT stores the transmittance from a point in the
/// atmosphere to the outer edge of the atmosphere in any direction,
/// parametrized by the point's radius and the cosine of the zenith angle
/// of the ray.
///
/// The multiscattering LUT stores the factor representing luminance scattered
/// towards the camera with scattering order >2, parametrized by the point's radius
/// and the cosine of the zenith angle of the sun.
///
/// The sky-view lut is essentially the actual skybox, storing the light scattered
/// towards the camera in every direction with a cubemap.
///
/// The aerial-view lut is a 3d LUT fit to the view frustum, which stores the luminance
/// scattered towards the camera at each point (RGB channels), alongside the average
/// transmittance to that point (A channel).
#[derive(Clone, Component, Reflect, ShaderType)]
#[type_path = "bevy_pbr::atmosphere::LutBasedAtmosphericScatteringSettings"]
pub struct Settings {
    /// The size of the sky-view LUT.
    pub sky_view_lut_size: UVec2,

    /// The number of points to sample along each ray when
    /// computing the sky-view LUT.
    pub sky_view_lut_samples: u32,

    /// The number of points to sample for each slice along the z-axis
    /// of the aerial-view LUT.
    pub aerial_view_lut_samples: u32,

    /// The size of the aerial-view LUT.
    pub aerial_view_lut_size: UVec3,

    /// The maximum distance from the camera to evaluate the
    /// aerial view LUT. The slices along the z-axis of the
    /// texture will be distributed linearly from the camera
    /// to this value.
    ///
    /// units: m
    pub aerial_view_lut_max_distance: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            sky_view_lut_size: UVec2::new(400, 200),
            sky_view_lut_samples: 16,
            aerial_view_lut_size: UVec3::new(32, 32, 32),
            aerial_view_lut_samples: 10,
            aerial_view_lut_max_distance: 3.2e4,
        }
    }
}

pub struct Layout {
    sky_view_lut: BindGroupLayout,
    aerial_view_lut: BindGroupLayout,
    sampler: Sampler,
}

impl FromWorld for Layout {
    fn from_world(world: &mut World) -> Self {
        
    }
}

#[derive(Resource, Default)]
pub struct AtmosphereTransforms {
    uniforms: DynamicUniformBuffer<AtmosphereTransform>,
}

impl AtmosphereTransforms {
    #[inline]
    pub fn uniforms(&self) -> &DynamicUniformBuffer<AtmosphereTransform> {
        &self.uniforms
    }
}

#[derive(ShaderType)]
pub struct AtmosphereTransform {
    world_from_atmosphere: Mat4,
    atmosphere_from_world: Mat4,
}

#[derive(Component)]
pub struct AtmosphereTransformsOffset {
    index: u32,
}

impl AtmosphereTransformsOffset {
    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }
}

pub(super) fn prepare_atmosphere_transforms(
    views: Query<(Entity, &ExtractedView), (With<Camera3d>, With<Atmosphere>)>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut atmo_uniforms: ResMut<AtmosphereTransforms>,
    mut commands: Commands,
) {
    let atmo_count = views.iter().len();
    let Some(mut writer) =
        atmo_uniforms
            .uniforms
            .get_writer(atmo_count, &render_device, &render_queue)
    else {
        return;
    };

    for (entity, view) in &views {
        let world_from_view = view.world_from_view.compute_matrix();
        let camera_z = world_from_view.z_axis.truncate();
        let camera_y = world_from_view.y_axis.truncate();
        let atmo_z = camera_z
            .with_y(0.0)
            .try_normalize()
            .unwrap_or_else(|| camera_y.with_y(0.0).normalize());
        let atmo_y = Vec3::Y;
        let atmo_x = atmo_y.cross(atmo_z).normalize();
        let world_from_atmosphere = Mat4::from_cols(
            atmo_x.extend(0.0),
            atmo_y.extend(0.0),
            atmo_z.extend(0.0),
            world_from_view.w_axis,
        );

        let atmosphere_from_world = world_from_atmosphere.inverse();

        commands.entity(entity).insert(AtmosphereTransformsOffset {
            index: writer.write(&AtmosphereTransform {
                world_from_atmosphere,
                atmosphere_from_world,
            }),
        });
    }
}

#[derive(Component)]
pub struct Luts {
    sky_view_lut: CachedTexture,
    aerial_view_lut: CachedTexture,
}

fn prepare_luts(
    atmospheres: Query<(Entity, &Settings), (With<Camera3d>, With<AtmosphericScattering>)>,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
    mut commands: Commands,
) {
    for (entity, settings) in &atmospheres {
        let sky_view_lut = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("sky_view_lut"),
                size: Extent3d {
                    width: settings.sky_view_lut_size.x,
                    height: settings.sky_view_lut_size.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        let aerial_view_lut = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("aerial_view_lut"),
                size: Extent3d {
                    width: settings.aerial_view_lut_size.x,
                    height: settings.aerial_view_lut_size.y,
                    depth_or_array_layers: settings.aerial_view_lut_size.z,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D3,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        commands.entity(entity).insert(Luts {
            sky_view_lut,
            aerial_view_lut ,
        });
    }
}

#[derive(Component)]
pub struct BindGroups {
    pub sky_view_lut: BindGroup,
    pub aerial_view_lut: BindGroup,
}

fn prepare_bind_groups(
    atmospheres: Query<(Entity, &Luts, &GpuAtmosphere)>,
    render_device: Res<RenderDevice>,
    layout: Res<Layout>,
    mut commands: Commands,
) {
    for (entity, core_luts) in &atmospheres {
        let transmittance_lut = render_device.create_bind_group(
            "transmittance_lut_bind_group",
            &layout.transmittance_lut,
            &BindGroupEntries::with_indices((,)),
        );

        let multiscattering_lut = render_device.create_bind_group(
            "multiscattering_lut_bind_group",
            &layout.multiscattering_lut,
            &BindGroupEntries::with_indices((todo!())),
        );

        commands.entity(entity).insert(BindGroups {
            transmittance_lut,
            multiscattering_lut,
        })
    }
}
