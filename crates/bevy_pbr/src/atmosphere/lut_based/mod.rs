use bevy_core_pipeline::core_3d::Camera3d;
use bevy_ecs::{
    entity::Entity,
    query::With,
    system::{Commands, Query, Res, ResMut},
};
use bevy_math::Mat4;
use bevy_render::{
    render_resource::DynamicUniformBuffer,
    renderer::{RenderDevice, RenderQueue},
    view::ExtractedView,
};

use super::Atmosphere;

mod node;

pub mod shaders {
    use bevy_asset::{weak_handle, Handle};
    use bevy_render::render_resource::Shader;

    pub const SKY_VIEW_LUT: Handle<Shader> = weak_handle!("f87e007a-bf4b-4f99-9ef0-ac21d369f0e5");
    pub const AERIAL_VIEW_LUT: Handle<Shader> =
        weak_handle!("a3daf030-4b64-49ae-a6a7-354489597cbe");
    pub const RESOLVE: Handle<Shader> = weak_handle!("09422f46-d0f7-41c1-be24-121c17d6e834");
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
