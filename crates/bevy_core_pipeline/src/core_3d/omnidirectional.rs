//! Extraction and helpers for omnidirectional (cubemap) cameras.

use core::marker::PhantomData;

use arrayvec::ArrayVec;
use bevy_app::{App, Plugin};
use bevy_asset::Assets;
use bevy_camera::{
    primitives::CubemapFrusta,
    visibility::{OmnidirectionalVisibleEntities, RenderLayers, VisibleEntities},
    ActiveCubemapSides, Camera, Camera3d, CameraMainTextureUsages, CompositingSpace, Exposure, Hdr,
    ImageRenderTarget, OmnidirectionalProjection, RenderTarget, Viewport,
};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    entity::{EntityHashMap, EntityHashSet},
    prelude::*,
    system::{Commands, Local, Res, ResMut},
};
use bevy_image::Image;
use bevy_math::{UVec2, UVec4};
use bevy_render::{
    camera::{CameraRenderGraph, ExtractedCamera, TemporalJitter},
    extract_component::ExtractComponent,
    extract_instances::{ExtractInstance, ExtractedInstances},
    render_resource::TextureFormat,
    sync_world::{MainEntity, RenderEntity},
    view::{
        ColorGrading, ExtractedView, Msaa, NoIndirectDrawing, RenderExtractedVisibleEntities,
        RetainedViewEntity, VisibilityExtractionSystemParam,
    },
    Extract, ExtractSchedule, RenderApp,
};
use bevy_transform::components::GlobalTransform;

use crate::tonemapping::Tonemapping;

/// Maps each main-world omnidirectional camera to its render-world face cameras.
///
/// Face entities are stable across frames (like point-light shadow views) so that
/// [`RenderVisibleEntities`] added/removed tracking and binned render phases remain
/// consistent. They are despawned when the omnidirectional camera is removed.
#[derive(Resource, Default, Deref, DerefMut)]
pub struct RenderOmnidirectionalCameras(EntityHashMap<ArrayVec<(Entity, u32), 6>>);

/// Extracts components from main-world cameras to render-world cameras.
///
/// Prefer this over [`bevy_render::extract_component::ExtractComponentPlugin`] for
/// components on cameras: omnidirectional cameras extract to up to six face
/// sub-cameras, and components must be copied onto each face.
pub struct ExtractCameraComponentPlugin<C, F = ()> {
    marker: PhantomData<fn() -> (C, F)>,
}

impl<C, F> Default for ExtractCameraComponentPlugin<C, F> {
    fn default() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<C, F> Plugin for ExtractCameraComponentPlugin<C, F>
where
    C: ExtractComponent<RenderApp, F>,
    C::Out: Clone + 'static,
    F: 'static + Send + Sync,
{
    fn build(&self, app: &mut App) {
        app.add_plugins(bevy_render::sync_component::SyncComponentPlugin::<C, F>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.add_systems(
            ExtractSchedule,
            extract_camera_components::<C, F>.after(extract_omnidirectional_cameras),
        );
    }
}

/// Extracts instances from cameras into the render world, expanding omnidirectional
/// cameras onto their face sub-cameras.
pub struct ExtractCameraInstancesPlugin<EI>
where
    EI: ExtractInstance + Clone,
{
    marker: PhantomData<fn() -> EI>,
}

impl<EI> ExtractCameraInstancesPlugin<EI>
where
    EI: ExtractInstance + Clone,
{
    /// Creates a new plugin that extracts instances for cameras (including omni faces).
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<EI> Default for ExtractCameraInstancesPlugin<EI>
where
    EI: ExtractInstance + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<EI> Plugin for ExtractCameraInstancesPlugin<EI>
where
    EI: ExtractInstance + Clone,
{
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.init_resource::<ExtractedInstances<EI>>();
        render_app.add_systems(
            ExtractSchedule,
            extract_instances_from_cameras::<EI>.after(extract_omnidirectional_cameras),
        );
    }
}

/// Copies a camera component onto each corresponding render-world view entity.
pub fn extract_camera_components<C, F>(
    mut commands: Commands,
    mut previous_to_spawn_len: Local<usize>,
    omnidirectional_cameras: Res<RenderOmnidirectionalCameras>,
    query: Extract<Query<(Entity, RenderEntity, C::QueryData), C::QueryFilter>>,
) where
    C: ExtractComponent<RenderApp, F>,
    C::Out: Clone + 'static,
    F: 'static,
{
    let mut to_spawn = Vec::with_capacity(*previous_to_spawn_len);

    for (camera, render_entity, row) in &query {
        let Some(extracted_component) = C::extract_component(row) else {
            continue;
        };

        match omnidirectional_cameras.get(&camera) {
            None => {
                to_spawn.push((render_entity, extracted_component));
            }
            Some(faces) => {
                for &(face_entity, _) in faces {
                    to_spawn.push((face_entity, extracted_component.clone()));
                }
            }
        }
    }

    *previous_to_spawn_len = to_spawn.len();
    commands.try_insert_batch(to_spawn);
}

/// Extracts instances attached to cameras into [`ExtractedInstances`].
///
/// Instances remain keyed by main-world entity. Face sub-cameras look up shared
/// data via [`RenderOmnidirectionalCameras`] when needed.
pub fn extract_instances_from_cameras<EI>(
    mut extracted_instances: ResMut<ExtractedInstances<EI>>,
    query: Extract<Query<(Entity, EI::QueryData), EI::QueryFilter>>,
) where
    EI: ExtractInstance + Clone,
{
    extracted_instances.clear();

    for (camera, row) in &query {
        if let Some(extract_instance) = EI::extract(row) {
            extracted_instances.insert(camera.into(), extract_instance);
        }
    }
}

/// Extracts all omnidirectional cameras into per-face render-world cameras.
pub fn extract_omnidirectional_cameras(
    mut commands: Commands,
    images: Extract<Res<Assets<Image>>>,
    query: Extract<
        Query<(
            Entity,
            &Camera,
            &Camera3d,
            &Tonemapping,
            &CameraRenderGraph,
            &GlobalTransform,
            &OmnidirectionalVisibleEntities,
            &CubemapFrusta,
            (
                Has<Hdr>,
                Option<&CompositingSpace>,
                Option<&ColorGrading>,
                Option<&Exposure>,
                Option<&TemporalJitter>,
                Option<&RenderLayers>,
            ),
            &OmnidirectionalProjection,
            &CameraMainTextureUsages,
            &ActiveCubemapSides,
            &RenderTarget,
            &Msaa,
            Has<NoIndirectDrawing>,
        )>,
    >,
    mut omnidirectional_cameras: ResMut<RenderOmnidirectionalCameras>,
    visibility_extraction_system_param: VisibilityExtractionSystemParam,
) {
    let mut live_cameras = EntityHashSet::default();

    for (
        camera_entity,
        camera,
        camera_3d,
        tonemapping,
        camera_render_graph,
        camera_transform,
        visible_entities,
        cubemap_frusta,
        (hdr, compositing_space, color_grading, exposure, temporal_jitter, render_layers),
        projection,
        main_texture_usages,
        active_cubemap_sides,
        render_target,
        msaa,
        no_indirect_drawing,
    ) in &query
    {
        if !camera.is_active {
            continue;
        }

        let RenderTarget::Image(ref image_target) = *render_target else {
            continue;
        };
        let Some(cubemap_image) = images.get(&image_target.handle) else {
            continue;
        };

        live_cameras.insert(camera_entity);

        let view_translation = GlobalTransform::from_translation(camera_transform.translation());
        let color_grading = color_grading.cloned().unwrap_or_default();
        let cubemap_projections = bevy_camera::CubemapFaceProjections::new(projection.near);
        let target_size = cubemap_image.size();
        let target_format = if hdr {
            TextureFormat::Rgba16Float
        } else if compositing_space.is_some_and(|s| *s == CompositingSpace::Srgb) {
            TextureFormat::Rgba8Unorm
        } else {
            cubemap_image.texture_descriptor.format
        };

        // Stable face entities, matching point-light shadow view entities.
        if !omnidirectional_cameras.contains_key(&camera_entity) {
            let mut faces = ArrayVec::new();
            for face_index in 0..6u32 {
                faces.push((commands.spawn_empty().id(), face_index));
            }
            omnidirectional_cameras.insert(camera_entity, faces);
        }
        let faces = omnidirectional_cameras
            .get(&camera_entity)
            .expect("omni face entities were just ensured")
            .clone();

        for (face_index, (view_rotation, frustum)) in cubemap_projections
            .rotations
            .iter()
            .zip(cubemap_frusta.iter())
            .enumerate()
        {
            let face_entity = faces[face_index].0;

            if !active_cubemap_sides.contains(ActiveCubemapSides::from_bits_retain(1 << face_index))
            {
                // Keep the entity, but skip rendering this face this frame.
                if let Ok(mut entity_commands) = commands.get_entity(face_entity) {
                    entity_commands.remove::<(ExtractedCamera, ExtractedView)>();
                }
                continue;
            }

            let face_visible_entities = visible_entities.get(face_index);
            let mut render_visible_entities = RenderExtractedVisibleEntities::default();
            populate_render_visible_entities(
                face_visible_entities,
                &mut render_visible_entities,
                &visibility_extraction_system_param,
            );

            let retained_view_entity =
                RetainedViewEntity::new(MainEntity::from(camera_entity), None, face_index as u32);

            let face_target = ImageRenderTarget {
                handle: image_target.handle.clone(),
                scale_factor: image_target.scale_factor,
                array_layer: Some(face_index as u32),
            };

            let mut entity_commands = commands.entity(face_entity);
            entity_commands.insert((
                MainEntity::from(camera_entity),
                ExtractedView {
                    retained_view_entity,
                    clip_from_view: cubemap_projections.projection,
                    world_from_view: view_translation * *view_rotation,
                    clip_from_world: None,
                    target_format,
                    viewport: UVec4::new(0, 0, target_size.x, target_size.y),
                    color_grading: color_grading.clone(),
                    invert_culling: camera.invert_culling,
                },
                ExtractedCamera {
                    target: Some(bevy_camera::NormalizedRenderTarget::Image(face_target)),
                    viewport: Some(Viewport {
                        physical_position: UVec2::ZERO,
                        physical_size: target_size,
                        depth: match &camera.viewport {
                            Some(viewport) => viewport.depth.clone(),
                            None => 0.0..1.0,
                        },
                    }),
                    physical_viewport_size: Some(target_size),
                    physical_target_size: Some(target_size),
                    schedule: camera_render_graph.0,
                    order: camera.order,
                    output_mode: camera.output_mode,
                    msaa_writeback: camera.msaa_writeback,
                    clear_color: camera.clear_color,
                    sorted_camera_index_for_target: 0,
                    exposure: exposure
                        .map(Exposure::exposure)
                        .unwrap_or_else(|| Exposure::default().exposure()),
                    hdr,
                    compositing_space: compositing_space.copied(),
                },
                camera_3d.clone(),
                *frustum,
                *main_texture_usages,
                *tonemapping,
                // Required by `prepare_view_targets` / depth / skybox. Face cameras are not
                // the synced render entity, so they miss ExtractComponentPlugin::<Msaa>.
                *msaa,
                render_visible_entities,
            ));

            if let Some(temporal_jitter) = temporal_jitter {
                entity_commands.insert(temporal_jitter.clone());
            } else {
                entity_commands.remove::<TemporalJitter>();
            }
            if let Some(render_layers) = render_layers {
                entity_commands.insert(render_layers.clone());
            } else {
                entity_commands.remove::<RenderLayers>();
            }
            // Omni faces always use CPU draw lists. GPU frustum culling is keyed off
            // per-face frusta that are still easy to get wrong for 90° cubemap views,
            // which shows up as "only one cube appears in the probe".
            let _ = no_indirect_drawing;
            entity_commands.insert(NoIndirectDrawing);
        }
    }

    // Despawn face cameras for omnidirectional cameras that are gone or inactive.
    omnidirectional_cameras.retain(|camera_entity, faces| {
        if live_cameras.contains(camera_entity) {
            return true;
        }
        for &(face_entity, _) in faces.iter() {
            commands.entity(face_entity).despawn();
        }
        false
    });
}

fn populate_render_visible_entities(
    visible_entities: &VisibleEntities,
    render_visible_entities: &mut RenderExtractedVisibleEntities,
    visibility_extraction_system_param: &VisibilityExtractionSystemParam,
) {
    for (visibility_class, visible_mesh_entities) in visible_entities.entities.iter() {
        let render_view_visible_entities = render_visible_entities
            .classes
            .entry(*visibility_class)
            .or_default();
        render_view_visible_entities.entities.clear();
        for main_entity in visible_mesh_entities {
            let render_entity = match visibility_extraction_system_param.mapper.get(*main_entity) {
                Ok(render_entity) => render_entity.entity(),
                Err(_) => Entity::PLACEHOLDER,
            };
            render_view_visible_entities
                .entities
                .push((render_entity, MainEntity::from(*main_entity)));
        }
    }
}
