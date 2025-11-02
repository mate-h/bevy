use bevy_ecs::{query::QueryItem, system::lifetimeless::Read, world::World};
use bevy_math::{UVec2, Vec3Swizzles};
use bevy_render::{
    diagnostic::RecordDiagnostics,
    render_graph::{NodeRunError, RenderGraphContext, RenderLabel, ViewNode},
    render_resource::{ComputePass, ComputePassDescriptor, PipelineCache, RenderPassDescriptor},
    renderer::RenderContext,
    view::{ViewTarget, ViewUniformOffset},
};

use crate::{ExtractedAtmosphere, GpuScatteringMedium, LightMeta, ScatteringMediumSampler, resources::GpuAtmosphere, ViewLightsUniformOffset};
use bevy_render::{
    extract_component::ComponentUniforms,
    render_asset::RenderAssets,
    render_resource::{BindGroupEntries, LoadOp, Operations, RenderPassColorAttachment, StoreOp},
    renderer::RenderDevice,
    view::{Msaa, ViewDepthTexture},
};

use super::{
    GpuAtmosphereSettings,
    resources::{
        AtmosphereLutBindGroups, AtmosphereLutPipelines, AtmosphereTextures, AtmosphereTransforms,
        RenderSkyBindGroupLayouts, RenderSkyPipelineId, ViewAtmospheres, AtmosphereSampler,
    },
};

#[derive(PartialEq, Eq, Debug, Copy, Clone, Hash, RenderLabel)]
pub enum AtmosphereNode {
    RenderLuts,
    RenderSky,
    Environment,
}

#[derive(Default)]
pub(super) struct AtmosphereLutsNode {}

impl ViewNode for AtmosphereLutsNode {
    type ViewQuery = (
        Read<ViewAtmospheres>,
        Read<ViewUniformOffset>,
        Read<ViewLightsUniformOffset>,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (
            view_atmospheres,
            view_uniforms_offset,
            lights_uniforms_offset,
        ): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipelines = world.resource::<AtmosphereLutPipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let (
            Some(transmittance_lut_pipeline),
            Some(multiscattering_lut_pipeline),
            Some(sky_view_lut_pipeline),
            Some(aerial_view_lut_pipeline),
        ) = (
            pipeline_cache.get_compute_pipeline(pipelines.transmittance_lut),
            pipeline_cache.get_compute_pipeline(pipelines.multiscattering_lut),
            pipeline_cache.get_compute_pipeline(pipelines.sky_view_lut),
            pipeline_cache.get_compute_pipeline(pipelines.aerial_view_lut),
        )
        else {
            return Ok(());
        };

        let diagnostics = render_context.diagnostic_recorder();
        let command_encoder = render_context.command_encoder();

        fn dispatch_2d(compute_pass: &mut ComputePass, size: UVec2) {
            const WORKGROUP_SIZE: u32 = 16;
            let workgroups_x = size.x.div_ceil(WORKGROUP_SIZE);
            let workgroups_y = size.y.div_ceil(WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Compute LUTs for each atmosphere
        for view_atmosphere in view_atmospheres.0.iter() {
            let Ok(entity) = world.get_entity(view_atmosphere.atmosphere_entity) else {
                continue;
            };
            let Some(settings) = entity.get::<GpuAtmosphereSettings>() else {
                continue;
            };
            let Some(bind_groups) = entity.get::<AtmosphereLutBindGroups>() else {
                continue;
            };

            let mut luts_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("atmosphere_luts"),
                timestamp_writes: None,
            });
            let pass_span = diagnostics.pass_span(&mut luts_pass, "atmosphere_luts");

            // Transmittance LUT
            luts_pass.set_pipeline(transmittance_lut_pipeline);
            luts_pass.set_bind_group(
                0,
                &bind_groups.transmittance_lut,
                &[
                    view_atmosphere.atmosphere_uniform_index,
                    view_atmosphere.settings_uniform_index,
                ],
            );
            dispatch_2d(&mut luts_pass, settings.transmittance_lut_size);

            // Multiscattering LUT
            luts_pass.set_pipeline(multiscattering_lut_pipeline);
            luts_pass.set_bind_group(
                0,
                &bind_groups.multiscattering_lut,
                &[
                    view_atmosphere.atmosphere_uniform_index,
                    view_atmosphere.settings_uniform_index,
                ],
            );
            luts_pass.dispatch_workgroups(
                settings.multiscattering_lut_size.x,
                settings.multiscattering_lut_size.y,
                1,
            );

            // Sky View LUT
            luts_pass.set_pipeline(sky_view_lut_pipeline);
            luts_pass.set_bind_group(
                0,
                &bind_groups.sky_view_lut,
                &[
                    view_atmosphere.atmosphere_uniform_index,
                    view_atmosphere.settings_uniform_index,
                    view_atmosphere.transform_offset.index(),
                    view_uniforms_offset.offset,
                    lights_uniforms_offset.offset,
                ],
            );
            dispatch_2d(&mut luts_pass, settings.sky_view_lut_size);

            // Aerial View LUT
            luts_pass.set_pipeline(aerial_view_lut_pipeline);
            luts_pass.set_bind_group(
                0,
                &bind_groups.aerial_view_lut,
                &[
                    view_atmosphere.atmosphere_uniform_index,
                    view_atmosphere.settings_uniform_index,
                    view_uniforms_offset.offset,
                    lights_uniforms_offset.offset,
                ],
            );
            dispatch_2d(&mut luts_pass, settings.aerial_view_lut_size.xy());

            pass_span.end(&mut luts_pass);
        }

        Ok(())
    }
}

#[derive(Default)]
pub(super) struct RenderSkyNode;

impl ViewNode for RenderSkyNode {
    type ViewQuery = (
        Read<ViewAtmospheres>,
        Read<ViewTarget>,
        Read<ViewDepthTexture>,
        Read<ViewUniformOffset>,
        Read<ViewLightsUniformOffset>,
        Read<RenderSkyPipelineId>,
        Read<Msaa>,
    );

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (
            view_atmospheres,
            view_target,
            view_depth_texture,
            view_uniforms_offset,
            lights_uniforms_offset,
            render_sky_pipeline_id,
            msaa,
        ): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let render_device = world.resource::<RenderDevice>();
        let Some(render_sky_pipeline) =
            pipeline_cache.get_render_pipeline(render_sky_pipeline_id.0)
        else {
            return Ok(());
        };

        let render_sky_layouts = world.resource::<RenderSkyBindGroupLayouts>();
        let atmosphere_uniforms = world.resource::<ComponentUniforms<GpuAtmosphere>>();
        let settings_uniforms = world.resource::<ComponentUniforms<GpuAtmosphereSettings>>();
        let atmosphere_transforms = world.resource::<AtmosphereTransforms>();
        let view_uniforms = world.resource::<bevy_render::view::ViewUniforms>();
        let lights_uniforms = world.resource::<LightMeta>();
        let gpu_media = world.resource::<RenderAssets<GpuScatteringMedium>>();
        let medium_sampler = world.resource::<ScatteringMediumSampler>();
        let atmosphere_sampler = world.resource::<AtmosphereSampler>();

        let Some(atmosphere_binding) = atmosphere_uniforms.binding() else {
            return Ok(());
        };

        let Some(settings_binding) = settings_uniforms.binding() else {
            return Ok(());
        };

        let Some(transforms_binding) = atmosphere_transforms.uniforms().binding() else {
            return Ok(());
        };

        let Some(view_binding) = view_uniforms.uniforms.binding() else {
            return Ok(());
        };

        let Some(lights_binding) = lights_uniforms.view_gpu_lights.binding() else {
            return Ok(());
        };

        let diagnostics = render_context.diagnostic_recorder();
        render_context
            .command_encoder()
            .push_debug_group("render_sky");
        let time_span = diagnostics.time_span(render_context.command_encoder(), "render_sky");

        let mut render_sky_pass =
            render_context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("render_sky"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: view_target.main_texture_view(),
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Load,
                            store: StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
        let pass_span = diagnostics.pass_span(&mut render_sky_pass, "render_sky");

        render_sky_pass.set_pipeline(render_sky_pipeline);

        // Iterate over all atmospheres for this view
        for view_atmosphere in view_atmospheres.0.iter() {
            let Ok(entity) = world.get_entity(view_atmosphere.atmosphere_entity) else {
                continue;
            };
            let Some(atmosphere) = entity.get::<ExtractedAtmosphere>() else {
                continue;
            };
            let Some(textures) = entity.get::<AtmosphereTextures>() else {
                continue;
            };

            let gpu_medium = match gpu_media.get(atmosphere.medium) {
                Some(medium) => medium,
                None => continue,
            };

            // Create render_sky bind group for this atmosphere
            let render_sky_bind_group = render_device.create_bind_group(
                None,
                &pipeline_cache.get_bind_group_layout(if *msaa == Msaa::Off {
                    &render_sky_layouts.render_sky
                } else {
                    &render_sky_layouts.render_sky_msaa
                }),
                &BindGroupEntries::with_indices((
                    // uniforms
                    (0, atmosphere_binding.clone()),
                    (1, settings_binding.clone()),
                    (2, transforms_binding.clone()),
                    (3, view_binding.clone()),
                    (4, lights_binding.clone()),
                    // scattering medium luts and sampler
                    (5, &gpu_medium.density_lut_view),
                    (6, &gpu_medium.scattering_lut_view),
                    (7, medium_sampler.sampler()),
                    // atmosphere luts and sampler
                    (8, &textures.transmittance_lut.default_view),
                    (9, &textures.multiscattering_lut.default_view),
                    (10, &textures.sky_view_lut.default_view),
                    (11, &textures.aerial_view_lut.default_view),
                    (12, &**atmosphere_sampler),
                    // view depth texture
                    (13, view_depth_texture.view()),
                )),
            );

            render_sky_pass.set_bind_group(
                0,
                &render_sky_bind_group,
                &[
                    view_atmosphere.atmosphere_uniform_index,
                    view_atmosphere.settings_uniform_index,
                    view_atmosphere.transform_offset.index(),
                    view_uniforms_offset.offset,
                    lights_uniforms_offset.offset,
                ],
            );

            render_sky_pass.draw(0..3, 0..1);
        }

        pass_span.end(&mut render_sky_pass);
        
        drop(render_sky_pass);
        let command_encoder = render_context.command_encoder();
        time_span.end(command_encoder);
        command_encoder.pop_debug_group();

        Ok(())
    }
}
