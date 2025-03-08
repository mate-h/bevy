use bevy_ecs::{query::{QueryItem, QueryState, With}, system::lifetimeless::Read, world::World};
use bevy_math::UVec2;
use bevy_render::{
    extract_component::DynamicUniformIndex,
    render_graph::{NodeRunError, RenderGraphContext, RenderLabel, ViewNode},
    render_resource::{ComputePass, ComputePassDescriptor, PipelineCache, RenderPassDescriptor},
    renderer::RenderContext,
    view::{ExtractedView, ViewTarget, ViewUniformOffset},
};

use crate::{atmosphere::lut_based, atmosphere::core, ViewLightsUniformOffset};

#[derive(PartialEq, Eq, Hash, Clone, Debug, RenderLabel)]
pub struct LutsLabel;

#[derive(Default)]
pub struct LutsNode {
    views: QueryState<(), With<ExtractedView>>
    atmospheres: QueryState<(core::UniformIndex, )
}

impl ViewNode for LutsNode {
    type ViewQuery = (
        Read<lut_based::Settings>,
        Read<DynamicUniformIndex<lut_based::Settings>>,
        Read<lut_based::BindGroups>,
        Read<ViewUniformOffset>,
        Read<ViewLightsUniformOffset>,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (
            settings,
            settings_offset, 
            bind_groups,
            view_uniforms_offset,
            lights_uniforms_offset,
        ): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipelines = world.resource::<lut_based::Pipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let (Some(sky_view_lut_pipeline), Some(aerial_view_lut_pipeline)) = (
            pipeline_cache.get_compute_pipeline(pipelines.sky_view_lut),
            pipeline_cache.get_compute_pipeline(pipelines.aerial_view_lut),
        ) else {
            return Ok(());
        };

        let command_encoder = render_context.command_encoder();

        let mut luts_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("lut_based_atmosphere_luts_pass"),
            timestamp_writes: None,
        });

        fn dispatch_2d(compute_pass: &mut ComputePass, size: UVec2) {
            const WORKGROUP_SIZE: u32 = 16;
            let workgroups_x = size.x.div_ceil(WORKGROUP_SIZE);
            let workgroups_y = size.y.div_ceil(WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Sky View LUT
        let offsets = &[
            core_uniforms_offset.index(),
            view_uniforms_offset.offset,
            lights_uniforms_offset.offset,
            lut_based_uniforms_offset.index(),
        ];

        luts_pass.set_pipeline(sky_view_lut_pipeline);
        luts_pass.set_bind_group(
            0,
            &bind_groups.sky_view_lut,
            offsets,
        );

        dispatch_2d(&mut luts_pass, settings.sky_view_lut_size);

        // Aerial View LUT

        luts_pass.set_pipeline(aerial_view_lut_pipeline);
        luts_pass.set_bind_group(
            0,
            &bind_groups.aerial_view_lut,
            offsets,
        );

        dispatch_2d(&mut luts_pass, settings.aerial_view_lut_size.xy());

        Ok(())
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Debug, RenderLabel)]
pub struct ResolveLabel;

#[derive(Default)]
pub(super) struct ResolveNode;

impl ViewNode for ResolveNode {
    type ViewQuery = (
        Read<AtmosphereBindGroups>,
        Read<ViewTarget>,
        Read<DynamicUniformIndex<ScatteringProfile>>,
        Read<DynamicUniformIndex<lut_based::Uniforms>>,
        Read<AtmosphereTransformsOffset>,
        Read<ViewUniformOffset>,
        Read<ViewLightsUniformOffset>,
        Read<RenderSkyPipelineId>,
    );

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (
            atmosphere_bind_groups,
            view_target,
            atmosphere_uniforms_offset,
            settings_uniforms_offset,
            atmosphere_transforms_offset,
            view_uniforms_offset,
            lights_uniforms_offset,
            render_sky_pipeline_id,
        ): QueryItem<'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(render_sky_pipeline) =
            pipeline_cache.get_render_pipeline(render_sky_pipeline_id.0)
        else {
            return Ok(());
        }; //TODO: warning

        let mut render_sky_pass =
            render_context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("render_sky_pass"),
                    color_attachments: &[Some(view_target.get_color_attachment())],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

        render_sky_pass.set_pipeline(render_sky_pipeline);
        render_sky_pass.set_bind_group(
            0,
            &atmosphere_bind_groups.render_sky,
            &[
                atmosphere_uniforms_offset.index(),
                settings_uniforms_offset.index(),
                atmosphere_transforms_offset.index(),
                view_uniforms_offset.offset,
                lights_uniforms_offset.offset,
            ],
        );
        render_sky_pass.draw(0..3, 0..1);

        Ok(())
    }
}
