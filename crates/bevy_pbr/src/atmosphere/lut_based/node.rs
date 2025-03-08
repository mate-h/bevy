use bevy_ecs::{
    query::{QueryItem, QueryState, With},
    system::lifetimeless::Read,
    world::{FromWorld, World},
};
use bevy_math::{UVec2, Vec3Swizzles};
use bevy_render::{
    extract_component::DynamicUniformIndex,
    render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel, ViewNode},
    render_resource::{ComputePass, ComputePassDescriptor, PipelineCache, RenderPassDescriptor},
    renderer::RenderContext,
    view::{ExtractedView, ViewTarget, ViewUniformOffset},
};

use crate::{
    atmosphere::{core, lut_based},
    AtmosphericScattering, AtmosphericScatteringSettings, ExtractedAtmosphere,
    ViewLightsUniformOffset,
};

#[derive(PartialEq, Eq, Hash, Clone, Debug, RenderLabel)]
pub struct LutsLabel;

pub struct LutsNode {
    views: QueryState<
        (
            Read<AtmosphericScattering>,
            Read<AtmosphericScatteringSettings>,
            Read<lut_based::BindGroups>,
            Read<DynamicUniformIndex<lut_based::Uniforms>>,
            Read<ViewUniformOffset>,
            Read<ViewLightsUniformOffset>,
        ),
        With<ExtractedView>,
    >,
    atmospheres: QueryState<Read<core::UniformsIndex>, With<ExtractedAtmosphere>>,
}

impl FromWorld for LutsNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            views: world.query_filtered(),
            atmospheres: world.query_filtered(),
        }
    }
}

impl Node for LutsNode {
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let views_query = self.views.query_manual(world);
        let atmospheres_query = self.atmospheres.query_manual(world);

        let Ok((
            AtmosphericScattering(atmosphere),
            AtmosphericScatteringSettings::LutBased(settings),
            bind_groups,
            lut_based_uniforms_offset,
            view_uniforms_offset,
            lights_uniforms_offset,
        )) = views_query.get(graph.view_entity())
        else {
            return Ok(());
        };

        let Ok(core_uniforms_offset) = atmospheres_query.get(*atmosphere) else {
            return Ok(());
        };

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
        luts_pass.set_bind_group(0, &bind_groups.sky_view_lut, offsets);

        dispatch_2d(&mut luts_pass, settings.sky_view_lut_size);

        // Aerial View LUT

        luts_pass.set_pipeline(aerial_view_lut_pipeline);
        luts_pass.set_bind_group(0, &bind_groups.aerial_view_lut, offsets);

        dispatch_2d(&mut luts_pass, settings.aerial_view_lut_size.xy());

        Ok(())
    }

    fn update(&mut self, world: &mut World) {
        self.views.update_archetypes(world);
        self.atmospheres.update_archetypes(world);
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Debug, RenderLabel)]
pub struct ResolveLabel;

pub struct ResolveNode {
    views: QueryState<
        (
            Read<AtmosphericScattering>,
            Read<lut_based::BindGroups>,
            Read<DynamicUniformIndex<lut_based::Uniforms>>,
            Read<ViewUniformOffset>,
            Read<ViewLightsUniformOffset>,
            Read<lut_based::ResolvePipelineId>,
            Read<ViewTarget>,
        ),
        With<ExtractedView>,
    >,
    atmospheres: QueryState<Read<core::UniformsIndex>, With<ExtractedAtmosphere>>,
}

impl FromWorld for ResolveNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            views: world.query_filtered(),
            atmospheres: world.query_filtered(),
        }
    }
}

impl Node for ResolveNode {
    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let views_query = self.views.query_manual(world);
        let atmospheres_query = self.atmospheres.query_manual(world);

        let Ok((
            AtmosphericScattering(atmosphere),
            bind_groups,
            lut_based_uniforms_offset,
            view_uniforms_offset,
            lights_uniforms_offset,
            resolve_pipeline_id,
            view_target,
        )) = views_query.get(graph.view_entity())
        else {
            return Ok(());
        };

        let Ok(core_uniforms_offset) = atmospheres_query.get(*atmosphere) else {
            return Ok(());
        };

        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(render_sky_pipeline) =
            pipeline_cache.get_render_pipeline(resolve_pipeline_id.id())
        else {
            return Ok(());
        };

        let mut render_sky_pass =
            render_context
                .command_encoder()
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("resolve_atmosphere"),
                    color_attachments: &[Some(view_target.get_color_attachment())],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });

        render_sky_pass.set_pipeline(render_sky_pipeline);
        render_sky_pass.set_bind_group(
            0,
            &bind_groups.resolve_atmosphere,
            &[
                core_uniforms_offset.index(),
                view_uniforms_offset.offset,
                lights_uniforms_offset.offset,
                lut_based_uniforms_offset.index(),
            ],
        );
        render_sky_pass.draw(0..3, 0..1);

        Ok(())
    }

    fn update(&mut self, world: &mut World) {
        self.views.update_archetypes(world);
        self.atmospheres.update_archetypes(world);
    }
}
