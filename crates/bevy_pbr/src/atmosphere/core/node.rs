use bevy_ecs::{
    query::QueryState,
    system::lifetimeless::Read,
    world::{FromWorld, World},
};
use bevy_render::{
    render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel},
    render_resource::{ComputePassDescriptor, PipelineCache},
    renderer::RenderContext,
};

use crate::atmosphere::core;

#[derive(RenderLabel, PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct LutsLabel;

pub(super) struct LutsNode {
    query_state: QueryState<(Read<core::Settings>, Read<core::BindGroups>)>,
}

impl FromWorld for LutsNode {
    fn from_world(world: &mut World) -> Self {
        let query_state = world.query::<(Read<core::Settings>, Read<core::BindGroups>)>();
        Self { query_state }
    }
}

impl Node for LutsNode {
    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let pipelines = world.resource::<core::Pipelines>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let (Some(transmittance_lut_pipeline), Some(multiscattering_lut_pipeline)) = (
            pipeline_cache.get_compute_pipeline(pipelines.transmittance_lut),
            pipeline_cache.get_compute_pipeline(pipelines.multiscattering_lut),
        ) else {
            return Ok(());
        };

        let mut core_luts_pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("atmosphere_core_luts_pass"),
                    timestamp_writes: None,
                });

        core_luts_pass.set_pipeline(&transmittance_lut_pipeline);

        for (core_settings, core_bind_groups) in &self.query_state.query_manual(world) {
            core_luts_pass.set_bind_group(
                0,
                &core_bind_groups.transmittance_lut,
                &[todo!("uniform offsets!!!!!!")],
            );

            core_luts_pass.dispatch_workgroups(todo!(), todo!(), todo!());
        }

        core_luts_pass.set_pipeline(&multiscattering_lut_pipeline);

        for (core_settings, core_bind_groups) in &self.query_state.query_manual(world) {
            core_luts_pass.set_bind_group(
                0,
                &core_bind_groups.multiscattering_lut,
                &[todo!("uniform offsets!!!!!!")],
            );

            core_luts_pass.dispatch_workgroups(todo!(), todo!(), todo!());
        }

        Ok(())
    }

    fn update(&mut self, world: &mut World) {
        self.query_state.update_archetypes(world);
    }
}
