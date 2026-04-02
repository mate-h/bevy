use bevy_camera::{MainPassResolutionOverride, Viewport};
use bevy_ecs::query::{Changed, Or, With};
use bevy_ecs::system::{Query, Res};
use bevy_math::{UVec2, Vec3Swizzles};
use bevy_render::{
    camera::ExtractedCamera,
    extract_component::DynamicUniformIndex,
    render_asset::RenderAssets,
    render_resource::{
        BindGroupEntries, ComputePass, ComputePassDescriptor, PipelineCache, RenderPassDescriptor,
    },
    renderer::{RenderContext, RenderDevice, ViewQuery},
    view::{Msaa, ViewDepthTexture, ViewTarget, ViewUniformOffset},
};

use super::{resources::GpuAtmosphere, GpuAtmosphereSettings};
use crate::resources::{AtmosphereLutBindGroups, AtmosphereTransforms, RenderSkyBindGroupLayouts};
use crate::{
    ExtractedAtmosphere, GpuScatteringMedium, ScatteringMediumSampler, ViewLightsUniformOffset,
};

use super::resources::{
    AtmosphereLutPipelines, AtmosphereTextures, RenderSkyPipelineId, ViewAtmosphereLutBindGroups,
    ViewAtmosphereTextures, ViewAtmospheres,
};

fn dispatch_2d(compute_pass: &mut ComputePass, size: UVec2) {
    const WORKGROUP_SIZE: u32 = 16;
    let workgroups_x = size.x.div_ceil(WORKGROUP_SIZE);
    let workgroups_y = size.y.div_ceil(WORKGROUP_SIZE);
    compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}

/// Computes transmittance and multiscattering LUTs once per atmosphere, before any view is rendered.
/// These LUTs are view-independent, so they must not be recomputed per view.
/// Only recomputes when atmosphere parameters (including medium) or settings have changed.
pub fn atmosphere_transmittance_multiscattering_luts(
    atmospheres: Query<
        (
            &AtmosphereLutBindGroups,
            &GpuAtmosphereSettings,
            &DynamicUniformIndex<GpuAtmosphere>,
            &DynamicUniformIndex<GpuAtmosphereSettings>,
        ),
        (
            With<ExtractedAtmosphere>,
            Or<(Changed<ExtractedAtmosphere>, Changed<GpuAtmosphereSettings>)>,
        ),
    >,
    pipelines: Res<AtmosphereLutPipelines>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    if atmospheres.is_empty() {
        return;
    }

    let (Some(transmittance_lut_pipeline), Some(multiscattering_lut_pipeline)) = (
        pipeline_cache.get_compute_pipeline(pipelines.transmittance_lut),
        pipeline_cache.get_compute_pipeline(pipelines.multiscattering_lut),
    ) else {
        return;
    };

    let command_encoder = ctx.command_encoder();

    {
        let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("atmosphere_transmittance_lut"),
            timestamp_writes: None,
        });
        pass.set_pipeline(transmittance_lut_pipeline);
        for (bind_groups, settings, atmosphere_idx, settings_idx) in &atmospheres {
            pass.set_bind_group(
                0,
                &bind_groups.transmittance_lut,
                &[atmosphere_idx.index(), settings_idx.index()],
            );
            dispatch_2d(&mut pass, settings.transmittance_lut_size);
        }
    }

    {
        let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("atmosphere_multiscattering_lut"),
            timestamp_writes: None,
        });
        pass.set_pipeline(multiscattering_lut_pipeline);
        for (bind_groups, settings, atmosphere_idx, settings_idx) in &atmospheres {
            pass.set_bind_group(
                0,
                &bind_groups.multiscattering_lut,
                &[atmosphere_idx.index(), settings_idx.index()],
            );
            pass.dispatch_workgroups(
                settings.multiscattering_lut_size.x,
                settings.multiscattering_lut_size.y,
                1,
            );
        }
    }
}

pub fn atmosphere_luts(
    view: ViewQuery<(
        &ViewAtmospheres,
        &ViewAtmosphereLutBindGroups,
        &ViewUniformOffset,
        &ViewLightsUniformOffset,
    )>,
    atmospheres: Query<(&GpuAtmosphereSettings,), With<ExtractedAtmosphere>>,
    pipelines: Res<AtmosphereLutPipelines>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let (view_atmospheres, bind_groups, view_uniforms_offset, lights_uniforms_offset) =
        view.into_inner();

    let (Some(sky_view_lut_pipeline), Some(aerial_view_lut_pipeline)) = (
        pipeline_cache.get_compute_pipeline(pipelines.sky_view_lut),
        pipeline_cache.get_compute_pipeline(pipelines.aerial_view_lut),
    ) else {
        return;
    };

    let command_encoder = ctx.command_encoder();

    // Sky view LUT: one per (view, atmosphere).
    {
        let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("atmosphere_sky_view_lut"),
            timestamp_writes: None,
        });
        pass.set_pipeline(sky_view_lut_pipeline);
        for (i, view_atmosphere) in view_atmospheres.0.iter().enumerate() {
            let Some(sky_view_bind_group) = bind_groups.sky_view_luts.get(i) else {
                continue;
            };
            let Ok((settings,)) = atmospheres.get(view_atmosphere.atmosphere_entity) else {
                continue;
            };
            pass.set_bind_group(
                0,
                sky_view_bind_group,
                &[
                    view_atmosphere.atmosphere_uniform_index,
                    view_atmosphere.settings_uniform_index,
                    view_atmosphere.transform_offset.index(),
                    view_uniforms_offset.offset,
                    lights_uniforms_offset.offset,
                ],
            );
            dispatch_2d(&mut pass, settings.sky_view_lut_size);
        }
    }

    // Aerial view LUT: one per view.
    {
        let mut pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("atmosphere_aerial_view_lut"),
            timestamp_writes: None,
        });
        pass.set_pipeline(aerial_view_lut_pipeline);
        let first = match view_atmospheres.0.first() {
            Some(v) => v,
            None => return,
        };
        let Ok((settings,)) = atmospheres.get(first.atmosphere_entity) else {
            return;
        };
        pass.set_bind_group(
            0,
            &bind_groups.aerial_view_lut,
            &[
                first.atmosphere_uniform_index,
                first.settings_uniform_index,
                first.transform_offset.index(),
                view_uniforms_offset.offset,
                lights_uniforms_offset.offset,
            ],
        );
        dispatch_2d(&mut pass, settings.aerial_view_lut_size.xy());
    }
}

pub fn render_sky(
    view: ViewQuery<(
        &ExtractedCamera,
        &ViewAtmospheres,
        &ViewAtmosphereTextures,
        &ViewTarget,
        &ViewDepthTexture,
        &Msaa,
        &ViewUniformOffset,
        &ViewLightsUniformOffset,
        &RenderSkyPipelineId,
        Option<&MainPassResolutionOverride>,
    )>,
    atmospheres: Query<(&ExtractedAtmosphere, &AtmosphereTextures), With<ExtractedAtmosphere>>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
    render_sky_layouts: Res<RenderSkyBindGroupLayouts>,
    atmosphere_transforms: Res<AtmosphereTransforms>,
    atmosphere_uniforms: Res<bevy_render::extract_component::ComponentUniforms<GpuAtmosphere>>,
    settings_uniforms: Res<
        bevy_render::extract_component::ComponentUniforms<GpuAtmosphereSettings>,
    >,
    view_uniforms: Res<bevy_render::view::ViewUniforms>,
    lights_uniforms: Res<crate::LightMeta>,
    gpu_media: Res<RenderAssets<GpuScatteringMedium>>,
    atmosphere_sampler: Res<super::resources::AtmosphereSampler>,
    medium_sampler: Res<ScatteringMediumSampler>,
    mut ctx: RenderContext,
) {
    let (
        camera,
        view_atmospheres,
        view_textures,
        view_target,
        view_depth_texture,
        msaa,
        view_uniforms_offset,
        lights_uniforms_offset,
        render_sky_pipeline_id,
        resolution_override,
    ) = view.into_inner();

    let Some(render_sky_pipeline) = pipeline_cache.get_render_pipeline(render_sky_pipeline_id.0)
    else {
        return;
    };

    let atmosphere_binding = match atmosphere_uniforms.binding() {
        Some(b) => b,
        None => return,
    };
    let settings_binding = match settings_uniforms.binding() {
        Some(b) => b,
        None => return,
    };
    let Some(transforms_binding) = atmosphere_transforms.uniforms().binding() else {
        return;
    };
    let Some(view_binding) = view_uniforms.uniforms.binding() else {
        return;
    };
    let Some(lights_binding) = lights_uniforms.view_gpu_lights.binding() else {
        return;
    };

    let command_encoder = ctx.command_encoder();

    let mut render_sky_pass = command_encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("render_sky"),
        color_attachments: &[Some(view_target.get_color_attachment())],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    if let Some(viewport) =
        Viewport::from_viewport_and_override(camera.viewport.as_ref(), resolution_override)
    {
        render_sky_pass.set_viewport(
            viewport.physical_position.x as f32,
            viewport.physical_position.y as f32,
            viewport.physical_size.x as f32,
            viewport.physical_size.y as f32,
            viewport.depth.start,
            viewport.depth.end,
        );
    }

    render_sky_pass.set_pipeline(render_sky_pipeline);

    for (i, view_atmosphere) in view_atmospheres.0.iter().enumerate() {
        // continue if not index 0
        // if i != 0 {
        //     continue;
        // }

        let Ok((atmosphere, textures)) = atmospheres.get(view_atmosphere.atmosphere_entity) else {
            continue;
        };

        let Some(gpu_medium) = gpu_media.get(atmosphere.medium) else {
            continue;
        };

        let Some(sky_view_lut) = view_textures.sky_view_luts.get(i) else {
            continue;
        };

        let render_sky_bind_group = render_device.create_bind_group(
            "render_sky_bind_group",
            &pipeline_cache.get_bind_group_layout(if *msaa == Msaa::Off {
                &render_sky_layouts.render_sky
            } else {
                &render_sky_layouts.render_sky_msaa
            }),
            &BindGroupEntries::with_indices((
                (0, atmosphere_binding.clone()),
                (1, settings_binding.clone()),
                (2, transforms_binding.clone()),
                (3, view_binding.clone()),
                (4, lights_binding.clone()),
                (5, &gpu_medium.density_lut_view),
                (6, &gpu_medium.scattering_lut_view),
                (7, medium_sampler.sampler()),
                (8, &textures.transmittance_lut.default_view),
                (9, &textures.multiscattering_lut.default_view),
                (10, &sky_view_lut.default_view),
                (11, &view_textures.aerial_view_lut.default_view),
                (12, &**atmosphere_sampler),
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
}
