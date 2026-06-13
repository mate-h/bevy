use bevy_asset::{load_embedded_asset, AssetServer, Handle};
use bevy_camera::CompositingSpace;
use bevy_ecs::prelude::*;
use bevy_mesh::VertexBufferLayout;
use bevy_render::{
    render_resource::{
        binding_types::{sampler, texture_2d, uniform_buffer},
        *,
    },
    view::ViewUniform,
};
use bevy_shader::{Shader, ShaderDefVal};
use bevy_utils::default;

/// Appends the writer-side compositing-encode shader def for `compositing_space`
/// onto `shader_defs`.
///
/// UI runs after tone mapping, so for a view whose resolved
/// [`CompositingSpace`] is [`Srgb`](CompositingSpace::Srgb) or
/// [`Oklab`](CompositingSpace::Oklab) the main texture already holds values in
/// that encoded space; each UI fragment must encode its straight-alpha output
/// the same way (the `SRGB_OUTPUT` / `OKLAB_OUTPUT` writer-encode def family
/// shared with sprites) so the terminal decode reads it correctly.
/// [`Linear`](CompositingSpace::Linear) and `None` push nothing, keeping the
/// default-path shader-def vector byte-identical.
pub(crate) fn push_compositing_space_defs(
    shader_defs: &mut Vec<ShaderDefVal>,
    compositing_space: Option<CompositingSpace>,
) {
    match compositing_space {
        Some(CompositingSpace::Srgb) => shader_defs.push("SRGB_OUTPUT".into()),
        Some(CompositingSpace::Oklab) => shader_defs.push("OKLAB_OUTPUT".into()),
        Some(CompositingSpace::Linear) | None => {}
    }
}

#[derive(Resource)]
pub struct UiPipeline {
    pub view_layout: BindGroupLayoutDescriptor,
    pub image_layout: BindGroupLayoutDescriptor,
    pub shader: Handle<Shader>,
}

pub fn init_ui_pipeline(mut commands: Commands, asset_server: Res<AssetServer>) {
    let view_layout = BindGroupLayoutDescriptor::new(
        "ui_view_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX_FRAGMENT,
            uniform_buffer::<ViewUniform>(true),
        ),
    );

    let image_layout = BindGroupLayoutDescriptor::new(
        "ui_image_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    );

    commands.insert_resource(UiPipeline {
        view_layout,
        image_layout,
        shader: load_embedded_asset!(asset_server.as_ref(), "ui.wgsl"),
    });
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct UiPipelineKey {
    pub target_format: TextureFormat,
    pub anti_alias: bool,
    /// Resolved [`CompositingSpace`] of the view this node renders into, driving
    /// the writer-side encode of the fragment output (see
    /// [`push_compositing_space_defs`]).
    pub compositing_space: Option<CompositingSpace>,
}

impl SpecializedRenderPipeline for UiPipeline {
    type Key = UiPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let vertex_layout = VertexBufferLayout::from_vertex_formats(
            VertexStepMode::Vertex,
            vec![
                // position
                VertexFormat::Float32x3,
                // uv
                VertexFormat::Float32x2,
                // color
                VertexFormat::Float32x4,
                // mode
                VertexFormat::Uint32,
                // border radius
                VertexFormat::Float32x4,
                // border thickness
                VertexFormat::Float32x4,
                // border size
                VertexFormat::Float32x2,
                // position relative to the center
                VertexFormat::Float32x2,
            ],
        );
        let mut shader_defs = if key.anti_alias {
            vec!["ANTI_ALIAS".into()]
        } else {
            Vec::new()
        };
        push_compositing_space_defs(&mut shader_defs, key.compositing_space);

        RenderPipelineDescriptor {
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: shader_defs.clone(),
                buffers: vec![vertex_layout],
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            layout: vec![self.view_layout.clone(), self.image_layout.clone()],
            label: Some("ui_pipeline".into()),
            ..default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A view with no compositing-space request, or a `Linear` one, must leave
    /// the shader-def vector untouched so the default UI pipeline compiles
    /// byte-identically.
    #[test]
    fn no_writer_encode_for_default_or_linear() {
        let baseline = vec![ShaderDefVal::from("ANTI_ALIAS")];

        let mut none_defs = baseline.clone();
        push_compositing_space_defs(&mut none_defs, None);
        assert_eq!(none_defs, baseline);

        let mut linear_defs = baseline.clone();
        push_compositing_space_defs(&mut linear_defs, Some(CompositingSpace::Linear));
        assert_eq!(linear_defs, baseline);
    }

    /// A resolved `Srgb` view appends exactly `SRGB_OUTPUT` (the writer-encode
    /// def family shared with sprites), nothing else.
    #[test]
    fn srgb_appends_srgb_output() {
        let mut defs = Vec::new();
        push_compositing_space_defs(&mut defs, Some(CompositingSpace::Srgb));
        assert_eq!(defs, vec![ShaderDefVal::from("SRGB_OUTPUT")]);
    }

    /// A resolved `Oklab` view appends exactly `OKLAB_OUTPUT`, nothing else.
    #[test]
    fn oklab_appends_oklab_output() {
        let mut defs = Vec::new();
        push_compositing_space_defs(&mut defs, Some(CompositingSpace::Oklab));
        assert_eq!(defs, vec![ShaderDefVal::from("OKLAB_OUTPUT")]);
    }

    /// The two encode spaces are mutually exclusive: each pushes one def and
    /// never the other.
    #[test]
    fn srgb_and_oklab_are_exclusive() {
        let mut srgb_defs = Vec::new();
        push_compositing_space_defs(&mut srgb_defs, Some(CompositingSpace::Srgb));
        assert!(srgb_defs.contains(&ShaderDefVal::from("SRGB_OUTPUT")));
        assert!(!srgb_defs.contains(&ShaderDefVal::from("OKLAB_OUTPUT")));

        let mut oklab_defs = Vec::new();
        push_compositing_space_defs(&mut oklab_defs, Some(CompositingSpace::Oklab));
        assert!(oklab_defs.contains(&ShaderDefVal::from("OKLAB_OUTPUT")));
        assert!(!oklab_defs.contains(&ShaderDefVal::from("SRGB_OUTPUT")));
    }
}
