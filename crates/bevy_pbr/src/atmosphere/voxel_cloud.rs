//! Voxel clouds: authored cloud volumes placed in the world (Nubis Cubed, Schneider, Advances in
//! Real-Time Rendering 2023, "Nubis Voxel Cloud Data Fields").

use bevy_asset::{AssetId, Handle};
use bevy_ecs::{component::Component, reflect::ReflectComponent, resource::Resource};
use bevy_image::Image;
use bevy_math::{Mat4, Vec3};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_transform::components::{GlobalTransform, Transform};

/// A voxel cloud: a Nubis voxel data field (NVDF) filling a box placed by the entity's
/// [`Transform`]. The box is centred on the entity; its size is the texture's voxel count times
/// `voxel_size` along local X, Y (up) and Z, before the transform's scale (see [`Self::size`]).
///
/// The data is an RGBA 3D texture laid out as `tools/nvdf_to_ktx2.py` writes it: R the
/// dimensional profile (0 no cloud .. 1 full cloud), G the detail type (0 wispy .. 1 billowy),
/// B the density scale, A the signed distance to the cloud (-256 m .. 4096 m mapped to 0 .. 1).
/// Width is X, height is Y (row 0 at the bottom of the box), depth is Z. The values are data, not
/// colors: load the image with `ImageLoaderSettings::is_srgb = false`, or the sampler
/// gamma-decodes them.
///
/// The cloud renderer takes the profile, type and density scale from the data inside the box and
/// erodes them with the detail noise as Nubis Cubed's voxel sampler does (`clouds.wesl`,
/// `cloud_column_at`). The voxel cloud belongs to the atmosphere's [`VolumetricCloudLayer`]
/// (`crate::VolumetricCloudLayer`): it uses the layer's optical properties and density, and
/// must lie within its altitude range; without a layer it is not drawn. One voxel cloud is
/// rendered (the first).
#[derive(Component, Clone, Debug, Reflect)]
#[reflect(Component, Default, Clone)]
#[require(Transform)]
pub struct VoxelCloud {
    /// The NVDF 3D texture.
    pub data: Handle<Image>,
    /// Edge of a voxel in metres (cubic voxels), before the transform's scale.
    pub voxel_size: f32,
}

impl VoxelCloud {
    /// The voxel size of the Nubis Voxel Cloud Pack: its clouds fill a domain of 512 x 64 x 512
    /// voxels, the Burning Shores cloud area of ~4 km x 4 km with ~500 m of vertical space
    /// (Schneider 2023): 4000 m / 512.
    pub const NUBIS_PACK_VOXEL_SIZE: f32 = 4000.0 / 512.0;

    /// A voxel cloud with the Nubis pack's voxel size ([`Self::NUBIS_PACK_VOXEL_SIZE`]).
    pub fn nubis_pack(data: Handle<Image>) -> Self {
        Self {
            data,
            voxel_size: Self::NUBIS_PACK_VOXEL_SIZE,
        }
    }

    /// Size of the box in metres for its loaded texture (voxel count times the voxel size), before
    /// the transform's scale.
    pub fn size(&self, data: &Image) -> Vec3 {
        let extent = data.texture_descriptor.size;
        Vec3::new(
            extent.width as f32,
            extent.height as f32,
            extent.depth_or_array_layers as f32,
        ) * self.voxel_size
    }

    /// The box as a transform of the unit cube centred on the origin: the entity's global
    /// transform with the box `size` ([`Self::size`]) applied in its local frame.
    pub fn bounds(&self, transform: &GlobalTransform, size: Vec3) -> GlobalTransform {
        transform.mul_transform(Transform::from_scale(size))
    }
}

impl Default for VoxelCloud {
    fn default() -> Self {
        Self::nubis_pack(Handle::default())
    }
}

/// Render world: the texture of the rendered voxel cloud, bound to every cloud pass (a fallback
/// 3D texture without one).
#[derive(Resource, Default)]
pub struct ExtractedVoxelCloud {
    pub image: Option<AssetId<Image>>,
}

/// Atmosphere space to the voxel cloud's texture coordinates: the box maps to [0, 1]^3.
pub(crate) fn uvw_from_atmosphere(
    cloud: &VoxelCloud,
    transform: &GlobalTransform,
    size: Vec3,
    atmosphere_to_world: Mat4,
) -> Mat4 {
    Mat4::from_translation(Vec3::splat(0.5))
        * cloud.bounds(transform, size).to_matrix().inverse()
        * atmosphere_to_world
}

/// The voxel cloud's texture view, or a fallback 3D texture before it has loaded, without a voxel
/// cloud, or when the image is not 3D.
pub(crate) fn voxel_cloud_view<'a>(
    extracted: &ExtractedVoxelCloud,
    images: &'a bevy_render::render_asset::RenderAssets<bevy_render::texture::GpuImage>,
    fallback: &'a bevy_render::texture::FallbackImage,
) -> &'a bevy_render::render_resource::TextureView {
    match extracted.image.and_then(|id| images.get(id)) {
        Some(gpu) if gpu.texture.dimension() == bevy_render::render_resource::TextureDimension::D3 => {
            &gpu.texture_view
        }
        Some(_) => {
            bevy_log::warn_once!("a VoxelCloud's data is not a 3D texture; it is not rendered");
            &fallback.d3.texture_view
        }
        None => &fallback.d3.texture_view,
    }
}
