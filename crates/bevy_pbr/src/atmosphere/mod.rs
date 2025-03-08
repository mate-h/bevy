//! Procedural Atmospheric Scattering.
//!
//! This plugin implements [Hillaire's 2020 paper](https://sebh.github.io/publications/egsr2020.pdf)
//! on real-time atmospheric scattering. While it *will* work simply as a
//! procedural skybox, it also does much more. It supports dynamic time-of-
//! -day, multiple directional lights, and since it's applied as a post-processing
//! effect *on top* of the existing skybox, a starry skybox would automatically
//! show based on the time of day. Scattering in front of terrain (similar
//! to distance fog, but more complex) is handled as well, and takes into
//! account the directional light color and direction.
//!
//! Adding the [`Atmosphere`] component to a 3d camera will enable the effect,
//! which by default is set to look similar to Earth's atmosphere. See the
//! documentation on the component itself for information regarding its fields.
//!
//! Performance-wise, the effect should be fairly cheap since the LUTs (Look
//! Up Tables) that encode most of the data are small, and take advantage of the
//! fact that the atmosphere is symmetric. Performance is also proportional to
//! the number of directional lights in the scene. In order to tune
//! performance more finely, the [`AtmosphereSettings`] camera component
//! manages the size of each LUT and the sample count for each ray.
//!
//! Given how similar it is to [`crate::volumetric_fog`], it might be expected
//! that these two modules would work together well. However for now using both
//! at once is untested, and might not be physically accurate. These may be
//! integrated into a single module in the future.
//!
//! [Shadertoy]: https://www.shadertoy.com/view/slSXRW
//!
//! [Unreal Engine Implementation]: https://github.com/sebh/UnrealEngineSkyAtmosphere

mod core;
mod lut_based;

use bevy_app::{App, Plugin};
use bevy_asset::{
    load_internal_asset, weak_handle, AsAssetId, Asset, AssetApp, AssetId, Assets, Handle,
};
use bevy_color::LinearRgba;
use bevy_core_pipeline::core_3d::Camera3d;
use bevy_ecs::{
    component::{require, Component},
    entity::Entity,
    query::{Changed, With},
    reflect::ReflectComponent,
    schedule::IntoSystemConfigs,
    system::{Commands, Query},
};
use bevy_math::Vec3;
use bevy_reflect::Reflect;
use bevy_render::{
    extract_component::{ExtractComponent, ExtractComponentPlugin},
    render_resource::{DownlevelFlags, Shader, ShaderType, TextureFormat, TextureUsages},
    renderer::{RenderAdapter, RenderDevice},
    settings::WgpuFeatures,
    sync_world::{RenderEntity, SyncToRenderWorld},
    Extract, ExtractSchedule, Render, RenderApp, RenderSet,
};
use tracing::warn;

use bitflags::bitflags;
use lut_based::LutBasedAtmospherePlugin;

mod shaders {
    use bevy_asset::{weak_handle, Handle};
    use bevy_render::render_resource::Shader;

    pub const TYPES: Handle<Shader> = weak_handle!("ef7e147e-30a0-4513-bae3-ddde2a6c20c5");
    pub const INTERNAL: Handle<Shader> = weak_handle!("b95e172a-bebb-435f-bc76-ebba2e2baf62");
    pub const FUNCTIONS: Handle<Shader> = weak_handle!("7ff93872-2ee9-4598-9f88-68b02fef605f");
    pub const BRUNETON_FUNCTIONS: Handle<Shader> =
        weak_handle!("e2dccbb0-7322-444a-983b-e74d0a08bcda");
}

use ::core::fmt;

use core::CoreAtmospherePlugin;
pub use core::{
    ExtractedAtmosphere, Luts as CoreAtmosphereLuts, Settings as AtmosphereSettings,
    UniformsBuffer as AtmosphereUniforms, UniformsIndex as AtmosphereUniformIndex,
};
pub use lut_based::{Luts as AuxAtmosphereLuts, Settings as LutBasedAtmosphericScatteringSettings};

#[doc(hidden)]
pub struct AtmospherePlugin;

impl Plugin for AtmospherePlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, shaders::TYPES, "types.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, shaders::INTERNAL, "internal.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, shaders::FUNCTIONS, "functions.wgsl", Shader::from_wgsl);
        load_internal_asset!(
            app,
            shaders::BRUNETON_FUNCTIONS,
            "bruneton_functions.wgsl",
            Shader::from_wgsl
        );

        app.init_asset::<ScatteringProfile>();
        app.world_mut()
            .resource_mut::<Assets<ScatteringProfile>>()
            .insert(&ScatteringProfile::EARTH_HANDLE, ScatteringProfile::EARTH);

        app.register_type::<ScatteringProfile>()
            .register_type::<Atmosphere>()
            .register_type::<AtmosphericScattering>()
            .register_type::<AtmosphericScatteringCameras>()
            .add_plugins((
                ExtractComponentPlugin::<AtmosphericScatteringSettings>::default(),
                CoreAtmospherePlugin,
                LutBasedAtmospherePlugin,
            ));
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        let device = render_app.world().resource::<RenderDevice>();
        let adapter = render_app.world().resource::<RenderAdapter>();
        if let Err(err) = validate_plugin(device, adapter) {
            warn!("AtmospherePlugin not loaded: {err}");
        }

        render_app
            .add_systems(ExtractSchedule, extract_atmospheric_scattering)
            .add_systems(
                Render,
                configure_camera_depth_usages.in_set(RenderSet::ManageViews),
            );
    }
}

bitflags! {
    #[derive(PartialEq, Eq)]
    pub struct AtmospherePluginInitError: u8 {
        const NO_DUAL_SOURCE_BLENDING = 1 << 0;
        const NO_COMPUTE = 1 << 1;
        const NO_RGBA_16_FLOAT_STORAGE = 1 << 2;
    }
}

impl fmt::Display for AtmospherePluginInitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GPU lacks support for: ")?;

        if self.contains(AtmospherePluginInitError::NO_DUAL_SOURCE_BLENDING) {
            write!(f, "dual-source blending, ")?;
        }

        if self.contains(AtmospherePluginInitError::NO_COMPUTE) {
            write!(f, "compute shaders, ")?;
        }

        if self.contains(AtmospherePluginInitError::NO_RGBA_16_FLOAT_STORAGE) {
            write!(f, "Rgba16Float storage textures")?;
        }

        Ok(())
    }
}

fn validate_plugin(
    device: &RenderDevice,
    adapter: &RenderAdapter,
) -> Result<(), AtmospherePluginInitError> {
    let mut err = AtmospherePluginInitError::empty();

    if !device
        .features()
        .contains(WgpuFeatures::DUAL_SOURCE_BLENDING)
    {
        err |= AtmospherePluginInitError::NO_DUAL_SOURCE_BLENDING;
    }

    if !adapter
        .get_downlevel_capabilities()
        .flags
        .contains(DownlevelFlags::COMPUTE_SHADERS)
    {
        err |= AtmospherePluginInitError::NO_COMPUTE;
    }

    if !adapter
        .get_texture_format_features(TextureFormat::Rgba16Float)
        .allowed_usages
        .contains(TextureUsages::STORAGE_BINDING)
    {
        err |= AtmospherePluginInitError::NO_RGBA_16_FLOAT_STORAGE;
    }

    if err.is_empty() {
        Ok(())
    } else {
        Err(err)
    }
}

#[derive(Component, Clone, PartialEq)]
pub struct Planet {
    /// Radius of the planet
    ///
    /// units: m
    pub radius: f32,

    /// Altitude at which we consider the atmosphere to 'end' for our
    /// calculations
    ///
    /// units: m
    pub space_altitude: f32,

    /// An approximation of the average albedo (or color, roughly) of the
    /// planet's surface. This is used when calculating multiscattering.
    /// The alpha channel is ignored.
    ///
    /// units: n/a
    pub ground_albedo: LinearRgba,
}

impl Planet {
    pub const EARTH: Self = Self {
        radius: 6_360_000.0,
        space_altitude: 100_000.0,
        ground_albedo: LinearRgba::rgb(0.3, 0.3, 0.3),
    };
}

impl Default for Planet {
    fn default() -> Self {
        Self::EARTH
    }
}

/// An asset that describes how a volume scatters light.
///
/// Most atmospheric particles scatter and absorb light in two main ways:
///
/// Rayleigh scattering occurs among very small particles, like individual gas
/// molecules. It's wavelength dependent, and causes colors to separate out as
/// light travels through the atmosphere. These particles *don't* absorb light.
///
/// Mie scattering occurs among slightly larger particles, like dust and sea spray.
/// These particles *do* absorb light, but Mie scattering and absorption is
/// *wavelength independent*.
///
/// Ozone acts differently from the other two, and is special-cased because
/// it's very important to the look of Earth's atmosphere. It's wavelength
/// dependent, but only *absorbs* light. Also, while the density of particles
/// participating in Rayleigh and Mie scattering falls off roughly exponentially
/// from the planet's surface, ozone only exists in a band centered at a fairly
/// high altitude.
#[derive(Clone, Reflect, ShaderType, Asset)]
pub struct ScatteringProfile {
    /// The rate of falloff of rayleigh particulate with respect to altitude:
    /// optical density = exp(-rayleigh_density_exp_scale * altitude in meters).
    ///
    /// THIS VALUE MUST BE POSITIVE
    ///
    /// units: m^-1
    pub rayleigh_density_exp_scale: f32,

    /// The scattering optical density of rayleigh particulate, or how
    /// much light it scatters per meter
    ///
    /// units: m^-1
    pub rayleigh_scattering: Vec3,

    /// The rate of falloff of mie particulate with respect to altitude:
    /// optical density = exp(-mie_density_exp_scale * altitude in meters)
    ///
    /// THIS VALUE MUST BE POSITIVE
    ///
    /// units: m^-1
    pub mie_density_exp_scale: f32,

    /// The scattering optical density of mie particulate, or how much light
    /// it scatters per meter.
    ///
    /// units: m^-1
    pub mie_scattering: f32,

    /// The absorbing optical density of mie particulate, or how much light
    /// it absorbs per meter.
    ///
    /// units: m^-1
    pub mie_absorption: f32,

    /// The "asymmetry" of mie scattering, or how much light tends to scatter
    /// forwards, rather than backwards or to the side.
    ///
    /// domain: (-1, 1)
    /// units: n/a
    pub mie_asymmetry: f32,

    /// The altitude at which the ozone layer is centered.
    ///
    /// units: m
    pub ozone_layer_altitude: f32,

    /// The width of the ozone layer
    ///
    /// units: m
    pub ozone_layer_width: f32,

    /// The optical density of ozone, or how much of each wavelength of
    /// light it absorbs per meter.
    ///
    /// units: m^-1
    pub ozone_absorption: Vec3,
}

impl ScatteringProfile {
    const EARTH_HANDLE: Handle<Self> = weak_handle!("e44ea1fe-143a-4712-8cae-aec86922f081");

    pub const EARTH: Self = Self {
        rayleigh_density_exp_scale: 1.0 / 8_000.0,
        rayleigh_scattering: Vec3::new(5.802e-6, 13.558e-6, 33.100e-6),
        mie_density_exp_scale: 1.0 / 1_200.0,
        mie_scattering: 3.996e-6,
        mie_absorption: 0.444e-6,
        mie_asymmetry: 0.8,
        ozone_layer_altitude: 25_000.0,
        ozone_layer_width: 30_000.0,
        ozone_absorption: Vec3::new(0.650e-6, 1.881e-6, 0.085e-6),
    };

    pub fn earth() -> Handle<Self> {
        Self::EARTH_HANDLE.clone()
    }

    pub fn with_density_multiplier(mut self, mult: f32) -> Self {
        self.rayleigh_scattering *= mult;
        self.mie_scattering *= mult;
        self.mie_absorption *= mult;
        self.ozone_absorption *= mult;
        self
    }

    // Given a conversion factor, convert the scattering profile's values to a new unit of length
    //
    // Example: to convert from meters to kilometers, choose a factor of 1e-3.
    pub fn with_length_unit(mut self, factor: f32) -> Self {
        self.rayleigh_density_exp_scale /= factor;
        self.rayleigh_scattering /= factor;
        self.mie_density_exp_scale /= factor;
        self.mie_scattering /= factor;
        self.mie_absorption /= factor;
        self.ozone_layer_altitude *= factor;
        self.ozone_layer_width *= factor;
        self.ozone_absorption /= factor;
        self
    }
}

#[derive(Component, Clone, Reflect)]
#[require(AtmosphereSettings, Planet, SyncToRenderWorld)]
pub struct Atmosphere(pub Handle<ScatteringProfile>);

impl Default for Atmosphere {
    fn default() -> Self {
        Self::earth()
    }
}

impl AsAssetId for Atmosphere {
    type Asset = ScatteringProfile;

    fn as_asset_id(&self) -> AssetId<Self::Asset> {
        self.0.id()
    }
}

impl Atmosphere {
    pub fn earth() -> Self {
        Self(ScatteringProfile::earth())
    }
}

#[derive(Component, Clone, Reflect)]
#[reflect(Component)]
#[require(AtmosphericScatteringSettings)]
#[relationship(relationship_target = AtmosphericScatteringCameras)]
pub struct AtmosphericScattering(pub Entity);

#[derive(Component, Clone, Reflect)]
#[reflect(Component)]
#[relationship_target(relationship = AtmosphericScattering)]
pub struct AtmosphericScatteringCameras(Vec<Entity>);

pub fn extract_atmospheric_scattering(
    views: Extract<Query<(RenderEntity, &AtmosphericScattering), With<Camera3d>>>,
    atmospheres: Extract<Query<RenderEntity, With<Atmosphere>>>,
    mut commands: Commands,
) {
    for (view_render_entity, AtmosphericScattering(atmosphere)) in &views {
        if let Ok(atmosphere_render_entity) = atmospheres.get(*atmosphere) {
            commands
                .entity(view_render_entity)
                .insert(AtmosphericScattering(atmosphere_render_entity));
        }
    }
}

#[derive(Component, Clone, Reflect, ExtractComponent)]
#[extract_component_filter(With<AtmosphericScattering>)]
#[reflect(Component)]
pub enum AtmosphericScatteringSettings {
    LutBased(LutBasedAtmosphericScatteringSettings),
    //this is an enum so that raytracing support can be added without a breaking change
}

impl Default for AtmosphericScatteringSettings {
    fn default() -> Self {
        Self::LutBased(Default::default())
    }
}

fn configure_camera_depth_usages(
    mut cameras: Query<&mut Camera3d, (Changed<Camera3d>, With<AtmosphericScattering>)>,
) {
    for mut camera in &mut cameras {
        camera.depth_texture_usages.0 |= TextureUsages::TEXTURE_BINDING.bits();
    }
}
