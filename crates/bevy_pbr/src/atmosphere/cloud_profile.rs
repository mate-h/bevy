//! Vertical cloud profiles: Nubis's height x type lookup with physical contents.
//!
//! One Rgba16Float texture, [`CLOUD_PROFILE_HEIGHT_RES`] texels per row against metres above the
//! shared cloud base (texel i at i / (N - 1) x [`CLOUD_PROFILE_RANGE_M`]), one row per cloud
//! type ([`CLOUD_PROFILE_TYPE_RES`]: 0 stratus, 0.5 stratocumulus, 1 cumulus):
//! - R, G: cloud fraction (peak 1) as a bottom and a top factor (Nubis's two lookups); times
//!   coverage it is the dimensional profile that the erosion turns into clouds;
//! - B: in-cloud liquid water (kg/m^3); divided by the layer's
//!   [`VolumetricCloudLayer::liquid_water`](super::VolumetricCloudLayer::liquid_water) it scales
//!   the eroded density;
//! - A: unused (1).
//!
//! A last row, filled on the GPU from the shape noise, holds the noise quantiles that make the
//! erosion threshold exact (`clouds.wesl` `cloud_erosion_threshold`). SI throughout: m, kg/m^3.

use bevy_asset::Handle;
use bevy_ecs::component::Component;
use bevy_image::Image;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};

/// Texels per row: height above the cloud base (texel i at i / (N - 1) x range).
pub const CLOUD_PROFILE_HEIGHT_RES: usize = 256;
/// Rows of cloud types (texel j at type j / (N - 1)).
pub const CLOUD_PROFILE_TYPE_RES: usize = 64;
/// Height range of the rows (m above the cloud base; must match `clouds.wesl`).
pub const CLOUD_PROFILE_RANGE_M: f32 = 4000.0;

/// The vertical structure of the clouds of a [`VolumetricCloudLayer`](super::VolumetricCloudLayer) (module docs).
#[derive(Clone, Component, Reflect, Default)]
#[reflect(Clone, Default)]
pub enum CloudProfiles {
    /// Baked offline from a surface-layer parcel ensemble (Neggers, Köhler & Beljaars 2009,
    /// J. Atmos. Sci. 66): the updrafts are the top 10 % of the surface-layer distribution with
    /// spreads from the surface fluxes, lifted with entrainment 1 / (tau w); cloud fraction is the
    /// share of saturated rising parcels plus the saturated mean state, liquid water their mean.
    /// Cloud fraction is the share of saturated rising parcels plus the saturated mean state,
    /// liquid water the parcels' mean, against height above the base. The type axis orders
    /// published LES cases by cloud depth, like Nubis's stratus .. cumulus axis, linear between:
    /// - 0: stratocumulus, DYCOMS-II RF01 (Stevens et al. 2005): an overcast deck ~270 m deep;
    /// - 0.5: shallow trade cumulus, BOMEX (Siebesma et al. 2003): the fraction peaks at the base
    ///   and falls off to ~1.4 km (within ~20 % of the LES above 1 km);
    /// - 1: deeper trade cumulus, RICO's equilibrium state (van Zanten et al. 2011, Fig. 4):
    ///   0.05 at the base, ~0.015 through the cloud layer, tops near 3 km (LES: 0.06, ~0.02,
    ///   ~2.7 km).
    /// The baker records its inputs in the KTX2 key/value data.
    #[default]
    Physical,
    /// The analytic envelope `x^bottom (1 - x)^top`, normalized to a peak of 1, of the height x
    /// in a column whose top rises with coverage from
    /// [`VolumetricCloudLayer::profile_top_min`] to the layer top, with uniform liquid water and
    /// Nubis's `1 - profile` erosion threshold (evaluated in the shader).
    ///
    /// [`VolumetricCloudLayer::profile_top_min`]: super::VolumetricCloudLayer::profile_top_min
    Envelope {
        /// Exponent of the rise above the base.
        bottom: f32,
        /// Exponent of the fall toward the top.
        top: f32,
    },
    /// A lookup image of the format above (Rgba16Float, 64 x 32), e.g. baked for another air
    /// mass or painted. Until it loads, or if its format differs (logged), [`Self::Physical`].
    Image(Handle<Image>),
    /// Nubis's vertical profile (Schneider, "Nubis, Evolved", SIGGRAPH 2022 Advances, slides 22,
    /// 26 and 28): the top-type and bottom-type gradients, height in the cloud layer by cloud
    /// type, multiplied (one weather type drives both), with Nubis's `1 - profile` erosion
    /// threshold and uniform liquid water. The 64 x 64 lookups were recovered from the deck and
    /// rebuilt as smooth functions (cloud-top boundary per type, gradients between base and top),
    /// within ~1.5 grey levels RMS of the original, then baked at this lookup's resolution.
    Nubis,
}

impl CloudProfiles {
    /// Adiabatic liquid water growing linearly above the base times an adiabatic fraction falling
    /// linearly toward the top (Warner 1955): `x (1 - x)`.
    pub const ADIABATIC: Self = Self::Envelope {
        bottom: 1.0,
        top: 1.0,
    };
    /// Nubis Evolved (SIGGRAPH 2022) envelope-model gradients: `x^2 (1 - x)^1.5`.
    pub const NUBIS_ENVELOPE: Self = Self::Envelope {
        bottom: 2.0,
        top: 1.5,
    };
}

/// The baked [`CloudProfiles::Physical`] set: an uncompressed Rgba16Float KTX2.
const PHYSICAL_KTX2: &[u8] = include_bytes!("cloud_profiles.ktx2");
/// The baked [`CloudProfiles::Nubis`] set (R bottom-type, G top-type gradient).
const NUBIS_KTX2: &[u8] = include_bytes!("cloud_profiles_nubis.ktx2");

/// The single level of an uncompressed KTX2 (vkFormat 97 = R16G16B16A16_SFLOAT,
/// [`CLOUD_PROFILE_HEIGHT_RES`] x [`CLOUD_PROFILE_TYPE_RES`], checked by a test).
fn ktx2_texels(ktx2: &'static [u8]) -> &'static [u8] {
    let u64_at = |o: usize| u64::from_le_bytes(ktx2[o..o + 8].try_into().unwrap()) as usize;
    let (offset, len) = (u64_at(80), u64_at(88));
    &ktx2[offset..offset + len]
}

/// Texels of [`CloudProfiles::Physical`].
pub(crate) fn physical_profile_texels() -> &'static [u8] {
    ktx2_texels(PHYSICAL_KTX2)
}

/// Texels of [`CloudProfiles::Nubis`].
pub(crate) fn nubis_profile_texels() -> &'static [u8] {
    ktx2_texels(NUBIS_KTX2)
}

/// IEEE half from f32 for values in [0, 1] (subnormals flush to 0).
pub(crate) fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7bff;
    }
    // Round to nearest on the 13 dropped mantissa bits (a carry into the exponent is correct).
    let mant = bits & 0x7f_ffff;
    let h = ((exp as u32) << 10) | (mant >> 13);
    sign | (h + ((mant >> 12) & 1)).min(0x7bff) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baked_sets_are_rgba16f_lookups() {
        for (ktx2, texels) in [
            (PHYSICAL_KTX2, physical_profile_texels()),
            (NUBIS_KTX2, nubis_profile_texels()),
        ] {
            let u32_at = |o: usize| u32::from_le_bytes(ktx2[o..o + 4].try_into().unwrap());
            assert_eq!(&ktx2[..12], b"\xabKTX 20\xbb\r\n\x1a\n");
            assert_eq!(u32_at(12), 97);
            assert_eq!(u32_at(20) as usize, CLOUD_PROFILE_HEIGHT_RES);
            assert_eq!(u32_at(24) as usize, CLOUD_PROFILE_TYPE_RES);
            assert_eq!(texels.len(), 8 * CLOUD_PROFILE_HEIGHT_RES * CLOUD_PROFILE_TYPE_RES);
        }
    }

    #[test]
    fn half_conversion() {
        assert_eq!(f32_to_f16(0.0), 0);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(0.5), 0x3800);
        assert_eq!(f32_to_f16(4.0), 0x4400);
    }
}
