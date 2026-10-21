//! Vertical cloud profiles: Nubis's height x type lookups, extended with liquid water.
//!
//! One Rgba16Float texture, [`CLOUD_PROFILE_HEIGHT_RES`] (x = height between the column's cloud
//! base, 0, and top, 1) by [`CLOUD_PROFILE_TYPE_RES`] (cloud type: 0 stratus, 0.5
//! stratocumulus, 1 cumulus):
//! - R, G: presence as a bottom and a top factor (Nubis's two lookups); their product is the cloud
//!   fraction at that height, and drives the erosion of the noise as the dimensional profile;
//! - B: in-cloud liquid water, peak 1, which scales the eroded density with height;
//! - A: unused (1).
//!
//! The engine only samples the lookups. Physically based sets are baked offline (a surface-layer
//! parcel ensemble on published LES cases) and loaded through [`CloudProfiles`]; without one,
//! every type holds the analytic envelope `x^a (1 - x)^b` of [`VolumetricCloudLayer`] with
//! uniform liquid water, which is today's look.

use bevy_asset::Handle;
use bevy_ecs::component::Component;
use bevy_image::Image;
use bevy_reflect::Reflect;

use super::VolumetricCloudLayer;

/// Samples along the height in the column (texel i at x = i / (N - 1)).
pub const CLOUD_PROFILE_HEIGHT_RES: usize = 64;
/// Samples along the cloud type (texel j at j / (N - 1)).
pub const CLOUD_PROFILE_TYPE_RES: usize = 32;

/// Vertical cloud profiles baked offline (module docs): an Rgba16Float image,
/// [`CLOUD_PROFILE_HEIGHT_RES`] x [`CLOUD_PROFILE_TYPE_RES`]. Add it to the [`Atmosphere`]
/// entity with the [`VolumetricCloudLayer`]; images of another size or format are ignored with a
/// warning, and the analytic envelope stays in use.
///
/// [`Atmosphere`]: bevy_light::Atmosphere
#[derive(Clone, Component, Reflect)]
#[reflect(Clone)]
pub struct CloudProfiles(pub Handle<Image>);

/// The lookup contents, row-major by type: `bottom[type * HEIGHT_RES + height]`.
#[derive(Clone, PartialEq)]
pub(crate) struct CloudProfileTables {
    pub bottom: Vec<f32>,
    pub top: Vec<f32>,
    pub liquid_water: Vec<f32>,
}

impl CloudProfileTables {
    /// Every type holds the analytic envelope `x^a (1 - x)^b`, peak-normalized (bottom `x^a`,
    /// top `(1 - x)^b / max`), with uniform liquid water.
    pub fn analytic(a: f32, b: f32) -> Self {
        let peak = a / (a + b).max(1e-6);
        let max_v = (peak.powf(a) * (1.0 - peak).powf(b)).max(1e-6);
        let n = CLOUD_PROFILE_TYPE_RES * CLOUD_PROFILE_HEIGHT_RES;
        let (mut bottom, mut top) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for _ in 0..CLOUD_PROFILE_TYPE_RES {
            for i in 0..CLOUD_PROFILE_HEIGHT_RES {
                let x = i as f32 / (CLOUD_PROFILE_HEIGHT_RES - 1) as f32;
                bottom.push(x.powf(a));
                top.push((1.0 - x).powf(b) / max_v);
            }
        }
        Self {
            bottom,
            top,
            liquid_water: vec![1.0; n],
        }
    }

    /// The analytic envelope of the layer's profile exponents.
    pub fn for_layer(layer: &VolumetricCloudLayer) -> Self {
        Self::analytic(layer.profile_bottom_exponent, layer.profile_top_exponent)
    }

    /// Rgba16Float texels (R bottom, G top, B liquid water, A 1).
    pub fn to_rgba16f_bytes(&self) -> Vec<u8> {
        self.bottom
            .iter()
            .zip(&self.top)
            .zip(&self.liquid_water)
            .flat_map(|((&b, &t), &l)| [b, t, l, 1.0].map(f32_to_f16))
            .flat_map(u16::to_le_bytes)
            .collect()
    }
}

/// IEEE half from f32 for the lookup values (finite, >= 0; subnormals flush to 0).
fn f32_to_f16(v: f32) -> u16 {
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
    fn analytic_matches_the_envelope() {
        let tables = CloudProfileTables::analytic(1.0, 1.0);
        for i in 0..CLOUD_PROFILE_HEIGHT_RES {
            let x = i as f32 / (CLOUD_PROFILE_HEIGHT_RES - 1) as f32;
            let v = tables.bottom[i] * tables.top[i];
            assert!((v - 4.0 * x * (1.0 - x)).abs() < 1e-5);
            assert_eq!(tables.liquid_water[i], 1.0);
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
