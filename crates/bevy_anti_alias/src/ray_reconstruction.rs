//! Shared ray-reconstruction guide textures used by DLSS-RR and MetalFX Temporal Denoised Scaler.

use bevy_ecs::component::Component;
use bevy_render::texture::CachedTexture;

/// Marker inserted by DLSS-RR / MetalFX prepare when a camera needs Solari guide buffers.
#[derive(Component, Clone, Copy, Default)]
pub struct RayReconstructionDenoiser;

/// Guide buffers produced by Solari (or similar) for neural ray-reconstruction denoisers.
#[derive(Component)]
pub struct ViewRayReconstructionGuideTextures {
    pub diffuse_albedo: CachedTexture,
    pub specular_albedo: CachedTexture,
    /// World/view-space normals in RGB; perceptual roughness may also be packed in A for DLSS Packed mode.
    pub normal_roughness: CachedTexture,
    /// Specular / reflection motion vectors (DLSS-RR). Unused by MetalFX v1.
    pub specular_motion_vectors: CachedTexture,
    /// Linear roughness in R (MetalFX). May mirror `normal_roughness.a`.
    pub roughness: CachedTexture,
}

/// Backward-compatible alias for DLSS integration sites.
pub type ViewDlssRayReconstructionTextures = ViewRayReconstructionGuideTextures;
