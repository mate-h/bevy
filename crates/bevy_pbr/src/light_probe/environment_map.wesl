import package::light_probe::{light_probe_iterator_new, light_probe_iterator_next};
import package::render::mesh_view_bindings as bindings;
import package::render::mesh_view_bindings::light_probes;
import package::render::mesh_view_types::{
    LIGHT_PROBE_FLAG_AFFECTS_LIGHTMAPPED_MESH_DIFFUSE, LIGHT_PROBE_FLAG_PARALLAX_CORRECT,
    LIGHT_PROBE_FLAG_MANSON_SLOAN_ENVIRONMENT_IBL
};
import package::render::pbr_lighting::{D_GGX, V_SmithGGXCorrelated, sample_visible_ggx, ggx_vndf_pdf, F_Schlick_vec, LightingInput, LayerLightingInput, LAYER_BASE, LAYER_CLEARCOAT};
import package::render::clustered_forward::ClusterableObjectIndexRanges;
import bevy_render::maths::quat_rotate;

/// The maximum representable value in a 32-bit floating point number.
const FLOAT_MAX: f32 = 3.40282347e+38;

struct EnvironmentMapLight {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
};

struct EnvironmentMapRadiances {
    irradiance: vec3<f32>,
    radiance: vec3<f32>,
    fresnel_radiance: vec3<f32>,
}

// https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling
// Experimental shade-time FIS. The reflection_probes harness enables this only
// when it binds an ordinary box-filtered radiance pyramid, never a GGX chain.
const SHADE_TIME_FIS: bool = false;
const FIS_SAMPLES: u32 = 64u;

// Integrate both Schlick coefficients so Fresnel remains inside the integral.
// Bounded visible-normal sampling and its matching reflected-direction PDF are
// shared with the renderer. Below-horizon samples contribute zero, not a
// renormalized weight. The correlated Smith BRDF matches the pathtracer.
fn filtered_importance_radiance(
    map: texture_cube<f32>,
    input: LayerLightingInput,
    world_position: vec3<f32>,
    light_from_world: mat4x4<f32>,
    bounds: vec3<f32>,
    parallax_correct: bool,
    rotation: vec4<f32>,
) -> mat2x3<f32> {
    let N = input.N;
    let V = normalize(2.0 * input.NdotV * N - input.R);
    let NoV = max(dot(N, V), 0.0001);
    let alpha = max(input.roughness, 0.001);
    let local_V = vec3(sqrt(max(0.0, 1.0 - NoV * NoV)), 0.0, NoV);
    let size = f32(textureDimensions(map, 0).x);
    let texel_solid_angle = 12.56637061436 / (6.0 * size * size);
    var a = vec3(0.0);
    var b = vec3(0.0);
    for (var i = 0u; i < FIS_SAMPLES; i++) {
        let xi = vec2((f32(i) + 0.5) / f32(FIS_SAMPLES),
            f32(reverseBits(i)) * 2.3283064365386963e-10);
        let L = sample_visible_ggx(xi, alpha, N, V);
        let NoL = dot(N, L);
        if NoL <= 0.0 { continue; }
        let H = normalize(V + L);
        let NoH = max(dot(N, H), 0.0);
        let pdf = max(ggx_vndf_pdf(local_V, NoH, alpha), 1e-8);
        let sample_solid_angle = 1.0 / (f32(FIS_SAMPLES) * pdf);
        let lod = clamp(0.5 * log2(sample_solid_angle / texel_solid_angle),
            0.0, f32(textureNumLevels(map) - 1u));
        let direction = compute_cubemap_sample_dir(world_position, L,
            light_from_world, bounds, parallax_correct, rotation);
        let Li = textureSampleLevel(map, bindings::environment_map_sampler, direction, lod).rgb;
        let weight = D_GGX(alpha, NoH) * V_SmithGGXCorrelated(alpha, NoV, NoL) * NoL / pdf;
        let Fc = pow(1.0 - saturate(dot(V, H)), 5.0);
        a += Li * (weight * (1.0 - Fc));
        b += Li * (weight * Fc);
    }
    return mat2x3(a / f32(FIS_SAMPLES), b / f32(FIS_SAMPLES));
}

/// Split-sum weight with the same multiple-scattering compensation as
/// `specular_multiscatter` and the pathtracer. Energy missing from the single
/// scatter is scaled back onto this lobe.
///
/// That energy used to be added as `FmsEms * irradiance` (Kulla–Conty). Irradiance
/// is a cosine lobe, so a small HDR source grew a halo a GGX integral does not have.
fn compensated_split_sum(F0: vec3<f32>, F_ab: vec2<f32>, specular_occlusion: f32) -> vec3<f32> {
    let ess = max(F_ab.x + F_ab.y, 1e-4);
    let single_scatter = (F0 * F_ab.x + F_ab.y) * specular_occlusion;
    let compensation = 1.0 + F0 * (1.0 / ess - 1.0);
    return single_scatter * compensation;
}

/// Computes the direction at which to sample the reflection probe.
///
/// * `light_from_world` is the matrix that transforms world space into light
///   probe space (a 1×1×1 cube centered on the origin).
///
/// * `parallax_correction_bounds` is the half-extents of the simulated
///   reflection boundaries used for parallax correction, in light probe space.
///   It's ignored if the `parallax_correct` parameter is false.
///
/// * `parallax_correct` is true if parallax correction is to be applied and
///   false otherwise.
fn compute_cubemap_sample_dir(
    world_ray_origin: vec3<f32>,
    world_ray_direction: vec3<f32>,
    light_from_world: mat4x4<f32>,
    parallax_correction_bounds: vec3<f32>,
    parallax_correct: bool,
    view_rotation: vec4<f32>,
) -> vec3<f32> {
    var sample_dir: vec3<f32>;

    // If we're supposed to parallax correct, then intersect with the light cube.
    if (parallax_correct) {
        // Compute the direction of the ray bouncing off the surface, in light
        // probe space.
        // Recall that light probe space is a 1×1×1 cube centered at the origin.
        let ray_origin = (light_from_world * vec4(world_ray_origin, 1.0)).xyz;
        let ray_direction = (light_from_world * vec4(world_ray_direction, 0.0)).xyz;

        // Solve for the intersection of that ray with each side of the cube.
        // Since our light probe is a 1×1×1 cube centered at the origin in light
        // probe space, the faces of the cube are at X = ±0.5, Y = ±0.5, and Z =
        // ±0.5.
        var t0 = (-parallax_correction_bounds - ray_origin) / ray_direction;
        var t1 = (parallax_correction_bounds - ray_origin) / ray_direction;

        // We're shooting the rays forward, so we need to rule out negative time
        // values. So, if t is negative, make it a large value so that we won't
        // choose it below.
        // We would use infinity here but WGSL forbids it:
        // https://github.com/gfx-rs/wgpu/issues/5515
        t0 = select(vec3(FLOAT_MAX), t0, t0 >= vec3(0.0));
        t1 = select(vec3(FLOAT_MAX), t1, t1 >= vec3(0.0));

        // Choose the minimum valid time value to find the intersection of the
        // first cube face.
        let t_min = min(t0, t1);
        let t = min(min(t_min.x, t_min.y), t_min.z);

        // Compute the sample direction. (It doesn't have to be normalized.)
        sample_dir = ray_origin + ray_direction * t;
    } else {
        // We treat the reflection as infinitely far away in the non-parallax
        // case, so the ray origin is irrelevant.
        sample_dir = (light_from_world * vec4(world_ray_direction, 0.0)).xyz;
    }

    // Rotating the world space ray direction by the environment map light view rotation,
    // it is equivalent to rotating the environment cubemap itself.
    sample_dir = quat_rotate(view_rotation, sample_dir);

    // Cubemaps are left-handed, so we negate the Z coordinate.
    sample_dir.z = -sample_dir.z;
    return sample_dir;
}

// Mip level for a prefiltered specular cubemap. Level `i` stores perceptual
// roughness `i / maxMip` (Karis 2013; the same linear schedule the Manson–Sloan
// filter writes). α = perceptual² is used only inside the GGX convolution.
fn specular_radiance_level(perceptual_roughness: f32, max_specular_mip: f32) -> f32 {
    return perceptual_roughness * max_specular_mip;
}

// GGX split-sum dominant reflection direction (Frostbite Listing 22).
// Used for both prefiltered and MS-generated specular maps — both store radiance
// indexed by direction, so the sample direction must match.
fn radiance_sample_direction_ggx_split_sum(N: vec3<f32>, R: vec3<f32>, mip_roughness: f32) -> vec3<f32> {
    let smoothness = saturate(1.0 - mip_roughness);
    let lerp_factor = smoothness * (sqrt(smoothness) + mip_roughness);
    return mix(N, R, lerp_factor);
}

// Define two versions of this function, one for the case in which there are
// multiple light probes and one for the case in which only the view light probe
// is present.

@if(MULTIPLE_LIGHT_PROBES_IN_ARRAY)

fn compute_radiances(
    input: LayerLightingInput,
    clusterable_object_index_ranges: ptr<function, ClusterableObjectIndexRanges>,
    world_position: vec3<f32>,
    found_diffuse_indirect: bool,
) -> EnvironmentMapRadiances {
    // Unpack.
    let N = input.N;
    let R = input.R;
    let perceptual_roughness = input.perceptual_roughness;
    let roughness = input.roughness;

    var radiances: EnvironmentMapRadiances;
    radiances.fresnel_radiance = vec3(0.0);

    // Find all reflection probes that contain the fragment. We're going to
    // accumulate all the radiance and irradiance from them in a weighted sum.
    var iterator = light_probe_iterator_new(
        world_position,
        /*is_irradiance_volume=*/ false,
        clusterable_object_index_ranges,
    );

    var total_weight = 0.0;
    var fis_a = vec3(0.0);
    radiances.irradiance = vec3(0.0);
    radiances.radiance = vec3(0.0);

    while (true) {
        var query_result = light_probe_iterator_next(&iterator);
        var view_rotation = vec4<f32>(0.0, 0.0, 0.0, 1.0);

        // If we reached the end of the light probe list, and we didn't find
        // enough reflection probes to reach a weight of 1.0, use the view
        // environment map if applicable. This allows for e.g. nice transitions
        // between the interior of a building and the outdoor environment map.
        if (query_result.texture_index < 0 && total_weight < 0.9999) {
            query_result.texture_index = light_probes.view_cubemap_index;
            query_result.intensity = light_probes.intensity_for_view;
            view_rotation = light_probes.view_rotation;
            query_result.light_from_world = mat4x4(
                vec4(1.0, 0.0, 0.0, 0.0),
                vec4(0.0, 1.0, 0.0, 0.0),
                vec4(0.0, 0.0, 1.0, 0.0),
                vec4(0.0, 0.0, 0.0, 1.0)
            );
            query_result.parallax_correction_bounds = vec3(0.0);
            query_result.flags = 0u;
            if light_probes.view_environment_map_affects_lightmapped_mesh_diffuse != 0u {
                query_result.flags |= LIGHT_PROBE_FLAG_AFFECTS_LIGHTMAPPED_MESH_DIFFUSE;
            }
            if light_probes.view_manson_sloan_environment_ibl != 0u {
                query_result.flags |= LIGHT_PROBE_FLAG_MANSON_SLOAN_ENVIRONMENT_IBL;
            }
            query_result.weight = 1.0 - total_weight;
        }

        // If we reached the end, we're done.
        if (query_result.texture_index < 0) {
            break;
        }

        // Split-sum approximation for image based lighting: https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
        let max_specular_mip = f32(textureNumLevels(
            bindings::specular_environment_maps[query_result.texture_index]) - 1u);
        // Both glTF-IBL-Sampler (prefiltered) and Esoterica/MS-generated maps encode mip `i`
        // as perceptual roughness `i/(n-1)` (α = perceptual² used only inside the GGX filter).
        let radiance_level = specular_radiance_level(perceptual_roughness, max_specular_mip);

        // If we're lightmapped, and we shouldn't accumulate diffuse light from the
        // environment map, note that.
        var enable_diffuse = !found_diffuse_indirect;
        @if(LIGHTMAP)
        enable_diffuse = enable_diffuse &&
            (query_result.flags & LIGHT_PROBE_FLAG_AFFECTS_LIGHTMAPPED_MESH_DIFFUSE) != 0u;

        let parallax_correct = (query_result.flags & LIGHT_PROBE_FLAG_PARALLAX_CORRECT) != 0u;

        if (enable_diffuse) {
            let irradiance_sample_dir = compute_cubemap_sample_dir(
                world_position,
                N,
                query_result.light_from_world,
                query_result.parallax_correction_bounds,
                parallax_correct,
                view_rotation,
            );
            radiances.irradiance += textureSampleLevel(
                bindings::diffuse_environment_maps[query_result.texture_index],
                bindings::environment_map_sampler,
                irradiance_sample_dir,
                0.0).rgb * query_result.intensity * query_result.weight;
        }

        var radiance_sample_dir =
            radiance_sample_direction_ggx_split_sum(N, R, roughness);
        radiance_sample_dir = compute_cubemap_sample_dir(
            world_position,
            radiance_sample_dir,
            query_result.light_from_world,
            query_result.parallax_correction_bounds,
            parallax_correct,
            view_rotation,
        );
        radiances.radiance +=
            textureSampleLevel(
                bindings::specular_environment_maps[query_result.texture_index],
                bindings::environment_map_sampler,
                radiance_sample_dir,
                radiance_level).rgb * query_result.intensity * query_result.weight;

        if SHADE_TIME_FIS {
            let filtered = filtered_importance_radiance(
                bindings::specular_environment_maps[query_result.texture_index], input,
                world_position, query_result.light_from_world,
                query_result.parallax_correction_bounds, parallax_correct, view_rotation);
            // Replace this probe's split-sum contribution below by accumulating
            // the independent Schlick integrals in a separate sum.
            fis_a += filtered[0] * query_result.intensity * query_result.weight;
            radiances.fresnel_radiance += filtered[1] * query_result.intensity * query_result.weight;
        }

        total_weight += query_result.weight;
    }

    if SHADE_TIME_FIS { radiances.radiance = fis_a; }
    if (total_weight != 0.0) {
        radiances.fresnel_radiance /= total_weight;
        radiances.irradiance /= total_weight;
        radiances.radiance /= total_weight;
    }

    return radiances;
}

@else   // MULTIPLE_LIGHT_PROBES_IN_ARRAY

fn compute_radiances(
    input: LayerLightingInput,
    clusterable_object_index_ranges: ptr<function, ClusterableObjectIndexRanges>,
    world_position: vec3<f32>,
    found_diffuse_indirect: bool,
) -> EnvironmentMapRadiances {
    // Unpack.
    let N = input.N;
    let R = input.R;
    let perceptual_roughness = input.perceptual_roughness;
    let roughness = input.roughness;

    var radiances: EnvironmentMapRadiances;
    radiances.fresnel_radiance = vec3(0.0);

    // If we have no light probe, bail.
    if (light_probes.view_cubemap_index < 0) {
        radiances.irradiance = vec3(0.0);
        radiances.radiance = vec3(0.0);
        return radiances;
    }

    // Split-sum approximation for image based lighting: https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
    // Technically we could use textureNumLevels(specular_environment_map) - 1 here, but we use a uniform
    // because textureNumLevels() does not work on WebGL2.
    // Same mip index as the multi-probe path: perceptual roughness times the last mip.
    let max_specular_mip = f32(light_probes.smallest_specular_mip_level_for_view);
    let radiance_level = specular_radiance_level(perceptual_roughness, max_specular_mip);

    let intensity = light_probes.intensity_for_view;

    // If we're lightmapped, and we shouldn't accumulate diffuse light from the
    // environment map, note that.
    var enable_diffuse = !found_diffuse_indirect;
    @if(LIGHTMAP)
    enable_diffuse = enable_diffuse &&
        light_probes.view_environment_map_affects_lightmapped_mesh_diffuse;

    if (enable_diffuse) {
        var irradiance_sample_dir = N;
        // Rotating the world space ray direction by the environment map light view rotation,
        // it is equivalent to rotating the environment cubemap itself.
        irradiance_sample_dir = quat_rotate(light_probes.view_rotation, irradiance_sample_dir);
        // Cube maps are left-handed so we negate the z coordinate.
        irradiance_sample_dir.z = -irradiance_sample_dir.z;
        radiances.irradiance = textureSampleLevel(
            bindings::diffuse_environment_map,
            bindings::environment_map_sampler,
            irradiance_sample_dir,
            0.0).rgb * intensity;
    }

    var radiance_sample_dir =
        radiance_sample_direction_ggx_split_sum(N, R, roughness);
    // Rotating the world space ray direction by the environment map light view rotation,
    // it is equivalent to rotating the environment cubemap itself.
    radiance_sample_dir = quat_rotate(light_probes.view_rotation, radiance_sample_dir);
    // Cube maps are left-handed so we negate the z coordinate.
    radiance_sample_dir.z = -radiance_sample_dir.z;
    radiances.radiance = textureSampleLevel(
        bindings::specular_environment_map,
        bindings::environment_map_sampler,
        radiance_sample_dir,
        radiance_level).rgb * intensity;

    if SHADE_TIME_FIS {
        let filtered = filtered_importance_radiance(bindings::specular_environment_map,
            input, world_position, mat4x4<f32>(
                vec4(1.0, 0.0, 0.0, 0.0), vec4(0.0, 1.0, 0.0, 0.0),
                vec4(0.0, 0.0, 1.0, 0.0), vec4(0.0, 0.0, 0.0, 1.0)),
            vec3(0.0), false, light_probes.view_rotation);
        radiances.radiance = filtered[0] * intensity;
        radiances.fresnel_radiance = filtered[1] * intensity;
    }
    return radiances;
}

@if(STANDARD_MATERIAL_CLEARCOAT)

/// Adds the environment map light from the clearcoat layer to that of the base
/// layer.
fn environment_map_light_clearcoat(
    out: ptr<function, EnvironmentMapLight>,
    input: ptr<function, LightingInput>,
    clusterable_object_index_ranges: ptr<function, ClusterableObjectIndexRanges>,
    found_diffuse_indirect: bool,
) {
    // Unpack.
    let world_position = (*input).P;
    let clearcoat_NdotV = (*input).layers[LAYER_CLEARCOAT].NdotV;
    let clearcoat_strength = (*input).clearcoat_strength;

    // Calculate the Fresnel term `Fc` for the clearcoat layer.
    // 0.04 is a hardcoded value for F0 from the Filament spec.
    let clearcoat_F0 = vec3<f32>(0.04);
    let Fc = F_Schlick_vec(clearcoat_F0, 1.0, clearcoat_NdotV) * clearcoat_strength;
    let inv_Fc = 1.0 - Fc;

    let clearcoat_radiances = compute_radiances(
        (*input).layers[LAYER_CLEARCOAT],
        clusterable_object_index_ranges,
        world_position,
        found_diffuse_indirect,
    );

    // Composite the clearcoat layer on top of the existing one.
    // These formulas are from Filament:
    // <https://google.github.io/filament/Filament.md.html#lighting/imagebasedlights/clearcoat>
    (*out).diffuse *= inv_Fc;
    if SHADE_TIME_FIS {
        (*out).specular = (*out).specular * inv_Fc * inv_Fc
            + (clearcoat_F0 * clearcoat_radiances.radiance + clearcoat_radiances.fresnel_radiance) * clearcoat_strength;
        return;
    }
    (*out).specular = (*out).specular * inv_Fc * inv_Fc + clearcoat_radiances.radiance * Fc;
}

fn environment_map_light(
    input: ptr<function, LightingInput>,
    clusterable_object_index_ranges: ptr<function, ClusterableObjectIndexRanges>,
    found_diffuse_indirect: bool,
) -> EnvironmentMapLight {
    // Unpack.
    let diffuse_color = (*input).diffuse_color;
    let metallic = (*input).metallic;
    let F_ab = (*input).F_ab;
    let F0_dielectric = (*input).F0_dielectric;
    let F0_metallic = (*input).F0_metallic;
    let world_position = (*input).P;

    var out: EnvironmentMapLight;

    let radiances = compute_radiances(
        (*input).layers[LAYER_BASE],
        clusterable_object_index_ranges,
        world_position,
        found_diffuse_indirect,
    );

    if (all(radiances.irradiance == vec3(0.0)) && all(radiances.radiance == vec3(0.0))) {
        out.diffuse = vec3(0.0);
        out.specular = vec3(0.0);
        return out;
    }

    // No real world material has specular values under 0.02, so we use this range as a
    // "pre-baked specular occlusion" that extinguishes the fresnel term, for artistic control.
    // See: https://google.github.io/filament/Filament.md.html#specularocclusion
    let F0_surface = mix(F0_dielectric, F0_metallic, metallic);
    let specular_occlusion = saturate(dot(F0_surface, vec3(50.0 * 0.33)));

    // Compensation is nonlinear in F0, so dielectric and metallic are evaluated
    // separately and then mixed, matching `lobe_reflectances` in the pathtracer.
    let specular_dielectric = compensated_split_sum(F0_dielectric, F_ab, specular_occlusion);
    let specular_metallic = compensated_split_sum(F0_metallic, F_ab, specular_occlusion);
    out.specular = mix(specular_dielectric, specular_metallic, metallic) * radiances.radiance;

    if SHADE_TIME_FIS {
        let compensation_dielectric = 1.0 + F0_dielectric * (1.0 / max(F_ab.x + F_ab.y, 1e-4) - 1.0);
        let compensation_metallic = 1.0 + F0_metallic * (1.0 / max(F_ab.x + F_ab.y, 1e-4) - 1.0);
        out.specular = mix(
            (F0_dielectric * radiances.radiance + radiances.fresnel_radiance) * compensation_dielectric,
            (F0_metallic * radiances.radiance + radiances.fresnel_radiance) * compensation_metallic,
            metallic) * specular_occlusion;
    }

    // `diffuse_color` already includes `(1 - metallic)`. Removing the compensated
    // dielectric specular albedo keeps a constant environment equal to the base color.
    if (!found_diffuse_indirect) {
        out.diffuse = diffuse_color * saturate(vec3(1.0) - specular_dielectric) * radiances.irradiance;
    } else {
        out.diffuse = vec3(0.0);
    }

    @if(STANDARD_MATERIAL_CLEARCOAT)
    environment_map_light_clearcoat(
        &out,
        input,
        clusterable_object_index_ranges,
        found_diffuse_indirect,
    );

    return out;
}
