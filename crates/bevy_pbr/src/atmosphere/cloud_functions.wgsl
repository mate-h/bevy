#define_import_path bevy_pbr::atmosphere::cloud_functions

#import bevy_pbr::atmosphere::{
    types::{Atmosphere, Clouds},
    bindings::{atmosphere, clouds, view, lights, globals},
    functions::{sample_transmittance_lut, rayleigh, henyey_greenstein, FRAC_PI, sample_multiscattering_lut},
    bruneton_functions::{ray_intersects_ground}
}

const CLOUD_COLOR: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);

// Structure to store cloud ray-marching results
struct CloudResult {
    // Scattered light from clouds
    inscattering: vec3<f32>,
    // Light transmitted through clouds
    transmittance: vec3<f32>,
}

// Hash function for noise generation
fn hash(p: vec3<f32>) -> vec3<f32> {
    var p2 = fract(p * vec3<f32>(443.8975, 397.2973, 491.1871));
    p2 += dot(p2, p2.yzx + 19.19);
    return fract(vec3<f32>((p2.x + p2.y) * p2.z, (p2.x + p2.z) * p2.y, (p2.y + p2.z) * p2.x));
}

// Value noise function
fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    
    // Cubic interpolation
    let u = f * f * (3.0 - 2.0 * f);
    
    // Generate 8 corner points
    let a = hash(i);
    let b = hash(i + vec3<f32>(1.0, 0.0, 0.0));
    let c = hash(i + vec3<f32>(0.0, 1.0, 0.0));
    let d = hash(i + vec3<f32>(1.0, 1.0, 0.0));
    let e = hash(i + vec3<f32>(0.0, 0.0, 1.0));
    let f1 = hash(i + vec3<f32>(1.0, 0.0, 1.0));
    let g = hash(i + vec3<f32>(0.0, 1.0, 1.0));
    let h = hash(i + vec3<f32>(1.0, 1.0, 1.0));
    
    // Interpolate along x-axis
    let k0 = a.x;
    let k1 = b.x;
    let k2 = c.x;
    let k3 = d.x;
    let k4 = e.x;
    let k5 = f1.x;
    let k6 = g.x;
    let k7 = h.x;
    
    // Trilinear interpolation
    let w0 = mix(mix(k0, k1, u.x), mix(k2, k3, u.x), u.y);
    let w1 = mix(mix(k4, k5, u.x), mix(k6, k7, u.x), u.y);
    
    return mix(w0, w1, u.z);
}

// Fractal Brownian Motion (FBM) for cloud shape
fn fbm(p: vec3<f32>, octaves: i32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    
    for (var i = 0; i < octaves; i++) {
        value += amplitude * value_noise(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
        
        if (i >= octaves) { break; }
    }
    
    return value;
}

// Cloud density at a specific world position
fn cloud_density(pos: vec3<f32>, time: f32) -> f32 {
    // Transform to cloud space where bottom of cloud layer is at y=0
    var cloud_pos = pos;
    
    // Apply wind movement to clouds over time
    // Can't use swizzle assignment in WGSL, so update components individually
    cloud_pos.x += clouds.wind_speed.x * time;
    cloud_pos.z += clouds.wind_speed.y * time;
    
    // Add some high-frequency temporal variation to create subtle movement
    let temporal_offset = vec3<f32>(
        sin(time * 0.1) * 300.0,
        cos(time * 0.07) * 200.0, 
        sin(time * 0.15) * 250.0
    );
    
    // Offset the position for the detail noise to create more variation
    let detail_pos = cloud_pos + temporal_offset;
    
    // Sample base cloud shape noise with more octaves for better structure
    let base_noise = fbm(cloud_pos * clouds.shape_scale, 5);
    
    // Sample multiple detail noise layers at different scales and frequencies
    let detail_noise1 = fbm(detail_pos * clouds.detail_scale, 3);
    let detail_noise2 = fbm(detail_pos * clouds.detail_scale * 2.0, 2);
    
    // Create worley-like patterns for more realistic cloud edges
    let detail_pos2 = detail_pos * clouds.detail_scale * 3.0;
    let cellular_noise = 1.0 - abs(sin(detail_pos2.x * 3.0) * sin(detail_pos2.z * 3.0) * sin(detail_pos2.y * 3.0));
    
    // Create horizontal variation - more clouds in some areas, fewer in others
    let variation = (sin(cloud_pos.x * 0.0001 + time * 0.01) * sin(cloud_pos.z * 0.0001)) * 0.1;
    
    // Better height-based density profile for realistic cloud shapes
    let height = (pos.y - (atmosphere.bottom_radius + clouds.altitude)) / clouds.thickness;
    
    // Create different density profiles at different heights
    // - Bottom: sharp cutoff for flat bottoms
    // - Middle: maximum density
    // - Top: feathered and wispy
    let base_height_gradient = 1.0 - abs(2.0 * height - 1.0);
    let bottom_shape = smoothstep(0.0, 0.2, height);
    let top_shape = smoothstep(1.0, 0.7, height);
    
    // Combine with a stronger ramp at the bottom for the typical flat-bottom cloud look
    let height_gradient = base_height_gradient * bottom_shape * top_shape;
    
    // Final cloud shape from base noise, affected by coverage
    let coverage_adjusted = clouds.coverage + variation;
    var density = base_noise - (1.0 - coverage_adjusted);
    density = max(0.0, density);
    
    // Apply different types of detail erosion based on height
    // - Bottom: stronger erosion (sharper edges)
    // - Middle: medium erosion 
    // - Top: lighter erosion (wispy)
    let bottom_detail = mix(detail_noise1, cellular_noise, 0.7);
    let top_detail = mix(detail_noise1, detail_noise2, 0.6);
    let detail_mix = mix(bottom_detail, top_detail, height);
    
    // Erode cloud edges with the detail noise - stronger at the edges
    let edge_factor = 1.0 - smoothstep(0.0, 0.2, density);
    let erosion_strength = clouds.detail_strength * (1.0 + edge_factor * 1.5);
    density = max(0.0, density - (1.0 - detail_mix) * erosion_strength);
    
    // Apply height-based shaping for realistic cloud profiles
    density *= height_gradient;
    
    // Reduce density very slightly with horizontal distance from viewer for fade-out
    let horizontal_dist_sq = cloud_pos.x * cloud_pos.x + cloud_pos.z * cloud_pos.z;
    let dist_fade = 1.0 - smoothstep(10000000.0, 90000000.0, horizontal_dist_sq) * 0.2;
    
    // Apply global density multiplier
    return density * clouds.density * dist_fade;
}

// Sample light scattering and absorption through clouds
fn sample_cloud_light(pos: vec3<f32>, time: f32) -> f32 {
    var light_density = 0.0;
    
    // Accumulate light only when there's a light in the scene
    if (lights.n_directional_lights > 0u) {
        // Use first directional light (typically the sun)
        let light_dir = lights.directional_lights[0].direction_to_light;
        
        // Calculate step size with smaller steps at the beginning for better detail near cloud edges
        let base_step_size = clouds.thickness / f32(clouds.light_samples);
        
        // Start position for ray marching
        var light_pos = pos;
        
        // Light accumulation with distance-based weighting
        var accumulated_density = 0.0;
        var accumulated_weight = 0.0;
        
        // Silver lining effect - stronger at edges
        var silver_lining = 0.0;
        
        for (var i = 0u; i < clouds.light_samples; i++) {
            // Use a smaller step at the beginning for more detail close to the sample point
            let step_progress = f32(i) / f32(clouds.light_samples);
            let current_step = base_step_size * (0.5 + step_progress);
            
            // March to next position
            light_pos += light_dir * current_step;
            
            // Check if we've exited the cloud layer
            let height = (light_pos.y - (atmosphere.bottom_radius + clouds.altitude)) / clouds.thickness;
            if (height < 0.0 || height > 1.0) {
                break;
            }
            
            // Get cloud density at this point
            let sample_density = cloud_density(light_pos, time);
            
            // Silver lining calculation - light contribution near edges
            if (i < 2u && sample_density > 0.0) {
                silver_lining = max(silver_lining, sample_density * 0.2);
            }
            
            // Accumulate weighted density with distance falloff
            // (closer samples have stronger impact on shadowing)
            let weight = 1.0 - step_progress * 0.4; // 0.6-1.0 range gives less harsh falloff
            accumulated_density += sample_density * current_step * weight;
            accumulated_weight += weight;
            
            if (i >= clouds.light_samples) {
                break;
            }
        }
        
        // Normalize by total weight
        if (accumulated_weight > 0.0) {
            light_density = accumulated_density / accumulated_weight;
        }
        
        // Add silver lining effect to simulate light scattering at cloud edges
        light_density = max(0.0, light_density - silver_lining);
    }
    
    // Non-linear Beer's law for more realistic light extinction
    // Lower densities let more light through, while higher densities create darker shadows
    let extinction_strength = 2.2 + light_density * 0.8; // Stronger extinction for denser parts
    return exp(-light_density * extinction_strength);
}

// Ray-sphere intersection to determine when ray enters cloud layer
fn ray_sphere_intersection(ray_origin: vec3<f32>, ray_dir: vec3<f32>, sphere_center: vec3<f32>, sphere_radius: f32) -> vec2<f32> {
    let oc = ray_origin - sphere_center;
    let a = dot(ray_dir, ray_dir);
    let b = 2.0 * dot(oc, ray_dir);
    let c = dot(oc, oc) - sphere_radius * sphere_radius;
    let discriminant = b * b - 4.0 * a * c;
    
    if (discriminant < 0.0) {
        return vec2<f32>(-1.0); // No intersection
    }
    
    let t1 = (-b - sqrt(discriminant)) / (2.0 * a);
    let t2 = (-b + sqrt(discriminant)) / (2.0 * a);
    
    return vec2<f32>(t1, t2);
}

// Sample clouds via ray-marching through cloud volume
fn sample_clouds(ray_origin: vec3<f32>, ray_dir: vec3<f32>, time: f32) -> CloudResult {
    // Find intersection with cloud layer
    let earth_center = vec3<f32>(0.0, -atmosphere.bottom_radius, 0.0);
    let inner_radius = atmosphere.bottom_radius + clouds.altitude;
    let outer_radius = inner_radius + clouds.thickness;
    
    let inner_intersection = ray_sphere_intersection(ray_origin, ray_dir, earth_center, inner_radius);
    let outer_intersection = ray_sphere_intersection(ray_origin, ray_dir, earth_center, outer_radius);
    
    var t_start = max(0.0, inner_intersection.y);
    var t_end = outer_intersection.x;
    
    // Check if we're inside the cloud layer
    let height = length(ray_origin - earth_center) - atmosphere.bottom_radius;
    if (height >= clouds.altitude && height <= clouds.altitude + clouds.thickness) {
        t_start = 0.0;
        t_end = max(outer_intersection.x, inner_intersection.x);
    }
    
    // No intersection with cloud layer
    if (t_end <= t_start || t_end < 0.0) {
        return CloudResult(vec3<f32>(0.0), vec3<f32>(1.0));
    }
    
    // Use adaptive ray marching - more samples close to the camera, fewer far away
    let distance_factor = min(1.0, t_end / 100000.0);
    let adaptive_step_count = max(8u, clouds.ray_march_steps - u32(distance_factor * f32(clouds.ray_march_steps) * 0.5));
    
    // Prepare for ray marching - using the same approach as the sky rendering
    let sample_count = f32(adaptive_step_count);
    var total_inscattering = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    var prev_t = t_start;
    
    // Use dithered sampling to reduce banding artifacts
    let dither_offset = (sin(ray_dir.x * 13.0 + ray_dir.y * 17.0 + ray_dir.z * 19.0) * 0.5 + 0.5) * ((t_end - t_start) / sample_count) * 0.8;
    
    // Ray march through clouds
    for (var s = 0.0; s < sample_count; s += 1.0) {
        // Use consistent midpoint sampling as in the sky rendering
        // with a bit of dithering to reduce banding
        let t_i = t_start + ((t_end - t_start) * (s + 0.3)) / sample_count + dither_offset * (s % 2.0);
        let dt_i = t_i - prev_t;
        prev_t = t_i;
        
        // Get the current sample position
        let pos = ray_origin + ray_dir * t_i;
        
        // Sample cloud density at this position
        var density = cloud_density(pos, time);
        
        // Skip empty space
        if (density > 0.0) {
            // Apply density refinement for smoother appearance in low-density areas
            density = smoothstep(0.0, 0.05, density) * density;
            
            // Calculate position parameters for the current sample point
            let local_r = length(pos - earth_center);
            let local_up = normalize(pos - earth_center);
            
            // Calculate direct light contribution
            let light_energy = sample_cloud_light(pos, time);
            
            // Get light color from both direct light and multiscattering
            var direct_light = vec3<f32>(0.0);
            var mu_light: f32 = 0.0;
            
            for (var light_i = 0u; light_i < min(lights.n_directional_lights, 2u); light_i++) {
                let light = &lights.directional_lights[light_i];
                let neg_LdotV = dot((*light).direction_to_light, ray_dir);
                mu_light = dot((*light).direction_to_light, local_up);
                
                // Apply more nuanced phase function for anisotropic scattering
                let forward_scatter = henyey_greenstein(neg_LdotV, 0.6);  // Strong forward scattering
                let side_scatter = henyey_greenstein(neg_LdotV, 0.0);     // Isotropic for general illumination
                let back_scatter = henyey_greenstein(neg_LdotV, -0.2);    // Slight back scattering
                
                // Blend phase functions - gives better results than single HG function
                let phase = mix(forward_scatter, mix(side_scatter, back_scatter, 0.3), 0.7);
                
                // Add directional light contribution
                direct_light += (*light).color.rgb * phase * 0.1 * light_energy;
            }
            
            // Sample multiscattering for ambient light - using the same approach as in atmospheric scattering
            let ms_contribution = sample_multiscattering_lut(local_r, 0.0);
            
            // Calculate the cloud scattering coefficient
            // This is the color that will be scattered at this point
            let cloud_scattering_coeff = CLOUD_COLOR * (density * clouds.density);
            let cloud_extinction_coeff = cloud_scattering_coeff;
            
            // Calculate optical depth for this segment
            let optical_depth = cloud_extinction_coeff * dt_i;
            let transmittance = exp(-optical_depth);
            
            // Calculate inscattering for this segment
            // Use the multiscattering as a proxy for the ambient light contribution
            // Adjust intensity of multiscattering to balance with direct light
            let scattering_source = direct_light + ms_contribution * 0.2;
            let inscattering = scattering_source * cloud_scattering_coeff;
            
            // Integrate using the same analytical approach as in the sky rendering
            // This gives more accurate results than discrete accumulation
            let cloud_extinction = max(vec3<f32>(0.000001), cloud_extinction_coeff); // Avoid division by zero
            let s_int = (inscattering - inscattering * transmittance) / cloud_extinction;
            
            // Add inscattering contribution for this segment, attenuated by throughput so far
            total_inscattering += throughput * s_int * clouds.brightness;
            
            // Update throughput for the next segment
            throughput *= transmittance;
            
            // Early exit if clouds are opaque (optimization)
            if (all(throughput < vec3<f32>(0.005))) {
                break;
            }
        }
    }
    
    return CloudResult(total_inscattering, throughput);
}
