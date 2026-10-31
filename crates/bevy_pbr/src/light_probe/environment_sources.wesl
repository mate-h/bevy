//! Extracts bright, disk-like sources from a generated environment map.
//!
//! Every texel brighter than a threshold `t` splits into a residual that stays in
//! the map and an excess that becomes light, keeping its chroma:
//! `e = L max(0, 1 - t / Y(L))`. Excess is summed into a coarse grid of cells
//! (moments add up exactly). Connected groups of cells, across face seams,
//! become disk-shaped directional lights, read back to the main world. Kept
//! sources leave the map: the residual replaces the copy of mip 0, so every
//! downstream filter sees `L - e`.
//!
//! For a uniform disk of angular radius θ, the mean cosine to its center is
//! `(1 + cos θ) / 2`, so `θ = acos(2 R̄ - 1)` with `R̄` the luminance-weighted
//! mean resultant length of the excess directions.

import package::render::utils::{dir_to_cube_uv, sample_cube_dir};
import bevy_render::maths::PI;

const MAX_GRID: u32 = 16u;
const MAX_CELLS: u32 = 6u * MAX_GRID * MAX_GRID;
const MAX_SOURCES: u32 = 4u;
/// Threshold candidates, half an octave apart below the peak luminance.
const CANDIDATES: u32 = 32u;
/// More connected components than sources may be examined, since some are rejected.
const MAX_ATTEMPTS: u32 = 8u;
const SELECT_THREADS: u32 = 256u;
const CELLS_PER_THREAD: u32 = MAX_CELLS / SELECT_THREADS;
const UNLABELED: u32 = 0xFFFFFFFFu;

struct SourceParams {
    base_size: u32,
    grid: u32,
    max_sources: u32,
    _pad: u32,
    dynamic_range: f32,
    min_isotropy: f32,
    min_energy_fraction: f32,
    _pad2: f32,
}

struct Cell {
    /// Σ e Ω, RGB.
    energy: vec3<f32>,
    /// Σ Y(e) Ω.
    weight: f32,
    /// Σ Y(e) Ω d, in cubemap space.
    moment: vec3<f32>,
    max_luminance: f32,
    /// Σ Y Ω and Σ Ω of the full radiance.
    luminance: f32,
    solid_angle: f32,
    kept: u32,
    _pad: u32,
    /// The four neighboring cells, across face seams.
    neighbors: vec4<u32>,
    /// Σ min(Y, t_k) Ω for every threshold candidate.
    clamped: array<f32, CANDIDATES>,
}

struct Globals {
    max_luminance: f32,
    mean_luminance: f32,
    threshold: f32,
    total_weight: f32,
}

struct Source {
    /// World-space direction toward the light, before the environment map's rotation.
    direction: vec3<f32>,
    angular_size: f32,
    /// Normal-incidence illuminance, in environment map radiance units.
    illuminance: vec3<f32>,
    isotropy: f32,
}

struct Sources {
    count: u32,
    threshold: f32,
    extracted_fraction: f32,
    _pad: f32,
    sources: array<Source, MAX_SOURCES>,
}

@group(0) @binding(0) var source_cubemap: texture_2d_array<f32>;
@group(0) @binding(1) var residual_cubemap: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: SourceParams;
@group(0) @binding(3) var<storage, read_write> cells: array<Cell, MAX_CELLS>;
@group(0) @binding(4) var<storage, read_write> globals: Globals;
@group(0) @binding(5) var<storage, read_write> output: Sources;

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

/// Exact solid angle of a cubemap texel: the integral of `(1 + x² + y²)^{-3/2}`
/// over the texel, from the corner function `atan2(xy, sqrt(1 + x² + y²))`.
fn texel_solid_angle(coords: vec2<u32>, size: u32) -> f32 {
    let inv = 2.0 / f32(size);
    let lo = vec2<f32>(coords) * inv - 1.0;
    let hi = lo + inv;
    return corner(hi.x, hi.y) - corner(lo.x, hi.y) - corner(hi.x, lo.y) + corner(lo.x, lo.y);
}

fn corner(x: f32, y: f32) -> f32 {
    return atan2(x * y, sqrt(1.0 + x * x + y * y));
}

fn texel_direction(coords: vec2<u32>, face: u32, size: u32) -> vec3<f32> {
    return sample_cube_dir((vec2<f32>(coords) + 0.5) / f32(size), face);
}

fn cell_index(face: u32, cell: vec2<u32>) -> u32 {
    return (face * params.grid + cell.y) * params.grid + cell.x;
}

fn cell_count() -> u32 {
    return 6u * params.grid * params.grid;
}

/// Face and cell coordinates of the cell with the given index.
fn cell_coords(index: u32) -> vec3<u32> {
    let per_face = params.grid * params.grid;
    let face = index / per_face;
    let rem = index - face * per_face;
    return vec3(rem % params.grid, rem / params.grid, face);
}

/// Texel range `[start, start + span)` covered by a cell.
fn cell_span() -> u32 {
    return params.base_size / params.grid;
}

/// Threads per cell in the per-cell passes: one workgroup per cell, the cell's
/// texels spread over its threads, then reduced in workgroup memory.
const CELL_THREADS: u32 = 64u;

var<workgroup> stats_partial: array<vec3<f32>, CELL_THREADS>;

@compute
@workgroup_size(64, 1, 1)
fn cell_stats(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_index) thread: u32,
) {
    let index = workgroup.x;
    let coords = cell_coords(index);
    let span = cell_span();
    // Σ Y Ω, Σ Ω and max Y.
    var partial = vec3(0.0);
    for (var t = thread; t < span * span; t += CELL_THREADS) {
        let texel = coords.xy * span + vec2(t % span, t / span);
        let L = textureLoad(source_cubemap, texel, coords.z, 0).rgb;
        let omega = texel_solid_angle(texel, params.base_size);
        let Y = luminance(max(L, vec3(0.0)));
        partial = vec3(partial.x + Y * omega, partial.y + omega, max(partial.z, Y));
    }
    stats_partial[thread] = partial;
    workgroupBarrier();
    for (var stride = CELL_THREADS / 2u; stride > 0u; stride /= 2u) {
        if thread < stride {
            let other = stats_partial[thread + stride];
            let mine = stats_partial[thread];
            stats_partial[thread] = vec3(mine.x + other.x, mine.y + other.y, max(mine.z, other.z));
        }
        workgroupBarrier();
    }
    if thread == 0u {
        cells[index].luminance = stats_partial[0].x;
        cells[index].solid_angle = stats_partial[0].y;
        cells[index].max_luminance = stats_partial[0].z;
        cells[index].neighbors = vec4(
            cell_neighbor(index, vec2(1, 0)),
            cell_neighbor(index, vec2(-1, 0)),
            cell_neighbor(index, vec2(0, 1)),
            cell_neighbor(index, vec2(0, -1)),
        );
    }
}

var<workgroup> reduce_a: array<f32, SELECT_THREADS>;
var<workgroup> reduce_b: array<f32, SELECT_THREADS>;
var<workgroup> peak_luminance: f32;

/// The largest luminance over all cells, the same in every invocation.
fn workgroup_peak_luminance(thread: u32, threads: u32) -> f32 {
    var peak = 0.0;
    for (var index = thread; index < cell_count(); index += threads) {
        peak = max(peak, cells[index].max_luminance);
    }
    reduce_a[thread] = peak;
    workgroupBarrier();
    for (var stride = threads / 2u; stride > 0u; stride /= 2u) {
        if thread < stride {
            reduce_a[thread] = max(reduce_a[thread], reduce_a[thread + stride]);
        }
        workgroupBarrier();
    }
    if thread == 0u {
        peak_luminance = reduce_a[0];
    }
    return workgroupUniformLoad(&peak_luminance);
}

fn candidate_threshold(peak: f32, k: u32) -> f32 {
    return peak * exp2(-0.5 * f32(k));
}

var<workgroup> texel_bin: array<u32, CELL_THREADS>;
var<workgroup> texel_luminance: array<f32, CELL_THREADS>;
var<workgroup> texel_solid_angles: array<f32, CELL_THREADS>;
var<workgroup> bin_luminance: array<f32, CANDIDATES>;
var<workgroup> bin_solid_angle: array<f32, CANDIDATES>;

/// Σ min(Y, t_k) Ω of each cell, for every threshold candidate, from a histogram:
/// a texel in bin `j` (`t_{j+1} < Y ≤ t_j`) contributes `Y Ω` to the candidates
/// `k ≤ j` and `t_k Ω` to the rest, so
/// `Σ min(Y, t_k) Ω = Σ_{j ≥ k} (Σ Y Ω)_j + t_k Σ_{j < k} (Σ Ω)_j` exactly.
@compute
@workgroup_size(64, 1, 1)
fn cell_clamped(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_index) thread: u32,
) {
    let index = workgroup.x;
    let peak = workgroup_peak_luminance(thread, CELL_THREADS);
    let coords = cell_coords(index);
    let span = cell_span();
    var luminance_sum = 0.0;
    var solid_angle_sum = 0.0;
    for (var first = 0u; first < span * span; first += CELL_THREADS) {
        let t = first + thread;
        var bin = CANDIDATES;
        var weighted = 0.0;
        var omega = 0.0;
        if t < span * span {
            let texel = coords.xy * span + vec2(t % span, t / span);
            let Y = luminance(max(textureLoad(source_cubemap, texel, coords.z, 0).rgb, vec3(0.0)));
            omega = texel_solid_angle(texel, params.base_size);
            weighted = Y * omega;
            // Texels darker than every candidate go in the last bin: min is Y for all.
            bin = select(
                CANDIDATES - 1u,
                u32(clamp(floor(-2.0 * log2(Y / peak)), 0.0, f32(CANDIDATES - 1u))),
                Y > 0.0 && peak > 0.0,
            );
        }
        texel_bin[thread] = bin;
        texel_luminance[thread] = weighted;
        texel_solid_angles[thread] = omega;
        workgroupBarrier();
        if thread < CANDIDATES {
            for (var i = 0u; i < CELL_THREADS; i++) {
                if texel_bin[i] == thread {
                    luminance_sum += texel_luminance[i];
                    solid_angle_sum += texel_solid_angles[i];
                }
            }
        }
        workgroupBarrier();
    }
    if thread < CANDIDATES {
        bin_luminance[thread] = luminance_sum;
        bin_solid_angle[thread] = solid_angle_sum;
    }
    workgroupBarrier();
    if thread < CANDIDATES {
        var clamped = 0.0;
        for (var j = 0u; j < CANDIDATES; j++) {
            clamped += select(candidate_threshold(peak, thread) * bin_solid_angle[j], bin_luminance[j], j >= thread);
        }
        cells[index].clamped[thread] = clamped;
    }
}

/// Largest candidate threshold whose clamped map has `peak / mean ≤ dynamic_range`.
/// Infinite when the map already meets the target; 0 (extract everything bright)
/// when no candidate does, as for an emitter on a black background.
///
/// Thread `t` sums candidate `t % 32` over every eighth cell, then the 8 partial
/// sums of each candidate are added up.
@compute
@workgroup_size(256, 1, 1)
fn choose_threshold(@builtin(local_invocation_index) thread: u32) {
    let peak = workgroup_peak_luminance(thread, SELECT_THREADS);
    let k = thread % CANDIDATES;
    let group = thread / CANDIDATES;
    var clamped = 0.0;
    var total = 0.0;
    for (var index = group; index < cell_count(); index += SELECT_THREADS / CANDIDATES) {
        clamped += cells[index].clamped[k];
        if k == 0u {
            total += cells[index].luminance;
        }
    }
    reduce_a[thread] = clamped;
    reduce_b[thread] = total;
    workgroupBarrier();
    if thread < CANDIDATES {
        var sum = 0.0;
        for (var g = 0u; g < SELECT_THREADS / CANDIDATES; g++) {
            sum += reduce_a[g * CANDIDATES + thread];
        }
        reduce_a[thread] = candidate_threshold(peak, thread) / max(sum / (4.0 * PI), 1e-30);
    }
    if thread == 0u {
        var sum = 0.0;
        for (var g = 0u; g < SELECT_THREADS / CANDIDATES; g++) {
            sum += reduce_b[g * CANDIDATES];
        }
        reduce_b[0] = sum;
    }
    workgroupBarrier();
    if thread == 0u {
        let mean = reduce_b[0] / (4.0 * PI);
        var threshold = 0.0;
        if peak <= 0.0 || peak <= params.dynamic_range * mean {
            threshold = 3.0e38;
        } else {
            for (var i = 0u; i < CANDIDATES; i++) {
                if reduce_a[i] <= params.dynamic_range {
                    threshold = candidate_threshold(peak, i);
                    break;
                }
            }
        }
        globals.max_luminance = peak;
        globals.mean_luminance = mean;
        globals.threshold = threshold;
    }
}

var<workgroup> excess_partial: array<vec4<f32>, CELL_THREADS>;
var<workgroup> moment_partial: array<vec4<f32>, CELL_THREADS>;

@compute
@workgroup_size(64, 1, 1)
fn cell_excess(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_index) thread: u32,
) {
    let index = workgroup.x;
    let coords = cell_coords(index);
    let span = cell_span();
    let threshold = globals.threshold;
    var energy_weight = vec4(0.0);
    var moment = vec4(0.0);
    for (var t = thread; t < span * span; t += CELL_THREADS) {
        let texel = coords.xy * span + vec2(t % span, t / span);
        let L = max(textureLoad(source_cubemap, texel, coords.z, 0).rgb, vec3(0.0));
        let Y = luminance(L);
        if Y <= threshold {
            continue;
        }
        let omega = texel_solid_angle(texel, params.base_size);
        let e = L * (1.0 - threshold / Y);
        let w = luminance(e) * omega;
        energy_weight += vec4(e * omega, w);
        moment += vec4(w * texel_direction(texel, coords.z, params.base_size), 0.0);
    }
    excess_partial[thread] = energy_weight;
    moment_partial[thread] = moment;
    workgroupBarrier();
    for (var stride = CELL_THREADS / 2u; stride > 0u; stride /= 2u) {
        if thread < stride {
            excess_partial[thread] += excess_partial[thread + stride];
            moment_partial[thread] += moment_partial[thread + stride];
        }
        workgroupBarrier();
    }
    if thread == 0u {
        cells[index].energy = excess_partial[0].xyz;
        cells[index].weight = excess_partial[0].w;
        cells[index].moment = moment_partial[0].xyz;
        cells[index].kept = 0u;
    }
}

var<workgroup> labels: array<u32, MAX_CELLS>;
var<workgroup> reduce_index: array<u32, SELECT_THREADS>;
var<workgroup> reduce_vector: array<vec4<f32>, SELECT_THREADS>;
var<workgroup> changed: atomic<u32>;
// Copies of per-workgroup decisions, read with `workgroupUniformLoad` so the
// loops around barriers stay in uniform control flow.
var<workgroup> changed_value: u32;
var<workgroup> accepted_value: u32;

fn cell_neighbor(index: u32, step: vec2<i32>) -> u32 {
    let coords = cell_coords(index);
    let uv = (vec2<f32>(coords.xy) + 0.5 + vec2<f32>(step)) / f32(params.grid);
    let direction = sample_cube_dir(uv, coords.z);
    let cube_uv = dir_to_cube_uv(direction);
    let cell = min(vec2<u32>(cube_uv.uv * f32(params.grid)), vec2(params.grid - 1u));
    return cell_index(cube_uv.face, cell);
}

/// Sums a vec4 over the workgroup into `reduce_vector[0]`.
fn reduce_sum(thread: u32, value: vec4<f32>) -> vec4<f32> {
    reduce_vector[thread] = value;
    workgroupBarrier();
    for (var stride = SELECT_THREADS / 2u; stride > 0u; stride /= 2u) {
        if thread < stride {
            reduce_vector[thread] += reduce_vector[thread + stride];
        }
        workgroupBarrier();
    }
    let result = reduce_vector[0];
    workgroupBarrier();
    return result;
}

@compute
@workgroup_size(256, 1, 1)
fn select_sources(@builtin(local_invocation_index) thread: u32) {
    let count = cell_count();
    var total_weight = 0.0;
    for (var j = 0u; j < CELLS_PER_THREAD; j++) {
        let index = thread + j * SELECT_THREADS;
        labels[index] = UNLABELED;
        if index < count {
            total_weight += cells[index].weight;
        }
    }
    total_weight = reduce_sum(thread, vec4(total_weight, 0.0, 0.0, 0.0)).x;

    var kept_count = 0u;
    var kept_weight = 0.0;
    for (var attempt = 0u; attempt < MAX_ATTEMPTS; attempt++) {
        // Seed: the unlabeled cell with the most excess.
        var best = 0.0;
        var best_index = UNLABELED;
        for (var j = 0u; j < CELLS_PER_THREAD; j++) {
            let index = thread + j * SELECT_THREADS;
            if index < count && labels[index] == UNLABELED && cells[index].weight > best {
                best = cells[index].weight;
                best_index = index;
            }
        }
        reduce_a[thread] = best;
        reduce_index[thread] = best_index;
        workgroupBarrier();
        for (var stride = SELECT_THREADS / 2u; stride > 0u; stride /= 2u) {
            if thread < stride && reduce_a[thread + stride] > reduce_a[thread] {
                reduce_a[thread] = reduce_a[thread + stride];
                reduce_index[thread] = reduce_index[thread + stride];
            }
            workgroupBarrier();
        }
        let seed = workgroupUniformLoad(&reduce_index[0]);
        if seed == UNLABELED || kept_count >= params.max_sources {
            break;
        }
        if thread == 0u {
            labels[seed] = attempt;
        }
        workgroupBarrier();

        // Flood fill across cells with excess, crossing face seams.
        for (var iteration = 0u; iteration < 6u * MAX_GRID; iteration++) {
            if thread == 0u {
                atomicStore(&changed, 0u);
            }
            workgroupBarrier();
            for (var j = 0u; j < CELLS_PER_THREAD; j++) {
                let index = thread + j * SELECT_THREADS;
                if index >= count || labels[index] != UNLABELED || cells[index].weight <= 0.0 {
                    continue;
                }
                let neighbors = cells[index].neighbors;
                if labels[neighbors.x] == attempt
                    || labels[neighbors.y] == attempt
                    || labels[neighbors.z] == attempt
                    || labels[neighbors.w] == attempt {
                    labels[index] = attempt;
                    atomicStore(&changed, 1u);
                }
            }
            workgroupBarrier();
            if thread == 0u {
                changed_value = atomicLoad(&changed);
            }
            if workgroupUniformLoad(&changed_value) == 0u {
                break;
            }
        }

        // Component moments.
        var energy_weight = vec4(0.0);
        var moment = vec4(0.0);
        for (var j = 0u; j < CELLS_PER_THREAD; j++) {
            let index = thread + j * SELECT_THREADS;
            if index < count && labels[index] == attempt {
                energy_weight += vec4(cells[index].energy, cells[index].weight);
                moment += vec4(cells[index].moment, 0.0);
            }
        }
        energy_weight = reduce_sum(thread, energy_weight);
        moment = reduce_sum(thread, moment);
        let weight = energy_weight.w;
        let resultant = length(moment.xyz);
        let center = moment.xyz / max(resultant, 1e-30);
        let mean_cos = min(resultant / max(weight, 1e-30), 1.0);

        // Tangent-plane second moments of the cells' centroids: equal for a disk.
        let up = select(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), abs(center.y) < 0.9);
        let basis_x = normalize(cross(center, up));
        let basis_y = cross(center, basis_x);
        var covariance = vec4(0.0);
        for (var j = 0u; j < CELLS_PER_THREAD; j++) {
            let index = thread + j * SELECT_THREADS;
            if index < count && labels[index] == attempt && cells[index].weight > 0.0 {
                let w = cells[index].weight;
                let d = normalize(cells[index].moment);
                let offset = vec2(dot(d, basis_x), dot(d, basis_y));
                covariance += w * vec4(offset.x * offset.x, offset.y * offset.y, offset.x * offset.y, 0.0);
            }
        }
        covariance = reduce_sum(thread, covariance) / max(weight, 1e-30);
        let trace = covariance.x + covariance.y;
        let determinant = covariance.x * covariance.y - covariance.z * covariance.z;
        let root = sqrt(max(trace * trace * 0.25 - determinant, 0.0));
        let major = trace * 0.5 + root;
        let minor = trace * 0.5 - root;
        // A component inside one or two cells has no measurable shape: call it a disk.
        let cell_angle = 2.0 / f32(params.grid);
        let isotropy = select(1.0, minor / major, major > cell_angle * cell_angle);

        if thread == 0u {
            accepted_value = select(0u, 1u, isotropy >= params.min_isotropy
                && weight >= params.min_energy_fraction * total_weight);
        }
        let accepted = workgroupUniformLoad(&accepted_value) != 0u;
        if accepted {
            for (var j = 0u; j < CELLS_PER_THREAD; j++) {
                let index = thread + j * SELECT_THREADS;
                if index < count && labels[index] == attempt {
                    cells[index].kept = 1u;
                }
            }
            if thread == 0u {
                var source: Source;
                // Bevy samples cubemaps with z flipped.
                source.direction = center * vec3(1.0, 1.0, -1.0);
                source.angular_size = 2.0 * acos(clamp(2.0 * mean_cos - 1.0, -1.0, 1.0));
                source.illuminance = energy_weight.xyz * mean_cos;
                source.isotropy = isotropy;
                output.sources[kept_count] = source;
            }
            kept_count += 1u;
            kept_weight += weight;
        }
        workgroupBarrier();
    }

    if thread == 0u {
        output.count = kept_count;
        output.threshold = globals.threshold;
        output.extracted_fraction = kept_weight / max(total_weight, 1e-30);
        globals.total_weight = total_weight;
    }
}

@compute
@workgroup_size(8, 8, 1)
fn write_residual(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = params.base_size;
    if any(id.xy >= vec2(size)) {
        return;
    }
    var L = textureLoad(source_cubemap, id.xy, id.z, 0).rgb;
    let cell = id.xy / cell_span();
    let Y = luminance(max(L, vec3(0.0)));
    let threshold = globals.threshold;
    if cells[cell_index(id.z, cell)].kept != 0u && Y > threshold {
        // L - e = L t / Y.
        L *= threshold / Y;
    }
    textureStore(residual_cubemap, id.xy, id.z, vec4(L, 1.0));
}
