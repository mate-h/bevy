//! Pure metric-v2 scorer. RGB inputs are linear exposed radiance. Hashes use
//! FNV-1a over little-endian f64/reference bits and mask memberships; they are
//! reproducibility identifiers, not cryptographic provenance hashes.
pub const MASK_NAMES: [&str; 6] = ["geometry", "core", "skirt", "lit", "dark", "near_black"];

#[derive(Clone, Debug, PartialEq)]
pub struct Metric {
    pub value: Option<f64>,
    pub reason: Option<&'static str>,
    pub floor_bound: bool,
}
impl Metric {
    fn some(value: f64, floor_bound: bool) -> Self {
        if !value.is_finite() {
            return Self::none("numerical overflow or underflow");
        }
        Self {
            value: Some(value),
            reason: None,
            floor_bound,
        }
    }
    fn none(reason: &'static str) -> Self {
        Self {
            value: None,
            reason: Some(reason),
            floor_bound: false,
        }
    }
}
#[derive(Clone, Debug)]
pub struct Region {
    pub population: usize,
    pub elements: usize,
    pub mask_hash: u64,
    pub reference_mean: Metric,
    pub relative_denominator: Metric,
    pub rmse: Metric,
    pub relative_rmse: Metric,
    pub peak_normalized_rmse: Metric,
    pub signed_bias: Metric,
    pub absolute_bias: Metric,
    pub positive_excess: Metric,
    pub p99_absolute_rgb_error: Metric,
    pub maximum_error: Metric,
    pub leakage_mean: Metric,
    pub leakage_p99: Metric,
}
#[derive(Clone, Debug)]
pub struct Report {
    pub version: u32,
    pub width: usize,
    pub height: usize,
    pub peak: f64,
    pub epsilon: f64,
    pub reference_hash: u64,
    pub unpooled: [Region; 6],
    pub pooled_4x4: [Region; 6],
    /// All-G positive excess / all-G reference energy, independent of dark mask.
    pub integrated_positive_excess: Metric,
    pub integrated_energy_error: Metric,
}
#[derive(Clone)]
struct Element {
    candidate: [f64; 3],
    reference: [f64; 3],
    population: usize,
    index: usize,
}
fn luminance(rgb: [f64; 3]) -> f64 {
    rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722
}
fn hash_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    for b in bytes {
        hash = (hash ^ u64::from(*b)).wrapping_mul(1099511628211);
    }
    hash
}
const HASH: u64 = 14695981039346656037;
fn member(mask: usize, y: f64, peak: f64) -> bool {
    match mask {
        0 => true,
        1 => y > 0.2 * peak,
        2 => y > 0.02 * peak && y <= 0.2 * peak,
        3 => y > 0.02 * peak,
        4 => y <= 0.02 * peak,
        5 => y <= 0.001 * peak,
        _ => unreachable!(),
    }
}
/// Nearest-rank percentile with integer population weights (no interpolation).
fn percentile(mut values: Vec<(f64, usize)>, population: usize) -> f64 {
    values.sort_by(|a, b| a.0.total_cmp(&b.0));
    let rank = population - population / 100;
    let mut count = 0;
    for (v, weight) in values {
        count += weight;
        if count >= rank {
            return v;
        }
    }
    unreachable!()
}
fn region(elements: &[Element], mask: usize, peak: f64) -> Region {
    let selected: Vec<_> = elements
        .iter()
        .filter(|e| member(mask, luminance(e.reference), peak))
        .collect();
    let population = selected.iter().map(|e| e.population).sum::<usize>();
    let mut mask_hash = HASH;
    for e in &selected {
        mask_hash = hash_bytes(mask_hash, &(e.index as u64).to_le_bytes());
        mask_hash = hash_bytes(mask_hash, &(e.population as u64).to_le_bytes());
    }
    let empty = || Metric::none("empty mask");
    let mut out = Region {
        population,
        elements: selected.len(),
        mask_hash,
        reference_mean: empty(),
        relative_denominator: empty(),
        rmse: empty(),
        relative_rmse: empty(),
        peak_normalized_rmse: empty(),
        signed_bias: empty(),
        absolute_bias: empty(),
        positive_excess: empty(),
        p99_absolute_rgb_error: empty(),
        maximum_error: empty(),
        leakage_mean: empty(),
        leakage_p99: empty(),
    };
    if population == 0 {
        return out;
    }
    let mut energy = 0.0;
    let mut signed = 0.0;
    let mut positive = 0.0;
    let mut square = 0.0;
    let mut errors = Vec::new();
    let mut leakage = Vec::new();
    let mut maximum: f64 = 0.0;
    for e in selected {
        let weight = e.population as f64;
        let delta_y = luminance(e.candidate) - luminance(e.reference);
        energy += weight * luminance(e.reference);
        signed += weight * delta_y;
        positive += weight * delta_y.max(0.0);
        leakage.push((delta_y.max(0.0), e.population));
        for channel in 0..3 {
            let error = (e.candidate[channel] - e.reference[channel]).abs();
            square += weight * error * error;
            maximum = maximum.max(error);
            errors.push((error, e.population));
        }
    }
    let mean = energy / population as f64;
    let rmse = (square / (3 * population) as f64).sqrt();
    out.reference_mean = Metric::some(mean, false);
    out.rmse = Metric::some(rmse, false);
    out.maximum_error = Metric::some(maximum, false);
    out.p99_absolute_rgb_error = Metric::some(percentile(errors, 3 * population), false);
    if peak == 0.0 {
        let undefined = || Metric::none("all-black reference");
        out.relative_denominator = undefined();
        out.relative_rmse = undefined();
        out.peak_normalized_rmse = undefined();
        out.signed_bias = undefined();
        out.absolute_bias = undefined();
        out.positive_excess = undefined();
        out.leakage_mean = undefined();
        out.leakage_p99 = undefined();
    } else {
        let epsilon = 1e-6 * peak;
        let floor = mean <= epsilon;
        let denominator = mean.max(epsilon);
        out.relative_denominator = Metric::some(denominator, floor);
        out.relative_rmse = Metric::some(rmse / denominator, floor);
        out.peak_normalized_rmse = Metric::some(rmse / peak, false);
        let bias = signed / (population as f64 * denominator);
        out.signed_bias = Metric::some(bias, floor);
        out.absolute_bias = Metric::some(bias.abs(), floor);
        out.positive_excess = Metric::some(positive / (population as f64 * denominator), floor);
        out.leakage_mean = Metric::some(positive / (population as f64 * peak), false);
        out.leakage_p99 = Metric::some(percentile(leakage, population) / peak, false);
    }
    out
}
/// Inputs must share width/height and contain widened finite, nonnegative f32
/// linear RGB (at most f32::MAX). f64 arithmetic protects capture-scale sums.
/// Geometry membership is supplied independently of either image. Tiles start at
/// (0,0), average only G pixels, and retain their original pixel populations.
pub fn score(
    candidate: &[[f64; 3]],
    fixed_reference: &[[f64; 3]],
    geometry_mask: &[bool],
    width: usize,
    height: usize,
) -> Result<Report, &'static str> {
    let size = width
        .checked_mul(height)
        .ok_or("image dimensions overflow")?;
    if size == 0
        || candidate.len() != size
        || fixed_reference.len() != size
        || geometry_mask.len() != size
    {
        return Err("invalid image dimensions or input lengths");
    }
    if candidate
        .iter()
        .chain(fixed_reference)
        .flatten()
        .any(|v| !v.is_finite() || *v < 0.0 || *v > f64::from(f32::MAX))
    {
        return Err("RGB must be finite, nonnegative, and at most f32::MAX");
    }
    let mut reference_hash = hash_bytes(
        hash_bytes(HASH, &(width as u64).to_le_bytes()),
        &(height as u64).to_le_bytes(),
    );
    let mut elements = Vec::new();
    for i in 0..size {
        reference_hash = hash_bytes(reference_hash, &[u8::from(geometry_mask[i])]);
        for v in fixed_reference[i] {
            reference_hash = hash_bytes(reference_hash, &v.to_le_bytes());
        }
        if geometry_mask[i] {
            elements.push(Element {
                candidate: candidate[i],
                reference: fixed_reference[i],
                population: 1,
                index: i,
            });
        }
    }
    let peak = elements
        .iter()
        .map(|e| luminance(e.reference))
        .fold(0.0, f64::max);
    let mut pooled = Vec::new();
    for ty in (0..height).step_by(4) {
        for tx in (0..width).step_by(4) {
            let mut e = Element {
                candidate: [0.0; 3],
                reference: [0.0; 3],
                population: 0,
                index: ty * width + tx,
            };
            for y in ty..(ty + 4).min(height) {
                for x in tx..(tx + 4).min(width) {
                    let i = y * width + x;
                    if geometry_mask[i] {
                        e.population += 1;
                        for c in 0..3 {
                            e.candidate[c] += candidate[i][c];
                            e.reference[c] += fixed_reference[i][c];
                        }
                    }
                }
            }
            if e.population > 0 {
                for c in 0..3 {
                    e.candidate[c] /= e.population as f64;
                    e.reference[c] /= e.population as f64;
                }
                pooled.push(e);
            }
        }
    }
    let unpooled = std::array::from_fn(|mask| region(&elements, mask, peak));
    let integrated_positive_excess = integrated_excess(&elements);
    let energy: f64 = elements.iter().map(|e| luminance(e.reference)).sum();
    let signed: f64 = elements
        .iter()
        .map(|e| luminance(e.candidate) - luminance(e.reference))
        .sum();
    let integrated_energy_error = if energy > 0.0 {
        Metric::some(signed / energy, false)
    } else {
        Metric::none(if elements.is_empty() {
            "empty mask"
        } else {
            "all-black reference"
        })
    };
    Ok(Report {
        version: 2,
        width,
        height,
        peak,
        epsilon: 1e-6 * peak,
        reference_hash,
        unpooled,
        pooled_4x4: std::array::from_fn(|mask| region(&pooled, mask, peak)),
        integrated_positive_excess,
        integrated_energy_error,
    })
}

fn integrated_excess(elements: &[Element]) -> Metric {
    let reference_energy: f64 = elements
        .iter()
        .map(|e| luminance(e.reference) * e.population as f64)
        .sum();
    let positive_energy: f64 = elements
        .iter()
        .map(|e| (luminance(e.candidate) - luminance(e.reference)).max(0.0) * e.population as f64)
        .sum();
    if elements.is_empty() {
        Metric::none("empty mask")
    } else if reference_energy == 0.0 {
        Metric::none("all-black reference")
    } else {
        Metric::some(positive_energy / reference_energy, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }
    #[test]
    fn equality_and_signed_bias() {
        let r = [[1.0; 3], [0.1; 3], [0.0; 3]];
        for (scale, bias) in [(1.0, 0.0), (2.0, 1.0), (0.0, -1.0)] {
            let c = r.map(|rgb| rgb.map(|v| v * scale));
            let report = score(&c, &r, &[true; 3], 3, 1).unwrap();
            close(report.unpooled[0].signed_bias.value.unwrap(), bias);
            if scale == 1.0 {
                close(report.unpooled[0].rmse.value.unwrap(), 0.0);
            }
        }
    }
    #[test]
    fn halo_cannot_change_masks_or_normalizers() {
        let r = [[1.0; 3], [0.1; 3], [0.0; 3]];
        let mut c = r;
        c[2] = [10.0; 3];
        let a = score(&r, &r, &[true; 3], 3, 1).unwrap();
        let b = score(&c, &r, &[true; 3], 3, 1).unwrap();
        assert_eq!(a.reference_hash, b.reference_hash);
        assert_eq!(a.peak, b.peak);
        for (aa, bb) in a
            .unpooled
            .iter()
            .chain(&a.pooled_4x4)
            .zip(b.unpooled.iter().chain(&b.pooled_4x4))
        {
            assert_eq!(aa.population, bb.population);
            assert_eq!(aa.mask_hash, bb.mask_hash);
            assert_eq!(aa.relative_denominator, bb.relative_denominator);
        }
        assert!(b.unpooled[4].leakage_mean.value.unwrap() > 0.0);
        assert!(b.unpooled[4].relative_rmse.floor_bound);
    }
    #[test]
    fn black_and_empty_masks() {
        let a = score(&[[1.0; 3]], &[[0.0; 3]], &[true], 1, 1).unwrap();
        close(a.unpooled[4].rmse.value.unwrap(), 1.0);
        assert_eq!(
            a.unpooled[0].relative_rmse.reason,
            Some("all-black reference")
        );
        assert_eq!(a.unpooled[1].rmse.reason, Some("empty mask"));
        let empty = score(&[[0.0; 3]], &[[0.0; 3]], &[false], 1, 1).unwrap();
        assert_eq!(empty.unpooled[0].rmse.value, None);
    }
    #[test]
    fn origin_tiles_and_partial_population_weighting() {
        let r = [[1.0; 3]; 5];
        let mut c = r;
        c[4] = [3.0; 3];
        let a = score(&c, &r, &[true; 5], 5, 1).unwrap();
        assert_eq!(a.pooled_4x4[0].elements, 2);
        assert_eq!(a.pooled_4x4[0].population, 5);
        close(a.pooled_4x4[0].rmse.value.unwrap(), (4.0_f64 / 5.0).sqrt());
        close(a.pooled_4x4[0].signed_bias.value.unwrap(), 0.4);
    }
    #[test]
    fn thresholds_use_fixed_peak_even_after_pooling() {
        // Achromatic luminance is exactly the scalar within floating precision.
        let r = [[1.0; 3], [0.2; 3], [0.02; 3], [0.001; 3], [0.1; 3]];
        let a = score(&r, &r, &[true; 5], 5, 1).unwrap();
        assert!(!member(1, 0.2, 1.0));
        assert!(member(2, 0.2, 1.0));
        assert!(!member(2, 0.02, 1.0));
        assert!(member(4, 0.02, 1.0));
        assert!(member(5, 0.001, 1.0));
        // Last one-pixel tile is a skirt using pixel peak 1, not pooled peak.
        assert_eq!(a.pooled_4x4[2].population, 1);
        let outside = score(
            &[[100.0; 3], [1.0; 3]],
            &[[100.0; 3], [1.0; 3]],
            &[false, true],
            2,
            1,
        )
        .unwrap();
        close(outside.peak, 1.0);
    }
    #[test]
    fn percentile_weights_and_geometry_identity() {
        let r = [[1.0; 3]; 100];
        let mut c = r;
        c[99] = [101.0; 3];
        let a = score(&c, &r, &[true; 100], 100, 1).unwrap();
        close(a.unpooled[0].p99_absolute_rgb_error.value.unwrap(), 0.0);
        let b = score(&c, &r, &[false; 100], 100, 1).unwrap();
        assert_ne!(a.reference_hash, b.reference_hash);
    }
    #[test]
    fn integrated_excess_uses_energy_without_floor() {
        let elements = [
            Element {
                candidate: [2.0; 3],
                reference: [1.0; 3],
                population: 1,
                index: 0,
            },
            Element {
                candidate: [0.0; 3],
                reference: [0.0; 3],
                population: 2_000_000,
                index: 1,
            },
        ];
        let region = region(&elements, 0, 1.0);
        assert!(region.positive_excess.floor_bound);
        assert!(region.positive_excess.value.unwrap() < 0.5);
        close(integrated_excess(&elements).value.unwrap(), 1.0);
    }
    #[test]
    fn bad_inputs() {
        assert!(score(&[[f64::MAX; 3]], &[[1.0; 3]], &[true], 1, 1).is_err());
        assert!(score(&[], &[], &[], 0, 0).is_err());
        assert!(score(&[], &[], &[], usize::MAX, 2).is_err());
        assert!(score(&[[f64::NAN; 3]], &[[0.0; 3]], &[true], 1, 1).is_err());
        assert!(score(&[[0.0; 3]], &[[0.0; 3]], &[], 1, 1).is_err());
    }
}

impl Metric {
    pub fn display(&self) -> String {
        self.value
            .map_or_else(|| "null".into(), |v| format!("{v:.6}"))
    }
    fn json(&self) -> String {
        let reason = self
            .reason
            .map_or_else(|| "null".into(), |r| format!("\"{r}\""));
        format!(
            "{{\"value\":{},\"reason\":{reason},\"floor_bound\":{}}}",
            self.value.map_or_else(|| "null".into(), |v| v.to_string()),
            self.floor_bound
        )
    }
}
impl Region {
    fn json(&self) -> String {
        let metrics = [
            ("reference_mean", &self.reference_mean),
            ("relative_denominator", &self.relative_denominator),
            ("rmse", &self.rmse),
            ("relative_rmse", &self.relative_rmse),
            ("peak_normalized_rmse", &self.peak_normalized_rmse),
            ("signed_bias", &self.signed_bias),
            ("absolute_bias", &self.absolute_bias),
            ("positive_excess", &self.positive_excess),
            ("p99_absolute_rgb_error", &self.p99_absolute_rgb_error),
            ("maximum_error", &self.maximum_error),
            ("leakage_mean", &self.leakage_mean),
            ("leakage_p99", &self.leakage_p99),
        ];
        let fields = metrics
            .iter()
            .map(|(name, m)| format!("\"{name}\":{}", m.json()))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"population\":{},\"elements\":{},\"mask_hash\":\"{:016x}\",{fields}}}",
            self.population, self.elements, self.mask_hash
        )
    }
}
impl Report {
    /// Versioned metric artifact only: capture synchronization and model/source
    /// provenance must be supplied by a future qualified-reference workflow.
    pub fn json(&self) -> String {
        self.json_with_geometry_role(false)
    }
    pub fn json_with_geometry_role(&self, eroded: bool) -> String {
        let geometry_role = if eroded {
            "eroded_1px_8_neighbors"
        } else {
            "complete_geometry"
        };
        let regions = |values: &[Region; 6]| {
            MASK_NAMES
                .iter()
                .zip(values)
                .map(|(name, r)| format!("\"{name}\":{}", r.json()))
                .collect::<Vec<_>>()
                .join(",")
        };
        format!(concat!("{{\"metric_version\":{},\"width\":{},\"height\":{},\"geometry_role\":\"{}\",\"bias_convention\":\"candidate_minus_fixed_reference\",",
            "\"reference_qualified\":false,\"actual_spp\":null,\"rng_seed\":null,",
            "\"source_model_dfg_hashes\":null,\"peak\":{},\"epsilon\":{},",
            "\"rmse_definition\":\"sqrt(sum_population_weighted_RGB_squared_error/(3*population))\",",
            "\"relative_rmse_definition\":\"rmse/max(reference_mean,epsilon)\",",
            "\"signed_bias_definition\":\"sum(candidate_Y-reference_Y)/max(sum(reference_Y),population*epsilon)\",",
            "\"integrated_positive_excess_definition\":\"sum(max(candidate_Y-reference_Y,0))/sum(reference_Y)\",",
            "\"core_threshold_fraction\":0.2,\"lit_threshold_fraction\":0.02,\"near_black_threshold_fraction\":0.001,",
            "\"pool_size\":4,\"pool_origin\":[0,0],\"pool_weighting\":\"geometry_pixel_population\",",
            "\"percentile\":\"nearest_rank_population_weighted_RGB_components\",",
            "\"hash_algorithm\":\"fnv1a64_le\",\"reference_hash\":\"{:016x}\",",
            "\"unpooled\":{{{}}},\"pooled_4x4\":{{{}}},\"integrated_positive_excess\":{},\"integrated_energy_error\":{}}}"),
            self.version, self.width, self.height, geometry_role, self.peak, self.epsilon, self.reference_hash, regions(&self.unpooled), regions(&self.pooled_4x4), self.integrated_positive_excess.json(), self.integrated_energy_error.json())
    }
}
