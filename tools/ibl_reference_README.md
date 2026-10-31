# Reference capture and convergence diagnostics

This stage does not launch the GPU by default and never labels captures qualified.
Old stress results and logs are untouched. Run from the repository root after
building the `reflection_probes` example with Solari enabled.

Print the default 7-case × 3-budget × 4-seed plan without writing anything:

```sh
python3 tools/ibl_reference.py capture --output /tmp/ibl-reference-eval --dry-run
```

Add `--execute` (remove `--dry-run`) to run it. `--cases`, `--spp`, `--seeds`,
`--binary`, `--filter`, and `--timeout` customize the plan. Timeout defaults to
3600 seconds per process; failed/timed-out runs retain their files. Budgets must
be positive integers no greater than 16777216 (exact float accumulation range).
The runner removes inherited `IBL_PROFILE`, sets `BEVY_ASSET_ROOT`, and resolves
relative binary paths from the repository so it works from any directory. Existing run directories cause a
preflight failure; failed runs retain invocation metadata and process logs and
must be collected under a new output root. Each run must export exactly one
manifest under its `IBL_CAPTURE_DIR`. Successful runs validate actual spp/seed.

The four default seeds are only the evaluation group. A separate capture plan is
required for a fixed-reference group, for example `--seeds 100 101 102 103 --spp
8192` in a separate root. More samples do not by themselves prove convergence.
Do not include seeds 100–103 in the evaluation group or reuse them as independent
replicates. Four seeds is a minimum diagnostic sample size, not a power guarantee.

Analyze **one case at a time**, passing directories that directly contain their
manifest (not invocation parent directories):

```sh
python3 tools/ibl_reference.py analyze /path/to/capture1 /path/to/capture2 \
  --reference-seeds 100 101 102 103 --allowed-error 0.05 \
  --claimed-improvement 0.03 > /tmp/reference-diagnostics.json
```

The analyzer requires schema version 1, `rgb32float-le`, tightly packed RGB f32
little-endian reference radiance and binary u8 geometry. It validates byte counts,
finite RGB, dimensions, geometry, seed/budget, and matching scene/model/build
identity. Unknown hashes/material remain unresolved; equality of unknown values
is not provenance validation. Capture-file hashes identify the fixed reference
inputs. No candidate image enters the reference construction.

High-budget designated reference seeds are averaged before pooling. The fixed
pixel reference peak sets thresholds; geometry pixels are averaged in 4×4 tiles
anchored at image origin and tile populations weight errors. Lit/core/skirt/dark
membership uses that fixed pooled reference. Black reference relative metrics
are null. Reports retain absolute RGB RMSE and floor flags. This bounded analyzer
is a reference-repeatability diagnostic, not the complete metric-v2 candidate
scorer (no p99, erosion, halo or scene-image gate).

Student-t two-sided 95% half-widths are computed across independent evaluation
**run metrics**, with at least four runs. They are conditional on the fixed
reference and do not account for its uncertainty. Successive budgets use paired
seed differences with the same immutable masks, and do not assume budgets are
independent. The optional dimensionless decision limit is the smaller of 20% of
allowed metric error and one third of claimed improvement, applied uniformly to
relative RMSE and bias as an explicitly simplified diagnostic. For different
metric tolerances, analyze/report those separately rather than interpreting this
single threshold as the full protocol. Normal/t assumptions and multiple
comparisons remain review considerations; a small half-width cannot establish
absence of systematic error or matching BRDF models.

`reference_qualified` always remains false. Independent fixed-reference pool
uncertainty, complete decision metrics, known provenance and model agreement,
reference budget stability and the intended case suite still require protocol
review. Missing reference groups, insufficient seeds and unspecified thresholds
are explicitly unresolved. These files provide no architecture-ranking approval.

Tests (synthetic captures and mocked processes only):

```sh
python3 -B -m unittest discover -s tools -p 'test_ibl_reference.py'
```
