#!/usr/bin/env bash
# Run from any directory after building reflection_probes with bevy_solari.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
filter=${IBL_FILTER:-fis}
# Metric v2 reverses the legacy bias sign and uses fixed-reference masks. Keep
# new logs separate from legacy captures/results, even with an explicit directory.
output=${1:-/tmp/ibl-stress-v2}
mkdir -p "$output"
cd "$repo"
for case in furnace_back_50 wide_front_20 wide_front_50 wide_back_50 narrow_front_20 narrow_front_50 narrow_back_50; do
    IFS=_ read -r source view roughness <<< "$case"
    log="$output/${filter}_${source}_${view}_r${roughness}_metrics_v2.log"
    RUST_LOG=info BEVY_ASSET_ROOT="$repo" IBL_FILTER="$filter" IBL_MEASURE=1 \
        IBL_STRESS="stress_$source" IBL_VIEW="$view" IBL_ROUGHNESS="0.$roughness" \
        target/debug/examples/reflection_probes > "$log" 2>&1
    if [[ "$filter" == fis ]]; then
        rg -q 'Enabled shade-time FIS' "$log"
    fi
    # Reject legacy reports before displaying metrics. Undefined v2 statistics
    # are literal null; do not coerce them to zero or require numeric values.
    rg -Fq 'Metric v2 (candidate minus fixed reference; positive bias means excess)' "$log"
    rg -q 'lit relRMSE' "$log"
    python3 - "$log" "${log%.log}.json" <<'PYTHON'
import json
from pathlib import Path
import sys

log, output = map(Path, sys.argv[1:])
prefix = "IBL_METRIC_V2 "
records = [line.split(prefix, 1)[1] for line in log.read_text().splitlines() if prefix in line]
if len(records) != 1:
    raise SystemExit(f"{log}: expected exactly one metric-v2 JSON report")

def reject_nonfinite(value):
    raise ValueError(f"non-finite JSON metric: {value}")

report = json.loads(records[0], parse_constant=reject_nonfinite)
if report.get("metric_version") != 2 or report.get("bias_convention") != "candidate_minus_fixed_reference":
    raise SystemExit(f"{log}: unsupported metric schema or bias convention")
output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
PYTHON
    rg 'IBL stress|Metric v2|IBL_METRIC_V2|lit relRMSE|core bias' "$log"
done
