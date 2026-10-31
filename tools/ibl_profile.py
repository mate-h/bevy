#!/usr/bin/env python3
"""Profile a warmed 640² IBL sphere with xctrace, retaining JSON, deleting traces.

Build first: cargo build --example reflection_probes --features bevy_solari --offline
Run: python tools/ibl_profile.py --output /tmp/ibl-gpu.json
The opaque pass includes the sphere and skybox. Durations are GPU intervals,
not CPU encoding time, frame time, or latency from submission to execution.
"""
import argparse
from collections import Counter, defaultdict
import json
import os
import re
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET


def summarize(values):
    values = sorted(values)
    return {
        "observations": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": values[int((len(values) - 1) * 0.95)],
    }


def extract_gpu_timings(xml_path, pid):
    root = ET.parse(xml_path).getroot()
    ids = {node.attrib["id"]: node for node in root.iter() if "id" in node.attrib}
    schema = root.find(".//schema")
    columns = [col.findtext("mnemonic") for col in schema.findall("col")]
    encoders = defaultdict(list)
    for row in root.findall(".//row"):
        nodes = [ids[node.attrib["ref"]] if "ref" in node.attrib else node for node in row]
        c = dict(zip(columns, nodes))
        if c["process"].attrib.get("fmt") != f"reflection_probes ({pid})":
            continue
        if not c["event-label"].attrib.get("fmt", "").startswith("main_opaque_pass_3d:"):
            continue
        if c["event-depth"].text != "0" or c["state"].text != "Active":
            continue
        start = int(c["start"].text)
        end = start + int(c["duration"].text)
        encoders[c["encoder-id"].text].append((start, end, c["channel-name"].text))
    if not encoders:
        processes = Counter()
        labels = Counter()
        for row in root.findall(".//row"):
            c = dict(zip(columns, [ids[n.attrib["ref"]] if "ref" in n.attrib else n for n in row]))
            processes[c["process"].attrib.get("fmt", "")] += 1
            if "reflection_probes" in c["process"].attrib.get("fmt", ""):
                labels[c["event-label"].attrib.get("fmt", "").split("     ")[0]] += 1
        raise RuntimeError(f"No target opaque-pass GPU intervals for PID {pid}; processes={processes.most_common(8)}; target labels={labels.most_common(8)}")
    first = min(start for rows in encoders.values() for start, _, _ in rows)
    last = max(end for rows in encoders.values() for _, end, _ in rows)
    # Trim trace startup/shutdown and reject incomplete pass executions.
    margin = (last - first) * 0.1
    observations = []
    for rows in encoders.values():
        start = min(row[0] for row in rows)
        end = max(row[1] for row in rows)
        channels = {row[2] for row in rows}
        if start < first + margin or end > last - margin or not {"Vertex", "Fragment"} <= channels:
            continue
        # Sum the union of active intervals, rather than double-counting overlaps.
        def active_ms(channel):
            intervals = sorted((a, b) for a, b, c in rows if c == channel)
            total = 0
            previous_end = -1
            for a, b in intervals:
                total += max(0, b - max(a, previous_end))
                previous_end = max(previous_end, b)
            return total / 1e6
        observations.append({"envelope_ms": (end - start) / 1e6,
                             "fragment_active_ms": active_ms("Fragment"),
                             "vertex_active_ms": active_ms("Vertex")})
    if len(observations) < 30:
        raise RuntimeError(f"Only {len(observations)} complete GPU observations")
    return {key: summarize([row[key] for row in observations]) for key in observations[0]}


def record(repo, variant, samples, seconds):
    # TemporaryDirectory removes the trace, XML export, and logs even on failure.
    with tempfile.TemporaryDirectory(prefix="bevy-ibl-gpu-") as temporary:
        directory = Path(temporary)
        environment = {key: os.environ[key] for key in ("PATH", "HOME", "TMPDIR", "LANG") if key in os.environ}
        environment.update(RUST_LOG="info", BEVY_ASSET_ROOT=str(repo), IBL_PROFILE="1",
                           IBL_FILTER=variant, IBL_FIS_SAMPLES=str(samples),
                           IBL_STRESS="stress_narrow", IBL_ROUGHNESS="0.5", IBL_VIEW="back")
        with (directory / "app.log").open("w+") as log:
            app = subprocess.Popen([str(repo / "target/debug/examples/reflection_probes")],
                                   cwd=repo, env=environment, stdout=log, stderr=subprocess.STDOUT)
            try:
                time.sleep(12)
                if app.poll() is not None:
                    raise RuntimeError("Profiling application exited before capture")
                log.flush()
                text = (directory / "app.log").read_text()
                required = "Prepared FIS radiance pyramid" if variant == "fis" else "Prepared sharp"
                if required not in text or (variant == "fis" and f"Enabled shade-time FIS ({samples} samples)" not in text):
                    raise RuntimeError("Environment or shader not ready for capture")
                trace = directory / "gpu.trace"
                subprocess.run(["xcrun", "xctrace", "record", "--template", "Metal System Trace",
                                "--attach", str(app.pid), "--time-limit", f"{seconds}s", "--output", str(trace),
                                "--no-prompt"], check=True, capture_output=True, timeout=180)
                xml = directory / "gpu.xml"
                subprocess.run(["xcrun", "xctrace", "export", "--input", str(trace), "--xpath",
                                '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]',
                                "--output", str(xml)], check=True, capture_output=True, timeout=180)
                timing = extract_gpu_timings(xml, app.pid)
                adapter = re.search(r'AdapterInfo \{ name: "([^"]+)"', text)
                timing["adapter"] = adapter.group(1) if adapter else "unknown"
                return timing
            finally:
                app.terminate()
                try:
                    app.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    app.kill()
                    app.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="Skip completed repeat/variant pairs in the output")
    args = parser.parse_args()
    if args.seconds < 2 or args.repeats < 1:
        parser.error("Use at least 2 seconds and 1 repeat")
    repo = Path(__file__).resolve().parents[1]
    result = {"backend": "Metal", "resolution": [640, 640], "case": "narrow_back_r50",
              "method": "xctrace Metal System Trace; warmed at least 8 seconds; opaque pass including skybox; central 80% of complete encoder executions",
              "trace_seconds": args.seconds, "runs": []}
    if args.resume and args.output.exists():
        result = json.loads(args.output.read_text())
        if result.get("trace_seconds") != args.seconds or result.get("case") != "narrow_back_r50" or result.get("resolution") != [640, 640]:
            parser.error("Existing output uses a different capture configuration")
    for repeat in range(args.repeats):
        variants = [("karis", 0), ("fis", 64), ("fis", 256)]
        if repeat % 2:
            variants.reverse()
        for variant, samples in variants:
            if any(r["filter"] == variant and r["samples"] == samples and r["repeat"] == repeat + 1 for r in result["runs"]):
                continue
            print(f"Recording {variant} {samples}, repeat {repeat + 1} (trace deleted after extraction)", flush=True)
            timing = record(repo, variant, samples, args.seconds)
            result["runs"].append({"filter": variant, "samples": samples, "repeat": repeat + 1, "warmup_seconds": 12, **timing})
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result["runs"][-1]), flush=True)


if __name__ == "__main__":
    main()
