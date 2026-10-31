#!/usr/bin/env python3
"""Profile disk lights, source extraction and the generator passes with xctrace.

Build first: cargo build --example reflection_probes --features bevy_solari,area_light_luts,debug --offline
(`debug` keeps render pass labels, which the opaque-pass timing needs.)
Run: python tools/ibl_profile_sources.py --output /tmp/ibl-sources-gpu.json

Each variant runs the 640² sphere (narrow back r0.5) with `IBL_PROFILE=1` and
records GPU intervals of the opaque pass and of the generator's compute passes.
Traces are deleted after extraction. Variants alternate order between repeats.
"""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET


def ktraces():
    """xctrace leaves a raw instruments*.ktrace (0.5-4 GB) in TMPDIR per recording."""
    return set(Path(tempfile.gettempdir()).glob("instruments*.ktrace"))


def delete_new_ktraces(before):
    # A failed or interrupted recording can still be writing its ktrace, so look twice.
    for attempt in range(2):
        for path in ktraces() - before:
            path.unlink(missing_ok=True)
        if attempt == 0:
            time.sleep(3)

SUN = {"IBL_SUN_LUX": "54000", "IBL_SUN_DIR": "0.5,0.6,0.62"}
VARIANTS = {
    "karis": {},
    "sun_point": {**SUN, "IBL_SUN_SIZE_DEG": "0"},
    "sun_disk_0.5_ltc": {**SUN, "IBL_SUN_SIZE_DEG": "0.5"},
    "sun_disk_24_ltc": {**SUN, "IBL_SUN_SIZE_DEG": "24"},
    "sun_disk_24_rep": {**SUN, "IBL_SUN_SIZE_DEG": "24", "IBL_SUN_METHOD": "rep"},
    "stretch": {"IBL_FILTER": "stretch"},
    "ms": {"IBL_FILTER": "ms"},
    "ms_irradiance_sampled": {"IBL_FILTER": "ms", "IBL_IRRADIANCE": "sampled"},
    "ms_extract_lights": {"IBL_FILTER": "ms", "IBL_EXTRACT": "1"},
    "ms_extract_shading": {"IBL_FILTER": "ms", "IBL_EXTRACT": "shading"},
}
PASSES = ["main_opaque_pass_3d", "lightprobe_sources", "lightprobe_copy",
          "lightprobe_quadratic_downsample", "lightprobe_radiance_map", "lightprobe_irradiance_map"]


def summarize(values):
    values = sorted(values)
    return {"observations": len(values), "median_ms": statistics.median(values),
            "p95_ms": values[int((len(values) - 1) * 0.95)]}


def extract(xml_path, pid):
    root = ET.parse(xml_path).getroot()
    ids = {node.attrib["id"]: node for node in root.iter() if "id" in node.attrib}
    columns = [col.findtext("mnemonic") for col in root.find(".//schema").findall("col")]
    encoders = {}
    for row in root.findall(".//row"):
        c = dict(zip(columns, [ids[n.attrib["ref"]] if "ref" in n.attrib else n for n in row]))
        if c["process"].attrib.get("fmt") != f"reflection_probes ({pid})":
            continue
        label = c["event-label"].attrib.get("fmt", "")
        name = next((p for p in PASSES if label.startswith(p)), None)
        if name is None or c["event-depth"].text != "0" or c["state"].text != "Active":
            continue
        start = int(c["start"].text)
        end = start + int(c["duration"].text)
        key = (name, c["encoder-id"].text)
        lo, hi = encoders.get(key, (start, end))
        encoders[key] = (min(lo, start), max(hi, end))
    if not encoders:
        raise RuntimeError("No GPU intervals for the target process")
    first = min(lo for lo, _ in encoders.values())
    last = max(hi for _, hi in encoders.values())
    margin = (last - first) * 0.1
    per_pass = {}
    for (name, _), (lo, hi) in encoders.items():
        if lo >= first + margin and hi <= last - margin:
            per_pass.setdefault(name, []).append((hi - lo) / 1e6)
    return {name: summarize(values) for name, values in per_pass.items() if len(values) >= 10}


def record(repo, name, overrides, seconds):
    before = ktraces()
    try:
        return _record(repo, name, overrides, seconds)
    finally:
        # The trace and its raw ktrace are deleted once their data is extracted.
        delete_new_ktraces(before)


def _record(repo, name, overrides, seconds):
    with tempfile.TemporaryDirectory(prefix="bevy-ibl-sources-gpu-") as temporary:
        directory = Path(temporary)
        env = {key: os.environ[key] for key in ("PATH", "HOME", "TMPDIR", "LANG") if key in os.environ}
        env.update(RUST_LOG="info", BEVY_ASSET_ROOT=str(repo), IBL_PROFILE="1", IBL_FILTER="karis",
                   IBL_STRESS="stress_narrow", IBL_ROUGHNESS="0.5", IBL_VIEW="back")
        env.update(overrides)
        with (directory / "app.log").open("w+") as log:
            app = subprocess.Popen([str(repo / "target/debug/examples/reflection_probes")],
                                   cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                time.sleep(12)
                if app.poll() is not None:
                    raise RuntimeError(f"{name}: application exited before capture")
                trace = directory / "gpu.trace"
                subprocess.run(["xcrun", "xctrace", "record", "--template", "Metal System Trace",
                                "--attach", str(app.pid), "--time-limit", f"{seconds}s", "--output", str(trace),
                                "--no-prompt"], check=True, capture_output=True, timeout=180)
                xml = directory / "gpu.xml"
                subprocess.run(["xcrun", "xctrace", "export", "--input", str(trace), "--xpath",
                                '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]',
                                "--output", str(xml)], check=True, capture_output=True, timeout=180)
                timing = extract(xml, app.pid)
                text = (directory / "app.log").read_text()
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
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    result = {"backend": "Metal", "case": "narrow_back_r50", "resolution": [640, 640],
              "method": "xctrace Metal System Trace; 12 s warmup; per-pass encoder envelopes, central 80%",
              "runs": []}
    for repeat in range(args.repeats):
        names = list(VARIANTS)
        if repeat % 2:
            names.reverse()
        for name in names:
            print(f"Recording {name}, repeat {repeat + 1}", flush=True)
            try:
                timing = record(repo, name, VARIANTS[name], args.seconds)
            except Exception as error:  # Keep going; record the failure.
                timing = {"error": str(error)}
            result["runs"].append({"variant": name, "repeat": repeat + 1, **timing})
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
