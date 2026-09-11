#!/usr/bin/env python3
"""Serial, throughput-only local runs; stdlib orchestration, not Python inference.

Build `cargo build --release --locked --example qualify` before running. The Rust
harness enforces the development-host restriction. No builds/download preparation
should run concurrently. This is not the complete weighted model-selection eval.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def snapshot():
    return {
        "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "load_average": list(os.getloadavg()),
        "power_profile": Path("/sys/firmware/acpi/platform_profile").read_text().strip(),
        "ac_online": Path("/sys/class/power_supply/ACAD/online").read_text().strip(),
        "cpu_governor": Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip(),
        "gpu": command("nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.free,utilization.gpu,temperature.gpu", "--format=csv,noheader"),
        "gpu_compute_processes": command("nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"),
        "processes": command("ps", "-eo", "pid,comm,pcpu", "--sort=-pcpu"),
    }


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new results directory")
    parser.add_argument("--power-mode", required=True, help="record/check; does not change host policy")
    parser.add_argument("--cpu-manifest", type=Path)
    parser.add_argument("--cuda-manifest", type=Path)
    parser.add_argument("--cooldown-seconds", type=int, default=15)
    args = parser.parse_args()
    if "CI" in os.environ:
        parser.error("local measurement only, never CI")
    if args.cooldown_seconds < 0:
        parser.error("cooldown must be nonnegative")
    executable = ROOT / "target/release/examples/qualify"
    if not executable.is_file():
        parser.error("build the release qualify example first")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    source_paths = sorted({ROOT / "Cargo.toml", ROOT / "Cargo.lock", ROOT / "build.rs", Path(__file__).resolve(),
                           *ROOT.glob("src/**/*.rs"), *ROOT.glob("examples/**/*.rs")})
    metadata = {
        "scope": "serial local throughput baseline, synthetic token IDs, no retrieval quality scores",
        "git_head": command("git", "rev-parse", "HEAD"),
        "git_status": command("git", "status", "--short"),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
        "executable_sha256": sha256(executable),
        "rustc": command("rustc", "--version"),
        "uname": command("uname", "-a"),
        "cooldown_seconds": args.cooldown_seconds,
        "process_repeats": 3,
        "inference_repeats": 5,
        "batches": [1, 8],
        "token_lengths": [64, 256, 512],
        "cache_policy": "no cache flush; artifact hashing precedes session loads",
        "sampling": "10-ms harness sampler; peaks are lower bounds",
        "environment": {k: v for k, v in os.environ.items() if k in
                        ["CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "ORT_NUM_THREADS", "REE_DEVICE"]},
    }
    write_json(args.output / "invocation.json", metadata)
    cells = {}
    runs = []
    expected_cells = {(length, batch) for length in [64, 256, 512] for batch in [1, 8]}
    recipes = {}
    for device, manifest in [("cpu", args.cpu_manifest), ("cuda:0", args.cuda_manifest)]:
        for repeat in range(1, 4):
            time.sleep(args.cooldown_seconds)
            before = snapshot()
            if before["power_profile"] != args.power_mode or before["ac_online"] != "1":
                raise RuntimeError("power profile changed or machine is not on AC")
            if before["gpu_compute_processes"]:
                raise RuntimeError("competing GPU compute process; stop instead of collecting contaminated runs")
            name = f"{device.replace(':', '-')}-{repeat}"
            invocation = [str(executable), "--device", device, "--power-mode", args.power_mode,
                          "--batches", "1,8", "--repeats", "5", "--load-repeats", "3"]
            if manifest:
                invocation += ["--manifest", str(manifest.resolve())]
            write_json(args.output / f"{name}-before.json", {"command": invocation, "host": before})
            print(f"Starting {name}", flush=True)
            start = time.monotonic()
            with (args.output / f"{name}.jsonl").open("x") as out, (args.output / f"{name}.stderr").open("x") as err:
                result = subprocess.run(invocation, cwd=ROOT, stdout=out, stderr=err, timeout=1800)
            after = snapshot()
            write_json(args.output / f"{name}-after.json", {"host": after, "returncode": result.returncode,
                                                          "wall_seconds": time.monotonic() - start})
            result.check_returncode()
            if after["power_profile"] != args.power_mode or after["ac_online"] != "1":
                raise RuntimeError("power profile/AC changed during measurement")
            events = [json.loads(line) for line in (args.output / f"{name}.jsonl").read_text().splitlines()]
            if events[-1]["type"] != "qualification_completed" or any(e["type"] == "probe_failed" for e in events):
                raise RuntimeError("incomplete baseline: retain raw output, do not summarize as success")
            info = next(e for e in events if e["type"] == "qualification_started")
            recipe = {k: info[k] for k in ["model", "revision", "model_sha256", "tokenizer_sha256", "precision",
                                          "recipe", "threads", "power_mode", "runtime"]}
            if recipes.setdefault(device, recipe) != recipe:
                raise RuntimeError("recipe/thread settings changed between process runs")
            probes = [e for e in events if e["type"] == "throughput"]
            if len(probes) != 6 or {(e["tokens"], e["batch"]) for e in probes} != expected_cells:
                raise RuntimeError("missing or duplicate throughput cells")
            for probe in probes:
                cells.setdefault((device, probe["tokens"], probe["batch"]), []).append(probe["embeddings_per_second"])
            runs.append({"file": f"{name}.jsonl", "sha256": sha256(args.output / f"{name}.jsonl")})
            print(f"Completed {name}", flush=True)
    write_json(args.output / "summary.json", {
        "scope": metadata["scope"], "recipes": recipes, "runs": runs,
        "aggregation": "median of three within-process mean embeddings/second values, per cell",
        "cells": [{"device": device, "tokens": tokens, "batch": batch,
                   "process_embeddings_per_second": values,
                   "median_embeddings_per_second": statistics.median(values),
                   "min_embeddings_per_second": min(values), "max_embeddings_per_second": max(values)}
                  for (device, tokens, batch), values in sorted(cells.items())],
    })
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
