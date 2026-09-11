#!/usr/bin/env python3
"""Frozen two-corpus exploratory retrieval/compatibility run. No Python inference.

All encoding and ranking use the Rust examples. Python orchestrates serial jobs
and paired bootstrap analysis over their per-query metrics. No downloads/installs.
"""
import argparse
import array
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("baseline", ROOT / "benchmarks/run-baseline.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
COMBINATIONS = ["cpu/cpu", "cuda/cuda", "cpu/cuda", "cuda/cpu", "cpu/mixed", "cuda/mixed"]
REPLICATES = 10000


def dump_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def status(directory, **value):
    value["time_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, directory / "status.json")


def bootstrap_indices(corpus_hash, count, replicates=REPLICATES):
    if count < 1 or count > 100000 or replicates < 1:
        raise ValueError("invalid bootstrap shape")
    prefix = f"ree-bootstrap-v1\0{20260909}\0{corpus_hash}\0".encode()
    threshold = (1 << 64) - ((1 << 64) % count)
    result = array.array("I")
    counter = 0
    while len(result) < count * replicates:
        digest = hashlib.sha256(prefix + struct.pack("<Q", counter)).digest()
        counter += 1
        for draw in struct.unpack("<4Q", digest):
            if draw < threshold:
                result.append(draw % count)
                if len(result) == count * replicates:
                    break
    return result


def index_bytes(indices):
    result = array.array("I", indices)
    if result.itemsize != 4:
        raise ValueError("platform u32 array size differs")
    if sys.byteorder != "little":
        result.byteswap()
    return result.tobytes()


def read_indices(path, expected_hash, count):
    data = path.read_bytes()
    if len(data) != REPLICATES * count * 4 or hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError("bootstrap index file hash/shape mismatch")
    indices = array.array("I")
    indices.frombytes(data)
    if sys.byteorder != "little":
        indices.byteswap()
    if any(i >= count for i in indices):
        raise ValueError("out-of-range bootstrap index")
    return indices


def percentile(values, p):
    values = sorted(values)
    position = p * (len(values) - 1)
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def interval(values):
    return [percentile(values, 0.025), percentile(values, 0.975)]


def bootstrap_means(values, indices):
    n = len(values)
    if not n or len(indices) % n or any(i >= n for i in indices):
        raise ValueError("invalid paired bootstrap inputs")
    return [sum(values[i] for i in indices[start:start+n]) / n for start in range(0, len(indices), n)]


def overlap(a, b):
    if len(a) != len(b) or not a:
        raise ValueError("ranking query coverage mismatch")
    top1, top10 = [], []
    for left, right in zip(a, b):
        if not left or not right or len(set(left)) != len(left) or len(set(right)) != len(right):
            raise ValueError("invalid top-k lists")
        top1.append(left[0] == right[0])
        top10.append(len(set(left) & set(right)) / min(10, len(left), len(right)))
    return {"top1_agreement": sum(top1)/len(top1), "top10_overlap": sum(top10)/len(top10)}


def verify_export(directory, log, manifest, corpus):
    events = [json.loads(line) for line in log.read_text().splitlines()]
    if not events or events[-1]["type"] != "qualification_completed":
        raise ValueError("partial encoding log; preserve it and retry in a separate run directory")
    metadata_path = directory / "metadata.json"
    event = next(e for e in events if e["type"] == "vector_export_completed")
    if baseline.sha256(metadata_path) != event["metadata_sha256"]:
        raise ValueError("export completion hash mismatch")
    meta = json.loads(metadata_path.read_text())
    info = meta["information"]
    for field in ["name", "revision", "precision", "recipe", "document_prompt", "query_prompt"]:
        if info["model" if field == "name" else field] != manifest[field]:
            raise ValueError(f"export recipe changed: {field}")
    if info["corpus_sha256"] != corpus["sha256"] or info["model_sha256"] != manifest["model"]["sha256"] or info["tokenizer_sha256"] != manifest["tokenizer"]["sha256"]:
        raise ValueError("export artifact/corpus binding mismatch")
    if info["threads"] != 8 or info["power_mode"] != "balanced":
        raise ValueError("export runtime settings differ from plan")
    if meta["queries"]["ids"] != corpus["query_ids"] or meta["documents"]["ids"] != corpus["document_ids"]:
        raise ValueError("export ID coverage/order mismatch")
    for key, filename in [("documents", "documents.f32"), ("queries", "queries.f32"), ("single_documents", "single-documents.f32")]:
        file = directory / filename
        if file.stat().st_size != len(meta[key]["ids"]) * meta["dimensions"] * 4 or baseline.sha256(file) != meta[key]["sha256"]:
            raise ValueError("export matrix hash/shape mismatch")
    return meta


def run_command(command, stdout, stderr, result_dir, job, check_power=True):
    before = baseline.snapshot()
    if before["ac_online"] != "1" or before["power_profile"] != "balanced" or before["gpu_compute_processes"]:
        raise RuntimeError("AC/profile/competing GPU preflight failed")
    dump_new(result_dir / (job + "-before.json"), {"command": command, "host": before})
    start = time.monotonic()
    with stdout.open("x") as out, stderr.open("x") as err:
        child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, start_new_session=True)
        status(result_dir, state="running", job=job, runner_pid=os.getpid(), child_pid=child.pid,
               stdout=str(stdout), stderr=str(stderr))
        try:
            while child.poll() is None:
                if time.monotonic() - start > 7200:
                    raise TimeoutError("single job exceeded two hours")
                if check_power and (Path("/sys/class/power_supply/ACAD/online").read_text().strip() != "1" or
                                    Path("/sys/firmware/acpi/platform_profile").read_text().strip() != "balanced"):
                    raise RuntimeError("AC or power profile changed during inference")
                time.sleep(2)
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
    dump_new(result_dir / (job + "-after.json"), {"host": baseline.snapshot(), "exit_code": child.returncode,
                                                  "wall_seconds": time.monotonic() - start})
    if child.returncode:
        raise RuntimeError(f"{job} failed with code {child.returncode}; see {stderr}")


def analyze(result_dir, plan):
    records, samples = [], {}
    for corpus in plan["corpora"]:
        indices = read_indices(Path(corpus["bootstrap_path"]), corpus["bootstrap_sha256"], corpus["queries"])
        for candidate in ["arctic", "granite", "e5", "jina-code", "bge-m3", "arctic-256"]:
            job = f"{candidate}-{corpus['id']}-matrix"
            events = [json.loads(line) for line in (result_dir / (job + ".jsonl")).read_text().splitlines()]
            if events[-1]["type"] != "compatibility_completed":
                raise ValueError("incomplete compatibility report")
            start = events[0]
            if start["corpus_sha256"] != corpus["sha256"]:
                raise ValueError("compatibility corpus binding mismatch")
            rows, ranks, boot, values = {}, {}, {}, {}
            for combination in COMBINATIONS:
                queries = [e for e in events if e["type"] == "compatibility_query" and e["combination"] == combination]
                if [q["query"] for q in queries] != corpus["query_ids"]:
                    raise ValueError("compatibility query coverage/order mismatch")
                summaries = [e for e in events if e["type"] == "compatibility_quality" and e["combination"] == combination]
                if len(summaries) != 1 or summaries[0]["queries"] != len(queries):
                    raise ValueError("unexpected domain/count in pilot")
                values[combination] = [q["ndcg_at_10"] for q in queries]
                boot[combination] = bootstrap_means(values[combination], indices)
                ranks[combination] = [q["top_10"] for q in queries]
                rows[combination] = {k: summaries[0][k] for k in ["ndcg_at_10", "recall_at_10", "mrr_at_10"]}
                if abs(rows[combination]["ndcg_at_10"] - sum(values[combination])/len(queries)) > 1e-12:
                    raise ValueError("per-query/summary metric mismatch")
                rows[combination]["ndcg_ci95"] = interval(boot[combination])
                samples[(corpus["id"], candidate, combination)] = boot[combination]
            reference = max(rows["cpu/cpu"]["ndcg_at_10"], rows["cuda/cuda"]["ndcg_at_10"])
            checks = {}
            for combination in COMBINATIONS[2:]:
                loss = reference - rows[combination]["ndcg_at_10"]
                distribution = [max(c, g)-m for c, g, m in zip(boot["cpu/cpu"], boot["cuda/cuda"], boot[combination])]
                checks[combination] = {"ndcg_loss_vs_better_homogeneous": loss, "loss_ci95": interval(distribution),
                    "passes_pilot_loss_threshold": loss <= 0.01, "threshold": 0.01,
                    "vs_cpu_homogeneous": overlap(ranks[combination], ranks["cpu/cpu"]),
                    "vs_cuda_homogeneous": overlap(ranks[combination], ranks["cuda/cuda"])}
            records.append({"candidate": candidate, "corpus": corpus["id"], "dimensions": start["dimensions"],
                "qualities": rows, "mixed_checks": checks,
                "homogeneous_cuda_minus_cpu_ndcg": rows["cuda/cuda"]["ndcg_at_10"] - rows["cpu/cpu"]["ndcg_at_10"],
                "homogeneous_cuda_minus_cpu_ci95": interval([g-c for c,g in zip(boot["cpu/cpu"],boot["cuda/cuda"])]),
                "vector_drift": next(e for e in events if e["type"] == "vector_drift"),
                "token_statistics": start["token_statistics"], "mixed_cuda_documents": start["mixed_cuda_documents"],
                "raw_file": job + ".jsonl", "raw_sha256": baseline.sha256(result_dir / (job + ".jsonl"))})
    lookup = {(r["corpus"],r["candidate"]):r for r in records}
    for record in records:
        reference = lookup[(record["corpus"],"arctic")]
        record["paired_delta_vs_arctic768"] = {}
        for combination in COMBINATIONS:
            record["paired_delta_vs_arctic768"][combination] = {
                "ndcg_delta": record["qualities"][combination]["ndcg_at_10"] - reference["qualities"][combination]["ndcg_at_10"],
                "ci95": interval([a-b for a,b in zip(samples[(record["corpus"],record["candidate"],combination)],
                                                         samples[(record["corpus"],"arctic",combination)])])}
    dump_new(result_dir / "summary.json", {"scope": "two-corpus exploratory pilot; no weighted score or shipping winner",
        "plan_sha256": baseline.sha256(result_dir / "plan.json"), "bootstrap_replicates": REPLICATES,
        "ci_policy": "paired query bootstrap; unadjusted descriptive percentile intervals", "results": records})


def freeze(args):
    audit = json.loads((ROOT / "benchmarks/datasets/small-beir-prepared.json").read_text())
    recipes = json.loads((ROOT / "benchmarks/candidates/recipes.json").read_text())
    source_paths = sorted({ROOT / "Cargo.toml", ROOT / "Cargo.lock", ROOT / "build.rs",
                           ROOT / "docs/retrieval-pilot-v1.md", *ROOT.glob("src/**/*.rs"),
                           *ROOT.glob("examples/**/*.rs"), *ROOT.glob("benchmarks/*.py")})
    binaries = [ROOT / "target/release/examples/qualify", ROOT / "target/release/examples/score_compatibility"]
    for binary in binaries:
        if not binary.is_file():
            raise ValueError("build both release examples before freezing a run")
    workspace = Path(tempfile.mkdtemp(prefix="retrieval-pilot-", dir=args.scratch))
    corpora = []
    for name in ["nfcorpus", "scifact"]:
        recorded = next(r for r in audit["results"] if r["dataset"] == name)
        path = args.scratch / "prepared-small-beir-v1" / (name + "-corpus.json")
        if baseline.sha256(path) != recorded["sha256"]:
            raise ValueError("prepared corpus hash differs from recorded audit")
        corpus = json.loads(path.read_text())
        indices = index_bytes(bootstrap_indices(recorded["sha256"], len(corpus["queries"])))
        index_path = workspace / (name + "-bootstrap.u32")
        with index_path.open("xb") as stream:
            stream.write(indices)
        corpora.append({"id": name, "path": str(path), "sha256": recorded["sha256"], "documents": len(corpus["documents"]),
            "queries": len(corpus["queries"]), "document_ids": [d["id"] for d in corpus["documents"]],
            "query_ids": [q["id"] for q in corpus["queries"]], "bootstrap_path": str(index_path),
            "bootstrap_sha256": hashlib.sha256(indices).hexdigest()})
    candidates = []
    for name in ["arctic", "granite", "e5", "jina-code", "bge-m3"]:
        recipe = next(c for c in recipes["candidates"] if c["id"] == name)
        entry = {"id": name}
        for provider in ["cpu", "cuda"]:
            path = args.scratch / (name + "-" + provider + ".json")
            manifest = json.loads(path.read_text())
            if manifest != recipe[provider]:
                raise ValueError("materialized manifest differs from recorded recipe")
            entry[provider] = {"path": str(path), "sha256": baseline.sha256(path), "manifest": manifest}
        candidates.append(entry)
    expected = sum((c["documents"]+c["queries"]+32)*candidate["cpu"]["manifest"]["recipe"]["dimensions"]*4*2
                   for c in corpora for candidate in candidates) + sum(c["queries"]*REPLICATES*4 for c in corpora)
    if expected > 1024**3 or shutil.disk_usage(workspace).free < expected + 512*1024**2:
        raise ValueError("insufficient pilot scratch capacity or exceeded 1 GiB vector/index allowance")
    plan = {"version": 1, "scope": "retrieval-pilot-v1; exploratory exception authorized before real quality results",
        "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "git_head": baseline.command("git","rev-parse","HEAD"),
        "git_status": baseline.command("git","status","--short"), "python": sys.version,
        "source_sha256": {str(p.relative_to(ROOT)): baseline.sha256(p) for p in source_paths},
        "executable_sha256": {str(p.relative_to(ROOT)): baseline.sha256(p) for p in binaries},
        "protocol_sha256": baseline.sha256(ROOT / "docs/retrieval-pilot-v1.md"),
        "corpora": corpora, "candidates": candidates, "workspace": str(workspace),
        "expected_vector_index_bytes": expected, "document_batch": 8, "query_batch": 1,
        "max_tokens": 512, "threads": 8, "power_mode": "balanced", "bootstrap_replicates": REPLICATES,
        "seed": "20260909", "mixed_ndcg_loss_threshold": 0.01, "comparisons": COMBINATIONS}
    dump_new(args.output / "plan.json", plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="only completed/hash-matching exports can be reused")
    args = parser.parse_args()
    if "CI" in os.environ:
        parser.error("real retrieval inference must not run in CI")
    args.scratch, args.output = args.scratch.resolve(), args.output.resolve()
    if not args.resume:
        args.output.mkdir(parents=True, exist_ok=False)
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    lock_file = (args.output / ".runner.lock").open("a")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        plan = json.loads((args.output / "plan.json").read_text()) if args.resume else freeze(args)
        for relative, expected in {**plan["source_sha256"], **plan["executable_sha256"]}.items():
            if baseline.sha256(ROOT / relative) != expected:
                raise ValueError("source/executable changed since premeasurement freeze")
        for corpus in plan["corpora"]:
            if baseline.sha256(Path(corpus["path"])) != corpus["sha256"]:
                raise ValueError("corpus changed since freeze")
        workspace = Path(plan["workspace"])
        for candidate in plan["candidates"]:
            for corpus in plan["corpora"]:
                for provider in ["cpu", "cuda"]:
                    job = f"{candidate['id']}-{corpus['id']}-{provider}"
                    exported = workspace / job
                    log, err = args.output / (job + ".jsonl"), args.output / (job + ".stderr")
                    manifest = candidate[provider]
                    if baseline.sha256(Path(manifest["path"])) != manifest["sha256"]:
                        raise ValueError("manifest changed since freeze")
                    if exported.exists() or log.exists():
                        if not args.resume:
                            raise ValueError("refusing to overwrite existing run")
                        verify_export(exported, log, manifest["manifest"], corpus)
                        print("Reused", job, flush=True)
                        continue
                    time.sleep(5)
                    command = [str(ROOT / "target/release/examples/qualify"), "--manifest", manifest["path"],
                        "--corpus", corpus["path"], "--device", "cpu" if provider == "cpu" else "cuda:0",
                        "--power-mode", "balanced", "--quality-only", "--load-repeats", "1", "--export-vectors", str(exported)]
                    print("Encoding", job, flush=True)
                    run_command(command, log, err, args.output, job)
                    verify_export(exported, log, manifest["manifest"], corpus)
                    print("Completed", job, flush=True)
        for candidate in plan["candidates"]:
            for corpus in plan["corpora"]:
                for projected in ([False, True] if candidate["id"] == "arctic" else [False]):
                    name = candidate["id"] + ("-256" if projected else "")
                    job = f"{name}-{corpus['id']}-matrix"
                    log = args.output / (job + ".jsonl")
                    if args.resume and log.exists():
                        events = [json.loads(line) for line in log.read_text().splitlines()]
                        if events[-1]["type"] != "compatibility_completed":
                            raise ValueError("partial scoring log; retain it and retry separately")
                        continue
                    command = [str(ROOT / "target/release/examples/score_compatibility"), "--corpus", corpus["path"],
                        "--cpu-vectors", str(workspace / f"{candidate['id']}-{corpus['id']}-cpu"),
                        "--cuda-vectors", str(workspace / f"{candidate['id']}-{corpus['id']}-cuda")]
                    if projected:
                        command += ["--dimensions", "256"]
                    print("Scoring", job, flush=True)
                    run_command(command, log, args.output / (job + ".stderr"), args.output, job)
        status(args.output, state="analyzing", runner_pid=os.getpid())
        analyze(args.output, plan)
        status(args.output, state="completed", runner_pid=os.getpid(), summary_sha256=baseline.sha256(args.output / "summary.json"))
        print("Pilot completed:", args.output / "summary.json", flush=True)
    except BaseException as exc:
        status(args.output, state="failed", runner_pid=os.getpid(), error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        lock_file.close()


if __name__ == "__main__":
    main()
