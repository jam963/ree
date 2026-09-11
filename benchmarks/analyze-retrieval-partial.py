#!/usr/bin/env python3
"""Offline partial analysis of an interrupted pilot. Never starts model inference.

Only candidates with BOTH corpora and BOTH provider exports complete are included.
Missing candidates are explicitly pending, never replaced by zeros or omitted from
an alleged complete pilot. Uses the parent's frozen Rust scorer/bootstrap indices.
"""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pilot", ROOT / "benchmarks/run-retrieval-pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.run_dir, args.output = args.run_dir.resolve(), args.output.resolve()
    with (args.run_dir / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True, exist_ok=False)
        plan = json.loads((args.run_dir / "plan.json").read_text())
        workspace = Path(plan["workspace"])
        scorer = ROOT / "target/release/examples/score_compatibility"
        for relative, digest in plan["source_sha256"].items():
            if pilot.baseline.sha256(ROOT / relative) != digest:
                raise ValueError("parent source changed since the frozen pilot")
        if pilot.baseline.sha256(scorer) != plan["executable_sha256"][str(scorer.relative_to(ROOT))]:
            raise ValueError("scorer binary changed since freeze")
        complete, pending = [], []
        for candidate in plan["candidates"]:
            try:
                for corpus in plan["corpora"]:
                    for provider in ["cpu", "cuda"]:
                        name = f"{candidate['id']}-{corpus['id']}-{provider}"
                        pilot.verify_export(workspace / name, args.run_dir / (name + ".jsonl"),
                                            candidate[provider]["manifest"], corpus)
            except (OSError, ValueError, KeyError, StopIteration) as exc:
                pending.append({"candidate": candidate["id"], "reason": f"{type(exc).__name__}: {exc}"})
                continue
            complete.append(candidate)
        if not any(c["id"] == "arctic" for c in complete):
            raise ValueError("completed Arctic reference required for paired partial comparisons")
        pilot.dump_new(args.output / "analysis-plan.json", {
            "scope": "offline partial analysis only; no inference or performance measurement",
            "parent_plan_sha256": pilot.baseline.sha256(args.run_dir / "plan.json"),
            "parent_state": json.loads((args.run_dir / "status.json").read_text()),
            "analyzer_sha256": pilot.baseline.sha256(Path(__file__)),
            "scorer_sha256": pilot.baseline.sha256(scorer),
            "included_candidates": [c["id"] for c in complete], "pending_candidates": pending,
            "selection_rule": "both fixed corpora and both providers complete, independent of scores",
            "host_ac_online": Path("/sys/class/power_supply/ACAD/online").read_text().strip(),
            "host_power_profile": Path("/sys/firmware/acpi/platform_profile").read_text().strip(),
        })
        records, bootstraps = [], {}
        for corpus in plan["corpora"]:
            if pilot.baseline.sha256(Path(corpus["path"])) != corpus["sha256"]:
                raise ValueError("corpus changed since freeze")
            indices = pilot.read_indices(Path(corpus["bootstrap_path"]), corpus["bootstrap_sha256"], corpus["queries"])
            for candidate in complete:
                for projected in ([False, True] if candidate["id"] == "arctic" else [False]):
                    name = candidate["id"] + ("-256" if projected else "")
                    job = f"{name}-{corpus['id']}-matrix"
                    command = [str(scorer), "--corpus", corpus["path"],
                        "--cpu-vectors", str(workspace / f"{candidate['id']}-{corpus['id']}-cpu"),
                        "--cuda-vectors", str(workspace / f"{candidate['id']}-{corpus['id']}-cuda")]
                    if projected:
                        command += ["--dimensions", "256"]
                    pilot.dump_new(args.output / (job + "-command.json"), {"command": command, "no_inference": True})
                    print("Offline scoring", job, flush=True)
                    start = time.monotonic()
                    with (args.output / (job + ".jsonl")).open("x") as out, (args.output / (job + ".stderr")).open("x") as err:
                        subprocess.run(command, stdout=out, stderr=err, check=True, timeout=180)
                    events = [json.loads(line) for line in (args.output / (job + ".jsonl")).read_text().splitlines()]
                    if events[-1]["type"] != "compatibility_completed" or events[0]["corpus_sha256"] != corpus["sha256"]:
                        raise ValueError("incomplete/mismatched compatibility report")
                    rows, ranks, means = {}, {}, {}
                    for combination in pilot.COMBINATIONS:
                        queries = [e for e in events if e["type"] == "compatibility_query" and e["combination"] == combination]
                        if [q["query"] for q in queries] != corpus["query_ids"]:
                            raise ValueError("query coverage/order mismatch")
                        raw_summary = [e for e in events if e["type"] == "compatibility_quality" and e["combination"] == combination]
                        if len(raw_summary) != 1 or raw_summary[0]["queries"] != corpus["queries"]:
                            raise ValueError("domain/count mismatch")
                        values = [q["ndcg_at_10"] for q in queries]
                        rows[combination] = {k: raw_summary[0][k] for k in ["ndcg_at_10", "recall_at_10", "mrr_at_10"]}
                        if abs(rows[combination]["ndcg_at_10"] - sum(values)/len(values)) > 1e-12:
                            raise ValueError("summary differs from per-query metrics")
                        means[combination] = pilot.bootstrap_means(values, indices)
                        rows[combination]["ndcg_ci95"] = pilot.interval(means[combination])
                        ranks[combination] = [q["top_10"] for q in queries]
                        bootstraps[(corpus["id"], name, combination)] = means[combination]
                    better = max(rows["cpu/cpu"]["ndcg_at_10"], rows["cuda/cuda"]["ndcg_at_10"])
                    checks = {}
                    for combination in pilot.COMBINATIONS[2:]:
                        loss = better - rows[combination]["ndcg_at_10"]
                        losses = [max(c,g)-m for c,g,m in zip(means["cpu/cpu"], means["cuda/cuda"], means[combination])]
                        checks[combination] = {"ndcg_loss_vs_better_homogeneous": loss, "loss_ci95": pilot.interval(losses),
                            "threshold": plan["mixed_ndcg_loss_threshold"], "passes_pilot_loss_threshold": loss <= plan["mixed_ndcg_loss_threshold"],
                            "vs_cpu_homogeneous": pilot.overlap(ranks[combination], ranks["cpu/cpu"]),
                            "vs_cuda_homogeneous": pilot.overlap(ranks[combination], ranks["cuda/cuda"])}
                    records.append({"candidate": name, "corpus": corpus["id"], "dimensions": events[0]["dimensions"],
                        "qualities": rows, "mixed_checks": checks,
                        "homogeneous_cuda_minus_cpu_ndcg": rows["cuda/cuda"]["ndcg_at_10"]-rows["cpu/cpu"]["ndcg_at_10"],
                        "homogeneous_cuda_minus_cpu_ci95": pilot.interval([g-c for c,g in zip(means["cpu/cpu"],means["cuda/cuda"])]),
                        "vector_drift": next(e for e in events if e["type"] == "vector_drift"),
                        "token_statistics": events[0]["token_statistics"], "mixed_cuda_documents": events[0]["mixed_cuda_documents"],
                        "raw_file": job + ".jsonl", "raw_sha256": pilot.baseline.sha256(args.output / (job + ".jsonl")),
                        "offline_scoring_seconds": time.monotonic()-start})
        references = {r["corpus"]: r for r in records if r["candidate"] == "arctic"}
        for record in records:
            reference = references[record["corpus"]]
            record["paired_delta_vs_arctic768"] = {}
            for combination in pilot.COMBINATIONS:
                deltas = [a-b for a,b in zip(bootstraps[(record["corpus"],record["candidate"],combination)],
                                           bootstraps[(record["corpus"],"arctic",combination)])]
                record["paired_delta_vs_arctic768"][combination] = {
                    "ndcg_delta": record["qualities"][combination]["ndcg_at_10"]-reference["qualities"][combination]["ndcg_at_10"],
                    "ci95": pilot.interval(deltas)}
        pilot.dump_new(args.output / "summary.json", {"scope": "PARTIAL two-corpus exploratory pilot; no weighted score or winner",
            "parent_plan_sha256": pilot.baseline.sha256(args.run_dir / "plan.json"), "bootstrap_replicates": pilot.REPLICATES,
            "pending_candidates": pending, "results": records})
        print("Partial results:", args.output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
