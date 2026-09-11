#!/usr/bin/env python3
"""Prepare the three locked complete small BEIR corpora, without inference.

Only gzip decoding occurs here; ree's Rust prepare_corpus tool owns selection,
validation and final publication. No upstream scripts are imported or executed.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fetch_artifacts", ROOT / "benchmarks/fetch-artifacts.py")
artifacts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(artifacts)
MAX_BYTES = 128 * 1024 * 1024


def unpack(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with gzip.open(source, "rb") as stream, tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp:
            temp_path = Path(temp.name)
            digest = hashlib.sha256()
            total = 0
            while True:
                block = stream.read(min(1024 * 1024, MAX_BYTES + 1 - total))
                if not block:
                    break
                total += len(block)
                if total > MAX_BYTES:
                    raise ValueError("decoded input exceeds 128 MiB")
                digest.update(block)
                temp.write(block)
            temp.flush()
            os.fsync(temp.fileno())
        os.link(temp_path, destination)
        return digest.hexdigest()
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for normalized/prepared outputs")
    parser.add_argument("--lock", type=Path, default=ROOT / "benchmarks/datasets/small-beir-lock.json")
    args = parser.parse_args()
    root = args.scratch.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    lock = json.loads(args.lock.read_text())
    if {d["id"] for d in lock["datasets"]} != {"nfcorpus", "scifact", "arguana"}:
        parser.error("adapter is restricted to the reviewed three-dataset small-corpus lock")
    lock_hash = hashlib.sha256(args.lock.read_bytes()).hexdigest()
    adapter_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for entry in lock["files"]:
        path = root / entry["path"]
        if not path.resolve().is_relative_to(root):
            raise ValueError("locked input escapes scratch")
        artifacts.validate_file(path, entry)
    results = []
    for dataset in lock["datasets"]:
        name = dataset["id"]
        inputs = {}
        for upstream, output, kind in [("corpus", "documents", "documents"), ("queries", "queries", "queries")]:
            path = output_dir / "normalized" / name / (output + ".jsonl")
            digest = unpack(root / "datasets" / name / (upstream + ".jsonl.gz"), path)
            inputs[kind] = {"path": str(path.relative_to(output_dir)), "sha256": digest}
        qrels = root / "datasets" / (name + "-qrels") / "test.tsv"
        inputs["qrels"] = {"path": os.path.relpath(qrels, output_dir), "sha256": hashlib.sha256(qrels.read_bytes()).hexdigest()}
        manifest = {
            "version": "ree-subset-v1", "dataset": "BeIR/" + name,
            "source_url": f"https://huggingface.co/datasets/BeIR/{name}/tree/{dataset['corpus_revision']}",
            "revision": dataset["corpus_revision"], "split": "test", "domain": dataset["domain"],
            "license": dataset["declared_license"],
            "license_reference": f"https://huggingface.co/datasets/BeIR/{name}/blob/{dataset['corpus_revision']}/README.md",
            "preparation": json.dumps({"adapter": "benchmarks/prepare-small-beir.py", "adapter_sha256": adapter_hash,
                "operation": "gzip decode only; no text/ID/qrel transformations", "upstream_lock_sha256": lock_hash,
                "corpus_revision": dataset["corpus_revision"], "qrels_revision": dataset["qrels_revision"],
                "upstream_rights_review": dataset["upstream_rights_review"]}, sort_keys=True),
            "seed": "20260909", "query_count": 100, "document_count": 10000,
            "source_is_complete": True, "exclude_query_id": True, "hard_negatives_per_query": 20,
            **inputs, "negatives": None,
        }
        path = output_dir / (name + "-corpus-manifest.json")
        with path.open("x") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        output = output_dir / (name + "-corpus.json")
        command = [str(ROOT / "target/release/examples/prepare_corpus"), "--manifest", str(path),
                   "--scratch-dir", str(root), "--output", str(output)]
        run = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if run.returncode:
            results.append({"dataset": name, "status": "blocked", "exit_code": run.returncode,
                            "error": run.stderr, "preparation_manifest": manifest,
                            "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            print(f"Blocked {name}: {run.stderr.strip()}", flush=True)
            continue
        report = json.loads(run.stdout)
        corpus = json.loads(output.read_text())
        report.update(dataset=name, preparation_manifest=manifest,
                      manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      selected_query_ids=[q["id"] for q in corpus["queries"]],
                      positive_qrel_pairs=corpus["provenance"]["positive_qrel_pairs"],
                      retained_zero_qrel_pairs=corpus["provenance"]["retained_zero_qrel_pairs"],
                      source_documents=corpus["provenance"]["source_documents"],
                      eligible_queries=corpus["provenance"]["eligible_queries"],
                      full_corpus=corpus["provenance"]["full_corpus"])
        results.append(report)
        print(f"Prepared {name}: {report['documents']} documents / {report['queries']} queries", flush=True)
    with (output_dir / "small-beir-prepared.json").open("x") as stream:
        json.dump({"scope": "three prepared corpora; full suite/rights review pending; no retrieval scores",
                   "results": results}, stream, indent=2, sort_keys=True)
        stream.write("\n")
    if any(result.get("status") == "blocked" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
