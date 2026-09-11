"""Offline consistency checks for retained pilot evidence; no vector/model cache needed."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent / "results/retrieval-pilot-20260909"


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class Evidence(unittest.TestCase):
    def test_partial_coverage_is_explicit_and_bound_to_frozen_plan(self):
        summary = json.loads((ROOT / "partial-analysis/summary.json").read_text())
        self.assertIn("PARTIAL", summary["scope"])
        self.assertEqual({r["candidate"] for r in summary["pending_candidates"]}, {"jina-code", "bge-m3"})
        self.assertEqual(summary["parent_plan_sha256"], hashlib.sha256((ROOT / "plan.json").read_bytes()).hexdigest())
        self.assertEqual({(r["candidate"],r["corpus"]) for r in summary["results"]},
                         {(m,c) for m in ["arctic","arctic-256","granite","e5"] for c in ["nfcorpus","scifact"]})
        state = json.loads((ROOT / "status.json").read_text())
        if state["state"] != "completed":
            self.assertFalse((ROOT / "summary.json").exists(), "do not present an interrupted pilot as completed")

    def test_raw_metrics_rankings_and_loss_flags_are_consistent(self):
        summary = json.loads((ROOT / "partial-analysis/summary.json").read_text())
        for record in summary["results"]:
            path = ROOT / "partial-analysis" / record["raw_file"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),record["raw_sha256"])
            rows = events(path)
            self.assertEqual(rows[-1]["type"],"compatibility_completed")
            self.assertEqual(len(record["qualities"]),6)
            for combination, metrics in record["qualities"].items():
                queries = [r for r in rows if r["type"]=="compatibility_query" and r["combination"]==combination]
                self.assertEqual(len(queries),100)
                for metric in ["ndcg_at_10","recall_at_10","mrr_at_10"]:
                    self.assertAlmostEqual(metrics[metric],sum(q[metric] for q in queries)/100,places=12)
                if record["candidate"] != "arctic-256" and combination in ["cpu/cpu","cuda/cuda"]:
                    provider=combination.split('/')[0]
                    original=events(ROOT/f"{record['candidate']}-{record['corpus']}-{provider}.jsonl")
                    original={r["query"]:r for r in original if r["type"]=="quality_query"}
                    for query in queries:
                        self.assertEqual(query["top_10"],original[query["query"]]["top_10"])
            better=max(record["qualities"]["cpu/cpu"]["ndcg_at_10"],record["qualities"]["cuda/cuda"]["ndcg_at_10"])
            for combination, check in record["mixed_checks"].items():
                loss=better-record["qualities"][combination]["ndcg_at_10"]
                self.assertAlmostEqual(loss,check["ndcg_loss_vs_better_homogeneous"],places=12)
                self.assertEqual(check["passes_pilot_loss_threshold"],loss<=0.01)
                self.assertLessEqual(*check["loss_ci95"])

    def test_interruption_preserves_attempt_without_claiming_a_complete_export(self):
        pause=json.loads((ROOT/"pause.json").read_text())
        self.assertEqual((pause["completed_encoding_runs"],pause["remaining_encoding_runs"]),(12,8))
        for item in pause["archived_log_files"]:
            path=ROOT/"interrupted-attempts"/pause["interrupted_job"]/item["name"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),item["sha256"])
        path=ROOT/"interrupted-attempts"/pause["interrupted_job"]/(pause["interrupted_job"]+".jsonl")
        self.assertNotEqual(events(path)[-1]["type"],"qualification_completed")


class CompletedEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads((ROOT / "plan.json").read_text())
        cls.summary = json.loads((ROOT / "summary.json").read_text())
        cls.records = {(r["candidate"], r["corpus"]): r for r in cls.summary["results"]}

    def test_complete_coverage_and_artifact_bindings(self):
        state = json.loads((ROOT / "status.json").read_text())
        self.assertEqual(state["state"], "completed")
        self.assertEqual(state["summary_sha256"], hashlib.sha256((ROOT / "summary.json").read_bytes()).hexdigest())
        self.assertEqual(self.summary["plan_sha256"], hashlib.sha256((ROOT / "plan.json").read_bytes()).hexdigest())
        self.assertEqual(self.summary["bootstrap_replicates"], 10000)
        self.assertIn("no weighted score or shipping winner", self.summary["scope"])
        models = {c["id"] for c in self.plan["candidates"]} | {"arctic-256"}
        self.assertEqual(set(self.records), {(m, c["id"]) for m in models for c in self.plan["corpora"]})
        self.assertEqual(len(self.summary["results"]), 12)
        for candidate in self.plan["candidates"]:
            for corpus in self.plan["corpora"]:
                for provider in ["cpu", "cuda"]:
                    job = f"{candidate['id']}-{corpus['id']}-{provider}"
                    rows = events(ROOT / (job + ".jsonl"))
                    self.assertEqual(rows[-1]["type"], "qualification_completed")
                    self.assertFalse(any(e["type"] == "probe_failed" for e in rows))
                    info = next(e for e in rows if e["type"] == "qualification_started")
                    manifest = candidate[provider]["manifest"]
                    for field in ["revision", "precision", "recipe", "document_prompt", "query_prompt"]:
                        self.assertEqual(info[field], manifest[field])
                    self.assertEqual(info["model_sha256"], manifest["model"]["sha256"])
                    self.assertEqual(info["tokenizer_sha256"], manifest["tokenizer"]["sha256"])
                    self.assertEqual(info["corpus_sha256"], corpus["sha256"])
                    self.assertEqual(info["threads"], 8)
                    self.assertEqual(info["gpu"] is None, provider == "cpu")
                    for phase in ["before", "after"]:
                        snapshot = json.loads((ROOT / f"{job}-{phase}.json").read_text())
                        self.assertEqual(snapshot["host"]["ac_online"], "1")
                        self.assertEqual(snapshot["host"]["power_profile"], "balanced")
                        if phase == "after":
                            self.assertEqual(snapshot["exit_code"], 0)
                    exported = next(e for e in rows if e["type"] == "vector_export_completed")
                    self.assertEqual(exported["single_document_samples"], 32)
                    matrix = events(ROOT / self.records[candidate["id"], corpus["id"]]["raw_file"])
                    self.assertEqual(matrix[0][provider + "_export_sha256"], exported["metadata_sha256"])

    def test_all_7200_query_rows_and_2000_live_roundtrips(self):
        total = live = 0
        for record in self.records.values():
            corpus = next(c for c in self.plan["corpora"] if c["id"] == record["corpus"])
            path = ROOT / record["raw_file"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["raw_sha256"])
            rows = events(path)
            self.assertEqual(rows[0]["corpus_sha256"], corpus["sha256"])
            self.assertEqual(rows[-1]["type"], "compatibility_completed")
            self.assertEqual(set(record["qualities"]), set(self.plan["comparisons"]))
            self.assertEqual(sum(e["type"] == "compatibility_query" for e in rows), 600)
            expected_mixed = sum(hashlib.sha256(f"ree-mixed-index-v1\0{self.plan['seed']}\0{d}".encode()).digest()[0] & 1
                                 for d in corpus["document_ids"])
            self.assertEqual(record["mixed_cuda_documents"], expected_mixed)
            for combination, metrics in record["qualities"].items():
                queries = [e for e in rows if e["type"] == "compatibility_query" and e["combination"] == combination]
                self.assertEqual([q["query"] for q in queries], corpus["query_ids"])
                total += len(queries)
                for metric in ["ndcg_at_10", "recall_at_10", "mrr_at_10"]:
                    self.assertAlmostEqual(metrics[metric], sum(q[metric] for q in queries) / len(queries), places=12)
                for q in queries:
                    self.assertEqual(len(q["top_10"]), 10)
                    self.assertEqual(len(set(q["top_10"])), 10)
                    self.assertNotIn(q["query"], q["top_10"])
                    self.assertTrue(set(q["top_10"]).issubset(corpus["document_ids"]))
                if record["candidate"] != "arctic-256" and combination in ["cpu/cpu", "cuda/cuda"]:
                    provider = combination.split("/")[0]
                    original = events(ROOT / f"{record['candidate']}-{record['corpus']}-{provider}.jsonl")
                    original = {e["query"]: e for e in original if e["type"] == "quality_query"}
                    self.assertEqual(set(original), set(corpus["query_ids"]))
                    for q in queries:
                        for field in ["top_10", "ndcg_at_10", "recall_at_10", "mrr_at_10"]:
                            self.assertEqual(q[field], original[q["query"]][field])
                        live += 1
        self.assertEqual((total, live), (7200, 2000))

    def test_every_loss_flag_overlap_and_paired_point_delta(self):
        flags = 0
        for record in self.records.values():
            rows = events(ROOT / record["raw_file"])
            ranks = {c: [e["top_10"] for e in rows if e["type"] == "compatibility_query" and e["combination"] == c]
                     for c in self.plan["comparisons"]}
            qualities = record["qualities"]
            better = max(qualities[c]["ndcg_at_10"] for c in ["cpu/cpu", "cuda/cuda"])
            self.assertEqual(set(record["mixed_checks"]), set(self.plan["comparisons"][2:]))
            for combination, check in record["mixed_checks"].items():
                loss = better - qualities[combination]["ndcg_at_10"]
                self.assertAlmostEqual(loss, check["ndcg_loss_vs_better_homogeneous"], places=12)
                self.assertEqual(check["threshold"], self.plan["mixed_ndcg_loss_threshold"])
                self.assertEqual(check["passes_pilot_loss_threshold"], loss <= check["threshold"])
                self.assertLessEqual(*check["loss_ci95"])
                flags += not check["passes_pilot_loss_threshold"]
                for provider in ["cpu", "cuda"]:
                    pairs = list(zip(ranks[combination], ranks[provider + "/" + provider]))
                    overlap = check["vs_" + provider + "_homogeneous"]
                    self.assertEqual(overlap["top1_agreement"], sum(a[0] == b[0] for a, b in pairs) / len(pairs))
                    self.assertAlmostEqual(overlap["top10_overlap"], sum(len(set(a) & set(b)) / 10 for a, b in pairs) / len(pairs))
            reference = self.records["arctic", record["corpus"]]
            for combination, delta in record["paired_delta_vs_arctic768"].items():
                self.assertAlmostEqual(delta["ndcg_delta"], qualities[combination]["ndcg_at_10"] -
                                       reference["qualities"][combination]["ndcg_at_10"], places=12)
                self.assertLessEqual(*delta["ci95"])
        self.assertEqual(flags, 8)

    def test_partial_results_and_second_interruption_are_preserved(self):
        partial = json.loads((ROOT / "partial-analysis/summary.json").read_text())
        for record in partial["results"]:
            self.assertEqual({k: v for k, v in record.items() if k != "offline_scoring_seconds"},
                             self.records[record["candidate"], record["corpus"]])
        pause = json.loads((ROOT / "resume-pause.json").read_text())
        self.assertEqual((pause["completed_encoding_runs"], pause["remaining_encoding_runs"]), (15, 5))
        for item in pause["archived_log_files"]:
            path = ROOT / "interrupted-attempts" / pause["interrupted_job"] / item["name"]
            self.assertEqual(path.stat().st_size, item["bytes"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])
        interrupted = events(ROOT / "interrupted-attempts" / pause["interrupted_job"] / (pause["interrupted_job"] + ".jsonl"))
        self.assertNotEqual(interrupted[-1]["type"], "qualification_completed")
        self.assertEqual(events(ROOT / (pause["interrupted_job"] + ".jsonl"))[-1]["type"], "qualification_completed")


if __name__ == "__main__":
    unittest.main()
