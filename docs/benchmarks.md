# Local model and storage qualification

Performance/quality results are specific to the development AMD Ryzen 9 8945HS
and RTX 4070 Laptop GPU, not a cross-hardware matrix. The qualification executable
refuses CI and checks both hardware identities. Ordinary unit tests never run it.

## Model harness

```sh
cargo run --release --locked --example qualify -- \
  --device cpu --power-mode balanced --batches 1,2,4,8,16 --repeats 5 \
  --corpus benchmarks/smoke.json > arctic-cpu.jsonl
cargo run --release --locked --example qualify -- \
  --device cuda:0 --power-mode performance --batches 1,2,4,8,16 --repeats 5 \
  --corpus benchmarks/smoke.json > arctic-cuda.jsonl
```

Use the **actual** power mode, idle the machine, record competing processes, and
repeat runs. The harness reports artifact/revision/hash, precision, pooling,
provider/device, thread count, load+warm-up time, 64/256/512-token workloads,
per-repeat latency/variance, throughput, post-batch free VRAM, and per-domain
binary-qrel Recall@10/MRR@10/nDCG@10. It uses ree's actual ONNX implementation.
The checked-in six-document corpus is only a smoke test, **not BEIR/MIRACL/CoIR
qualification**. OOM stops increasing that sequence-length probe; it does not
extrapolate results beyond successful batches. Peak VRAM, repeated cold loads,
and formal stability confidence intervals still need instrumentation.

A candidate manifest supplies exact local artifacts for Granite 107M
Multilingual, E5-small, BGE-M3, or Jina v2 Base Code without adding models to the
production CLI:

```json
{
  "name": "intfloat/multilingual-e5-small",
  "revision": "REPLACE_WITH_EXACT_40_HEX_COMMIT",
  "precision": "int8",
  "model": {"path": "/absolute/model.onnx", "sha256": "REPLACE_WITH_SHA256"},
  "tokenizer": {"path": "/absolute/tokenizer.json", "sha256": "REPLACE_WITH_SHA256"},
  "recipe": {"dimensions": 384, "pooling": "mean", "pad_id": 1, "cls_id": 0, "sep_id": 2},
  "document_prompt": "passage: ",
  "query_prompt": "query: "
}
```

Pass `--manifest candidate.json`. Verify the candidate's pinned model card,
output tensors, pooling, normalization, prompts, and special IDs before using a
manifest. The harness supports CLS/attention-mask mean pooling of token outputs;
a candidate needing another graph contract must extend/test the same runtime
boundary, not silently use another inference library. Candidate artifacts are
user-provided and hash-checked, never downloaded from a moving revision. No
model-supplied Python is executed.

Corpus format is the checked-in JSON structure: unique document IDs/text, query
IDs/text, nonempty lists of relevant document IDs, and a `domain` tag. Prepare
manageable, reproducible BEIR, MIRACL language, and code-retrieval subsets (up to
10,000 documents per run), retain dataset licenses, hashes, and selection seeds,
and group metrics by domain. Inputs are truncated to 512 tokens in this harness.
Do not compare smoke scores with published model-card benchmarks.

Before freezing the production revision, run all five planned candidates with
comparable corpora and CPU/GPU artifacts where supported; investigate code or
throughput regressions. Use PLAN.md's 45/20/15/15/5 quality/multilingual/code/local
throughput/footprint weighting. A weighted decision requires comparable measured
inputs; the harness intentionally does not manufacture missing scores.

## Real correctness tests

```sh
cargo test --test model_real cpu_token_windows_and_repeatability -- --ignored --test-threads=1
cargo test --test model_real cpu_int8_cuda_fp16_cosine_and_ranking_parity -- --ignored --test-threads=1
cargo test --test cuda_real_oom -- --ignored --test-threads=1
cargo test --test cuda_e2e -- --ignored --test-threads=1
```

The CPU test checks exact multilingual token windows, normalization, and repeated
output stability. CUDA accuracy uses the same FP16 artifact on CPU as reference
(cosine >=0.999) and tests multiple batch/padded shapes. INT8/FP16 quantization
compatibility is separate (provisional cosine >=0.94 plus query top-1 agreement).
The original 0.98 INT8/FP16 gate failed due to quantization drift also present on
CPU; it was not a CUDA accuracy failure. See [runtime notes](runtime.md).
Real OOM tests constrain the ONNX arena, not physical device memory, and never
reset the GPU. Run real tests serially; all are development-machine-only.

## SQLite scale harness

```sh
cargo run --release --locked --example scale -- --db /scratch/ree-1m.db --chunks 1000000
cargo run --release --locked --example scale -- --db /scratch/ree-5m.db --chunks 5000000
cargo run --release --locked --example scale -- --db /scratch/ree-10m.db --chunks 10000000
```

A new path is required. These databases are explicitly marked
`schema_metadata.synthetic_vectors=true`; vectors are deterministic random unit
vectors, **not usable embeddings**. Reports include insertion rate, database
size, and five exact top-10 scan latencies. Expect more than 3 GB per million
768-dimensional vectors plus metadata/index/WAL overhead. Provision sufficient
disk and use idle-machine repeated runs. Do not load a synthetic benchmark into
your production database or report it as model throughput.

This harness measures storage, not million-file discovery, peak memory, or full
pipeline/rebuild scalability. Those release gates require separate real/synthetic
file corpora and process-level resource measurements.
