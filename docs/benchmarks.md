# Local model and storage qualification

Performance/quality results are specific to the development AMD Ryzen 9 8945HS
and RTX 4070 Laptop GPU, not a cross-hardware matrix. The qualification executable
refuses CI and checks both hardware identities. Ordinary unit tests never run it.

**Arctic M v2 at 768 dimensions is selected for beta**, by the user's
[2026-09-10 decision](model-decision-v1.md). Broader comparative evaluation is
deliberately deferred, not a blocker on this model choice. Do not start more
comparative evaluations unless requested. Selected-model beta work follows the
[roadmap](beta-roadmap.md).

The [original evaluation protocol](evaluation-protocol.md) and tooling below are
retained for provenance and possible future work. The [completed pilot](../benchmarks/results/retrieval-pilot-20260909/README.md)
covers all five candidates on two corpora; missing dataset locks, mining adapters
and full weighted aggregation remain uncompleted, and mixed-recipe regressions
remain documented. Neither the product decision nor the smoke reports establishes
a full-suite quality/compatibility pass.

## Model harness

All five [candidate recipes and artifact locks](../benchmarks/candidates/README.md)
now have actual CPU/CUDA capability checks and repeated, same-build speed runs.
See [recorded baselines](../benchmarks/results/README.md#2026-09-09-idle-window-five-candidate-speed-baselines).
Use `benchmarks/run-baseline.py` for serial three-process CPU/GPU repetitions after
building; it checks AC/current power profile and competing GPU compute processes,
records source/binary hashes and raw host snapshots, and never runs quality scoring.
It does not disable background OS work or change power policy. Build/provision
before starting, not concurrently.

```sh
cargo run --release --locked --example qualify -- \
  --device cpu --power-mode balanced --batches 1,2,4,8,16 --repeats 5 \
  --corpus benchmarks/smoke.json > arctic-cpu.jsonl
cargo run --release --locked --example qualify -- \
  --device cuda:0 --power-mode performance --batches 1,2,4,8,16 --repeats 5 \
  --corpus benchmarks/smoke.json > arctic-cuda.jsonl
```

Use the **actual** power mode, idle the machine, record competing processes, and
repeat runs. The harness reports artifact/revision/hash, precision, pooling/prompts,
provider/device, actual thread count, CUDA runtime/driver API/cuDNN versions,
64/256/512-token workloads, per-repeat latency/variance, output stability,
throughput, post-batch free VRAM, per-query top-10 rankings, and per-domain
Recall@10/MRR@10/nDCG@10 (positive qrels for Recall/MRR, linear graded gains for
nDCG; binary when grades are omitted). It uses ree's actual ONNX implementation. Host
snapshots record the platform profile, governor, load average, and GPU compute
processes (including the harness itself after initialization).
The checked-in six-document corpus is only a smoke test, **not BEIR/MIRACL/CoIR
qualification**. OOM stops increasing that sequence-length probe and recreates
the session before the next length; it does not extrapolate results beyond
successful batches. Non-OOM inference errors abort qualification.

`--load-repeats N` (default 3) recreates sessions and reports individual load plus
warm-up times. These are **not cold-cache loads**: artifact verification has
already read the files, OS caches are not flushed, and ORT process-global state
may persist. Run separate processes under documented cache conditions for cold
startup qualification; ree does not drop the host's caches.

A 10-ms sampler reports process RSS, lifetime RSS high-water mark, per-process
NVML compute memory and total device-used memory for each measured phase.
**Sampled peaks are lower bounds** and can miss brief allocation spikes. Total
GPU usage includes other processes; process RSS excludes filesystem/page-cache
memory. Unavailable readings are null and GPU sampling failures are counted.
Sampling adds overhead, so do not directly compare these results with older
unsampled runs. Full confidence intervals and saturation qualification remain
open.

For Matryoshka comparison, repeat the identical corpus with `--dimensions 256`.
This truncates **and renormalizes** vectors for quality scoring only; native
inference still produces 768 dimensions. Only use truncation for candidates
whose model contracts support it. `--quality-only --corpus PATH` skips synthetic
throughput probes but still loads and warms the model.

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
manifest. The harness supports CLS/attention-mask mean pooling of token outputs.
A reviewed export can name its rank-3 token tensor with optional recipe field
`token_output` (Granite uses `logits`); shape/dimensions are still checked and
production defaults are unchanged. Optional manifest `sidecars` is an array of
`{path, sha256}` entries, all verified before session creation. Audit the graph's
external references against that list; the harness does not discover undeclared
files automatically. Paths resolve relative to the manifest. Unknown manifest/
recipe fields are rejected. `--validate-only` tests special IDs, graph/provider
loading, mixed-length shapes and individual/batched vector drift without scoring
a corpus. Candidate artifacts are hash-checked; the provisioning helper uses
immutable revisions and runs no model-supplied Python.

Prepare pinned local inputs without inference or downloads:

```sh
cargo run --locked --example prepare_corpus -- \
  --manifest /scratch/ree-eval/task-manifest.json \
  --scratch-dir /scratch --output /scratch/ree-eval/task.json
```

The [protocol](evaluation-protocol.md#offline-preparer) documents the manifest,
BEIR-style normalized inputs, resource limits, and review requirements. The tool
retains all selected-query judgments and pinned hard negatives before deterministic
random fill; it refuses missing positives, short negative quotas, bad checksums,
and overwriting an existing output. Complete small corpora can omit mined negatives
only with an explicit reviewed completeness assertion. It does not download,
convert upstream formats, mine negatives, or verify their semantic difficulty.
The separate [small BEIR adapter](../benchmarks/datasets/README.md) now verifies
pinned gzip inputs and prepares NFCorpus/SciFact. ArguAna is blocked by five missing
source document IDs in its qrels; the failed audit is retained, not silently fixed.

Corpus format is the checked-in JSON structure: unique document IDs/text, query
IDs/text, nonempty lists of relevant document IDs, and a `domain` tag. An optional
query `relevance` map must assign a positive integer grade to exactly the IDs in
`relevant`; omitted/empty means binary. Top-level `exclude_query_id` defaults to
false for legacy corpora; pin it explicitly for real datasets, especially BEIR
self-matches. Positive qrels conflicting with exclusion are rejected. Ranking
score ties use ascending document ID, independent of corpus order. These policies
are reported as metric version 2; older smoke runs used corpus-order ties. Prepare
manageable, reproducible BEIR, MIRACL language, and code-retrieval subsets (up to
10,000 documents per run), retain dataset licenses, hashes, and selection seeds,
and group metrics by domain. Inputs are truncated to 512 tokens in this harness.
An optional top-level `provenance` JSON object records dataset URLs, revisions,
license references, original hashes and selection seeds. The corpus file's
SHA-256 is reported. IDs must be unique/nonempty, relevance lists must be
nonempty/unique and reference existing documents, and domains must be nonempty;
validation happens before loading inference. Corpus JSON is capped at 128 MiB.
Do not compare smoke scores with published model-card benchmarks.

Before freezing the production revision, run all five planned candidates with
comparable corpora and CPU/GPU artifacts where supported; investigate code or
throughput regressions. Use PLAN.md's 45/20/15/15/5 quality/multilingual/code/local
throughput/footprint weighting. A weighted decision requires comparable measured
inputs; the harness intentionally does not manufacture missing scores.

## Retrieval command latency

`examples/retrieve.rs` measures the production retrieval path on an existing real
indexed corpus. It opens the database read-only, refuses synthetic databases,
and does not ingest, rebuild, or run comparative model evaluations.

```sh
cargo run --release --locked --example retrieve -- \
  --db /path/to/ree.db --query 'writer lock timeout' --mode semantic --repeat 5
cargo run --release --locked --example retrieve -- \
  --db /path/to/ree.db --query 'SQLITE_BUSY' --mode lexical --repeat 5
cargo run --release --locked --example retrieve -- \
  --db /path/to/ree.db --query 'writer lock timeout' --mode hybrid \
  --root /absolute/source/root --repeat 5 --fresh-engine
```

Semantic/hybrid first use may download pinned artifacts; lexical mode never
loads models. Schema 1 supports semantic mode; explicitly run `ree migrate`
with the same database before lexical/hybrid measurements. This backfill is a
separate writer operation, not performed by the harness.

Reports label new versus reused model sessions and separate database open,
tokenizer provisioning/tokenization, model initialization/warmup, query inference,
SQL, hydration, and total elapsed milliseconds. `--fresh-engine` creates a new
model session each iteration, **not** a cold OS/disk cache. Use separate executable
invocations and an explicitly recorded cache protocol to measure cold process
startup; do not infer it from the reused-session measurements. Capture device,
corpus size, filter selectivity, mode, candidate count, and cache state alongside
reports. Query text and retrieved text are not included in benchmark events.

The existing million-vector synthetic scan result is not a new end-to-end search
latency qualification. This harness has no new large-corpus latency or hybrid
quality result attached to its implementation.

## Retrieval exports and mixed-precision/index scoring

The [two-corpus pilot](retrieval-pilot-v1.md) is an explicit exploratory exception
to the full-suite scoring embargo, not a shipping decision. On an otherwise idle
development machine, on AC with the existing balanced profile, start a **new** run:

```sh
cargo build --release --locked --example qualify --example score_compatibility
python3 benchmarks/run-retrieval-pilot.py \
  --scratch /path/to/provisioned-scratch --output /path/to/new-results-directory
```

The runner freezes source/executable/corpus/recipe/bootstrap hashes before inference,
then performs 20 serial encodings and offline six-combination matrices. It polls AC/
profile conditions, terminates its child if they change, and does not change power
policy. Completed hash-matching caches can be reused with `--resume`; incomplete
attempts must be archived separately, never mistaken for finished exports. Preserve
frozen source and executables until a paused run is resumed.

`qualify --quality-only --corpus PATH --export-vectors NEW_DIRECTORY` exports native
f32 little-endian document/query vectors, 32 single-document diagnostic vectors,
ordered IDs, input-token hashes and truncation statistics. Final metadata is the
completion marker; incomplete or corrupt exports fail offline validation. This is
a qualification-cache format, not a new production SQLite or embedding contract.
Dimension truncation happens in the offline scorer, not in native exports:

```sh
target/release/examples/score_compatibility --corpus PATH \
  --cpu-vectors CPU_EXPORT --cuda-vectors CUDA_EXPORT
# The planned Arctic-only projection, from the same 768-dimensional exports:
target/release/examples/score_compatibility --corpus PATH \
  --cpu-vectors CPU_EXPORT --cuda-vectors CUDA_EXPORT --dimensions 256
```

The scorer runs no model/GPU inference and verifies corpus, IDs, token hashes,
recipe/provider compatibility, binary shape/hash and unit vectors. It evaluates
CPU/CPU, CUDA/CUDA, CPU/CUDA, CUDA/CPU, plus both query providers against one
approximately 50/50 mixed-artifact index. Per-query rankings and graded metrics
are retained. The pilot's deterministic SHA-256 bootstrap indices are generated
before inference, shared across all candidates, and hash-identified in the plan.

The [recorded pilot](../benchmarks/results/retrieval-pilot-20260909/README.md) is now
complete: 20 encodings, 12 six-way comparisons and 10,000 paired bootstrap replicates
per corpus. Two power-condition interruptions were preserved separately; resumes
reused verified complete exports with unchanged frozen sources/executables. All
2,000 native homogeneous query rankings/metrics reproduce their live logs exactly.
Eight of 48 mixed-row point checks flag on SciFact; completion does not establish
shipping compatibility. No winner or weighted full-suite score is manufactured.
Do not invoke `--resume` on this completed output or overwrite its summary.

The earlier `partial-analysis/` is retained unchanged as historical evidence.
`benchmarks/analyze-retrieval-partial.py` remains an offline tool for an interrupted
run: it includes only candidates with both corpora/providers complete, explicitly
retains other candidates as pending and does not change the parent runner state.

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
CPU; it was not a CUDA accuracy failure. The later real-corpus pilot observed
Arctic document cosine as low as 0.91572 and mixed-ranking losses despite some high
cosines. The 0.94 smoke bound is therefore not a universal artifact-compatibility
guarantee. See [runtime notes](runtime.md) and the pilot results.
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

This harness measures storage, not million-file discovery or full pipeline
scalability. The production CLI rejects ingestion/rebuild into databases marked
`synthetic_vectors=true` to prevent mixing fake and real vectors.

## Full synthetic pipeline scale

```sh
cargo run --release --locked --example pipeline_scale -- \
  --scratch-dir /tmp --files 1000000 > pipeline-1m.jsonl
```

Development-Ryzen-only, never CI. Creates a private workspace on the requested
filesystem and removes it on success/failure; `--keep` explicitly retains it.
The preflight reserves about 14,000 bytes per file plus headroom. This is an
estimate, not a filesystem quota; ensure sufficient free RAM as well for tmpfs.
The real pinned tokenizer is prewarmed (downloaded/verified if necessary), but
vectors are **synthetic unit vectors**, not inference outputs. Databases are
marked synthetic and must never be used for retrieval.

Measures file creation, the production bounded extraction/tokenization/writer
pipeline, a hash no-op (asserting no embeddings were generated), 0.5% changes and
0.5% deletions, then stored-input rebuild/activation. Sources are removed before
rebuild to prove they are not reopened and to free scratch space for shadow
vectors. Reports phase latency, sampled/lifetime process RSS, document/vector
coverage and checkpointed database size. SQLite can retain freed pages after
retiring old generations; reported size is not a compacted/vacuumed estimate.

The recorded 1M run exercises the full pipeline with bounded process memory,
not real-model throughput, system-wide peak RAM, or SSD behavior. Repeat under
idle-machine conditions before treating its latency as a qualified baseline.
See [recorded results](../benchmarks/results/README.md).
