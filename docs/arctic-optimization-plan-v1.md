# Arctic optimization audit and plan v1

Date: **2026-09-10**. Status: **audit complete; first copy-removal patch now
implemented and offline-tested; scheduler fixes and measurements pending**.
Scope: the selected [Arctic 768 beta recipe](model-decision-v1.md), not another
model-selection exercise. The findings and line references below describe the
pre-patch audit; no new model inference or performance measurement ran for it.

**Implementation follow-up, 2026-09-10:** token pooling now borrows the extracted
output and copies/converts only CLS slices. Mean pooling retains per-token
conversion, division and accumulation order without full-tensor materialization.
Nine offline fixtures cover bitwise f32/f16 outputs, padding, shapes, normalization
and the CLS conversion count. Checks: 111 Rust tests pass (five ignored), 19 Python
tests pass, Clippy and formatting pass. Model pins, scheduler and normalization
source are unchanged. Frozen release binaries and a pre-edit source archive were
preserved. No measured speedup or real-model equivalence is claimed; see
[patch evidence](../benchmarks/results/arctic-cls-copy-20260910/README.md).

## Existing evidence: larger batches are not uniformly faster

Rechecked all six Arctic baseline log hashes and all 180 timed samples; all
summary medians reproduce. The audited runtime/scheduler source hashes match the
baseline's recorded versions. Rates below are **embeddings/second**, medians of
three process means, eight threads, balanced power:

| Provider | Tokens | Batch 1 | Batch 8 | Batch-8 rate change |
| --- | ---: | ---: | ---: | ---: |
| CPU INT8 | 64 | 75.64 | 99.10 | +31.0% |
| CPU INT8 | 256 | 21.29 | 19.73 | −7.4% |
| CPU INT8 | 512 | 9.27 | 8.14 | −12.3% |
| CUDA FP16 | 64 | 377.66 | 1,012.15 | +168.0% |
| CUDA FP16 | 256 | 207.58 | 267.64 | +28.9% |
| CUDA FP16 | 512 | 132.61 | 119.87 | −9.6% |

These synthetic-token measurements call `Onnx::infer` directly. They do **not**
exercise production calibration, tokenization, extraction, or SQLite ingestion.
They justify investigating length-aware behavior, not declaring batch 1 globally
optimal or selecting larger defaults. Only batches 1/8 have repeated same-build
evidence here; old smoke probes are not a substitute for a controlled optimum.
See [audit evidence](../benchmarks/results/arctic-scheduler-audit-20260910/README.md).

## Findings

### 1. Automatic growth also affects CPU and underfilled calls — replayed

`src/model/mod.rs:140–154, 232–238` enables growth whenever `batch_size` is unset,
not only for CUDA. Every successful sub-batch increments the growth counter,
regardless of occupancy or measured speed. After 32 singleton CPU calls, the
limits grow from **8 items / 4,096 padded tokens to 9 / 4,352**. An explicit
batch-size-8 control remains at 8 / 4,096.

Thus successful tiny batches are being treated as evidence for a larger limit.
CPU starts without a calibration pass, and INT8 batch shape is already known to
affect vectors. This is not evidence that CPU batching should become singleton;
it is a reason to stop treating blind growth as a qualified tuning strategy.
The replay demonstrates behavior, not an observed OOM or retrieval failure.

### 2. Cached limits can exceed the shape actually revalidated — replayed

`src/model/mod.rs:273–317` accepts cached item limits up to 64, but the default
calibration request is capped at 16. With a cache of **64 / 32,768**, the scheduler
calls the injected runtime once at **16 × 512**, retains the 64-item limit, then
issues **64 × 512** for the first suitable workload. Such a cache can originate
from an earlier explicit larger-batch run; it is not necessarily corrupt.

Also, an accepted cache of **16 / 512** turns a nominal 16-item calibration request
into **16 individual 512-token runtime calls**. Reported request size and actual
inference batch shape are not interchangeable. OOM recovery still exists; this
finding does not establish corruption, but weakens the claimed cache revalidation.

The current CUDA recovery fixture sets `batch_size=Some(4)`; the scheduler unit
fixture disables growth. The ordinary tests pass but do not cover these default
cache/growth paths. The standalone replay intentionally records current behavior;
its assertions are **not** desired-behavior tests to preserve after a fix.

### 3. CLS pooling needlessly copies every token output — source-confirmed

`src/model/onnx.rs:167–193` clones the entire f32 output (or converts the entire
f16 output) before selecting each item's first token for CLS pooling. Both pinned
Arctic graphs expose f32 `token_embeddings`, as recorded in the graph inventory.

At **batch 8 × 512 × 768**, that extra f32 materialization is **12 MiB**, although
the final CLS vectors contain only **24 KiB**. At batch 16 it is 24 MiB versus
48 KiB. These are allocation-size calculations, **not measured speedups**.

A CLS-specific path can read just the first-token slices from the extracted
output view, copy those into the final vectors, and keep normalization unchanged.
Retain the existing mean-pooling behavior and f16 handling for qualification
recipes. This removes ree's extra host copy; it does not claim to eliminate
model computation, runtime-owned outputs or GPU transfers. Do not switch to the
graph's `sentence_embedding` output without separately establishing its contract.

### 4. Smaller, shape-preserving allocation opportunities — source-confirmed

- `src/model/onnx.rs:138–155`: padded ID/mask buffers are allocated and then cloned
  into owned tensors. Ownership can be transferred instead of duplicating them.
- `src/pipeline/sync.rs:235–252`: successful vectors are deep-copied per document
  before being borrowed by the writer. A slice can avoid this copy; the separate
  per-document retry path must retain its owned results and error behavior.
- `src/model/mod.rs:199–220`: sorting and per-batch input clones are additional
  overhead, but changing the runtime input interface is a larger follow-up.

Do not remove finite/unit-vector validation, change accumulation order, or alter
transaction boundaries to save small amounts of work. Start with the output-copy
fix rather than combining every allocation change into one patch.

### 5. End-to-end stage evidence is missing — structural findings, not measured bottlenecks

`src/pipeline/sync.rs:183, 311–414` overlaps two extraction workers, but the consumer
serializes tokenization, inference and document writes. Pending groups flush at
32 documents or after reaching 8 MiB; a group of one-chunk documents cannot supply
more than 32 items even if the scheduler grows a larger item limit. Different
input roots flush separately. The byte threshold is checked after a document is
materialized; it is not a hard bound on an individual large document.

`src/storage/sqlite.rs:183–188, 348–389` updates `last_seen_run` per visited document
and repeatedly executes insertion statements inside per-document transactions.
A hash no-op is therefore not write-free. Statement reuse is worth measuring;
relaxing durability or changing document atomicity is not part of this plan.

The million-file result uses fake vectors on tmpfs: it cannot identify the real
Arctic pipeline's bottleneck or SSD transaction cost. Tokenization concurrency,
SQLite write batching and queue redesign should wait for stage measurements.

`Scheduler::inference_ms` currently includes scheduler calibration calls and
failed inference attempts, but excludes factory/session initialization, sorting,
packing, input cloning and scheduler-side vector checks. It truncates each call
to integer milliseconds. It must not be interpreted as isolated ONNX kernel time
or subtracted from total elapsed time to identify a specific other stage.

## Small implementation sequence

1. **First patch: shape-preserving CLS output-copy removal.** Add offline f32/f16
   fixture tests for exact old/new normalized CLS outputs, multiple items/padded
   lengths, invalid shapes and retained mean-pooling behavior. Keep all model
   pins, token IDs/order, scheduler logic, provider options and normalization
   arithmetic unchanged. Compare fixed batches: faster post-processing can still
   indirectly change timing-based automatic calibration decisions, so this is not
   a promise of identical production grouping under `auto`. This is the lowest-risk
   concrete optimization found.
2. **Second patch: calibration/cache and growth discipline.** Add injected-runtime
   regressions before fixing behavior: effective cached limits cannot exceed what
   was actually exercised; nominal requests must not masquerade as larger runtime
   batches; underfilled successes must not automatically raise limits. Treat CPU
   policy separately rather than inheriting CUDA growth. Preserve explicit user
   caps, low-memory reduction, session recreation, fallback, ordering and atomicity.
   Any changed grouping must be reported as a batch-policy change, not assumed
   numerically neutral for INT8. Do not choose new defaults from the table alone.
3. **Add minimal stage/shape counters, then one short Arctic-only check.** Separate
   initialization/calibration from workload inference; report actual batch sizes,
   real/padded tokens, tokenizer time, write time and queue wait. Concurrent-worker
   CPU/service times must not be added as if they were disjoint wall-clock stages.
   Use only a small fixed local workload and the relevant shapes, before/after the
   patch, with bounded repetitions and a declared time cap. No corpus download,
   full pilot rerun, weighted comparison or broad saturation sweep is needed.

Do not start the real check during this audit. When implementing, preserve a
reviewed baseline source snapshot and the frozen release examples before edits;
new measurements need new source/binary hashes and output directories. Do not
reset, clean, stash or overwrite unrelated dirty work. Real checks remain serial,
on an idle machine with recorded consistent power conditions; no automatic
package installation or GPU reset. For a copy-only change, any unexpected
same-provider/same-shape output mismatch must be investigated, not dismissed as
the already accepted mixed-precision limitation.

Stop after the small changes if they do not deliver a reproducible practical
benefit. GPU I/O binding/output-graph changes, parallel tokenization, cross-root
batching, thread-count searches and writer redesign are **not** part of the first
patch. Bounded large-document staging remains the next separate roadmap item.

## Verification completed for this audit

- `cargo test --locked --offline --all-targets`: **102 passed, five ignored**.
- Offline scheduler replay: four expected observations, including the explicit
  batch-size control; injected runtimes/probes, no model inference.
- Python evidence tests: **19 passed**. Formatting and diff whitespace checks pass.
- All 48 frozen pilot source hashes and both frozen release executable hashes
  still match after the debug test build and offline replay.

No implementation speedup, new recommended batch limit, or new retrieval
compatibility guarantee is claimed.
