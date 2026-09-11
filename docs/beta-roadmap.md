# Roadmap to the first installable beta

Execution order agreed on 2026-09-09. This prioritizes the remaining work in
[PLAN.md](../PLAN.md); it does not declare the original plan complete or replace
its correctness requirements. See [implementation status](implementation-status.md)
for what is already implemented and verified.

**Scope extension:** the subsequent user request to build retrieval adds
[semantic/lexical/hybrid search](retrieval.md) and an additive lexical-index
migration. It supersedes this roadmap's original no-search scope, without
reopening model selection or waiving the remaining release gates below.

Work proceeds through the following gates. Fix correctness, data-loss and
security defects whenever discovered rather than waiting for the hardening gate.
Performance changes must preserve deterministic chunking, complete-document
atomicity, source isolation, and CPU fallback semantics.

## 1. Select and freeze the shipping model — selected for beta

**Decision, 2026-09-10:** the user selected **Arctic M v2, 768 dimensions** and
explicitly deferred broader model evaluation. The existing revision, CPU INT8 /
CUDA FP16 artifacts, prompts, pooling and normalization are retained. See the
[beta model decision](model-decision-v1.md) for exact pins and known risks.
The beta model-choice question is closed by that decision, not by passing every
empirical gate. Mixed-precision quality remains a documented limitation.

The following is the **original evaluation plan**, retained for provenance and
possible future work. Its full-suite comparisons are deferred, not prerequisites
to this beta selection; do not start more comparative evaluations unless requested.

- Prepare pinned, hash-verified candidate artifacts and validate tokenizer IDs,
  prompts, output tensors, pooling, normalization, licenses, and CPU/GPU support
  for Arctic M v2, Granite 107M Multilingual, multilingual-e5-small, BGE-M3, and
  Jina v2 Base Code. Use ree's actual Rust runtime, not model-supplied Python.
- Before comparing results, freeze a reproducible evaluation protocol: dataset
  revisions/licenses/hashes, subset selection seeds, qrels, domain balancing,
  metric aggregation, and decision thresholds. Include multi-domain BEIR,
  multilingual MIRACL, code retrieval, and ree-specific technical material.
  Preserve relevant documents and meaningful negatives when selecting subsets.
- Measure quality, baseline CPU/GPU throughput, startup and footprint on the
  development Ryzen 9 8945HS / RTX 4070 Laptop only. Keep workloads and runtime
  settings comparable and record unsupported artifact/provider combinations as
  limitations, not invented scores. Schedule real measurements serially while
  the machine is otherwise idle; check scratch capacity before downloading.
- Investigate Arctic's measured INT8/FP16 drift using retrieval metrics and
  rankings, including CPU/GPU mixed-precision compatibility after fallback.
  Same-artifact FP16 parity alone does not qualify the CPU artifact.
- Evaluate Arctic's 768 versus 256 dimensions on the same meaningful corpora.
  Dimension truncation in the harness is not yet a production storage option.
- Apply PLAN.md's 45/20/15/15/5 quality/multilingual/code/throughput/footprint
  weighting with aggregation rules fixed before results are compared. Document
  material per-domain regressions rather than hiding them in a single score.

**Original evidence-based exit (not fully satisfied; beta choice now recorded
separately):** a checked-in decision backed by raw results; selected model/tokenizer
revision, artifact hashes and precisions, dimensions, prompts, pooling and
normalization pinned in production; correctness/SQL interoperability tests pass.
If the choice changes existing chunk or vector contracts, provide an explicit
migration or reingestion requirement before beta rather than mixing recipes.

Baseline speed measurements belong here because speed affects model selection.
Deep optimization of the selected model belongs in gate 2. The existing smoke
corpus and synthetic vectors cannot establish full empirical model qualification.

**Progress, 2026-09-09:** [evaluation protocol v1](evaluation-protocol.md) now
specifies strata, deterministic selection, negative retention, aggregation and
decision rules. The offline `prepare_corpus` tool verifies normalized input hashes,
preserves graded judgments and pinned hard negatives, and publishes reproducible
corpora without overwriting existing files. Qualification now supports graded
nDCG and explicit self-match exclusion. Unit fixtures pass; these are preparation
mechanisms, not real corpus or model qualification.

**Idle-window follow-up, 2026-09-09:** all five [candidate artifact recipes](../benchmarks/candidates/README.md)
are now downloaded/hash-verified and load on CPU/CUDA through ree. Thirty serial
same-build speed runs completed (three per candidate/device), with raw results in
[the results inventory](../benchmarks/results/README.md). NFCorpus and SciFact
preparation passed; ArguAna is blocked by five missing qrel documents, preserved in
[the dataset audit](../benchmarks/datasets/README.md). No winner was declared by
these speed/preparation runs.

**Retrieval pilot, 2026-09-09:** the user authorized a separate, explicitly scoped
[two-corpus pilot](retrieval-pilot-v1.md) before completing the full suite lock.
Native vector exports, all CPU/CUDA pairings, mixed indexes, Arctic-256 projection,
and paired bootstrap analysis are complete for all five candidates: 20 encodings,
12 six-way comparisons and 7,200 query rows. [Completed results](../benchmarks/results/retrieval-pilot-20260909/README.md)
show Arctic leading on these two tasks, but mixed-recipe compatibility is not
cleared: SciFact flags Arctic INT8/FP16, E5 and Arctic 256. Granite's same-FP32
provider metrics match; BGE has small same-FP32 ordering differences and mixed
losses below the cutoff. Jina's mixed point checks pass but absolute quality is low
on these two tasks; code retrieval remains unmeasured. Both power interruptions
and all reused exports are preserved and verified. At pilot completion on September
9, no model or dimension choice had been frozen. The September 10 decision above
selects Arctic without turning these limited measurements into a full-suite win.

**Immediate next work:** continue the remaining beta work with Arctic 768, beginning
with the selected-model work below. No further comparative model evaluation is
required for the beta choice. The missing strata/mining/ree set/weighted aggregation
and further mixed-recipe characterization are deferred, with unchanged thresholds
and retained failure evidence. ArguAna must not be repaired by dropping qrels if
that work is revisited. Correctness, security, model/runtime notices and installation
safeguards remain in scope. Methodology is retained in [benchmarks.md](benchmarks.md).

## 2. Improve performance and GPU tuning

**Audit complete, 2026-09-10:** [Arctic optimization plan v1](arctic-optimization-plan-v1.md)
records the existing timing evidence, replayed cache/growth issues and a
shape-preserving CLS output-copy opportunity. The first copy-removal patch is now
[implemented and offline-tested](../benchmarks/results/arctic-cls-copy-20260910/README.md);
no new model inference or speed measurement has been performed. Next: address
calibration/growth regressions separately, then use short targeted stage
measurements rather than a broad sweep.

**Goal:** tune the selected shipping recipe against measured bottlenecks.

- Profile extraction, tokenization, inference, writes, hash no-ops, and rebuilds
  separately and end to end; distinguish cold processes from warm sessions.
- Measure CPU thread/batch saturation and GPU item/padded-token limits across
  short, medium, full-length, and mixed-length workloads. Record repeated-run
  variance, sampled peaks, and headroom without deliberately resetting the GPU.
- Improve scheduling, profile calibration/revalidation, tokenization concurrency
  and database batching where measurements justify the changes. Keep queues
  bounded and avoid competing CPU thread pools.
- Revalidate OOM reduction, session recreation, device-loss recovery, explicit
  CUDA failures, automatic CPU continuation, and retry/atomicity behavior.

**Exit:** reproducible before/after measurements for the selected model; justified
CPU/GPU defaults and profiles; no quality or synchronization regressions. Record
safe supported operating limits, not throughput extrapolations to other hardware.

## 3. Handle large individual documents

**Goal:** document length must not require retaining every chunk and vector in
RAM before a replacement can be committed.

- Introduce bounded staging/streaming for extracted text, token inputs and vectors
  where needed. Preserve exact tokenizer windows, overlap, offsets and ordering.
- Keep the old document active until the staged replacement is complete and
  validated; clean up or safely resume interrupted staging.
- Make relevant size/queue/staging limits configurable and fail clearly on disk
  exhaustion, oversized converter output, interruption or late-page failure.
- Test huge single documents as well as many small files, with process and
  system-wide memory/disk measurements. Do not equate the existing million-tiny-
  file result with large-document qualification.

**Exit:** bounded working memory under the configured limits, deterministic chunks,
and tested interruption/failure behavior that never exposes a partial document.

## 4. Finish extraction for the beta-supported formats

**Goal:** a tested, documented support matrix rather than implicit promises based
on installed helper names.

- Test real image/scanned-PDF OCR, including mixed PDFs and failed/blank pages.
  Tesseract is currently missing on the development host; helper installation
  must be explicit, never automatic behavior of ree.
- Add and validate the planned LibreOffice office conversion workflow with safe
  helper configuration and the same resource/failure controls.
- Improve HTML readability and mapping of extracted page/cell/section locations
  to chunks without modifying stored text merely to simplify offsets.
- Expand multilingual/encoding, malformed-document and helper-availability tests.
  Document supported extensions, required tools, warnings and resource limits.

**Exit:** supported formats have real representative fixtures and failure tests;
unsupported formats and missing helpers fail clearly while retaining prior data.

## 5. Security and release hardening

**Goal:** a dependable local beta with explicit trust boundaries.

- Audit URL/DNS/redirect protections, Git hooks/config/transports/symlinks, helper
  argument handling, converter/archive outputs, model checksums and temporary
  file/staging behavior. Resolve sandboxing policy rather than presenting process
  groups or byte limits as a sandbox.
- Test resource exhaustion, writer conflicts, crash recovery, migrations and
  backward-incompatible recipe/schema handling against the chosen beta contract.
- Repeat realistic pipeline/rebuild and storage tests on disk, not just tmpfs;
  complete planned scale measurements as capacity permits and record remaining
  limits honestly. Arrange adequate scratch capacity before 5M/10M-vector runs.
- Review dependency/runtime licenses and notices, JSON/exit-code contracts,
  diagnostics, backup instructions and known limitations.

**Exit:** no known unaddressed data-loss or critical security defects in supported
workflows; a reviewed test/support matrix and explicit remaining qualification
limits. Any gate deferral needs an explicit beta-scope decision, not a claim of
full PLAN.md or production-1.0 completion.

## 6. Build the release installation, then beta test

Only after the preceding gates:

- Commit and tag the selected pre-1.0 version; build versioned Linux archives from
  the locked source/runtime artifacts, including checksums and required notices.
- Bundle the executable and adjacent optional CUDA provider libraries. Supply
  a versioned system-wide installer, uninstall instructions, and upgrade/rollback
  behavior that leaves per-user databases/configuration/model caches untouched.
- Test the unpacked installation independently of the build tree and Cargo cache
  paths, including CPU-only startup and CUDA operation with the documented stack.
- Include generated man/completion files and a short beta checklist: initial
  ingestion, no-op, changes/deletions, failed-file preservation, rebuild, and
  diagnostics on representative local inputs.

**Exit:** an installable, version-locked beta for hands-on testing on the user's
machine. Installation rollback must not be confused with database rollback;
schema compatibility and backup requirements must be explicit.

## Scope discipline

This sequence does not add a search command, model marketplace, or new platform
matrix. Git authentication/branch selection, persistent glob deletion scopes,
non-UTF-8 path support and general user-selectable model migration remain tracked
limitations; they are not automatically added to the selected-model beta work. Review
whether any must block the beta at the hardening gate rather than silently
expanding or discarding scope now.
