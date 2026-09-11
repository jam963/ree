# Implementation status

This records implementation and verification, not a declaration that every gate
in PLAN.md is complete. The original design plan is preserved unchanged.

## Speedup v2 follow-up — 2026-09-10

[Speedup v2](speedup-v2.md) implements streaming query reuse, an optional private
Unix-socket supervisor with disposable engine/idle release, conditional strict-CUDA
CPU provisioning, high-resolution stage metrics, scheduler cache/growth safeguards,
and input/vector/statement reuse. Original model pins, singleton query policy,
transactional ingestion and retrieval snapshots remain.

CLS-only graph output and small-document preparation overlap are opt-in experiments,
not default performance claims. CPU overlap failed numerical equivalence and is
excluded (including `auto`); sustained CUDA ingestion timing was unstable. Graph
cache/warmup shortcuts, I/O binding, and a broad writer/queue redesign remain gated,
not silently enabled. See the [measured evidence](../benchmarks/results/speedup-20260910/README.md).
Historical verification counts and audit statements below describe their original
snapshots, not the current follow-up's test/build state.

## Current execution priority

The [beta roadmap](beta-roadmap.md) records the agreed order for remaining work:

1. Select and freeze the shipping model and embedding recipe — **Arctic 768 selected for beta, 2026-09-10**.
2. Improve performance and GPU tuning for that model.
3. Implement bounded large-document handling.
4. Finish and qualify extraction for the beta-supported formats.
5. Audit security and harden the release.
6. Package a versioned installation, then begin hands-on beta testing.

**Next:** continue the remaining beta work with **Arctic M v2, 768 dimensions**.
The user chose to ship Arctic rather than spend more time on broader evaluation;
[model decision v1](model-decision-v1.md) records the existing revision/artifact
pins and deliberately deferred empirical work. Model selection is no longer the
active blocker. Do not start more comparative evaluations unless requested.

The [completed pilot](../benchmarks/results/retrieval-pilot-20260909/README.md)
contains all 20 encodings and 12 six-way comparisons. Its mixed-precision/dimension
regressions remain findings, not waived test failures relabeled as passes. The
full twelve-stratum suite and weighted comparison are deferred for this beta
choice. Existing serialized `qualification: "provisional"` metadata continues to
describe incomplete empirical qualification; the model choice itself is settled.
That model decision itself changed no production recipe, schema or fallback
behavior; the later retrieval schema extension is described below. Correctness,
security, license/notice and packaging safeguards remain in scope. The remaining items
below are an inventory, not a competing execution order.

**Arctic audit, 2026-09-10:** [the focused optimization plan](arctic-optimization-plan-v1.md)
is ready. Offline replay confirms CPU growth after underfilled calls and cached
limits larger than the actual calibration batch. Source review found a redundant
12-MiB host output copy for batch-8/512-token CLS pooling, which retains only
24 KiB of vectors. This is an allocation calculation, not a measured speedup.
The first shape-preserving copy-removal patch is now implemented with nine
[offline output fixtures](../benchmarks/results/arctic-cls-copy-20260910/README.md).
Checks: 111 Rust tests pass (five ignored), 19 Python tests pass; Clippy and
formatting pass. Baseline source and both frozen release executables were
preserved before edits. Runtime output pooling changed; scheduler, normalization,
model recipe and frozen release binaries did not. Cache/growth fixes and short
stage measurements remain separate next steps. No new model inference ran and
no speed improvement has been measured.

## Retrieval implementation verification

Semantic, lexical, and hybrid retrieval are now implemented; see
[retrieval](retrieval.md). Verification: **131 Rust tests passed, six ignored**
with `cargo test --locked --all-targets`; Clippy with warnings denied, formatting,
and `git diff --check` passed. `cargo build --release --locked` produced
`target/release/ree`. An isolated release-binary smoke test initialized schema 2
and returned an offline empty hybrid result; man/completion generation also passed.
No personal database was migrated, no model artifacts were downloaded, and no
real-model/large-corpus performance or quality evaluation was run. The new ignored
CPU query/retrieval smoke test remains unexecuted. No installation or tagged
release was performed.

## Implemented

- Rust library/thin CLI; XDG/TOML/environment configuration; metadata precedence;
  JSONL events and exit codes; TTY-only progress; maintenance commands.
- Bundled SQLite + sqlite-vec, transactional schema v1-to-v2 migration, application
  writer locks plus transactional SQLite writes, source isolation, bounded run
  history, active-generation SQL, and documented public schema. Read-only v1
  compatibility remains for semantic search and diagnostics.
- Local semantic, model-free FTS5 lexical, and reciprocal-rank-fused hybrid search;
  pre-top-k root/media-type filters, stored text/location JSONL, read snapshots
  across ranking/hydration, and explicit model-free `ree migrate` backfill.
  Singleton query initialization skips ingestion throughput calibration/profiles.
  See [retrieval](retrieval.md); this is an explicit extension of the historical
  ingestion-only scope, not a new model-selection or quality/latency qualification.
- Streaming recursive discovery (no ignore rules), globs, files, stdin, encoding
  and binary detection, immutable extraction snapshots, two bounded extraction
  workers, hash-based unchanged fast paths, and cross-document inference batches.
- Exact pinned-tokenizer windows, stored token IDs, checked artifact downloads,
  local Arctic INT8 inference, CLS/L2/768 interoperability metadata.
- Dynamic NVML probing, free-VRAM device selection, CUDA provider preflight and
  profiled warm-up verification, lazy FP16 provisioning, item/padded-token limits,
  conservative probe/profile caching, OOM reduction/session recreation, CPU
  continuation in auto mode, and explicit CUDA refusing silent CPU fallback.
- Atomic complete-document replacements; seen-before-failure tracking;
  conservative deletion suppression on traversal/fail-fast errors; root removal.
- Native text/code/Markdown/structured text, HTML main/article heuristics,
  notebooks; bounded configured/built-in helper execution for PDF, DOCX, ODT,
  EPUB, and image OCR. PDFs preserve text and OCR textless pages sequentially,
  with page byte ranges, configurable limits, and known-helper version metadata.
  Helper workspaces have polled aggregate limits and per-file RLIMIT_FSIZE bounds.
- DNS-pinned bounded HTTP(S), protected-address rejection, no JS; HTTPS shallow
  Git clone/fetch/reset, hook/config/unsafe-transport controls, outside-checkout
  symlink rejection, and repository leases held through ingestion.
- Resumable shadow-generation rebuilding from stored token inputs, keyset
  pagination, coverage checks, transactional activation, and retired-vector
  cleanup. Actual subprocess-kill tests cover replacement and activation commits.
- Local-only candidate qualification and synthetic storage/full-pipeline scale
  tools; CPU/GPU version telemetry, sampled memory peaks, repeated session loads,
  output stability, validated/hash-identified corpora, per-query rankings, and
  optional dimension truncation for qualification. Production CLI ingestion and
  rebuild reject databases marked as containing synthetic benchmark vectors.
- Offline deterministic corpus preparation from hash-locked normalized JSONL/TSV,
  disk-backed ID/qrel validation, complete selected-query judgment retention,
  required pinned hard-negative quotas, bounded retained text, and atomic no-clobber
  output. Qualification preserves graded nDCG and explicit self-match policies.
  This does not yet provide upstream adapters, mined negatives, or real corpus locks.
- Ignored real model/helper tests; CI for ordinary tests/lints; generated
  man/completion tool; MIT/Apache-2.0 license files.

## Verified on the development machine

Date: 2026-09-08. Rust 1.90.0; ONNX Runtime 1.28.0; Ryzen 9 8945HS;
RTX 4070 Laptop 8,188 MiB; NVIDIA driver 610.57.04; installed CUDA runtime 13.3
and cuDNN 9.25. No system packages or CUDA compatibility libraries were installed.

- Ordinary tests pass (57 tests; four real-model/CUDA tests ignored by default),
  including 14 injected CUDA recovery tests and CUDA/NVML UUID ordinal mapping.
- `cargo clippy --all-targets --locked -- -D warnings` and formatting checks pass.
- Release benchmark examples build; man page and shell completions generate.
- Actual pinned CPU artifact download/hash verification, local document ingestion,
  normalized vectors, and multilingual token-window/repeatability test passed.
- An actual automatic run survived an FP16 download failure and committed on CPU.
- The original ONNX Runtime 1.22/CUDA 12 packaging mismatch was resolved by
  upgrading to 1.28/CUDA 13. Real FP16 ingestion now works with the installed
  stack and adjacent provider libraries, without an `LD_LIBRARY_PATH` override.
- Real FP16 CPU-reference/CUDA accuracy passed (minimum observed cosine 0.9999927),
  including multiple batch/padded shapes. INT8/FP16 drift is separately documented:
  observed minimum 0.95009; the original provisional 0.98 bound failed on CPU too.
  A provisional 0.94 quantization bound and smoke top-1 ranking agreement are
  regression checks, not full model-quality qualification.
- Real bounded CUDA arena OOM recovered by reducing 256-item batches to 8, with
  five session-recreating reductions and no CPU fallback. A 1-KiB arena verified
  strict initialization failure and automatic CPU continuation. No GPU reset or
  physical-VRAM exhaustion was attempted.
- The actual CLI's CUDA auto ingestion, hash no-op, change/deletion/rename,
  failed-file vector preservation, stored-input rebuild, and removal passed.
- Release Arctic CPU INT8 and CUDA FP16 harnesses completed 64/256/512-token
  batches 1/2/4/8/16 with five repeats and the checked-in smoke corpus under
  ONNX Runtime 1.28; raw results are recorded separately from the earlier CPU spike.
- Real `CUDA_VISIBLE_DEVICES=''` checks used CPU in automatic mode and returned
  fatal code 3 for forced explicit CUDA. CUDA-visible ordinals map to NVML UUIDs.
- A 1M-chunk synthetic SQLite run completed on tmpfs: 3.79 GB checkpointed size,
  about 196 seconds insertion, and roughly 1.26 seconds per exact top-10 scan.
  This is exploratory storage validation, not complete pipeline/SSD qualification.
- An HTTPS clone/ingestion of `octocat/Hello-World.git` succeeded; a subsequent
  fetch/update was an unchanged no-op without initializing inference. This is a
  remote smoke test, not comprehensive hostile-Git or branch-change coverage.

### Additional verification, 2026-09-09

- Formatting, all-target Clippy, and 78 ordinary all-target tests pass (five
  local model/helper tests ignored by default), including PDF workflow/resource-
  limit fixtures, resumed rebuild gaps, and SIGKILL at
  SQLite's pre-commit hook during actual replacement/activation operations.
  Readers and reopened writers retain the old complete data; checkpointed shadow
  vectors survive and activate successfully on retry.
- All four real CPU/CUDA tests were rerun serially and passed: token windows/
  repeatability, FP16 parity/INT8 drift bounds, bounded-arena OOM recovery, and
  the actual CLI's CUDA synchronization/failure-preservation/rebuild lifecycle.
- Real Poppler text extraction, page mapping, bounded page rendering, version
  probing, and page-limit rejection passed on a generated two-page PDF. Tesseract
  is absent: its missing-tool path passed, but real OCR accuracy is **unverified**.
- Updated release qualification harness ran Arctic CPU INT8 and CUDA FP16 at
  64/256/512 tokens, batches 1/2, three inference repeats and three session loads.
  CUDA reports runtime/driver API 13030 and cuDNN 92500. Samples found no repeated
  output drift; peak observed process VRAM was about 1.39 GB. A separate 256-dim
  smoke evaluation passed. These are instrumentation checks, not quality gates.
- **1,000,000-file full-pipeline synthetic run completed on tmpfs.** Real pinned
  tokenization, fake unit vectors: initial ingestion 161 s, hash no-op 21 s,
  change/deletion synchronization 28 s, stored-input rebuild 488 s. After 5,000
  deletions, all 995,000 remaining chunks had active vectors. Peak process RSS
  was about 346 MB; the checkpointed database occupied 8.36 GB (including freed
  retired-generation pages). Generated files/database were removed afterward.
  Process RSS excludes tmpfs filesystem/page-cache memory. This is not real-model
  inference throughput, a repeated idle-machine benchmark, or SSD qualification.

### Gate-1 preparation follow-up, 2026-09-09

- Added versioned premeasurement selection/aggregation/decision rules and an
  explicit suite-lock checklist; dataset/candidate qualification remains pending.
- Preparation tests cover deterministic hash/query selection, source-order
  independence, graded/zero judgment preservation, hard-negative quotas,
  missing/duplicate references, checksum/resource errors, and no-clobber output.
- Shared scoring tests cover linear graded nDCG, stable ID ties, self-match
  exclusion and invalid qrel/exclusion combinations. Old smoke corpora remain
  readable; new reports identify metric version 2.
- Formatting, all-target Clippy and all-target tests pass: 91 passed, five real
  model/helper tests ignored. An offline CLI fixture also verified manifest-relative
  inputs, graded output, output hashes, no-clobber publication, failure without
  output, and scratch cleanup. No datasets or model artifacts were downloaded and
  no real model benchmarks were run for this follow-up. The protocol is not a
  claim that the actual suite is locked.

### Idle-window candidate provisioning and speed, 2026-09-09

- Provisioned and hash-verified all five candidates: approximately 4.23 GiB of
  model/tokenizer/sidecar/config evidence, reusing the existing Arctic cache.
  No model-supplied Python, package installation, graph conversion, power-policy
  changes or GPU resets. Model-license declarations are pinned; release notices
  and upstream license review are still a separate gate.
- Ten actual CPU/CUDA capability checks pass through the Rust runtime. Added
  explicit qualification-only output selection for Granite's rank-3 `logits`
  tensor, manifest-relative paths, sidecar hashes, strict manifest parsing and
  special-token checks. Production's default output/embedding recipe is unchanged.
- Thirty same-release-build serial speed runs completed with 900 timed inference
  samples, 8 threads and balanced power, at lengths 64/256/512 and batches 1/8.
  Raw samples, source/binary hashes and host/resource snapshots are retained.
  These are inference baselines, not quality scores, ingestion rates or a winner.
- INT8 capability fixtures show mixed-length/individual batch dependence (minimum
  cosine about 0.9754 Arctic, 0.9891 E5, 0.9825 Jina, including empty input).
  Identical-shape timed repetitions were stable. Real ranking consequences and
  mixed CPU/GPU artifacts remain unqualified.
- NFCorpus (3,633 documents) and SciFact (5,183 documents) prepared with 100 queries
  each and all source documents retained. ArguAna preparation failed closed on
  five dangling qrel document IDs; no queries/judgments were silently removed.
  Source locks, normalized/corpus hashes, selected IDs and failure audit recorded.
  Upstream rights review and the rest of the twelve-stratum suite remain open.
- Formatting, all-target Clippy, 93 ordinary Rust tests and nine offline Python
  orchestration tests pass; five real-model/helper tests remain ignored in the
  ordinary run. Python tests use only the standard library, with no downloads or
  real model/GPU work, and are included in CI. The actual capability/speed runs
  above were local-only, not CI.

### Historical partial retrieval/compatibility pilot, 2026-09-09

- The user authorized an explicit, preregistered two-corpus exploratory exception
  to the full-suite scoring embargo. The plan froze input/recipe/source/binary
  hashes and 10,000 paired bootstrap index sets per corpus before inference.
- Added native f32 export with shape/hash/unit-vector checks and final completion
  metadata, exact token-input hashes/truncation audits, and 32-document batch-vs-
  individual diagnostics. Offline Rust scoring rejects incompatible corpus/token/
  recipe/provider bindings, and evaluates four CPU/CUDA pairings plus mixed indexes.
- Twelve encoding runs completed: Arctic, Granite and E5, both corpora/providers.
  AC disconnected/profile changed to quiet during Jina's first run; the watchdog
  terminated it. BGE had not started. Interrupted evidence was archived separately;
  completed exports are retained for guarded resume. No inference remains running.
- Offline partial analysis completed on saved vectors without new model inference.
  All 1,200 native homogeneous per-query rankings/metrics matched their original
  live-scoring results exactly. The partial matrix has 4,800 query rows including
  Arctic 256. Missing Jina/BGE rows are pending, not zeros or implicit exclusions.
- Arctic 768 led the completed candidates on these two tasks, but SciFact's CUDA-
  query/CPU-index loss was 1.19 nDCG points (95% paired interval −0.46..3.35),
  exceeding the pilot's 1-point cutoff as an uncertain point-estimate flag.
  E5's worst SciFact mixed-index loss was 4.70 points (2.55..7.36); Granite's
  same-FP32 provider metrics matched. Arctic 256 lost 2.04 CPU / 1.38 CUDA SciFact
  points versus 768. No shipping compatibility/dimension/winner gate is closed.
- Observed Arctic CPU/CUDA document cosine reached 0.91572, below the historical
  0.94 smoke bound, confirming that smoke/cosine checks did not qualify the CPU
  artifact. Batch dependence and cross-artifact ranking consequences remain
  separate concerns; no production embedding recipe or stored database changed.
- Before inference, formatting, all-target Clippy and 102 ordinary Rust tests passed
  (five real-model/helper tests ignored); fifteen offline Python tests now pass,
  including retained-evidence consistency checks. Actual CPU/CUDA smoke export
  roundtrips reproduced online rankings/metrics exactly and exercised Arctic 256.
  Frozen source/executable hashes were rechecked after the pause and still match
  the plan, so completed work can be reused without changing the measured code.

### Completed retrieval/compatibility pilot, 2026-09-09

- Resumed the same frozen run without rebuilding or changing measured sources,
  binaries, artifacts, corpora, batches or power policy. A second watchdog stop
  occurred when balanced changed to performance; both interrupted attempts remain
  archived separately. After the user restored balanced, all remaining jobs and
  analysis completed at 22:11:22 UTC. No inference remains running.
- All 20 exports validated: 90,800 finite/unit vectors, matching provider token
  hashes and fixed 32-document samples. All 2,000 native homogeneous query rankings
  and nDCG/Recall/MRR values exactly match the live encoding logs. The complete
  matrix has 7,200 query rows; all eight prior partial comparisons reproduce exactly.
- Regenerated the pre-frozen bootstrap draws byte-for-byte and independently
  recomputed all 204 intervals; maximum numerical difference was 1.12e-16.
  Nineteen offline Python tests pass. Rust tests were not rerun/rebuilt during
  continuation; the earlier 102-passed/five-ignored result remains historical.
- Arctic remains highest on the two measured tasks, not a shipping winner.
  BGE-M3 homogeneous nDCG points are 33.83/33.75 NFCorpus and 65.58/65.58 SciFact
  (CPU/CUDA); its same-FP32 mixed losses are at most 0.08 points. Small NFCorpus
  provider ordering changes involve identical text under different IDs and tiny
  f32 score differences; original ID-specific qrels remain untouched.
- Jina Code scores 10.09/9.16 NFCorpus and 19.88/20.13 SciFact; its eight mixed
  point checks are below the cutoff, but intervals do not prove equivalence and
  code retrieval remains unmeasured. Across all native/projected recipes, eight
  of 48 compatibility point checks flag, all on SciFact (Arctic 768: one; Arctic
  256: three; E5: four). The earlier regressions remain unresolved.
- Full tables, interval caveats, interruption/resume history and verification are
  in the recorded pilot. The production model/schema/runtime recipe is unchanged;
  full-suite measurement, license review and weighted selection are still pending.

See `benchmarks/results/` for explicitly labeled local smoke/scale reports when
present. None of these checks freezes the winning model or supplies missing
BEIR/MIRACL/code benchmark results.

## Remaining release gates / known scope differences

1. **Deferred for beta selection by user decision, 2026-09-10:** the remaining
   multi-domain/multilingual/code model comparison and weighted aggregation.
   Arctic 768 is selected, not declared the full-suite measured winner. The
   observed CPU INT8/CUDA FP16 mixed-ranking loss and batch dependence remain
   documented beta limitations. Do not reopen comparative evaluation unless
   requested; functional correctness/recovery requirements remain in scope.
2. Complete cold-process/cache-controlled startup, repeated variance, and true
   CPU/GPU saturation qualification. Runtime versions, sampled peak VRAM/RSS,
   session reload timing, and output repeatability are now instrumented. Sampling
   is a lower bound, not exact peak VRAM. Device discovery, initialization,
   warm-up, OOM, device-loss, strict-mode, and retry/atomicity paths have injected
   coverage and safe real CUDA tests. Profiles remain conservative heuristics
   with empirical checks, not a fully qualified tuner.
3. Tokenization currently runs on the controlled inference side, not parallel
   extraction workers. Individual extracted documents and their chunks/vectors
   are bounded but materialized, not streamed through a disk staging generation.
   Hashing is always performed; there is no stat-only skip shortcut.
4. Qualify real scanned-PDF OCR accuracy with Tesseract installed; add LibreOffice
   conversion workflows, stronger HTML readability/chunk location mapping, and
   broader hostile archive/converter tests. PDF rendering and known-helper version
   probing are implemented. External tools are not OS-sandboxed. Several non-PDF
   resource limits remain fixed rather than fully configurable.
5. Glob expansion creates independent roots, not a persistent pattern deletion
   scope. Remote Git currently requires noninteractive HTTPS `.git` URLs and the
   default branch. Non-UTF-8 path identities are reported unsupported.
6. Rebuild covers the current pinned tokenizer/semantic model contract. General
   cross-revision migration/rechunk orchestration needs explicit compatibility
   design; the implementation rejects mismatched tokenizer revisions rather than
   silently changing stored boundaries.
7. Complete 5M/10M-vector, migration-upgrade, repeated scale/SSD, real-model
   full-pipeline/rebuild, system-wide peak-memory, and latency qualification.
   Million-file synthetic pipeline/rebuild and transactional crash-injection
   checks now pass. Larger vector databases need more scratch capacity: available
   SSD space was about 12 GiB and tmpfs 16 GiB, insufficient for the 10M run.
8. Publish reproducible Linux release archives and third-party runtime notices,
   test installation/upgrade paths, and audit security/resource behavior before
   calling this a production 1.0 release. Structured events are implemented;
   full tracing instrumentation and a richer progress UI remain future work.
