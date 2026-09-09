# Implementation status

This records implementation and verification, not a declaration that every gate
in PLAN.md is complete. The original design plan is preserved unchanged.

## Implemented

- Rust library/thin CLI; XDG/TOML/environment configuration; metadata precedence;
  JSONL events and exit codes; TTY-only progress; maintenance commands.
- Bundled SQLite + sqlite-vec, schema v1 initialization/version checks, application
  writer locks plus transactional SQLite writes, source isolation, bounded run
  history, active-generation SQL, and documented public schema.
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
  EPUB, and image OCR. Scanned PDFs without text are reported unsupported.
- DNS-pinned bounded HTTP(S), protected-address rejection, no JS; HTTPS shallow
  Git clone/fetch/reset, hook/config/unsafe-transport controls, outside-checkout
  symlink rejection, and repository leases held through ingestion.
- Resumable shadow-generation rebuilding from stored token inputs, coverage
  checks, transactional activation, and retired-vector cleanup.
- Local-only candidate qualification and synthetic scale tools; ignored real
  model parity tests; CI for ordinary tests/lints; generated man/completion tool;
  MIT/Apache-2.0 license files.

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

See `benchmarks/results/` for explicitly labeled local smoke/scale reports when
present. None of these checks freezes the winning model or supplies missing
BEIR/MIRACL/code benchmark results.

## Remaining release gates / known scope differences

1. Qualify all five candidate models on meaningful, reproducible multi-domain,
   multilingual, and code corpora. Arctic is pinned **provisionally**, not declared
   the measured winner. CUDA 13 correctness/recovery works on the development
   host; complete multi-domain and alternative-model quality qualification is
   still required, particularly given measured CPU INT8 quantization drift.
2. Measure peak VRAM, full runtime/driver/CUDA/cuDNN telemetry, repeated cold/warm
   variance, and true CPU/GPU saturation. Device discovery, initialization,
   warm-up, OOM, device-loss, strict-mode, and retry/atomicity paths now have
   injected coverage and safe real CUDA tests. Current profiles are
   conservative heuristics with empirical checks, not a fully qualified tuner.
3. Tokenization currently runs on the controlled inference side, not parallel
   extraction workers. Individual extracted documents and their chunks/vectors
   are bounded but materialized, not streamed through a disk staging generation.
   Hashing is always performed; there is no stat-only skip shortcut.
4. Add scanned-PDF OCR rendering, LibreOffice conversion workflows, stronger
   readability/location mapping, extractor version probing, and more hostile
   archive/converter tests. External tools are not OS-sandboxed. Several resource
   limits are fixed constants rather than fully configurable settings.
5. Glob expansion creates independent roots, not a persistent pattern deletion
   scope. Remote Git currently requires noninteractive HTTPS `.git` URLs and the
   default branch. Non-UTF-8 path identities are reported unsupported.
6. Rebuild covers the current pinned tokenizer/semantic model contract. General
   cross-revision migration/rechunk orchestration needs explicit compatibility
   design; the implementation rejects mismatched tokenizer revisions rather than
   silently changing stored boundaries.
7. Complete million-file, 5M/10M-vector, crash-injection, migration-upgrade, full
   pipeline/rebuild, peak-memory, and latency qualification. A synthetic vector
   insertion/scan result alone does not meet all scale gates.
8. Publish reproducible Linux release archives and third-party runtime notices,
   test installation/upgrade paths, and audit security/resource behavior before
   calling this a production 1.0 release. Structured events are implemented;
   full tracing instrumentation and a richer progress UI remain future work.
