# Arctic scheduler audit — 2026-09-10

**Offline audit and behavior replay, not a new performance or retrieval run.**
The user requested an audit and a small optimization plan before lengthy work.
See [findings and proposed changes](../../../docs/arctic-optimization-plan-v1.md).
No production source, model recipe, database schema or frozen pilot executable
was changed. No model inference, model download or saturation sweep was run.

## Retained evidence

- `baseline-analysis.json`: independently recalculated rates from all six existing
  Arctic process logs (180 timed samples), their source/summary bindings and the
  reviewed source hashes. At 512 tokens, batch 8 has lower median throughput than
  batch 1 on both providers; this is not a global recommendation for singleton
  inference and does not measure the production scheduler or ingestion pipeline.
- `scheduler-replay.rs`: standalone source using the actual production Scheduler
  with an injected runtime/factory/device probe. Profiles and configuration are
  created in private temporary directories, never the user's real cache/database.
- `scheduler-replay.jsonl`: four reproduced observations:
  1. Default CPU limits grow from 8/4,096 to 9/4,352 after 32 singleton calls.
  2. Explicit batch size 8 stays at 8/4,096 under the same calls (control).
  3. Cached 64/32,768 is calibrated at 16×512, retained, then used at 64×512.
  4. Cached 16/512 makes the nominal 16-item probe execute 16 singletons.
- `replay-verification.json`: source/log hashes, actual linked debug-library paths
  and hashes, offline Cargo dependency-discovery command, rustc version/compile
  command and successful replay exit. The standalone temporary binary was removed
  after execution. `compile.*` and `scheduler-replay.stderr` retain diagnostics.

The replay's assertions describe **historical current behavior**, not the desired
policy. Do not add them unchanged as permanent regression expectations after
fixing the scheduler. Reproduction requires the audited source/dependency build;
use the recorded compile command with a new temporary output path. It must not be
mistaken for a model/GPU timing harness. The fake CUDA path can query best-effort
runtime version telemetry, but does not create any ONNX/model inference session.

## Test and integrity checks

- `cargo test --locked --offline --all-targets`: 102 passed, five real model/helper
  tests ignored. This rebuilt debug tests, not the frozen release examples.
- `python3 -m unittest discover -s benchmarks -p 'test_*.py' -v`: 19 passed.
- Formatting and diff whitespace checks passed; all 48 frozen pilot source hashes
  and both release executable hashes still match.

The audit identifies concrete allocation/control-flow opportunities and missing
stage measurements. It does not establish an end-to-end bottleneck, a new batch
optimum, a measured speedup or a new CPU/CUDA compatibility guarantee. Existing
mixed-precision limitations remain governed by the beta model decision.
