# Speedup v2 evidence — September 10, 2026

Implementation/specification: `docs/speedup-v2.md`. This is a measured local
follow-up, not a new model-selection or general retrieval qualification report.
Pre-existing dirty source, the executable and provider libraries were preserved
before implementation. Builds used normal `target/`; no other coding session was
active. Original audit/benchmark reports have not been rewritten.

## Frozen final search measurement

- Before executable: `106b1c41b0844ec32dafb7850be7b55e5fc0f3127c10d74bab23a77c516737f7`.
- After executable: `c48cec7e7b32a260d8a784d469f07f559252dccb57ba8ebef9235d9fd50ba0b0`.
- Final source identity: `source-manifest.json` (includes pre-existing uncommitted
  source; HEAD alone is not a build identifier).
- RTX 4070 Laptop GPU; CUDA FP16, pinned Arctic M v2 / ORT 1.28.0, fixed top ten,
  five existing queries against a private copy of the existing 256-document DB.
- No model download. Strict CUDA, no fallback; one GPU-consuming experiment at a
  time. Fresh-process trials alternate before/after order, four per query per
  variant. Artifacts and OS file caches are warm: **cold here means a fresh process**,
  not a network download, cold disk, or rebooted driver.
- Warm runs: three independent engine lifetimes, 100 sequential queries each;
  results below pool 300 observations, not 300 independent processes. No concurrent
  throughput or large-index claim follows from these samples.
- AC/power profile and GPU activity were sampled during the final search run;
  see `verification.json` and its hashed raw telemetry path. Desktop activity and
  dynamic clocks are not eliminated. Small-sample tail statistics are descriptive.

| Path | n | Median | Upper-order p95 |
|---|---:|---:|---:|
| Fresh standalone before | 20 | 2.401 s | 3.701 s |
| Fresh standalone after | 20 | 2.237 s | 2.461 s |
| Warm stdin JSONL | 300 | 9.51 ms | 13.97 ms |
| Warm persistent socket | 300 | 9.47 ms | 12.69 ms |
| Socket CLI, including client process | 30 | 13.93 ms | 16.92 ms |
| Socket first model response | 3 | 1.884 s | 1.885 s |
| Socket reload after idle | 5 | 1.899 s | 1.956 s |

Socket first/reload excludes worker supervisor startup; standalone and socket CLI
include process completion; persistent paths end at terminal response receipt.
Do not subtract these different scopes as a stage breakdown. Queue wait has a
separate opt-in metric. Each of **681 final search responses** matched the fixed
query's before-build ordered top ten. That is a fixture check, not broad ranking
quality or mixed-precision qualification.

Warm median <25 ms and p95 <50 ms gates pass on this fixture. Cold median improves
about **6.8%**, not the draft's >=10% target. Earlier staged repetitions showed
6–9% rather than a materially different result. CPU rescue elimination is retained
for strict CUDA's explicit no-fallback policy; graph-cache/warmup shortcuts have
not been substituted to manufacture a larger gain.

Five one-second idle/reload cycles removed the engine PID and its GPU allocation,
while keeping the listener available. Measured median observation time was 1.175 s
(includes process-status/NVIDIA polling). Every reload succeeded. This tests full
process teardown on a functioning development-host driver, not a hard guarantee
of timely reaping a wedged kernel/driver.

## Ingestion experiments — intermediate builds, not a final throughput claim

`ingestion-intermediate.json` and `ingestion-parity.jsonl` retain the complete
experiment, including its failure. These were collected earlier in this change;
that intermediate after executable was not frozen separately. They are **not**
represented as final-binary performance qualification.

Fixed batch eight, existing 256-document fixture, isolated databases:

| Fresh process path | Repetitions/variant | Before median | Default after median |
|---|---:|---:|---:|
| Strict CUDA | 5 | 3.7005 s | 3.4627 s |
| CPU INT8 | 3 | 15.0912 s | 15.2035 s |

Default/CLS CUDA vectors were bitwise equal. CUDA preparation-overlap comparisons
passed the selected numerical gate: minimum cosine about 0.99999906, maximum
absolute component difference about 0.00019361. Gate: cosine >=0.99999 and maximum
component difference <=0.0005.

**CPU preparation overlap failed**: one comparison had minimum cosine 0.98348594
and maximum component difference 0.02659188. The attempt is preserved, not omitted
or averaged away. CPU and `auto` are consequently excluded by code and a regression
test; `REE_PIPELINE_OVERLAP=1` can only select eligible explicit-CUDA preparation.
Default CPU batching is still provisional and is not qualified across every larger
workload/grouping by these fixed-batch tests.

Retained-engine CUDA forced-reingestion timings varied substantially between
lifetimes (default approximately 1.8, 2.7, then 13.6 seconds). Those runs reported
no fallback, but did not capture enough contemporaneous resource telemetry to
identify the cause or qualify throughput. CPU trials were also too few for a
benefit claim. **No >=10% sustained-ingestion win is established.** Do not promote
overlap or a writer/queue redesign on these numbers. Large-document/multi-chunk
stress and a stable, fully observed sustained rerun remain work to do.

## CLS-only graph experiment: numerical pass, default-enable no-go

`cls-{cuda,cpu}.jsonl` contain same-provider comparisons at batches 1/8/16 and
lengths 64/256/512, including padding. All nine shapes/provider were bitwise equal,
with finite unit vectors and unchanged host normalization. Independent Python
and Rust wire transforms produce the same pinned derivative hashes.

Qualifier executable SHA-256:
`3034c094930e028cd80113863d1c9c3869a3bff0ff7696cd848165d0a51abf6b`.
It was built before the final **transport-only** control-frame fix; inference,
transform, and qualifier sources did not change afterward. Its source snapshot
is preserved separately with the raw local evidence.

CUDA verification required the new CLS Gather to run on CUDA. The seven residual
CPU warmup nodes were small shape operations; shapes are included in the trace
summary. The trace exposed no explicit transfer events and **did not measure
physical transfer bytes**. `transfer_bytes_measured:false` makes that gap explicit.
Output element counts must not be presented as measured PCIe transfer savings.

Three timing repetitions per shape are a microbenchmark, not a default-enable
gate. Short warm end-to-end queries did not improve with the derivative in the
initial streaming/socket matrix, while additional checksum verification increased
startup. Sustained ingestion was unstable. Therefore `REE_CLS_OUTPUT=1` remains
opt-in pending practical benefit, transfer measurement, and wider shapes/corpora.
No I/O binding, optimized graph cache, CUDA Graphs or TensorRT change is enabled.

## Verification and reproduction

Final offline verification: **151 Rust tests passed; six real-model tests ignored;
20 Python tests passed; Clippy with warnings denied, formatting, diff checks,
release build and CLI documentation generation passed.** Logs are included here.
The last added subprocess regression verifies stdin cancellation is still read
with one executing and two queued searches; extra search frames backpressure
without repeated allocation. Earlier queued search frames can naturally block
later controls in a single byte stream if the producer exceeds admission limits.

```sh
cargo test --locked --offline --all-targets
cargo clippy --locked --offline --all-targets -- -D warnings
cargo fmt --all -- --check
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s benchmarks -p 'test_*.py' -v
cargo run --locked --offline --example generate_docs
cargo build --release --locked --offline --bin ree
```

Development-host-only model checks, with existing artifacts, run providers serially:

```sh
cargo run --release --locked --offline --example qualify_cls -- \
  --cache /path/to/existing/ree-cache --derived-cache /path/to/scratch --device cuda
# Then repeat with --device cpu, never concurrently with the CUDA experiment.
```

`examples/ingest_latency.rs` supports fresh isolated databases and retained-engine
forced trials; ordinary correctness tests remain offline. See source argument
help. Use explicit disposable DB/output paths, not personal indexes.

Local evidence root:
`/home/jacob/.cache/ree-speedup-implementation-GHDkUcOH/`.
It contains preserved source/executables/libraries, incremental implementation
patch, scripts (`baseline.py`, `measure.py`, `ingestion.py`), complete JSONL and
telemetry, failed attempts, source snapshots and logs. `verification.json` hashes
the referenced raw files; `search-identity.json` identifies the final run. Earlier
query runs and the pre-control-fix executable are retained separately. These local
raw files are not bundled in a clone of this repository; small summaries and
parity/test evidence are included here. No persistent personal database was modified.

Remaining gates are explicit in the implementation spec: prolonged slow-client/
contention soak, multi-chunk/large-document bounds and RSS evidence, stable sustained
ingestion, real transfer-byte measurement, broader ranking/precision qualification,
and any later cache/warmup/I/O-binding proposal. This work does not clear beta.
