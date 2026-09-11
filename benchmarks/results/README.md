# Recorded local exploratory results

These reports were produced on 2026-09-08 using this implementation, Rust 1.90.0
release builds, and the development Ryzen 9 8945HS. Platform profile was `quiet`,
CPU governor `powersave`. They are smoke/scale checks, **not the complete model
qualification gate** and not measurements of other hardware.

- `arctic-cpu-smoke.jsonl`: historical ONNX Runtime 1.22 CPU spike; pinned Arctic
  INT8, actual Rust ONNX path, batches
  1/2/4 at 64/256/512 tokens, three repeats, followed by the six-document,
  five-query `benchmarks/smoke.json` corpus. No compilation or scale harness ran
  concurrently. These tiny quality scores are deliberately easy and must not be
  presented as BEIR, MIRACL, or code benchmark results. Batches were not increased
  to saturation. Load+warm-up time excludes artifact hashing/download.
- `scale-1m.json`: 1,000,000 synthetic normalized 768-dimensional vectors with
  metadata and chunk text. Database was on tmpfs, not the SSD. Approximately
  196 seconds insertion, 3.79 GB checkpointed size, and 1.26-second exact top-10
  scans. The generated synthetic database was removed after recording results.
  The report records its limited concurrency caveat and missing peak-RAM probe.

- `arctic-cpu-ort128-smoke.jsonl` and `arctic-cuda-ort128-smoke.jsonl`: current
  ONNX Runtime 1.28 CPU INT8 / CUDA FP16, batches 1/2/4/8/16, five repeats per
  64/256/512-token length, followed by the same tiny smoke corpus. Runs were
  serial, with no concurrent build or scale workload. CUDA 13.3 and cuDNN 9.25
  were already installed. TF32 was disabled and strict skip-layer norm enabled.
  These runs are not full quality or saturation qualification.
- `cuda13-validation.json`: 57 ordinary tests and four real tests passed,
  including actual CUDA CLI synchronization/rebuild and five OOM reductions from
  256 to 8 items under a 1.25-GiB arena. Same-artifact CPU/GPU FP16 cosine exceeded
  0.99999 on the tested inputs. It separately records the failed initial 0.98
  INT8/FP16 bound and measured quantization drift down to 0.95009.

The initial ONNX Runtime 1.22/CUDA 12 packaging mismatch is resolved: current
builds use ONNX Runtime 1.28 and the installed CUDA 13 stack, without compatibility
packages or library-path overrides. Earlier failed-download/provider cases
verified fallback; current real CUDA and injected recovery tests pass.

## 2026-09-09 instrumentation and pipeline scale

The following release-build runs used the same host/runtime, now with platform
profile `balanced` and CPU governor `powersave`. Runs were serial, without a
concurrent ree build. The interactive host was not placed into a controlled idle
state; load averages and GPU compute-process snapshots are embedded in reports.
No machine-wide caches were flushed. Do not compare latency directly with the
older quiet-mode, unsampled reports above.

- `arctic-cpu-telemetry-smoke.jsonl` and
  `arctic-cuda-telemetry-smoke.jsonl`: batches 1/2 at 64/256/512 tokens, three
  inference repeats and three session recreations, six-document smoke corpus.
  No repeat-to-repeat vector changes were observed. CUDA reported runtime and
  driver API version 13030 and cuDNN 92500. Sampled peak process GPU memory was
  1,386,217,472 bytes; sampling can miss transient peaks. CPU loads were about
  0.32–0.35 seconds; CUDA loads about 0.73–0.79 seconds. These are session reloads
  after artifact hashing, **not cold-cache/process qualification**.
- `arctic-cuda-256-smoke.jsonl`: same tiny corpus, full native inference followed
  by truncation/L2 normalization to 256 dimensions. Smoke top-1 judgments passed.
  This validates the evaluation path, **not the Matryoshka quality-loss gate**.
- `pipeline-1m-synthetic.jsonl`: one million tiny files in 1,000 directories on
  tmpfs. Production discovery/extraction/synchronization/storage/rebuild, actual
  prewarmed pinned tokenizer, explicitly fake unit vectors. File creation 5.43 s;
  initial ingestion 161.03 s; hash no-op 21.39 s; 5,000 modifications and 5,000
  deletions synchronized in 28.39 s; rebuild of the remaining 995,000 stored
  chunks 488.48 s. Sources were removed before rebuild. Vector coverage and
  SQLite quick-check passed. Peak process RSS 345,612,288 bytes; checkpointed DB
  8,357,187,584 bytes, including freed retired-generation pages. The private
  workspace was removed automatically. **RSS does not include tmpfs storage or
  filesystem page cache**. This is a single synthetic pipeline check, not real
  embedding throughput, system-wide peak-memory measurement, SSD latency, or
  proof of 5M/10M-vector scalability.

## 2026-09-09 idle-window five-candidate speed baselines

`candidate-baselines-20260909.json` indexes **30 completed serial process runs**:
three CPU and three CUDA runs per candidate, five timed repeats per shape,
64/256/512-token inputs, batches 1/8. All five used the **same release executable**,
8 inference threads, AC power, balanced platform profile, powersave governor,
ONNX Runtime 1.28 and the installed CUDA 13.3/cuDNN 9.25 stack. No builds,
provisioning or other ree measurements ran concurrently. Desktop/background OS
processes were not disabled; these are user-idle-window observations, not isolated
laboratory or randomized-order measurements. Resource sampling remained enabled.

Median embeddings/second across three process means, **batch 8**:

| Candidate (CPU / CUDA precision) | CPU 64 tokens | CUDA 64 tokens | CPU 512 tokens | CUDA 512 tokens |
| --- | ---: | ---: | ---: | ---: |
| Arctic (INT8 / FP16) | 99.10 | 1,012.15 | 8.14 | 119.87 |
| Granite (FP32 / FP32) | 383.83 | 2,426.07 | 28.57 | 236.26 |
| E5-small (INT8 / FP16) | 347.02 | 4,135.33 | 16.54 | 656.99 |
| BGE-M3 (FP32 / FP32) | 14.41 | 106.76 | 1.57 | 17.38 |
| Jina Code (INT8 / FP16) | 93.68 | 938.96 | 7.02 | 110.40 |

These are **synthetic-token inference rates**, excluding tokenization, extraction,
and SQLite writes, not end-to-end ingestion throughput or a weighted model winner.
Every raw latency sample, per-cell process range, session-load sample, command,
source/binary hash and host/resource snapshot is retained under
`{candidate}-candidate-baseline-20260909/`. Session loads follow artifact hashing;
OS caches were not flushed, so these are not cold-cache startup measurements.
Run order was fixed, not randomized; inspect variation/headroom before treating
medians as supported operating limits. Peak sampled process VRAM was about
1.40/1.21/0.71/3.65/1.16 GiB for Arctic/Granite/E5/BGE/Jina respectively; sampled
peaks can miss transient allocations. No OOM/probe failures or identical-shape
repeat vector drift occurred. No physical VRAM exhaustion or GPU resets attempted.

- `candidate-preflight-20260909/`: ten real Rust tokenizer/graph/provider capability
  checks. Mixed-length versus individual INT8 inference exhibited vector drift;
  see [candidate notes](../candidates/README.md). These checks do not establish
  cross-artifact retrieval compatibility.
- `arctic-baseline-20260909/`: earlier six-run Arctic-only baseline, before adding
  explicit candidate output-name/sidecar support. Retained separately; **not**
  mixed into the same-build five-candidate aggregate.

Pinned artifact recipes are in [candidates](../candidates/README.md). The first
[real corpus preparations](../datasets/README.md) include two successful complete
small corpora and a blocked ArguAna integrity audit; those preparation runs did
not score models.

## Completed retrieval/compatibility pilot, 2026-09-09

A subsequent user request authorized a separately preregistered two-corpus pilot.
See [the detailed assessment](retrieval-pilot-20260909/README.md),
[all metric/interval tables](retrieval-pilot-20260909/metrics.md) and
`retrieval-pilot-20260909/summary.json`. All **20 encoding jobs and 12 six-way
comparisons completed** at 22:11:22 UTC, with 7,200 retained query rows and 10,000
pre-frozen paired bootstrap replicates per corpus. All 2,000 native homogeneous
query rankings/metrics reproduce their live encoding logs exactly. Independently
recomputed intervals and 19 offline Python tests pass.

All five native recipes plus Arctic 256 cover CPU/CPU, CUDA/CUDA, both cross-provider
pairings and approximately 50/50 mixed indexes. Arctic has the highest homogeneous
scores on these two tasks, but SciFact flags its CUDA-query/CPU-index loss of 1.19
nDCG points (95% interval −0.46..3.35). E5's worst mixed-index loss is 4.70 points
(2.55..7.36). Eight of 48 point checks flag, including three Arctic-256 rows.
Granite's same-FP32 provider metrics match; BGE's same-FP32 mixed losses are at most
0.08 points, with small NFCorpus near-tie ordering differences. Jina's mixed point
checks pass but its homogeneous scores are substantially lower on these tasks;
code retrieval was not tested. Passing point checks do not prove equivalence.

Two power-condition interruptions and the earlier `partial-analysis/` remain
archived unchanged. Both resumes used validated complete caches, unchanged frozen
sources/binaries and AC/balanced power; no machine policy was changed. Nothing is
running and this completed output must not be resumed or overwritten. These are
exploratory task-specific results, not a weighted winner or shipping qualification.

Remaining measurements and methodology are in [benchmarks.md](../../docs/benchmarks.md).
