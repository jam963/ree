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

Remaining measurements and methodology are in [benchmarks.md](../../docs/benchmarks.md).
