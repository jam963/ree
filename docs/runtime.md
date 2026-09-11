# ONNX Runtime and GPU packaging

`ort`/`ort-sys` are pinned to `2.0.0-rc.13`, using ONNX Runtime 1.28.0.
The initial 1.22 spike targeted CUDA 12; it was replaced so the application works
with the development machine's existing CUDA 13 installation.
The verified Linux prebuilt distribution is obtained by ort-sys during Cargo
build. SQLite and the core ONNX runtime are statically linked. CUDA is optional
at execution time; its provider dependencies are dynamically loaded.

For a CUDA-capable Linux release, put these shared libraries alongside `ree`:

- `libonnxruntime_providers_shared.so`
- `libonnxruntime_providers_cuda.so`

Cargo copies/symlinks these into `target/release/`; **dereference symlinks when
packaging**. The binary has `$ORIGIN` runtime search paths. Also install a
compatible NVIDIA driver and CUDA **13.x** runtime/cuBLAS/cuRAND libraries through
your normal system tooling (cuDNN **9.x** for operators requiring it). CUDA major
versions have different SONAMEs; the runtime must match the installed stack.
The development host's `/opt/cuda` libraries resolve through its existing dynamic
linker configuration, with no environment override or compatibility installation.
ree never installs system libraries. TensorRT/TensorRT RTX are unused and their
provider libraries need not be shipped. Distribute the ONNX Runtime and NVIDIA
license notices appropriate to any libraries you package.

`cargo install --path . --locked` installs the binary and supports CPU execution.
Cargo does not install the adjacent optional provider libraries; copy/package
those separately for GPU support. The [local beta builder/installer](installation.md)
now packages both providers with notices and versioned installation/rollback.
No public production/CI release is published. Cross-compilation, cross-distribution
portability and non-x86_64 Linux qualification remain untested.

## Automatic selection

NVML and the CUDA driver are loaded dynamically, without calling `nvidia-smi`.
CUDA-visible ordinals are joined to NVML devices by UUID, so `CUDA_VISIBLE_DEVICES`
and device ordering cannot silently select or sample the wrong physical GPU.
`cuda:N` names the CUDA-visible ordinal. Unmatched devices (including unsupported
MIG layouts) are ignored conservatively. Accessible devices are
ranked by free VRAM after headroom: max(512 MiB, 15% total VRAM), further bounded
by the configured memory fraction. Devices require compute capability >=7.5 and
more than 1 GiB usable memory as a preliminary filter. Provider registration
then checks actual runtime compatibility before the FP16 artifact is downloaded.

A CUDA session sets `device_id` and arena memory limit, performs a full-length
warm-up, and checks the ONNX profile for executed CUDA nodes. A successful NVML
probe or compiled provider alone is not reported as CUDA activation. A missing
provider/runtime, model-load error, or warm-up failure falls back to the CPU INT8
artifact in `auto`. Explicit `cuda`/`cuda:N` refuses CPU fallback and skips CPU
artifact provisioning/checksums entirely. Automatic mode still provisions rescue
weights first; a later fallback re-verifies them before creating the CPU session.

## Batching and recovery

Inputs are tokenized before scheduling; batches are sorted by token length and
bounded by item count and padded tokens. A GPU warm-up probe grows candidate
limits geometrically, compares measured rates, stops with headroom, and writes a
profile keyed by policy, model/graph variant, precision, GPU UUID, driver, runtime,
and sequence length. The `bounded-calibration-v2` policy clamps cached item/token
hints to the shape actually probed, with an initial cap of 16 (up to 64 explicitly).
Split probes and OOM/low-memory reductions cannot restore unvalidated larger limits.
Calibration checks a five-second budget between candidates; a running inference
call itself is not preempted. Blind post-success growth is disabled on CPU and GPU.
Profiles remain hints, not static memory guarantees or evidence of an optimal batch.
NVML is retained, and free VRAM is sampled before each batch.

OOM discards uncommitted output, reduces both limits, tears down/recreates the
session, and retries smaller batches. A one-item OOM or non-OOM provider failure
switches automatic runs to CPU. Explicit CUDA returns a failure instead. Vectors
are checked for dimension, finiteness, and unit norm before publishing a complete
document. Per-run events record reductions, fallbacks, final limits, and inference
milliseconds. Runtime events also report the thread budget and CUDA runtime,
CUDA driver API, and cuDNN version queries (null when unavailable/not used).
The local qualification harness samples NVML per-process/device memory and RSS;
these sampled peaks are lower bounds, not exact instantaneous allocation peaks.
Cold-cache startup and complete saturation qualification remain release work.

## Reusable search and experiments

Streaming and optional Unix-socket search keep a query-only engine child. Idle
expiry exits/reaps it to release CUDA context allocations; it does not reset the
GPU. `REE_TIMING=1` adds overlapping nanosecond stage counters. Experimental
`REE_CLS_OUTPUT=1` uses a checksum-verified CLS-only derivative with unchanged host
normalization; it is off by default. Preparation overlap is likewise opt-in and
restricted to explicit CUDA after an INT8 equivalence failure. See
[speedup v2](speedup-v2.md) for lifecycle, tests, benchmark gates and remaining work.

## Correctness and recovery tests

CUDA accuracy is checked against **the same FP16 artifact on CPU**, with cosine
>=0.999, plus GPU batch/padding-shape stability >=0.999. Measured minimum cosine
on the current smoke inputs was 0.9999927. TF32 is disabled and strict fused
skip-layer normalization is enabled; these options are recorded in model metadata.

INT8-to-FP16 compatibility is a separate quantization check: the initial provisional
0.98 bound failed, and comparison against CPU FP16 confirmed this was INT8 artifact
drift, not CUDA error. Observed cosines ranged from about 0.950 to 0.983, including
long repeated-token probes. The regression bound is now **provisionally 0.94**,
with unchanged top-1 rankings required for smoke queries. This does not qualify
retrieval quality on arbitrary corpora; the multi-model/corpus gate remains open.

The later [two-corpus pilot](../benchmarks/results/retrieval-pilot-20260909/README.md)
observed Arctic CPU INT8/CUDA FP16 document cosine as low as **0.91572** and a
SciFact CUDA-query/CPU-index nDCG loss of **1.19 points** versus the better
homogeneous reference (paired 95% interval −0.46..3.35). The 0.94 smoke regression
bound is therefore not universal. Functional/atomic fallback recovery remains
separate from retrieval-ranking compatibility; neither the CPU artifact nor all
production batch/fallback shapes are qualified by the smoke tests. The production
recipe is unchanged. The user selected Arctic 768 for beta on 2026-09-10 with
broader evaluation deferred; see [the decision](model-decision-v1.md). This product
choice retains existing fallback behavior but does not clear the observed
mixed-precision ranking risk or qualify every production batch shape.

`tests/cuda_recovery.rs` injects discovery/provider/model/warm-up failures,
multiple GPUs, OOM, device loss, invalid vectors, VRAM shrinkage, strict mode,
and database retries. `tests/cuda_real_oom.rs` additionally triggers real bounded
CUDA arena exhaustion: 256 full-length inputs reduced to batches of 8 under a
1.25-GiB arena, without CPU fallback. An impossible 1-KiB arena checks real strict
initialization rejection and automatic CPU continuation. No GPU reset is attempted.
`tests/cuda_e2e.rs` tests the actual CLI's CUDA ingestion, no-op, partial-failure
preservation, deletion/rename, rebuild, and removal on this machine.

All real-GPU tests are ignored in ordinary CI and must be run serially locally.
See [benchmark methodology](benchmarks.md) and [recorded results](../benchmarks/results/).
