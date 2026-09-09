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
those separately for GPU support. Release production/CI packaging is not yet
published. Cross-compilation and non-x86_64 Linux qualification remain untested.

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
artifact in `auto`. Explicit `cuda`/`cuda:N` refuses CPU fallback.

## Batching and recovery

Inputs are tokenized before scheduling; batches are sorted by token length and
bounded by item count and padded tokens. A GPU warm-up probe grows candidate
limits geometrically, compares measured rates, stops with headroom, and writes a
profile keyed by model, precision, GPU UUID, driver, runtime, and sequence length.
Profiles are revalidated by inference, not trusted as static memory guarantees.
NVML is retained across batches rather than repeatedly initialized.
The initial probe cap is 16 items (up to 64 for an explicit request); subsequent
stable batches can grow slowly. This is deliberately conservative, not a claim
of local saturation on all workloads. Free VRAM is sampled before each batch.

OOM discards uncommitted output, reduces both limits, tears down/recreates the
session, and retries smaller batches. A one-item OOM or non-OOM provider failure
switches automatic runs to CPU. Explicit CUDA returns a failure instead. Vectors
are checked for dimension, finiteness, and unit norm before publishing a complete
document. Per-run events record reductions, fallbacks, final limits, and inference
milliseconds. Full CUDA/cuDNN version telemetry and peak-VRAM sampling remain
release work.

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
