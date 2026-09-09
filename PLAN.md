# ree implementation plan

> **Status:** design plan; the on-disk schema may change before `ree 1.0`.
> **Research snapshot:** 2026-09-08.

## 1. Product definition

`ree` is a Linux-first, local-first command-line embedding tool. It accepts files, directories, globs, stdin, one-page URLs, and Git repositories; extracts meaningful text; splits each document into overlapping token windows; generates embeddings with a local model; and synchronizes the chunks and vectors into a user-global SQLite database.

`ree` is responsible for ingestion and embedding, not retrieval. The database schema and vector index are nevertheless a documented public interface so agents, scripts, and other applications can query them directly.

The primary interface is intentionally terse:

```bash
ree ./docs
ree document.pdf
ree 'src/**/*.rs'
ree https://example.com/article
ree https://github.com/org/repository.git
cat document.md | ree -
```

The design priorities, in order, are:

1. Correct, deterministic embeddings and synchronization.
2. High ingestion throughput on both CPU and GPU.
3. Zero-configuration operation.
4. Machine-friendly behavior for agents and scripts.
5. A small operational footprint: one executable, one database, no daemon.
6. Broad text extraction without embedding meaningless binary data.

## 2. Decisions established by the design interview

- Linux is the initial supported platform.
- The zero-configuration command is `ree <input>`.
- Inputs include files, recursive directories, globs, stdin, URLs, and local or remote Git repositories.
- `ree` attempts every discovered file; it does not apply `.gitignore` or conventional directory excludes. Unsupported and binary files are detected, reported, and skipped.
- The initial chunker is a 512-token window with a 64-token overlap. It never crosses document boundaries and preserves exact extracted text.
- Inference is local. One opinionated model is supported initially.
- GPU acceleration is selected automatically and falls back safely to CPU.
- Data lives in one user-global SQLite database; there are no user-facing collections.
- Re-ingestion replaces changed content and removes confirmed deletions within the same source root. It never deletes unrelated inputs.
- Failures are reported and processing continues by default. `--fail-fast` is available.
- Default output is machine-readable. Human progress bars are opt-in.
- Maintenance commands manage sources and embeddings but do not perform retrieval.
- Configuration and data locations follow the XDG base-directory specification.
- The public schema is versioned and automatically migrated, but remains unstable until 1.0.
- The project will use the conventional Rust dual license, `MIT OR Apache-2.0`.

## 3. Embedding-model research and selection

### 3.1 Evaluation criteria

The default model must balance:

- English and cross-lingual retrieval quality.
- Performance across web prose, documentation, technical material, academic text, and source code.
- CPU throughput and startup footprint.
- GPU throughput and practical VRAM requirements.
- Permissive licensing.
- Stable local inference from ONNX without executing model-supplied Python.
- Model artifacts small enough for a zero-configuration download.
- A 512-token minimum context window and deterministic tokenizer.
- Reasonable vector dimensions for millions of chunks.

No public benchmark perfectly covers all of ree's domains. MTEB/BEIR is the strongest general-domain signal, MIRACL/CLEF covers multilingual retrieval, and CoIR/CodeSearchNet-style tests are needed for source code. Published throughput numbers are generally not comparable because hardware, sequence length, precision, and batch size differ. Parameter counts and active transformer parameters are therefore only speed proxies; ree must run a reproducible local bake-off before the model revision is frozen.

### 3.2 Candidate comparison

The most relevant candidates are:

| Model | General and multilingual quality | Speed/size characteristics | Context / dimensions | Assessment for ree |
|---|---|---|---|---|
| **Snowflake Arctic Embed M v2.0** | Its model card reports BEIR 55.4, MIRACL-4 55.2, focused CLEF 51.7, and full CLEF 53.9. It leads the card's same-protocol comparison on BEIR and CLEF while staying close to BGE-M3 on MIRACL. | 305M total parameters but only 113M non-embedding parameters. Official ONNX artifacts are about 311 MB INT8 and 613 MB FP16. The relatively shallow compute body is favorable for throughput. | 8,192 tokens; 768 dimensions, with Matryoshka truncation to 256 at roughly 2–3% loss in the card's tests. | **Best overall fit:** strongest balance of multi-domain English, multilingual quality, permissive license, efficient compute, long context, and ready CPU/GPU ONNX artifacts. |
| Alibaba GTE Multilingual Base | Same-card comparison: BEIR 51.1, MIRACL-4 52.3, focused CLEF 47.7, full CLEF 53.1. More than 70 languages. | 305M total / 113M non-embedding parameters; similar efficient architecture. Official repository has a roughly 611 MB weights artifact, but Arctic provides a cleaner ready-made ONNX precision matrix. | 8,192 tokens; 768 dimensions with elastic representations. | Strong runner-up, but Arctic improves most reported general-domain scores without a compute penalty. |
| BAAI BGE-M3 | Same-card comparison: BEIR 48.8, MIRACL-4 56.8, focused CLEF 40.8, full CLEF 41.3. Supports more than 100 languages and dense, sparse, and ColBERT outputs. | 568M total / roughly 303M non-embedding parameters and 1,024-dimensional vectors. Materially heavier inference and storage. FastEmbed's default quantized BGE-M3 artifact is CPU-oriented and cannot be passed to CUDA. | 8,192 tokens; 1,024 dimensions. | Excellent multilingual specialist, but too heavy and operationally awkward for ree's dense-only, automatic CPU/GPU default. |
| Nomic Embed Text v2 MoE | Its model card reports BEIR 52.86 and strong multilingual retrieval. It supports roughly 100 languages and open training data. | 475M total / 305M active parameters, custom MoE implementation, and a roughly 1.9 GB published weights artifact. Rust support currently uses a specialized Candle path rather than the simplest ONNX path. | 512 tokens; 768 dimensions, truncatable to 256. | Strong quality but a larger download, shorter context, more compute, and greater runtime complexity than Arctic. |
| IBM Granite Embedding 107M Multilingual | Model card reports MIRACL-18 55.9 and broad technical training data, including Stack Exchange and scientific corpora. | 107M parameters, 384-dimensional output, and an approximately 428 MB ONNX artifact. IBM reports it as twice as fast as comparable-dimension models, though this is not an independent apples-to-apples measurement. | 512 tokens; 384 dimensions. | Best speed-oriented fallback and a good technical-domain candidate, but lower reported English retrieval quality and no long context. |
| multilingual-e5-small | Mature, widely deployed multilingual baseline with strong Mr. TyDi results and more than 12 million recent Hugging Face downloads. | 118M parameters; official ONNX artifacts include approximately 118 MB INT8 and 470 MB FP32. Fast and well supported by FastEmbed Rust. | 512 tokens; 384 dimensions. | Safest minimal implementation fallback, but older, shorter-context, and lower-capacity than Arctic. |
| Jina Embeddings v2 Base Code | Explicitly trained for English and 30 programming languages; intended for technical QA and code search. | 161M parameters and permissive Apache-2.0 license. | 8,192 tokens; 768 dimensions. | Best code-specific candidate, but it does not satisfy ree's broad natural-language requirement as a single default model. |
| Jina Embeddings v3 | Strong multilingual/task-adapter design. | Large model and `CC-BY-NC-4.0` license. | 8,192 tokens; 1,024 dimensions with truncation. | Rejected because the non-commercial license is unsuitable for a general-purpose CLI default. |
| Nomic Embed Text v1.5 | Good English MTEB results, long context, Matryoshka dimensions, and mature ONNX support. | About 137M parameters. | 8,192 tokens; 768 down to 64 dimensions. | Efficient English option, but not a broad multilingual default. |

Benchmark numbers above are model-card results, not a claim that every row was evaluated by one independent harness. The Arctic/GTE/BGE comparison is especially useful because those values were published together under the same table. Scores from the Granite and Nomic cards use different MIRACL task sets or evaluation revisions and must not be compared numerically without rerunning them.

### 3.3 Selected default

Use **`Snowflake/snowflake-arctic-embed-m-v2.0`**, pinned to an exact Hugging Face revision.

Use two official artifacts with equivalent output semantics:

- **CPU:** INT8 ONNX, approximately 311 MB.
- **CUDA:** FP16 ONNX, approximately 613 MB.

Use normalized dense embeddings. Start with the full 768 dimensions because SQLite storage remains manageable and it preserves maximum quality. Before 1.0, benchmark the 256-dimensional Matryoshka representation against 768 dimensions; 256 dimensions cuts vector storage from 3,072 to 1,024 bytes per chunk before database overhead and may be a better million-chunk default if the measured quality loss stays near the published 2–3%.

Although the model accepts 8,192 tokens, ree's default remains 512-token chunks with 64-token overlap. This controls latency, produces useful retrieval granularity, and keeps GPU batches efficient. Long context remains available for future chunking policies.

Document embeddings must use the model's passage/document prompt convention exactly as specified by the pinned model revision. The prompt is applied only to model input; it is not added to stored chunk text. The database records the prompt and pooling/normalization recipe so external query embedders can remain compatible.

### 3.4 Model qualification gate

Model qualification is deliberately local. It evaluates models only through ree's actual Rust inference path on the development machine—not on rented hardware, CI hardware, or a synthetic matrix of GPU sizes. The currently detected qualification hardware is:

- AMD Ryzen 9 8945HS, 8 cores / 16 threads.
- 30 GiB system RAM.
- NVIDIA GeForce RTX 4070 Laptop GPU with 8,188 MiB VRAM and compute capability 8.9.
- NVIDIA driver 610.57.04 at the time of this hardware probe.

Before freezing the default revision, run Arctic M v2.0, Granite 107M Multilingual, multilingual-e5-small, BGE-M3, and Jina v2 Base Code on this machine. Use the exact model artifacts and Rust runtime that ree would ship. Do not extrapolate throughput to other CPUs or GPUs.

The local benchmark should measure:

- Retrieval quality on manageable BEIR subsets representing web, biomedical, scientific, financial, and argumentative text.
- Multilingual quality on a manageable set of MIRACL languages spanning different scripts.
- Code retrieval on a manageable CoIR or CodeSearchNet subset.
- Quality on a ree-specific corpus containing documentation, configuration, logs, source code, Markdown, HTML, and multilingual prose.
- CPU throughput on the Ryzen 9 8945HS at token lengths 64, 256, and 512 and batches 1 through local saturation.
- CUDA throughput and peak VRAM on the RTX 4070 Laptop GPU at those same token lengths.
- The largest stable CPU and CUDA batches on this machine.
- Cold model-load time, warm startup time, output stability, and downloaded artifact size.

Run benchmarks with the machine otherwise idle and record power mode, driver/runtime versions, artifact precision, thread count, batch size, token count, and repeated-run variance. The resulting numbers characterize only this machine.

Selection weighting should be 45% retrieval quality, 20% multilingual quality, 15% code/technical quality, 15% locally measured throughput, and 5% footprint/operational simplicity. Arctic is the planned default; replace it only if this local bake-off shows a material regression in code retrieval or throughput on this machine.

### 3.5 Research sources

- Snowflake Arctic Embed M v2.0 model card and benchmark table: <https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0>
- Alibaba GTE Multilingual Base model card: <https://huggingface.co/Alibaba-NLP/gte-multilingual-base>
- BAAI BGE-M3 model card: <https://huggingface.co/BAAI/bge-m3>
- Nomic Embed Text v2 MoE model card: <https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe>
- IBM Granite 107M Multilingual model card: <https://huggingface.co/ibm-granite/granite-embedding-107m-multilingual>
- multilingual-e5-small model card: <https://huggingface.co/intfloat/multilingual-e5-small>
- Jina Embeddings v2 Base Code model card: <https://huggingface.co/jinaai/jina-embeddings-v2-base-code>
- Jina Embeddings v3 model card: <https://huggingface.co/jinaai/jina-embeddings-v3>
- Nomic Embed Text v1.5 model card: <https://huggingface.co/nomic-ai/nomic-embed-text-v1.5>
- FastEmbed Rust supported models and GPU caveats: <https://github.com/Anush008/fastembed-rs>
- MTEB benchmark project: <https://github.com/embeddings-benchmark/mteb>
- ONNX Runtime CUDA execution-provider options: <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html>

## 4. GPU discovery, VRAM-aware batching, and fallback

GPU support is a first-class part of the pipeline rather than a compile-time afterthought.

### 4.1 Device selection

Expose:

```text
--device auto|cpu|cuda|cuda:N
--batch-size N
--gpu-memory-fraction FLOAT
```

Defaults are `--device auto`, adaptive batch size, and a conservative GPU memory fraction.

On Linux, `auto` should:

1. Dynamically load NVML (`libnvidia-ml.so.1`) without making the CLI depend on an installed `nvidia-smi` executable.
2. Enumerate NVIDIA devices and read device name, UUID, compute capability where available, total VRAM, and current free VRAM.
3. Ignore devices that cannot support the packaged ONNX Runtime/CUDA combination.
4. Prefer the compatible device with the most usable free VRAM, not simply device zero.
5. Reserve headroom for the display server and other processes: at least 512 MiB and normally 15% of total VRAM, whichever is larger.
6. Attempt to create an ONNX Runtime CUDA execution-provider session using the FP16 model and an explicit `device_id` and `gpu_mem_limit`.
7. Verify provider activation rather than assuming that successful library discovery means the graph is running on CUDA.
8. Fall back to the CPU INT8 session if NVML, CUDA, cuDNN, the execution provider, the model load, or the warm-up inference fails.

GPU detection and fallback produce structured events but are not fatal in `auto` mode. An explicit `--device cuda` request fails rather than silently using CPU, which prevents surprising scheduled-job performance.

### 4.2 Runtime packaging

Use ONNX Runtime through a narrow internal runtime abstraction. Evaluate `ort` directly and FastEmbed Rust during the first implementation spike. FastEmbed is attractive for downloading, tokenization, and standard model plumbing, but ree must be able to load the pinned custom ONNX path and configure the CUDA provider fully. If FastEmbed obscures provider verification, memory limits, or recovery, use `ort` directly for inference and a dedicated tokenizer crate.

Linux release artifacts must document their CUDA/cuDNN compatibility. CUDA libraries should be dynamically loaded so the same ree binary still runs on CPU-only hosts. A missing or incompatible CUDA stack must never prevent CPU startup in automatic mode.

Cache both model artifacts only when useful: download CPU INT8 first; download FP16 lazily when a compatible CUDA device is found. Verify checksums before activation and keep model downloads crash-safe through temporary files and atomic rename.

### 4.3 Adaptive batching

Batching must account for both document count and padded token count. A batch of 256 short snippets is very different from 256 full 512-token chunks.

The scheduler should:

- Tokenize before batching.
- Bucket chunks by approximate token length to minimize padding.
- Bound each batch by `max_items` and `max_padded_tokens`.
- Keep a bounded queue so extraction cannot exhaust RAM while inference is busy.
- Use separate calibrated limits for CPU and each detected GPU.
- Maintain one inference worker per selected GPU initially; concurrent sessions on one GPU commonly waste VRAM and reduce predictability.

For a previously unseen GPU/model pair:

1. Compute a conservative initial batch from free VRAM, fixed model/session memory, sequence length, output size, and reserved headroom.
2. Run a warm-up/probe batch without committing its output.
3. Increase the padded-token budget geometrically while latency improves and reserved headroom remains.
4. Stop increasing before the memory limit is approached.
5. Persist the successful profile keyed by model revision, artifact precision, GPU UUID/model, driver/runtime version, and maximum sequence length.
6. Revalidate the cached profile cheaply because other processes can change available VRAM.

Do not rely on a formula alone. ONNX Runtime arenas, kernels, and temporary buffers make empirical probing necessary. Set the CUDA provider's `gpu_mem_limit`, while recognizing that ONNX Runtime documents this as an arena limit rather than a guarantee on total process GPU memory.

During ingestion, periodically sample free VRAM. Reduce the padded-token budget if available headroom shrinks. Increase it slowly only after a sustained stable interval. A user-specified `--batch-size` disables growth but not emergency OOM reduction unless strict behavior is explicitly added later.

### 4.4 CUDA error handling

Classify failures into:

- Out of memory/resource exhaustion.
- Invalid or incompatible CUDA/cuDNN/provider setup.
- Device lost/reset.
- Model or graph incompatibility.
- Unknown ONNX Runtime failure.

On inference OOM:

1. Discard the failed batch output.
2. Halve the padded-token and item limits.
3. Recreate the session if required to release the provider arena.
4. Retry the same batch once at the smaller size.
5. If a one-item batch still fails, tear down CUDA and continue that run on CPU in `auto` mode.

On device-lost or repeated provider errors, do not repeatedly retry CUDA. Tear down the GPU session, emit a `device_fallback` event, initialize CPU INT8, and resume from the first uncommitted chunk. Deterministic chunk IDs and transactional batch writes make retries idempotent.

Example event:

```json
{"type":"device_fallback","from":"cuda:0","to":"cpu","reason":"cuda_out_of_memory","retry_batch_items":16}
```

CUDA failure must not corrupt or partially replace a document. A document's new chunks become active only after all of its embeddings are available and validated for dimension and finite values.

### 4.5 GPU correctness and observability

Record per run:

- Selected provider and device.
- GPU name and total/free VRAM at startup.
- ONNX Runtime, CUDA, cuDNN, and driver versions where available.
- Model artifact precision.
- Initial and final batch limits.
- Number of OOM reductions and provider fallbacks.
- Tokenization, inference, and database throughput.

Tests must compare normalized CPU INT8 and GPU FP16 vectors within a documented cosine-similarity tolerance and verify that their nearest-neighbor rankings are acceptably stable.

## 5. Storage

Use bundled SQLite with `sqlite-vec` at:

```text
~/.local/share/ree/ree.db
```

Reasons:

- No daemon or external service.
- Transactional synchronization and migrations.
- Portable, inspectable single-file database.
- Direct SQL access from agents and applications.
- Efficient batch writes and metadata indexing.
- A vector SQL interface despite ree not implementing retrieval.

Put storage behind an internal Rust trait, but implement only SQLite initially. Benchmark `sqlite-vec` exact vector scans at 1M, 5M, and 10M chunks. If query latency is unacceptable, keep SQLite as the metadata/source-of-truth layer and add an optional ANN backend later rather than prematurely adding a service dependency.

Use XDG paths:

```text
~/.config/ree/config.toml
~/.local/share/ree/ree.db
~/.cache/ree/models/
~/.cache/ree/repositories/
```

Configuration precedence is CLI, environment, config file, then built-in defaults.

## 6. Public database schema

The schema is versioned, documented, migrated automatically, and treated as a public compatibility surface. Indicative tables are:

```text
schema_metadata
models
source_roots
documents
chunks
embedding_generations
embeddings
runs
run_failures
```

### Models

Store the pinned model ID/revision, tokenizer revision, artifact hash and precision, dimensions, distance metric, prompts, pooling, normalization, maximum tokens, and creation time.

### Source roots

Store stable root ID, source kind, canonical identity, source options, user metadata, and last attempted/successful synchronization.

### Documents

Store stable document ID, source-root ID, canonical absolute path or URI, content and extracted-text hashes, media type, extractor/version, file size/times, metadata, and last-seen run.

### Chunks

Store stable chunk ID, document ID, ordinal, exact extracted text, token offsets/count, chunk hash, and location data such as page, notebook cell, or HTML section when available.

### Embeddings

Associate chunk IDs and embedding-generation IDs with normalized `float32` vectors in a `sqlite-vec` virtual table. Provide documented SQL examples that join nearest vectors to chunk text, documents, and source metadata.

### Runs and failures

Persist run status and aggregate counts plus actionable failures. Bound or prune verbose history so a million-file installation does not accumulate an unbounded event log.

## 7. Source identity and synchronization

### Local files and directories

Canonical absolute paths are the public source identity. Each exact directory invocation is also an internal synchronization root.

For `ree /absolute/docs`:

- Stream every directory entry without applying ignore files.
- Process symlinked files.
- Do not recursively follow symlinked directories, preventing cycles and accidental traversal outside the root.
- Skip unchanged documents using stat metadata followed by content hashes when needed.
- Replace changed documents atomically.
- Add new documents.
- Delete documents confirmed absent from this exact root.
- Keep unrelated roots untouched.

A direct file invocation replaces only that file. Failed files keep their previous valid chunks. Binary/unsupported files are recorded as failures or skips rather than embedded.

### Stdin

Without `--source`, identify stdin as `stdin://sha256/<content-hash>`, deduplicating identical input. With `--source`, use a stable replaceable identity:

```bash
cat report.md | ree - --source reports/current
```

### URLs

Normalize the URL and use it as a stable source identity. Fetch one page, follow bounded redirects, and use readability/article extraction with cleaned visible HTML as fallback. Do not execute JavaScript. Apply download-size, timeout, redirect, and private-network protections.

### Git repositories

For a remote Git URL, shallow-clone the default branch into the cache, record repository URL/branch/commit, and update on later runs. Embed checked-out files only; do not embed history or `.git` object data and never execute repository content. Treat local repositories as directories, with binary detection naturally rejecting meaningless Git internals.

## 8. Extraction

Define a native extractor interface returning exact text, optional title/author/language/location metadata, extractor identity/version, and warnings.

Priority order:

1. Plain text and source code.
2. Markdown and HTML.
3. JSON, YAML, and XML.
4. PDF.
5. DOCX and ODT.
6. EPUB.
7. Jupyter notebooks.
8. OCR for images and scanned PDFs.

Handle text, source, Markdown, structured text, HTML readability, and notebooks natively where practical. Preserve formatting useful to code and configuration retrieval rather than flattening everything into prose.

Use established optional executables where they are faster and more reliable:

- `pdftotext` for PDF.
- LibreOffice for office documents.
- `pandoc` for convertible formats.
- Tesseract for OCR.

Missing tools produce a structured unsupported event and processing continues. ree never installs packages automatically.

Support configurable extractor argument arrays without shell interpolation:

```toml
[extractors.epub]
extensions = ["epub"]
command = ["pandoc", "{path}", "-t", "plain"]
timeout_seconds = 60
max_output_bytes = 67108864
```

Use MIME sniffing, NUL/control-byte ratios, encoding detection, extension hints, and extractor availability together to distinguish meaningful text from binary data.

## 9. Chunking

Initial defaults:

- 512 model tokens.
- 64-token overlap.
- No cross-document chunks.
- Exact extracted chunk text is persisted.
- Boundaries are deterministic for a model/tokenizer revision.
- `--chunk-size` and `--overlap` override defaults.

Use the pinned model tokenizer, not whitespace or character estimates. Stream very large documents where the extractor format permits it and enforce configurable limits where it does not.

## 10. Processing pipeline and performance

Use a bounded streaming pipeline:

```text
discovery
  -> type detection
  -> extraction
  -> tokenization and chunking
  -> length-bucketed embedding batches
  -> single batched database writer
```

Principles:

- Never materialize an entire directory listing.
- Hash while reading where possible.
- Bound every queue and temporary output.
- Run extraction concurrently.
- Avoid oversubscribing CPU threads when ONNX Runtime also uses a thread pool.
- Batch prepared SQLite statements in transactions.
- Keep one controlled SQLite writer.
- Commit complete document replacements atomically.
- Buffer JSON output.
- Measure before applying SQLite pragmas; likely settings include WAL, foreign keys, busy timeout, an appropriate cache, and batched commits.

Within one process, extraction and tokenization are parallel and model inference is batched. Across processes, readers are allowed but only one ingestion/rebuild writer may run. A second writer waits briefly, then fails clearly with lock-owner information. Use SQLite locking plus an application lock file for understandable diagnostics.

## 11. Failure and atomicity semantics

Default behavior continues after per-input failures:

- Commit successful additions and updates.
- Delete only files confirmed absent.
- Preserve previous embeddings for documents that fail extraction or embedding.
- Emit structured failures.
- Return a partial-success exit code.

`--fail-fast` stops scheduling new work after the first failure and safely completes or rolls back work already in flight.

Suggested exit codes:

```text
0  complete success
1  partial success
2  invalid arguments or configuration
3  fatal database/model/runtime error
4  writer lock unavailable
```

## 12. Rebuilds

`ree rebuild` regenerates vectors from stored chunks without reopening original sources. This supports model revisions, precision/runtime changes, normalization fixes, and cron/systemd scheduling.

Build into a separate embedding generation. Activate it transactionally only after every stored chunk has a valid finite vector of the expected dimension. On failure, keep the old generation active and preserve resumable progress where practical.

A model/tokenizer change that changes chunk boundaries is not a vector rebuild; it is an explicit full rechunk/reingest operation. The CLI must make that distinction clear.

## 13. CLI and event contract

### Ingestion options

```text
ree <INPUT>...
  --db PATH
  --metadata KEY=VALUE
  --metadata-json JSON
  --chunk-size N
  --overlap N
  --device auto|cpu|cuda|cuda:N
  --batch-size N
  --gpu-memory-fraction FLOAT
  --progress
  --quiet
  --verbose
  --fail-fast
  --source NAME
  --force
  --max-file-size SIZE
  --extractor NAME
```

### Maintenance commands

```text
ree status
ree sources
ree remove <SOURCE>
ree rebuild
ree doctor
```

`doctor` checks paths, schema, model checksums, SQLite vector support, writer locks, CUDA/NVML/ONNX compatibility, and optional extractors.

Default stdout is buffered JSON Lines. Progress and human diagnostics go to stderr. Default events are concise so million-file runs do not print a success line per file; `--verbose` enables per-file events.

```json
{"type":"started","run_id":"...","inputs":["/abs/docs"]}
{"type":"runtime","provider":"cuda","device":"NVIDIA ...","free_vram_bytes":7516192768}
{"type":"failed","source":"/abs/docs/bad.pdf","error_code":"missing_extractor","message":"pdftotext is unavailable"}
{"type":"completed","run_id":"...","documents":42000,"chunks":380000,"failed":1,"elapsed_ms":18234}
```

`--progress` enables a TTY-aware progress UI. `--quiet` suppresses non-error output. Event types, fields, and error codes are documented interfaces.

## 14. Security and resource controls

Set configurable limits for local files, URL responses, Git clone time/size, redirects, extractor runtime/output, archive expansion, queued documents, and inference memory.

Additionally:

- Never execute content from repositories.
- Never invoke extractors through a shell.
- Treat filenames and metadata as untrusted.
- Prevent path traversal from archives and converters.
- Verify downloaded model hashes.
- Use private temporary files and atomic rename.
- Restrict URL schemes and guard against unwanted access to local/private networks by default.
- Require explicit overrides for unusually large or protected documents.

## 15. Rust architecture

Turn the binary into a library with a thin entry point:

```text
src/
  main.rs
  lib.rs
  cli.rs
  config.rs
  error.rs
  events.rs
  input/
    mod.rs
    discover.rs
    local.rs
    stdin.rs
    url.rs
    git.rs
  extract/
    mod.rs
    text.rs
    html.rs
    structured.rs
    notebook.rs
    external.rs
  chunk/
    mod.rs
    token_window.rs
  model/
    mod.rs
    download.rs
    device.rs
    batch.rs
    onnx.rs
  storage/
    mod.rs
    sqlite.rs
    migrations.rs
    schema.rs
  pipeline/
    mod.rs
    sync.rs
    rebuild.rs
```

Likely dependency areas include Clap, Tokio, Serde/JSON/TOML, bundled SQLite, sqlite-vec, ONNX Runtime, a tokenizer, reqwest with rustls, HTML readability, BLAKE3, MIME/encoding detection, NVML dynamic loading, filesystem locks, tracing, progress reporting, and XDG paths. Remove `qdrant-client` after the storage spike confirms SQLite.

Keep runtime-specific types behind `model::Runtime` so CPU/GPU artifacts, provider setup, and fallback do not leak into discovery, chunking, or storage.

## 16. Testing

### Unit tests

Test input classification, URL normalization, text/binary detection, Unicode and legacy encodings, stable IDs, metadata precedence, token boundaries and overlap, content hashes, batch packing, VRAM-budget calculations, schema migrations, and exit-code mapping.

### Integration tests

Test initial ingestion, no-op re-ingestion, changed-file replacement, deletion and rename, source-root isolation, partial failures, `--fail-fast`, stdin identities, local HTTP URL extraction, Git updates, external extractor limits, direct vector SQL, concurrent writer rejection, crash recovery, rebuild activation, and rebuild resume.

GPU tests must include:

- No NVIDIA driver/NVML present.
- Driver present but CUDA provider unavailable.
- Multiple GPUs with different free VRAM.
- FP16 model-load failure.
- Warm-up OOM.
- Mid-run OOM and successful reduced-batch retry.
- Device loss and CPU continuation.
- Explicit CUDA mode failing rather than silently falling back.
- Idempotent writes after retry.
- CPU/GPU vector similarity and ranking parity.

Use dependency-injected device and runtime probes so failure paths run in ordinary CI. Run real CUDA tests and all model-performance measurements only on this development machine's RTX 4070 Laptop GPU; CI should test simulated behavior rather than qualify model performance.

### Scale and performance tests

Measure cold/warm startup, discovery rate, extraction throughput, CPU and GPU embeddings per second, tokens per second, SQLite insertion rate, no-op synchronization, rebuild rate, peak host RAM/VRAM, and vector SQL latency at 1M, 5M, and 10M chunks. Include huge individual files and millions of tiny files.

## 17. Delivery phases

### Phase 0: benchmark and runtime spikes

- Build the model qualification harness.
- Run qualification only on this machine's Ryzen 9 8945HS and RTX 4070 Laptop GPU.
- Validate Arctic CPU INT8 and CUDA FP16 output compatibility from Rust.
- Measure Arctic, Granite, E5-small, BGE-M3, and Jina Code on local quality and throughput; do not construct a cross-hardware performance matrix.
- Validate automatic CPU fallback locally by disabling or mocking CUDA availability.
- Validate bundled SQLite plus sqlite-vec at million-vector scale on this machine.
- Pin the winning model revision and artifact hashes.

### Phase 1: foundation

- Restructure into library and binary.
- Implement CLI, XDG configuration, errors, JSON events, and tracing.
- Add SQLite schema, migrations, and writer locking.
- Publish the first schema documentation.

### Phase 2: local ingestion

- Add file/directory/glob/stdin discovery.
- Add binary/type/encoding detection and plain-text extraction.
- Implement deterministic tokenizer windows.
- Add automatic model download, CPU inference, batching, and transactional writes.
- Add hashing and unchanged-file fast paths.

Milestone: `ree ./docs` works end to end with zero configuration.

### Phase 3: GPU acceleration

- Implement NVML discovery and multi-GPU selection.
- Add CUDA provider setup and verification.
- Add FP16 artifact selection, VRAM budgets, adaptive length-bucketed batching, profile caching, OOM reduction, and CPU fallback.
- Add real and simulated CUDA failure tests and runtime metrics.

### Phase 4: synchronization and maintenance

- Add source-root tracking and deleted-file cleanup.
- Implement partial failure semantics.
- Add metadata and `status`, `sources`, `remove`, and `doctor`.
- Stabilize exit codes and event formats.

### Phase 5: rich extraction

- Add Markdown, HTML, structured data, and notebooks.
- Add secure external extractor support.
- Integrate PDF, office, EPUB, and OCR workflows.
- Add limits and hostile-input tests.

### Phase 6: remote inputs

- Add bounded single-page URL/readability ingestion.
- Add cached shallow Git clones and updates.
- Add remote-source metadata and security controls.

### Phase 7: rebuilds

- Add shadow embedding generations, checkpoint/resume, and atomic activation.
- Document cron and systemd scheduling.
- Clearly separate re-embedding from rechunk/reingest.

### Phase 8: scale hardening and release

- Profile million-file and multi-million-chunk corpora.
- Tune queues, inference, database writes, memory, and startup.
- Test crash recovery and migrations.
- Publish direct SQL retrieval examples, model interoperability instructions, man pages, and shell completions.
- Ship Linux binaries and `cargo install` support where ONNX packaging permits.
- Add `MIT` and `Apache-2.0` license files.

## 18. Initial definition of done

The first useful release is done when:

1. `ree ./docs` works with no prior configuration.
2. The pinned model downloads and verifies automatically.
3. CPU inference uses the selected INT8 artifact efficiently.
4. A compatible NVIDIA GPU is detected automatically, selected by usable VRAM, and uses FP16 inference.
5. Batches adapt to token lengths and available VRAM.
6. CUDA initialization, OOM, and device failures fall back safely in automatic mode without losing work or corrupting data.
7. Supported files are extracted, chunked, and embedded through bounded concurrent stages.
8. Repeated runs skip unchanged files and synchronize changes and deletions within the correct root.
9. Failed files retain previous valid embeddings and produce structured output.
10. SQLite vectors and metadata are directly queryable through a documented, migrated schema.
11. `ree rebuild` creates and atomically activates a complete new embedding generation.
12. Optional extractors fail safely and clearly.
13. The actual Rust runtime is benchmarked on multi-domain, multilingual, and code corpora using only this development machine.
14. The pipeline is tested on this machine with a synthetic or real corpus large enough to exercise million-scale behavior.
15. Agent-facing JSON and exit codes are documented and stable enough for automation.
