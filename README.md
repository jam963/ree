# ree

**r**ust, **e**mbed **e**verything. Linux-first, local inference, SQLite storage,
no mandatory daemon. `ree` ingests, synchronizes, and searches documents.

**Local beta: 0.1.0-beta.1.** Build a versioned installable archive with
`python3 packaging/build.py`, then run its `python3 install.py install` to install
under `~/.local` (or use `--prefix /usr/local` with sudo for system-wide installation).
See [installation, rollback, and first-use checklist](docs/installation.md).
This feedback beta deliberately precedes the remaining qualification gates.

```sh
cargo build --release --locked
./target/release/ree ./docs
./target/release/ree document.pdf 'src/**/*.rs'
./target/release/ree https://example.com/article
./target/release/ree https://github.com/org/repository.git
cat report.md | ./target/release/ree - --source reports/current
./target/release/ree search "how does CUDA fallback work?"
./target/release/ree search "SQLITE_BUSY" --mode lexical
./target/release/ree search "writer lock timeout" --mode hybrid --limit 10
```

First use downloads checksum-verified, revision-pinned Snowflake Arctic Embed M
v2 artifacts (17 MB tokenizer, 311 MB CPU INT8; 613 MB FP16 lazily for CUDA).
Explicit CUDA skips CPU rescue provisioning; `auto` retains it.
Passages use no prefix, CLS pooling, L2 normalization, and 768 dimensions.
**Arctic 768 is selected for the beta.** Broader model evaluation is deliberately
deferred; mixed CPU INT8 / CUDA FP16 retrieval remains a documented limitation.
See the [model decision and pinned recipe](docs/model-decision-v1.md).

## Behavior

- Recursive discovery includes dotfiles, ignored files, and hidden directories.
  File symlinks are processed; directory symlinks are not traversed.
- Two bounded extraction workers feed tokenizer windows and length-bucketed
  inference batches; complete document replacements are transactional.
- Default windows contain up to 512 model tokens, including CLS/SEP, with 64
  content-token overlap. Exact extracted text and model input IDs are stored.
- Repeated runs hash inputs and skip unchanged extraction/inference. Changed
  files replace old chunks; complete directory walks prune confirmed deletions
  **only in that exact source root**. Failed files retain their old vectors.
- Plain text/code, Markdown, structured text, HTML, and notebooks work natively.
  PDFs use `pdfinfo`/`pdftotext`, with bounded `pdftoppm` + `tesseract` OCR for
  textless pages; DOCX/ODT/EPUB use `pandoc`; images use `tesseract`.
  Missing helpers are reported, never installed automatically.
- HTTP(S) fetches enforce size/time/redirect limits and DNS-pinned private-network
  protection. `--allow-private-network` explicitly permits local HTTP endpoints.
  Remote Git requires HTTPS URLs ending in `.git`; hooks/submodules are disabled.
- Default stdout is JSON Lines. `--verbose` adds document events; `--quiet`
  retains errors only; `--progress` enables TTY-only stderr progress.

```sh
ree migrate                   # schema 2 + lexical backfill; no model/re-embedding
ree status
ree sources
ree remove /absolute/source/root
ree rebuild                   # stored token inputs; originals are not reopened
ree doctor                    # offline diagnostics, no downloads
ree --help
```

Search defaults to semantic retrieval; `--mode lexical` uses FTS5 without loading
models, and `--mode hybrid` fuses both rankings. `--root ID_OR_PATH` and
`--media-type TYPE` filter **before** top-k. Search is database-read-only, takes no
writer lock, and returns exact stored chunks with source/location metadata as
JSONL. Schema-1 databases support semantic search immediately; run `ree migrate`
with the same `--db` for lexical/hybrid search. See [retrieval](docs/retrieval.md)
for scoring, query limits, concurrency, and mixed-precision limitations.

For repeated queries, `ree search --stream` accepts versioned JSONL on stdin and
reuses an engine. An optional `ree worker` exposes the same service on a private
Unix socket; route explicitly with `ree search "query" --socket PATH`. Both release
the engine process after five idle minutes by default. Standalone behavior stays
the default; no service is installed or auto-started. See [speedup v2](docs/speedup-v2.md)
for framing, lifecycle, bounds, measured latency, and opt-in experiments.

Exit codes: **0** success, **1** partial success, **2** invalid arguments/config,
**3** fatal storage/model/runtime failure, **4** writer lock unavailable.

## Paths and configuration

| Purpose | Default |
|---|---|
| Config | `~/.config/ree/config.toml` |
| Database | `~/.local/share/ree/ree.db` |
| Models / profiles | `~/.cache/ree/models/`, `~/.cache/ree/profiles/` |
| Git checkouts | `~/.cache/ree/repositories/` |

Absolute `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, and `XDG_CACHE_HOME` override these
bases. CLI options override `REE_*` environment variables, then the config file,
then defaults. See [configuration](docs/configuration.md).

SQLite uses bundled `sqlite-vec`, foreign keys, WAL, migrations, and a two-second
writer lock wait. WAL/SHM sidecars and a `.lock` file exist during operation;
checkpoint/close before copying just `ree.db`.

## Build, GPU support, and tests

Rust 1.88+ and a Linux C/C++ toolchain are required. Cargo downloads verified
ONNX Runtime 1.28 build artifacts. The core runtime is statically linked: the
CPU binary does not require an installed CUDA stack. GPU acceleration uses the
packaged provider libraries and **CUDA 13.x** (cuDNN 9.x for operators that need
it). It works with the development machine's installed CUDA 13 stack; no CUDA 12
compatibility installation or `LD_LIBRARY_PATH` override is needed.
See [runtime packaging](docs/runtime.md); `cargo install` alone installs the CPU
executable, not the optional provider libraries.

```sh
cargo test --locked --all-targets
cargo clippy --all-targets --locked -- -D warnings
cargo fmt --all -- --check
```

Ordinary tests use injected runtimes, not downloaded models. Real-model tests
are ignored by default and restricted to the development machine.

- [Public schema and direct vector SQL](docs/schema.md)
- [JSON events and synchronization contract](docs/contract.md)
- [Local qualification and scale harnesses](docs/benchmarks.md)
- [Implementation status and remaining release gates](docs/implementation-status.md)

This is a pre-1.0 implementation, not a claim that all performance/security
qualification in [PLAN.md](PLAN.md) is complete. Schema version 2 remains unstable.

Licensed under **MIT OR Apache-2.0**.
