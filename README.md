# ree

**r**ust, **e**mbed **e**verything. Linux-first, local inference, SQLite storage,
no daemon. `ree` ingests and synchronizes documents; it does not implement search.

```sh
cargo build --release --locked
./target/release/ree ./docs
./target/release/ree document.pdf 'src/**/*.rs'
./target/release/ree https://example.com/article
./target/release/ree https://github.com/org/repository.git
cat report.md | ./target/release/ree - --source reports/current
```

First use downloads checksum-verified, revision-pinned Snowflake Arctic Embed M
v2 artifacts (17 MB tokenizer, 311 MB CPU INT8; 613 MB FP16 lazily for CUDA).
Passages use no prefix, CLS pooling, L2 normalization, and 768 dimensions.
The revision is **provisional pending the local qualification gate**.

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
  PDF uses `pdftotext`; DOCX/ODT/EPUB use `pandoc`; images use `tesseract`.
  Missing helpers are reported, never installed automatically.
- HTTP(S) fetches enforce size/time/redirect limits and DNS-pinned private-network
  protection. `--allow-private-network` explicitly permits local HTTP endpoints.
  Remote Git requires HTTPS URLs ending in `.git`; hooks/submodules are disabled.
- Default stdout is JSON Lines. `--verbose` adds document events; `--quiet`
  retains errors only; `--progress` enables TTY-only stderr progress.

```sh
ree status
ree sources
ree remove /absolute/source/root
ree rebuild                   # stored token inputs; originals are not reopened
ree doctor                    # offline diagnostics, no downloads
ree --help
```

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
cargo test --locked
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
qualification in [PLAN.md](PLAN.md) is complete. Schema version 1 remains unstable.

Licensed under **MIT OR Apache-2.0**.
