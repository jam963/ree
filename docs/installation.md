# Local beta installation and first-use checklist

## Scope decision — September 11, 2026

The user chose to checkpoint development, install a local beta, and gather
qualitative usage feedback **before** continuing the remaining performance,
large-document, extraction and hardening work. This supersedes the original
beta roadmap's requirement to complete all gates before installation. It does
not declare those gates passed or authorize a public/production release.

Version: **0.1.0-beta.1**. Supported installation target: the development machine,
**x86_64 Linux/glibc**, CPU and its existing CUDA 13 stack. This is not a portable
many-distribution build; `build-info.json` records toolchain, libc symbol
requirements and dynamic dependencies. Rust is needed to build, not to run an
unpacked package. Python **3.11+** is needed only for building/install management.
The installed launcher uses `/bin/sh` and GNU `readlink` (coreutils).
No mandatory daemon, automatic service, system-package install or model download
occurs during installation.

## Build and install

From a committed checkout with Cargo dependencies and the pinned ONNX Runtime
build artifacts already cached:

```sh
python3 packaging/build.py
# Explicitly allow Cargo to download build dependencies if necessary:
# python3 packaging/build.py --online
```

The builder refuses a dirty tree by default, uses Cargo.lock, copies (rather than
symlinks) both optional ONNX providers, and includes generated man/completions,
docs, build/source identity, checksums, third-party notices and dependency sources.
`--allow-dirty --output /absolute/scratch/path` is for development testing only;
such manifests are explicitly marked dirty. Builds do not overwrite existing
archives. Container metadata/order are deterministic for identical payloads;
bit-for-bit reproducible Rust compilation across hosts is **not** claimed.

```sh
cd dist
sha256sum -c ree-0.1.0-beta.1-x86_64-unknown-linux-gnu.tar.gz.sha256
tar -xzf ree-0.1.0-beta.1-x86_64-unknown-linux-gnu.tar.gz
cd ree-0.1.0-beta.1-x86_64-unknown-linux-gnu
python3 install.py verify
python3 install.py install                 # default prefix: ~/.local
# Alternative, explicitly system-wide:
# sudo python3 install.py install --prefix /usr/local
ree --version
ree doctor
```

Extract only trusted archives; checksums detect corruption, **not publisher
authenticity**. The installer verifies the complete extracted file inventory and
refuses symlinks/special files, mismatched hashes and unrelated destination paths.
Run it from the extracted package, not the checkout's `packaging/` folder. The
package directory must contain only its original payload and manifest.

The default command is `~/.local/bin/ree`; add `~/.local/bin` to PATH if your shell
does not already include it. The installer never edits shell configuration.
System-wide installation uses `/usr/local/bin/ree`. Do not run ordinary ree
commands with sudo. Prefixes and managed directories must belong to the installing
UID and not be group/world writable; symlinked prefix components are rejected.

Layout (relative to the prefix):

- `lib/ree/releases/0.1.0-beta.1/`: versioned payload, executable and adjacent providers.
- `lib/ree/current`: atomically switched symlink to the selected release.
- `bin/ree`: symlink through `current` to a small launcher. It resolves the version
  directory and execs its real `ree` binary, so ONNX Runtime finds adjacent providers.
  No Cargo cache/build-tree dependency.
- `share/man/man1/ree.1`, Bash/Zsh/Fish completions and `share/doc/ree`: managed links.
- `lib/ree/.install.lock`: serializes install/activation/uninstall operations.

No NVIDIA libraries are bundled. CUDA uses the existing system driver, CUDA 13
runtime/cuBLAS/cuRAND and cuDNN where required. CPU execution does not need these
libraries. See [runtime](runtime.md) for the tested stack and provider behavior.
Optional extraction helpers are also not bundled: Poppler for PDF text, Tesseract
for OCR, Pandoc for DOCX/ODT/EPUB. `ree doctor` reports availability.

## Start small

Use a small, trusted folder first—not your home directory or an entire vault.
Discovery includes hidden and ignored files. The database stores extracted text
as well as vectors, so protect it like the original documents.

```sh
ree --version
ree doctor
ree --progress /absolute/path/to/small-test-folder
ree status
ree sources
ree search "a question answered by those documents"
ree search "a distinctive phrase" --mode lexical
ree search "a question answered by those documents" --mode hybrid
ree /absolute/path/to/small-test-folder       # unchanged hash no-op
```

First dense ingestion/search downloads pinned model artifacts if not already
cached. By default `auto` selects CUDA when usable, with CPU fallback. To keep the
provider consistent while evaluating rankings on this machine, optionally use
`--device cuda` for **both** ingestion and queries; it refuses CPU fallback.
CPU INT8 and CUDA FP16 can rank differently, including in a mixed-provider index.
Do not enable the experimental overlap/CLS environment flags for initial feedback.

Output is JSONL; progress goes to terminal stderr. Errors and partial-success exit
codes matter. Try changing/deleting a file and rerunning the **same root**; verify
search reflects the change. Failed files should retain their previous indexed
contents. `ree remove /absolute/path/to/small-test-folder` removes that source
from the index, **not the original folder**. Stop short of large/untrusted helper
inputs until the remaining resource and security qualification is complete.

For fast repeated queries, optionally run `ree worker` in a separate terminal and
use `ree search "question" --socket "$XDG_RUNTIME_DIR/ree/worker.sock"`. Stop it
with Ctrl-C. It is never auto-started. See [speedup v2](speedup-v2.md) for private
socket requirements and streaming use.

Useful feedback: exact command, version, device/provider, input format and rough
size, expected vs actual search result, cold vs repeated-query delay, confusing
output/errors. Redact private paths/text before sharing logs.

## Database safety, upgrades and rollback

Default data is separate from the installation:
`~/.local/share/ree/ree.db`, `~/.config/ree/config.toml`, and `~/.cache/ree/`.
XDG and explicit path overrides still apply. The installer never opens/migrates,
backs up, deletes or changes these paths. Stop existing workers/ingestion before
upgrading; running processes keep their old code until restarted.

Schema 2 is pre-1.0 and unstable. Before an upgrade/migration, stop all ree
processes and back up the database. If `sqlite3` is installed, its backup command
includes committed WAL data without relying on copying just the main file:

```sh
sqlite3 "$HOME/.local/share/ree/ree.db" ".backup '/absolute/backup/ree-before-beta.db'"
```

Use the actual database path if overridden; choose a new backup filename. Without
SQLite tooling, with all clients stopped, copy the database **and any existing
`-wal`/`-shm` sidecars together** to a separate backup directory. Back up custom
configuration too. Do not make a bare main-file copy while clients are running.
Existing schema-1 databases retain semantic read compatibility; `ree migrate`
explicitly backfills lexical search without models/re-embedding. An ingestion or
other writer can also migrate under its normal existing behavior; installation
itself does not. See [schema](schema.md).

Install the next package with the same prefix. A different version is staged and
verified before activation; an existing version with different bytes is rejected.
Repeated installation of the identical package is safe. Previous release payloads
are retained. List or select them using any retained installer:

```sh
python3 ~/.local/lib/ree/current/install.py list
python3 ~/.local/lib/ree/current/install.py activate 0.1.0-beta.1
# Add --prefix /usr/local and sudo for a system-wide installation.
```

Activation changes executable/docs links only. **Executable rollback is not
schema/data rollback.** Restore a compatible stopped-database backup separately
if a future schema cannot be read by an older executable. Restore sidecars
consistently; never pair a restored main file with WAL from a different snapshot.

## Uninstall

```sh
python3 ~/.local/lib/ree/current/install.py uninstall
# System-wide:
# sudo python3 /usr/local/lib/ree/current/install.py uninstall --prefix /usr/local
```

Uninstall removes only managed command/docs/completion/current links. It does not
stop running workers or delete data/config/models. Versioned release files remain
for reinstall/rollback; after uninstall you may manually delete the dedicated
`PREFIX/lib/ree` directory if no longer needed. No purge-data operation is supplied.
Use the retained release's `install.py install` to reactivate after uninstall.

## Recheck an installed package

Offline installer/archive tests run in CI with temporary prefixes and fake payloads:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s packaging -p 'test_*.py'
python3 packaging/smoke.py --prefix "$HOME/.local" --output /tmp/ree-offline-smoke
```

The smoke output directory must be new. With no `--models`, only model-free
installed startup/migration/empty-query checks run. For serial real CPU/CUDA
checks on the development Ryzen/RTX host only (never CI), add
`--models "$HOME/.cache/ree/models"`. The harness copies existing models into
scratch, points download proxies at an unreachable local endpoint, uses isolated
HOME/XDG paths/databases,
checks all three retrieval modes plus stream/socket queries and sync/rebuild,
and removes its generated data afterward. Logs and `summary.json` remain in the
output directory. No performance or broad ranking qualification is implied.

## Qualification still deferred

Stable sustained throughput; prolonged transport contention/slow-client soak;
bounded disk staging for huge individual documents; real scanned-PDF OCR/office
qualification; comprehensive helper/security/resource review; larger disk-backed
scale tests; broad ranking/precision evaluation; public distribution license and
Linux portability review. This local beta is for feedback, not untrusted document
processing or a claim of production readiness.
