# Configuration

Use `--config PATH` / `REE_CONFIG` to select a TOML file. An explicitly selected
missing file is an error; a missing default file is normal. Unknown TOML fields
are rejected. Relative XDG environment paths are ignored per the XDG spec.

```toml
# ~/.config/ree/config.toml
# db = "/absolute/path/ree.db"
device = "auto"                # cpu, cuda, cuda:N
chunk_size = 512                # 3..8192, includes two special tokens
overlap = 64                    # strictly less than chunk_size - 2
# batch_size = 8                # positive explicit cap; OOM may reduce it
gpu_memory_fraction = 0.85      # >0 and <=0.95
max_file_size = "64MiB"         # positive integer; B/KiB/MiB/GiB (KB/MB/GB aliases)

[pdf]
ocr = true                     # OCR only pages without extractable text
max_pages = 1000
max_image_side = 2400           # rendered longest side in pixels; 128..8192
# Shared deadline across PDF text extraction/rendering/OCR steps:
timeout_seconds = 120
max_output_bytes = 67108864     # combined UTF-8 output including page separators
max_temp_bytes = 268435456      # input snapshot plus temporary helper output

[metadata]
project = "notes"

[extractors.epub]
extensions = ["epub"]
command = ["pandoc", "{path}", "-t", "plain"]
timeout_seconds = 60
max_output_bytes = 67108864
max_temp_bytes = 268435456
```

CLI > environment > file > defaults. Supported option environment variables:
`REE_DB`, `REE_CONFIG`, `REE_DEVICE`, `REE_BATCH_SIZE`,
`REE_GPU_MEMORY_FRACTION`, `REE_CHUNK_SIZE`, `REE_OVERLAP`, `REE_MAX_FILE_SIZE`.

Metadata precedence: extractor metadata, remote-source metadata, TOML metadata,
`--metadata-json` object, then repeated `--metadata KEY=VALUE`. `KEY=VALUE`
values are strings; use JSON to preserve numbers, booleans, arrays, and objects.
A processing recipe hash includes configured metadata and extractor settings,
so changes invalidate the unchanged fast path.

`--extractor text|html|notebook` forces a native extractor. Other names select a
configured extractor or built-in `pdf`, `docx`, `odt`, `epub`, or image extractor.
Commands are arrays, with literal `{path}` replacement in each argument. There
is no shell interpolation. Helpers receive a private immutable file snapshot,
not a source that can change during conversion. stdout/stderr are drained in
parallel; output size and wall-clock limits terminate the process group. Helpers
receive `LC_ALL=C` for predictable diagnostic/version parsing. Aggregate workspace
size is checked every 100 ms and on exit; a per-file `RLIMIT_FSIZE` also bounds
fast writes between samples (without loosening inherited process limits). These
bounds do not sandbox writes outside the workspace or prevent helper exploits.

Current fixed resource bounds: two extraction workers, discovery/result queues
of two items each, up to 32 pending documents or approximately 8 MiB of token/text
payload before inference, 30-second HTTP requests, five redirects, 120-second Git
commands, 512 MiB Git checkouts, 60-second non-PDF built-in extractors, and 64 MiB
non-PDF built-in helper stdout. One large document is still held in memory and is bounded by the file or
extractor limit. These are not yet all configurable; see implementation status.

## Search

`ree search` uses the shared `db`, cache, `device`, and `gpu_memory_fraction`
settings. Search-only `--mode`, `--limit`, `--root`, and `--media-type` are CLI
options, not TOML/environment settings. Dense queries have a fixed 512-token
limit including prompt/special tokens and always use singleton batches; ingestion
`chunk_size`, `overlap`, and `batch_size` do not change this policy. Lexical mode
requires no model. Standalone/streaming search and worker startup validate existing
configuration. Explicit socket clients instead use the worker's fixed configuration
and reject database/device/config/batch/memory overrides; they do not read local TOML.
See [retrieval](retrieval.md) for migration, scoring, and output details.

`search --stream` and `worker` accept `--idle-timeout 300s` (1..86400 seconds).
`worker --socket PATH` requires an absolute path in an existing private user-owned
directory; without it, a verified `XDG_RUNTIME_DIR/ree/worker.sock` is used. No worker
is auto-started. Protocol modes reject quiet/progress flags. Full wire/bounds and
lifecycle details are in [speedup v2](speedup-v2.md).

Developer experiments (only value `1` enables them; off by default):

- `REE_TIMING=1`: additive, overlapping nanosecond stage telemetry; no query text.
- `REE_CLS_OUTPUT=1`: verified CLS-output derivative, unchanged host normalization.
- `REE_PIPELINE_OVERLAP=1`: bounded small-document preparation overlap, **explicit
  CUDA only**, 512-token windows and overlap <=64. CPU/auto retain serial tokenization
  because the CPU regrouping experiment failed its equivalence gate.

These are not TOML/search request options or a promise of a faster default.
Original weights and logical embedding recipe remain unchanged.

## PDFs and helper versions

Built-in PDF extraction requires Poppler's `pdfinfo` and `pdftotext`. Textless
pages (including blank pages in mixed PDFs) additionally require `pdftoppm` and
`tesseract`; meaningful existing text is never replaced with OCR. Rendering is
sequential, one bounded-size PNG at a time. A missing helper or failed page fails
the complete document, preserving any previous valid embeddings. `ocr = false`
keeps existing text and records warnings for textless pages; an entirely textless
PDF is reported unsupported. Custom `extractors` mappings and a configured
`extractors.pdf` override take precedence over the built-in workflow.

PDF metadata includes half-open UTF-8 page byte ranges, one-based page numbers,
per-page OCR flags, warnings, and known-helper versions. Output retains page text
and a form-feed separator per page. `ree doctor` also reports known versions;
version queries are cached per process, bounded to two seconds, and unavailable
versions are null. Arbitrary custom commands are never invoked speculatively
with a guessed version flag. After upgrading an installed helper, use `--force`
to refresh extraction: the unchanged fast path does not invoke helper probes.

`--allow-private-network` is an explicit per-invocation exception to URL/Git IP
protection. Ambient HTTP proxies are disabled for untrusted source fetches.
Model downloads use HTTPS with pinned hashes; standard proxy settings apply to
those trusted artifact hosts.

## Scheduled rebuild

```cron
# Adjust executable path and schedule to your installation.
0 3 * * 0 /home/me/.local/bin/ree rebuild >> /home/me/ree-rebuild.jsonl 2>> /home/me/ree-rebuild.stderr
```

Or a user systemd service:

```ini
[Unit]
Description=Rebuild local ree embeddings
[Service]
Type=oneshot
ExecStart=%h/.local/bin/ree rebuild
```

Pair with a user timer using `OnCalendar=weekly` and `Persistent=true`. A rebuild
resumes an incomplete generation on its next invocation; a competing writer
returns exit code 4 after a short wait. Rebuild does not reopen sources. Use an
explicit ingestion with `--force` to extract/rechunk sources under a new policy.
