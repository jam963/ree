# CLI event and synchronization contract

Stdout is buffered JSON Lines, one object per event, no progress escape codes.
All events have `type`. Consumers should ignore unknown types/fields. Help and
version requests use Clap's ordinary text output. ONNX/helper diagnostics may
appear on stderr; they are not part of the machine-readable contract.

| Type | Important fields |
|---|---|
| `started` | `run_id`, `inputs` (ingestion), or `operation`, `generation` (rebuild) |
| `download` | `artifact`, `bytes`, `revision` |
| `download_retry` | `artifact`, failed `attempt`, `message` (at most three attempts; checksum failures are never activated) |
| `runtime` | `provider`, `device` (GPU details or null), `precision`, `runtime`, `runtime_versions`, `threads`, `batch_limits` |
| `device_fallback` | `from`, `to`, `reason`, optional `retry_batch_items` |
| `batch_reduced` | `reason`, `max_items`, `max_padded_tokens` |
| `document` | `source`, `chunks` (verbose only) |
| `failed` | `source`, `error_code`, `message` |
| `completed` | `run_id`, elapsed milliseconds, counters, `runtime_metrics` |
| `fatal` | `error_code`, `message` |
| `status` | `schema_version`, `sources`, `documents`, `chunks`, `embeddings`, `generation` |
| `source` | `id`, `kind`, `identity`, `metadata`, last attempted/successful run |
| `removed` | `source`, `roots` (zero is an idempotent no-op) |
| `migrated` | `schema_version` (explicit `ree migrate`; no model/download) |
| `search_result` | `rank`, chunk/document IDs, `text`, `uri`, `source_id`, `source`, `media_type`, extracted-text byte/token ranges, location/metadata, nullable `distance`, `bm25`, `fusion_score` |
| `search_completed` | `mode`, returned `results`, `generation`, `model_key`, `query_provider`, `query_tokens`, candidate limits/counts, `warnings`, stage `timings` |
| `doctor` | database/cache/model checks, NVML devices, provider compilation, lock, helpers |
| `rebuild_checkpoint` | `generation`, `embedded` (verbose only) |

Completed ingestion counters: `documents` and `chunks` count successfully
written replacements/additions, not total database size; `unchanged` counts
hash hits; `deleted` counts removed documents; `failed` includes skipped binary
and unsupported files. Thus an all-binary input returns partial code 1, not 0.

`runtime_versions` reports the integer values returned by CUDA runtime/driver
API and cuDNN version queries; unknown/not-used versions are null. For example,
CUDA `13030` means 13.3 and cuDNN `92500` means 9.25. These fields describe loaded
libraries, not a promise that every installed device supports that stack.

Current input codes: `size_limit`, `private_network`, `missing_extractor`,
`extractor_timeout`, `binary_input`, `unsupported`, `unsafe_symlink`,
`changed_while_reading`, `input_failed`. Fatal codes: `invalid_arguments`,
`invalid_configuration`, `writer_locked`, `fatal_error`. Rebuild history uses
`inference_failed`. Messages are actionable human text, not stable parsable IDs.

`--quiet` suppresses all but `failed`/`fatal`. `--verbose` enables per-document
and checkpoint output. `--progress` shows lightweight stderr status only when
stderr is a TTY; it is not a detailed per-stage progress bar.

Exit codes: 0 complete success; 1 partial success; 2 invalid arguments/config;
3 fatal database/model/runtime failure; 4 writer lock unavailable. `doctor`
returns 1 for a database-open/schema error; absent optional tools/models are
reported without making offline doctor fatal. A compiled CUDA provider is not
proof of activation: actual ingestion verifies CUDA nodes in a warm-up profile.

## Retrieval

See [retrieval](retrieval.md) for the full ranking, bounds, output, and migration
contract. Search defaults to semantic mode, with explicit lexical/hybrid modes.
Lexical search loads no model. Filters apply before top-k; results contain stored
text, not generated answers or freshly opened sources. Search uses read-only
snapshots without a writer lock and does not persist query history. No matches
is success (code 0); `--quiet` suppresses results as well as completion events.
Schema 1 remains readable for semantic search; lexical/hybrid requires the
transactional schema-2 FTS backfill via `ree migrate` or another writable command.

`query_provider` is the actual provider used after any fallback, not the provider
of every stored vector. Dense retrieval emits the summary warning
`mixed_int8_fp16_compatibility_unqualified` to preserve the known model-decision
limitation. Null provider/token count means no query inference ran.

## Persistent query protocol

Streaming stdin and optional Unix-socket search use versioned, request-ID-tagged
JSONL; ordinary single-shot output remains unchanged. Bounded queues, complete
terminal responses, cancellation, slow-consumer handling, private socket ownership,
and engine-process idle release are specified in [speedup v2](speedup-v2.md).
Control/rejection events can precede earlier search completion: correlate IDs.
Protocol modes reject `--quiet`/`--progress`; explicit socket CLI clients retain
ordinary quiet/error behavior. Queries never acquire the ingestion writer lock.

## Identity and deletion

- Exact canonical directory invocations are independent synchronization roots.
  Re-ingestion never deletes rows owned by other roots. Overlapping roots can
  contain duplicate document URIs. File symlink entries preserve their absolute
  entry paths; directory symlinks are not followed.
- File invocations only replace that file's own root. Globs expand to independent
  file/directory invocations; a glob is **not** a persistent deletion scope. If
  later glob matches omit a file, its standalone source remains until removed.
- Stdin uses `stdin://sha256/<hash>` or `stdin://source/<NAME>` with `--source`.
  `--source` requires exactly one stdin input. Identical anonymous input dedupes.
- URL identity removes fragments and uses URL-library normalization of scheme,
  host, and default port. Query parameters/order are preserved. Redirects do not
  change the original source identity; final URL is recorded in metadata.
- Remote Git identity is its normalized HTTPS URL ending in `.git`. Document
  URIs append `#<repository-relative-path>`. Shallow default-branch checkouts are
  updated; `.git`, submodule execution, hooks, history, and outside-root symlink
  targets are not ingested. Local repositories are ordinary directories.

A discovered failed file is marked seen before extraction/inference results are
handled; its previous data survives. Traversal errors (including a broken link)
disable all deletion reconciliation for that root, conservatively. Fail-fast
stops new scheduling after the first observed failure, drains bounded in-flight
work, and does not reconcile deletions. Complete documents already pending may
still commit. A fatal failure leaves already committed documents intact and the
run is marked interrupted when the next writer opens the database.

## Security limitations

ree never runs shell strings or repository code deliberately. HTTPS Git disables
ambient Git configuration, interactive credentials, redirects, unsafe transports,
and hooks. URL and Git connections pin DNS results checked against protected IP
ranges. Converters are trusted installed executables, **not sandboxed**: their
own parser vulnerabilities, memory use, and side effects remain a trust boundary.
Do not ingest hostile office/archive inputs without an OS sandbox. CLI limits
are safeguards, not a hard process/container memory quota. Local directory
inputs intentionally do not apply ignore files; choose roots accordingly.
