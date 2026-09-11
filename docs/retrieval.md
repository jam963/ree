# Retrieval

`ree search` searches the stored corpus locally, without reopening original files,
fetching source URLs, generating answers, or persisting query text/history. This
explicitly extends the original ingestion-only scope in historical `PLAN.md` and
`beta-roadmap.md`. The pinned Arctic 768 recipe and existing ingestion/fallback
behavior are unchanged.

```sh
ree search "how does CUDA fallback work?"
ree search "SQLITE_BUSY" --mode lexical
ree search "writer lock timeout" --mode hybrid --limit 20
ree search "OCR" --root /absolute/source/root --media-type application/pdf
ree search "transactional replacement" --device cpu
ree --db /path/to/ree.db search "scheduler"
```

Quote multiword queries. A file called `search` can still be ingested as
`ree ./search`. Options can follow `search`; global options such as `--db` and
`--device` also work before it. Search-only options have no environment/config-file
overrides. The default mode is `semantic` and the default limit is 10.

## Repeated queries

`ree search --stream` reads versioned JSONL requests and reuses a query-only engine.
`ree worker` optionally exposes the same service on a private Unix socket; route
explicitly with `ree search "query" --socket PATH`. Neither is required for ordinary
search. Both release their disposable engine process after an idle timeout while
the supervisor stays available. Each request reopens the read-only DB and retains
the snapshot/recipe validation described below. See [speedup v2](speedup-v2.md) for
framing, permissions, queue/output bounds, cancellation, failures and timings.

## Modes and scores

| Mode | Ranking | Model required? |
|---|---|---|
| `semantic` | Exact cosine distance; lower is better | Yes, except empty scopes |
| `lexical` | FTS5 BM25; lower (more negative) is better | No |
| `hybrid` | Reciprocal rank fusion; higher is better | Yes, except empty scopes |

Hybrid retrieves `min(4096, max(100, 4 * limit))` candidates independently from each
branch **after applying the same filters**. The score for a chunk is the sum of
`1 / (60 + rank)` across lists containing it, using one-based ranks. Chunk IDs are
deduplicated across branches; raw cosine distances and BM25 values are not added.
A missing lexical match does not remove a dense candidate. Scores missing from a
branch are null, not zero. Candidate depth is fixed and reported, not tuned as a
claim of quality; hybrid does not promise an exhaustive fused ranking of all rows.

These scores are not probabilities or relevance-confidence percentages. Semantic
search returns nearest chunks even for an unrelated query. Chunk-level retrieval
may return overlapping windows, several chunks from one document, and independent
copies from overlapping source roots. Document grouping, context expansion,
reranking, and approximate-nearest-neighbor indexing are not implemented.

Results sort by their mode's score and then chunk ID. Unfiltered dense KNN uses
sqlite-vec 0.1.6: when a distance tie crosses the candidate cutoff, **membership in
that tie is chosen by vec0**, not guaranteed to be the lexicographically smallest
IDs. Returned tied candidates are sorted by chunk ID. Filtered exact scans and
lexical queries use chunk ID as the SQL tie-breaker; fusion ties also use chunk ID.

## Query and filter bounds

- Queries must be nonblank, contain no NUL, and be at most 65,536 UTF-8 bytes.
- Dense encoding accepts at most **512 model tokens**, including the `query: `
  prefix and CLS/SEP. It rejects overflow without truncation or multiple windows.
  This is independent of ingestion's `chunk_size` and `overlap` configuration.
- The prefix is always added once by ree; don't manually prepend it to your query.
  User text, including leading/trailing whitespace, is otherwise preserved.
- Lexical/hybrid queries accept at most 512 whitespace-separated terms. Each is
  quoted/escaped as a literal FTS phrase, then joined by OR. `OR`, `NEAR`, column
  names, quotes, and `*` in input are **not** exposed as FTS query syntax.
- FTS5 `unicode61` still tokenizes punctuation and performs its normal case/diacritic
  handling. For example, `SQLITE_BUSY` is tokenized as adjacent words, not promised
  as an exact identifier/substring match. There is no language-specific stemming
  or specialized code/CJK analyzer in this version. Punctuation-only queries can
  return no lexical matches.
- `--limit` must be 1..1000. Filters must be nonblank and contain no NUL.
- `--root` selects one exact source-root ID or identity. Local paths also try their
  canonical identity. This is not recursive URI-prefix matching. An exact ID wins
  over an ambiguous identity; unknown roots return no matches.
- `--media-type` is an exact match on the stored document media type.

No eligible active chunks returns a successful empty response **without loading
or downloading tokenizer/model artifacts**. Byte/argument bounds still apply;
model-token overflow is checked only when dense encoding is necessary. A missing
or uninitialized database is an actionable error, not silently created by search.
Synthetic benchmark databases are rejected in all modes.

## Output

Stdout follows the existing JSONL event stream. Dense first use may emit download,
runtime, and fallback events before results. Consumers should ignore unknown event
types. Each hit is a `search_result` with:

- `rank`, `chunk_id`, `document_id`, zero-based `ordinal`, exact stored `text`;
- `uri`, `source_id`, source identity in `source`, and `media_type`;
- half-open `byte_start`/`byte_end` and `token_start`/`token_end` in extracted text;
- raw chunk `location`, `document_metadata`, and `source_metadata`;
- `locations`: available PDF page/notebook cell ranges overlapping the chunk;
- `distance`, `bm25`, and `fusion_score` (irrelevant/unavailable fields are null).

PDF pages are one-based and notebook cells zero-based. Byte ranges are **not**
original PDF/file bytes, and ree does not invent line/page mappings where absent.
Chunk IDs, not internal FTS integer IDs, are suitable for identifying returned
chunks. Replacement/rechunking can change those IDs.

A final `search_completed` event contains `mode`, `results` (returned count), active
`generation` and `model_key`, `query_provider`, `query_tokens`, `candidate_limit`,
`dense_candidates`, `lexical_candidates`, `warnings`, and `timings`:

- `tokenization_ms`: tokenizer provisioning/load plus tokenization;
- `initialization_ms`: model provisioning/load and provider-verification warmup;
- `embedding_ms`: query inference, including any runtime fallback;
- `sql_ms`: snapshot validation/scope checks and candidate retrieval;
- `hydration_ms`: reading/decoding result text and metadata;
- `elapsed_ms`: overall retrieval call, excluding CLI/config/database open and
  output flushing; timer resolution is milliseconds and zero is valid.

Generation/model are null if no active generation exists. Provider/token count are
null when no query was embedded, including lexical mode and empty scopes. Query
text itself is not copied into the summary or persisted in the database.
`--quiet` retains the existing errors-only behavior, suppressing results too.
Exit codes remain 0 for success (including no matches), 2 for invalid arguments,
and 3 for database/model/runtime failure. Search does not acquire a writer lock.

## Storage, migration, and concurrent writers

Semantic search works on existing schema-1 databases without migration. For
lexical/hybrid search:

```sh
ree --db /path/to/ree.db migrate
```

This explicitly acquires the writer lock and transactionally backfills the FTS
index from stored chunk text. No model or re-embedding is needed. New databases
and all writable opens use schema 2; read-only commands never migrate. Allocate
additional disk/WAL headroom for the index/backfill on large corpora. Failure rolls
back the migration; old ree binaries reject schema 2. See [schema](schema.md).

Search first checks scope/recipe in a short snapshot and releases it before model
initialization. After embedding, it begins a new snapshot, revalidates the active
recipe, and selects the active generation. Ranking and hydration both use that
snapshot. A compatible rebuild activation during embedding is allowed; an
unsupported new recipe fails rather than searching incompatible vectors.

Unfiltered dense search partitions by generation **inside** vec0 KNN. Filtered
search computes exact distances over eligible active rows before top-k; it does
not filter a global top-k result. Readers retain old complete rows during document
replacement/deletion or rebuild activation, and release their snapshot before
writing output. WAL/SHM sidecars may still be used by SQLite; database-read-only
does not mean the model cache or SQLite sidecar directory is never touched.

## Runtime and remaining qualifications

Queries use the same pinned tokenizer/artifacts, CLS pooling, L2 normalization,
and 768-dimensional vectors as ingestion. Interactive initialization uses singleton
batches and skips throughput calibration, profile reads/writes, and batch growth.
ONNX graph/provider verification warmup is retained. `auto` may fall back to CPU;
explicit CUDA still refuses CPU fallback. `--batch-size` controls ingestion, not
interactive query batching.

The existing [model decision](model-decision-v1.md) records unresolved INT8/FP16
and batch-shape ranking differences. Search reports the actual query provider and
`mixed_int8_fp16_compatibility_unqualified` whenever dense encoding occurred. This
is a general qualification warning, **not** a claim that the current index is
known to be mixed. Existing vectors do not have per-vector provider provenance,
so ree cannot automatically match the query provider to every stored vector or
promise precision-independent rankings.

Offline tests cover ranking, filters, migration/backfill/rollback, transactional
FTS cleanup, snapshot consistency, rebuild activation, output, argument handling,
and interactive scheduler behavior. An ignored development-host real-model test
covers query-token parity and a small CPU end-to-end retrieval smoke test. It is
not part of ordinary CI, and adding it does not claim a new real-model evaluation
has run. No comparative model selection or frozen pilot retuning is part of this
feature.

Use the [retrieval latency harness](benchmarks.md#retrieval-command-latency) to
measure first-session and reused-session timings on a real indexed corpus before
claiming interactive latency at scale. Exact vector scans remain the baseline;
there is no newly qualified million-chunk latency or hybrid quality result.
