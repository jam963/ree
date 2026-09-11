# Public SQLite schema, version 2

The authoritative DDL is [`src/storage/migrations.rs`](../src/storage/migrations.rs).
`PRAGMA user_version` and `schema_metadata.schema_version` are `2` for new/upgraded
databases. New databases and the additive v1-to-v2 migration are transactional;
newer unsupported versions are rejected, not modified. Version 2 adds a lexical
index without changing chunk IDs, vectors, model recipes, or generation semantics.

`ree migrate` acquires the writer lock and backfills from stored text; no originals
are opened and no model is downloaded or run. All writable opens also upgrade
older databases automatically. The backfill may take time and additional disk/WAL
space on large corpora; provision headroom and back up before upgrading. Failure
rolls back the migration. Earlier ree binaries reject a version-2 database.

Read-only opens never migrate. Schema 1 remains supported for semantic search,
status, sources, and doctor; lexical/hybrid search requires version 2 and fails
with an explicit `ree migrate` instruction otherwise. `status.schema_version`
reports the database's actual version.

Load sqlite-vec 0.1.6 into an external SQLite client before using vector SQL:

```sql
.load /path/to/vec0
PRAGMA foreign_keys = ON;
SELECT vec_version();
```

`ree` statically registers its own bundled extension. Ordinary external SQLite
clients do not inherit that registration. Treat these tables as a **read-only
public interface**; use `ree` for coordinated writes.

## Tables

- `models`: exact repository/revision, tokenizer revision, dimensions, timestamps,
  and `recipe_json` (prompts, pooling, normalization, distance, runtime, precision
  artifacts and SHA-256 hashes). CPU INT8 and CUDA FP16 share the same semantic
  recipe; a run can use both after fallback.
- `source_roots`: SHA-256 ID of the canonical identity, kind, source options,
  user metadata, and last attempted/successful run IDs. For Git, `options_json`
  contains repository URL, branch, and commit.
- `documents`: SHA-256 of root ID + NUL + URI, root foreign key, content/text
  SHA-256, extractor/version, processing recipe hash, size, modification time
  (nanoseconds since epoch as a decimal string), metadata, and last-seen run.
  Identity is unique **within a root**. Overlapping roots can own independent
  copies of the same public URI. Local directory symlink entries retain their
  absolute entry paths; direct file invocations resolve the symlink first.
- `chunks`: stable ID derived from document ID, processing recipe, ordinal, and
  exact text; zero-based ordinal; text/hash; half-open token and UTF-8 byte ranges;
  `token_ids_json` containing exact passage inputs including CLS/SEP; optional
  location JSON. Ranges refer to the extracted document, not original file bytes.
  Token offsets/count exclude special tokens. Notebook cell and PDF page ranges
  are available in document metadata rather than chunk `location_json`.
  PDF `pages` entries contain one-based `page`, half-open `byte_start`/`byte_end`
  (excluding the form-feed separator), and `ocr`. Notebook cells are zero-based.
- `embedding_generations`: model ID and `building`, `active`, or `retired` status.
  A partial unique index permits at most one active generation.
- `embedding_records`: integer `vector_id` mapping `(chunk_id,generation_id)` to
  a vector row. Foreign-key deletions trigger vector cleanup.
- `embeddings`: sqlite-vec `vec0`, `embedding float[768] distance_metric=cosine`,
  with an integer `generation_id` partition key. Row IDs match `vector_id`.
  Stored blobs are normalized float32, little-endian, 3,072 bytes per vector.
- `active_embeddings`: view of vector IDs/chunk IDs in the active generation.
- `chunk_search_ids` (v2): explicit `INTEGER PRIMARY KEY AUTOINCREMENT` mapping
  to unique chunk IDs, with cascading deletion. Unlike implicit `chunks.rowid`,
  these integer IDs survive VACUUM; they are internal, not public citation IDs.
  Complete document replacement may allocate new integer IDs even for identical
  chunk IDs.
- `chunk_search_content` (v2): view joining that mapping to exact `chunks.text`.
- `chunk_fts` (v2): FTS5 external-content index over the view using `unicode61`.
  Chunk insert, before-delete, and text-update triggers maintain it in the same
  transaction as document replacement/deletion. Before-delete captures old text
  before foreign-key cascades remove the mapping. Migration explicitly rebuilds
  the FTS index over all existing chunks; text is not duplicated in FTS.
- `runs`: UUID, kind, status, start/end times, aggregate JSON. A writer marks
  abandoned `running` runs `interrupted` on startup. Keep the latest 100 runs.
- `run_failures`: at most 1,000 persisted failures per run; all failure events
  still appear on stdout, and aggregate failure counts include every failure.
  Last-run references intentionally do not constrain history pruning.

## Nearest neighbors

`ree search "query text"` generates a compatible query vector locally and returns
stored result text/metadata; see [retrieval](retrieval.md). Direct SQL remains a
read-only public interface.

Alternatively, generate a compatible query vector externally: pinned tokenizer/model revision,
`query: ` prefix, CLS pooling, L2 normalization, 768 dimensions. Do not prepend a
query prompt to document embeddings. Bind `:query` as a JSON float array or
little-endian float32 blob.

```sql
SELECT e.distance, c.id AS chunk_id, c.text,
       d.uri, d.metadata_json, s.identity AS source, s.metadata_json AS source_metadata
FROM embeddings AS e
JOIN embedding_records AS r ON r.vector_id = e.rowid
JOIN chunks AS c ON c.id = r.chunk_id
JOIN documents AS d ON d.id = c.document_id
JOIN source_roots AS s ON s.id = d.source_root_id
WHERE e.embedding MATCH :query
  AND e.generation_id = (SELECT id FROM embedding_generations WHERE status = 'active')
  AND k = 10
ORDER BY e.distance;
```

**Filter the generation inside the vector query.** Filtering only after a KNN
query can return too few results while a rebuild has shadow vectors present.
For application-specific metadata filters, use an exact scan over eligible rows
or overfetch and account for filtering; do not assume a SQL join changes vec0's K.

```sql
SELECT c.text, d.uri, vec_distance_cosine(e.embedding, :query) AS distance
FROM active_embeddings AS a
JOIN embeddings AS e ON e.rowid = a.vector_id
JOIN chunks AS c ON c.id = a.chunk_id
JOIN documents AS d ON d.id = c.document_id
WHERE d.source_root_id = :root_id
ORDER BY distance LIMIT 10;
```

## Lexical SQL (version 2)

Bind `:match` as a safely generated FTS expression, not unescaped user text.
`ree search --mode lexical` treats whitespace-separated inputs as literal quoted
phrases joined by OR; it does not expose raw FTS syntax.

```sql
SELECT c.id, c.text, d.uri, bm25(chunk_fts) AS score
FROM chunk_fts
JOIN chunk_search_ids AS m ON m.id = chunk_fts.rowid
JOIN chunks AS c ON c.id = m.chunk_id
JOIN documents AS d ON d.id = c.document_id
JOIN active_embeddings AS a ON a.chunk_id = c.id
WHERE chunk_fts MATCH :match
ORDER BY score, c.id
LIMIT 10;
```

BM25 sorts ascending (lower is better). For consistency across multiple statements,
select generation, rank, and hydrate inside one short read transaction. The CLI
performs expensive query embedding before opening this final snapshot, then
revalidates the active recipe. It never holds a read transaction while emitting
results to a slow stdout consumer.

## Atomicity and rebuilds

A replacement validates all vectors before deleting old content. Chunks,
metadata, vector records, and vector rows commit in one document transaction.
Readers see either the old or new complete document. Root reconciliation is a
separate transaction after discovery finishes without traversal errors or an
interrupted/fail-fast schedule.

Rebuild creates/reuses a building generation, commits batches of at most 64
stored input sequences, and fills only missing chunks. Keyset pagination avoids
rescanning completed prefixes for every batch; a resumed process starts from the
beginning to cover any holes in checkpointed progress. Activation requires full
chunk coverage, retires the old generation, activates the new one, and removes
retired vectors in one transaction. Failure leaves the active generation intact.
Stored token IDs prevent subtle retokenization drift when exact chunk text has
whitespace or subword boundaries. A tokenizer change requires explicit source
reingestion/rechunking, not a vector-only rebuild.
