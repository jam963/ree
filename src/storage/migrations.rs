pub const VERSION: i64 = 1;
pub const INITIAL: &str = r#"
CREATE TABLE schema_metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
INSERT INTO schema_metadata VALUES('schema_version','1');
CREATE TABLE models(
 id TEXT PRIMARY KEY, model_id TEXT NOT NULL, revision TEXT NOT NULL,
 tokenizer_revision TEXT NOT NULL, recipe_json TEXT NOT NULL,
 dimensions INTEGER NOT NULL CHECK(dimensions=768), created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE source_roots(
 id TEXT PRIMARY KEY, kind TEXT NOT NULL, identity TEXT NOT NULL UNIQUE,
 options_json TEXT NOT NULL DEFAULT '{}', metadata_json TEXT NOT NULL DEFAULT '{}',
 last_attempted_run TEXT, last_successful_run TEXT
);
CREATE TABLE runs(
 id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
 started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, finished_at TEXT,
 counts_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE run_failures(
 id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
 source TEXT NOT NULL, error_code TEXT NOT NULL, message TEXT NOT NULL
);
CREATE TABLE documents(
 id TEXT PRIMARY KEY, source_root_id TEXT NOT NULL REFERENCES source_roots(id) ON DELETE CASCADE,
 uri TEXT NOT NULL, content_hash TEXT NOT NULL, text_hash TEXT NOT NULL,
 media_type TEXT NOT NULL, extractor TEXT NOT NULL, recipe TEXT NOT NULL,
 size_bytes INTEGER NOT NULL, modified_ns TEXT, metadata_json TEXT NOT NULL DEFAULT '{}',
 last_seen_run TEXT NOT NULL, UNIQUE(source_root_id,uri)
);
CREATE INDEX documents_root_seen ON documents(source_root_id,last_seen_run);
CREATE TABLE chunks(
 id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
 ordinal INTEGER NOT NULL, text TEXT NOT NULL, token_start INTEGER NOT NULL,
 token_end INTEGER NOT NULL, token_count INTEGER NOT NULL, byte_start INTEGER NOT NULL,
 byte_end INTEGER NOT NULL, chunk_hash TEXT NOT NULL, token_ids_json TEXT NOT NULL, location_json TEXT NOT NULL DEFAULT '{}',
 UNIQUE(document_id,ordinal)
);
CREATE TABLE embedding_generations(
 id INTEGER PRIMARY KEY, model_id TEXT NOT NULL REFERENCES models(id),
 status TEXT NOT NULL CHECK(status IN ('building','active','retired')),
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX one_active_generation ON embedding_generations(status) WHERE status='active';
CREATE TABLE embedding_records(
 vector_id INTEGER PRIMARY KEY, chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
 generation_id INTEGER NOT NULL REFERENCES embedding_generations(id) ON DELETE CASCADE,
 UNIQUE(chunk_id,generation_id)
);
CREATE INDEX embedding_records_generation ON embedding_records(generation_id,chunk_id);
CREATE VIRTUAL TABLE embeddings USING vec0(embedding float[768] distance_metric=cosine, generation_id integer partition key);
CREATE TRIGGER remove_vector AFTER DELETE ON embedding_records BEGIN
 DELETE FROM embeddings WHERE rowid=old.vector_id;
END;
CREATE VIEW active_embeddings AS
 SELECT r.vector_id,r.chunk_id,r.generation_id FROM embedding_records r
 JOIN embedding_generations g ON g.id=r.generation_id WHERE g.status='active';
PRAGMA user_version=1;
"#;
