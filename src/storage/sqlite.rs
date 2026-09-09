use super::{Store, migrations};
use crate::{chunk::Chunk, error::AppError, model, util::hash};
use anyhow::{Context, Result, ensure};
use fs2::FileExt;
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::{Value, json};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
    sync::Once,
    time::{Duration, Instant},
};

#[derive(Debug)]
pub struct WriterLock {
    _file: File,
}
impl WriterLock {
    pub fn acquire(db: &Path) -> Result<Self> {
        if let Some(p) = db.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(p)?;
        }
        // Resolve symlink aliases before deriving the lock path.
        let db = if db.exists() {
            db.canonicalize()?
        } else {
            let parent = db
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            parent
                .canonicalize()?
                .join(db.file_name().context("database requires a filename")?)
        };
        let mut name = db.as_os_str().to_owned();
        name.push(".lock");
        let mut f = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(name)?;
        let start = Instant::now();
        loop {
            match f.try_lock_exclusive() {
                Ok(()) => break,
                Err(e)
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        && start.elapsed() < Duration::from_secs(2) =>
                {
                    std::thread::sleep(Duration::from_millis(50))
                }
                Err(e) => {
                    let mut owner = String::new();
                    let _ = (&f).take(4096).read_to_string(&mut owner);
                    return Err(AppError::new(
                        4,
                        format!("writer lock unavailable ({e}); owner: {owner}"),
                    )
                    .into());
                }
            }
        }
        f.set_len(0)?;
        f.seek(SeekFrom::Start(0))?;
        write!(
            f,
            "pid={} started={:?}",
            std::process::id(),
            std::time::SystemTime::now()
        )?;
        f.sync_data()?;
        Ok(Self { _file: f })
    }
}

static VEC_INIT: Once = Once::new();
fn register_vec() {
    VEC_INIT.call_once(|| unsafe {
        // sqlite-vec exposes the standard SQLite extension entrypoint. SQLite owns
        // registration for the lifetime of this statically linked process.
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<
            unsafe extern "C" fn(),
            unsafe extern "C" fn(
                *mut rusqlite::ffi::sqlite3,
                *mut *mut std::ffi::c_char,
                *const rusqlite::ffi::sqlite3_api_routines,
            ) -> std::ffi::c_int,
        >(sqlite_vec::sqlite3_vec_init)));
    });
}

pub struct Database {
    pub conn: Connection,
}
#[derive(Debug)]
pub struct Document {
    pub id: String,
    pub root: String,
    pub uri: String,
    pub content_hash: String,
    pub text_hash: String,
    pub media_type: String,
    pub extractor: String,
    pub recipe: String,
    pub size: u64,
    pub modified_ns: Option<String>,
    pub metadata: Value,
    pub run: String,
}
impl Database {
    pub fn open(path: &Path, writable: bool) -> Result<Self> {
        register_vec();
        let conn = if writable {
            Connection::open(path)?
        } else {
            Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)?
        };
        conn.busy_timeout(Duration::from_secs(2))?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        let version: i64 = conn.pragma_query_value(None, "user_version", |r| r.get(0))?;
        ensure!(
            version <= migrations::VERSION,
            "database schema {version} is newer than supported schema {}",
            migrations::VERSION
        );
        if writable {
            conn.pragma_update(None, "journal_mode", "WAL")?;
            if version == 0 {
                conn.execute_batch("BEGIN IMMEDIATE")?;
                match conn.execute_batch(migrations::INITIAL) {
                    Ok(()) => conn.execute_batch("COMMIT")?,
                    Err(e) => {
                        let _ = conn.execute_batch("ROLLBACK");
                        return Err(e.into());
                    }
                }
            }
            conn.execute("UPDATE runs SET status='interrupted',finished_at=CURRENT_TIMESTAMP WHERE status='running'", [])?;
        } else {
            ensure!(
                version == migrations::VERSION,
                "database needs migration; run an ingestion first"
            );
        }
        Ok(Self { conn })
    }
    pub fn register_model(&self) -> Result<()> {
        self.conn.execute("INSERT OR IGNORE INTO models(id,model_id,revision,tokenizer_revision,recipe_json,dimensions) VALUES(?1,?2,?3,?3,?4,768)", params![model::MODEL_KEY,model::MODEL_ID,model::REVISION,model::recipe().to_string()])?;
        self.conn.execute("INSERT INTO embedding_generations(model_id,status) SELECT ?1,'active' WHERE NOT EXISTS(SELECT 1 FROM embedding_generations WHERE status='active')", [model::MODEL_KEY])?;
        Ok(())
    }
    pub fn start_run(&self, kind: &str) -> Result<String> {
        let id = uuid::Uuid::new_v4().to_string();
        self.conn.execute(
            "INSERT INTO runs(id,kind,status) VALUES(?1,?2,'running')",
            params![id, kind],
        )?;
        Ok(id)
    }
    pub fn finish_run(&self, run: &str, failed: usize, counts: &Value) -> Result<()> {
        self.conn.execute(
            "UPDATE runs SET status=?2,finished_at=CURRENT_TIMESTAMP,counts_json=?3 WHERE id=?1",
            params![
                run,
                if failed == 0 { "complete" } else { "partial" },
                counts.to_string()
            ],
        )?;
        self.conn.execute("DELETE FROM runs WHERE id NOT IN (SELECT id FROM runs ORDER BY started_at DESC,rowid DESC LIMIT 100)", [])?;
        Ok(())
    }
    pub fn failure(&self, run: &str, source: &str, code: &str, message: &str) -> Result<()> {
        self.conn.execute("INSERT INTO run_failures(run_id,source,error_code,message) SELECT ?1,?2,?3,?4 WHERE (SELECT count(*) FROM run_failures WHERE run_id=?1)<1000",params![run,source,code,message])?;
        Ok(())
    }
    pub fn root(&self, kind: &str, identity: &str, metadata: &Value, run: &str) -> Result<String> {
        let id = hash(identity.as_bytes());
        self.conn.execute("INSERT INTO source_roots(id,kind,identity,metadata_json,last_attempted_run) VALUES(?1,?2,?3,?4,?5) ON CONFLICT(identity) DO UPDATE SET metadata_json=excluded.metadata_json,last_attempted_run=excluded.last_attempted_run",params![id,kind,identity,metadata.to_string(),run])?;
        Ok(id)
    }
    pub fn seen(&self, id: &str, run: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE documents SET last_seen_run=?2 WHERE id=?1",
            params![id, run],
        )?;
        Ok(())
    }
    pub fn unchanged_content(&self, id: &str, content_hash: &str, recipe: &str) -> Result<bool> {
        Ok(self
            .conn
            .query_row(
                "SELECT content_hash=?2 AND recipe=?3 FROM documents WHERE id=?1",
                params![id, content_hash, recipe],
                |r| r.get(0),
            )
            .optional()?
            .unwrap_or(false))
    }
    pub fn unchanged(
        &self,
        id: &str,
        content_hash: &str,
        recipe: &str,
        metadata: &Value,
    ) -> Result<bool> {
        Ok(self.conn.query_row("SELECT content_hash=?2 AND recipe=?3 AND metadata_json=?4 FROM documents WHERE id=?1",params![id,content_hash,recipe,metadata.to_string()],|r|r.get(0)).optional()?.unwrap_or(false))
    }
    pub fn finish_root(
        &mut self,
        root: &str,
        run: &str,
        prune: bool,
        success: bool,
    ) -> Result<usize> {
        let tx = self.conn.transaction()?;
        let n = if prune {
            tx.execute(
                "DELETE FROM documents WHERE source_root_id=?1 AND last_seen_run<>?2",
                params![root, run],
            )?
        } else {
            0
        };
        if success {
            tx.execute(
                "UPDATE source_roots SET last_successful_run=?2 WHERE id=?1",
                params![root, run],
            )?;
        }
        tx.commit()?;
        Ok(n)
    }
    pub fn status(&self) -> Result<Value> {
        let count = |table: &str| -> Result<i64> {
            Ok(self
                .conn
                .query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))?)
        };
        let generation: Option<i64> = self
            .conn
            .query_row(
                "SELECT id FROM embedding_generations WHERE status='active'",
                [],
                |r| r.get(0),
            )
            .optional()?;
        Ok(
            json!({"type":"status","schema_version":migrations::VERSION,"sources":count("source_roots")?,"documents":count("documents")?,"chunks":count("chunks")?,"embeddings":count("active_embeddings")?,"generation":generation}),
        )
    }
    pub fn sources(&self) -> Result<Vec<Value>> {
        let mut stmt = self.conn.prepare("SELECT id,kind,identity,metadata_json,last_attempted_run,last_successful_run FROM source_roots ORDER BY identity")?;
        let rows = stmt.query_map([], |r| Ok(json!({"type":"source","id":r.get::<_,String>(0)?,"kind":r.get::<_,String>(1)?,"identity":r.get::<_,String>(2)?,"metadata":serde_json::from_str::<Value>(&r.get::<_,String>(3)?).unwrap_or(Value::Null),"last_attempted_run":r.get::<_,Option<String>>(4)?,"last_successful_run":r.get::<_,Option<String>>(5)?})))?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }
    pub fn remove(&mut self, source: &str) -> Result<usize> {
        let tx = self.conn.transaction()?;
        let n = tx.execute(
            "DELETE FROM source_roots WHERE id=?1 OR identity=?1",
            [source],
        )?;
        tx.commit()?;
        Ok(n)
    }
    pub fn active_generation(&self) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT id FROM embedding_generations WHERE status='active'",
            [],
            |r| r.get(0),
        )?)
    }
    pub fn rebuild_generation(&self) -> Result<i64> {
        if let Some(id) = self.conn.query_row("SELECT id FROM embedding_generations WHERE status='building' AND model_id=?1 ORDER BY id DESC LIMIT 1",[model::MODEL_KEY],|r|r.get(0)).optional()? { return Ok(id); }
        self.conn.execute(
            "INSERT INTO embedding_generations(model_id,status) VALUES(?1,'building')",
            [model::MODEL_KEY],
        )?;
        Ok(self.conn.last_insert_rowid())
    }
    pub fn pending_chunks(&self, generation: i64, limit: usize) -> Result<Vec<(String, Vec<i64>)>> {
        let mut stmt = self.conn.prepare("SELECT id,token_ids_json FROM chunks c WHERE NOT EXISTS(SELECT 1 FROM embedding_records r WHERE r.chunk_id=c.id AND r.generation_id=?1) ORDER BY id LIMIT ?2")?;
        let rows = stmt.query_map(params![generation, limit], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        })?;
        rows.map(|r| {
            let (id, ids) = r?;
            Ok((id, serde_json::from_str(&ids)?))
        })
        .collect()
    }
    pub fn write_vectors(
        &mut self,
        generation: i64,
        ids: &[String],
        vectors: &[Vec<f32>],
    ) -> Result<()> {
        ensure!(ids.len() == vectors.len(), "vector count mismatch");
        let tx = self.conn.transaction()?;
        for (id, v) in ids.iter().zip(vectors) {
            insert_vector(&tx, generation, id, v)?;
        }
        tx.commit()?;
        Ok(())
    }
    pub fn activate(&mut self, generation: i64) -> Result<()> {
        let tx = self.conn.transaction()?;
        let missing: i64 = tx.query_row("SELECT count(*) FROM chunks c WHERE NOT EXISTS(SELECT 1 FROM embedding_records r WHERE r.chunk_id=c.id AND r.generation_id=?1)",[generation],|r|r.get(0))?;
        ensure!(
            missing == 0,
            "cannot activate incomplete generation: {missing} missing vectors"
        );
        let building: bool = tx.query_row(
            "SELECT status='building' FROM embedding_generations WHERE id=?1",
            [generation],
            |r| r.get(0),
        )?;
        ensure!(building, "generation is not building");
        tx.execute(
            "UPDATE embedding_generations SET status='retired' WHERE status='active'",
            [],
        )?;
        tx.execute(
            "UPDATE embedding_generations SET status='active' WHERE id=?1",
            [generation],
        )?;
        // Retain only the active and resumable generation, bounding vector storage.
        tx.execute(
            "DELETE FROM embedding_generations WHERE status='retired'",
            [],
        )?;
        tx.commit()?;
        Ok(())
    }
}
fn insert_vector(
    tx: &rusqlite::Transaction<'_>,
    generation: i64,
    chunk: &str,
    vector: &[f32],
) -> Result<()> {
    model::validate_vector(vector)?;
    tx.execute(
        "INSERT INTO embedding_records(chunk_id,generation_id) VALUES(?1,?2)",
        params![chunk, generation],
    )?;
    let id = tx.last_insert_rowid();
    let blob: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();
    tx.execute(
        "INSERT INTO embeddings(rowid,embedding,generation_id) VALUES(?1,?2,?3)",
        params![id, blob, generation],
    )?;
    Ok(())
}
impl Store for Database {
    fn replace_document(
        &mut self,
        d: &Document,
        chunks: &[Chunk],
        vectors: &[Vec<f32>],
    ) -> Result<()> {
        ensure!(chunks.len() == vectors.len(), "chunk/vector count mismatch");
        for v in vectors {
            model::validate_vector(v)?;
        }
        let generation = self.active_generation()?;
        let tx = self.conn.transaction()?;
        tx.execute("DELETE FROM documents WHERE id=?1", [&d.id])?;
        tx.execute("INSERT INTO documents(id,source_root_id,uri,content_hash,text_hash,media_type,extractor,recipe,size_bytes,modified_ns,metadata_json,last_seen_run) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",params![d.id,d.root,d.uri,d.content_hash,d.text_hash,d.media_type,d.extractor,d.recipe,d.size,d.modified_ns,d.metadata.to_string(),d.run])?;
        for (i, (c, v)) in chunks.iter().zip(vectors).enumerate() {
            let id = hash(format!("{}\0{}\0{i}\0{}", d.id, d.recipe, c.text).as_bytes());
            tx.execute("INSERT INTO chunks(id,document_id,ordinal,text,token_start,token_end,token_count,byte_start,byte_end,chunk_hash,token_ids_json) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",params![id,d.id,i,c.text,c.token_start,c.token_end,c.token_end-c.token_start,c.byte_start,c.byte_end,hash(c.text.as_bytes()),serde_json::to_string(&c.input_ids)?])?;
            insert_vector(&tx, generation, &id, v)?;
        }
        tx.commit()?;
        Ok(())
    }
}
