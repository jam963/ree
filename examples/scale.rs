//! Storage-only synthetic load test. Never produces usable model embeddings.
use anyhow::{Result, ensure};
use clap::Parser;
use ree::{
    chunk::Chunk,
    model,
    storage::{Database, Document, Store, WriterLock},
    util::hash,
};
use serde_json::json;
use std::{path::PathBuf, time::Instant};
#[derive(Parser)]
struct Args {
    /// Must be a new database: synthetic vectors must never mix with real data.
    #[arg(long)]
    db: PathBuf,
    #[arg(long, default_value_t = 1000000)]
    chunks: usize,
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !args.db.exists(),
        "scale benchmark requires a new database path"
    );
    ensure!(args.chunks > 0, "chunks must be positive");
    let _lock = WriterLock::acquire(&args.db)?;
    let mut db = Database::open(&args.db, true)?;
    db.conn.execute(
        "INSERT INTO schema_metadata VALUES('synthetic_vectors','true')",
        [],
    )?;
    db.register_model()?;
    let run = db.start_run("synthetic-scale")?;
    let root = db.root(
        "benchmark",
        "synthetic://scale",
        &json!({"synthetic":true}),
        &run,
    )?;
    let start = Instant::now();
    let mut seed = 42u64;
    for first in (0..args.chunks).step_by(1000) {
        let end = (first + 1000).min(args.chunks);
        let mut chunks = Vec::new();
        let mut vectors = Vec::new();
        for index in first..end {
            let text = format!("Synthetic scale test chunk {index}");
            chunks.push(Chunk {
                text: text.clone(),
                token_start: 0,
                token_end: 1,
                byte_start: 0,
                byte_end: text.len(),
                input_ids: vec![0, 42, 2],
            });
            let mut vector = Vec::with_capacity(768);
            for _ in 0..768 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                vector.push(((seed >> 32) as u32 as f64 / u32::MAX as f64 - 0.5) as f32);
            }
            model::normalize(&mut vector)?;
            vectors.push(vector);
        }
        let id = hash(format!("synthetic:{first}").as_bytes());
        let document = Document {
            id,
            root: root.clone(),
            uri: format!("synthetic://scale/{first}"),
            content_hash: "synthetic".into(),
            text_hash: "synthetic".into(),
            media_type: "text/plain".into(),
            extractor: "synthetic-scale".into(),
            recipe: "synthetic-not-a-model".into(),
            size: 0,
            modified_ns: None,
            metadata: json!({"synthetic":true}),
            run: run.clone(),
        };
        db.replace_document(&document, &chunks, &vectors)?;
        if end % 100000 == 0 {
            eprintln!("inserted {end} synthetic chunks");
        }
    }
    let insert_seconds = start.elapsed().as_secs_f64();
    let mut query = vec![0f32; 768];
    query[0] = 1.;
    let blob: Vec<u8> = query.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut samples = Vec::new();
    let generation = db.active_generation()?;
    for _ in 0..5 {
        let start = Instant::now();
        let mut stmt=db.conn.prepare("SELECT rowid,distance FROM embeddings WHERE embedding MATCH ?1 AND generation_id=?2 AND k=10 ORDER BY distance")?;
        let rows = stmt
            .query_map(rusqlite::params![blob, generation], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        ensure!(rows.len() == args.chunks.min(10), "missing nearest vectors");
        samples.push(start.elapsed().as_secs_f64());
    }
    db.conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")?;
    let report = json!({"type":"synthetic_scale","chunks":args.chunks,"insert_seconds":insert_seconds,"chunks_per_second":args.chunks as f64/insert_seconds,"query_seconds":samples,"database_bytes":args.db.metadata()?.len(),"warning":"synthetic vectors; not model quality or inference measurements"});
    db.finish_run(&run, 0, &report)?;
    println!("{report}");
    Ok(())
}
