//! Local synthetic million-file pipeline test. Real tokenizer, FAKE vectors.
//! Uses the production discovery/extraction/sync/writer/rebuild paths.
#[path = "support/telemetry.rs"]
mod telemetry;
use anyhow::{Result, ensure};
use clap::Parser;
use ree::{
    chunk::Chunk,
    cli::Options,
    config::Config,
    events::Events,
    model::{Engine, LocalEngine},
    pipeline,
    storage::{Database, WriterLock},
};
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Parser)]
struct Args {
    /// Existing scratch filesystem. Generated workspace is private and removed
    /// on completion/failure unless --keep is supplied.
    #[arg(long, default_value = "/tmp")]
    scratch_dir: PathBuf,
    #[arg(long, default_value_t = 1000000)]
    files: usize,
    #[arg(long)]
    keep: bool,
}
struct SyntheticEngine {
    tokenizer: LocalEngine,
    embedded: usize,
}
impl Engine for SyntheticEngine {
    fn chunk(&mut self, text: &str, events: &mut Events) -> Result<Vec<Chunk>> {
        self.tokenizer.chunk(text, events)
    }
    fn embed(&mut self, inputs: &[Vec<i64>], _: &mut Events) -> Result<Vec<Vec<f32>>> {
        self.embedded += inputs.len();
        Ok(inputs
            .iter()
            .map(|ids| {
                let mut v = vec![0.; 768];
                v[ids
                    .iter()
                    .fold(0usize, |n, &id| n.wrapping_add(id as usize))
                    % 768] = 1.;
                v
            })
            .collect())
    }
    fn metrics(&self) -> Value {
        json!({"synthetic_vectors":true,"embedded":self.embedded})
    }
}
fn file(root: &Path, index: usize) -> PathBuf {
    root.join(format!("{:06}", index / 1000))
        .join(format!("{index:09}.txt"))
}
fn free_bytes(path: &Path) -> Result<u64> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(path.as_os_str().as_bytes())?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    ensure!(
        unsafe { libc::statvfs(path.as_ptr(), stats.as_mut_ptr()) } == 0,
        "cannot query scratch filesystem"
    );
    let stats = unsafe { stats.assume_init() };
    Ok(stats.f_bavail.saturating_mul(stats.f_frsize))
}
fn measure(label: &str, work: impl FnOnce() -> Result<()>) -> Result<()> {
    let sampler = telemetry::Sampler::start(None);
    let start = Instant::now();
    work()?;
    println!(
        "{}",
        json!({"type":"pipeline_scale_phase","phase":label,"elapsed_seconds":start.elapsed().as_secs_f64(),"resources":sampler.finish()})
    );
    Ok(())
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        std::env::var_os("CI").is_none()
            && std::fs::read_to_string("/proc/cpuinfo")?.contains("AMD Ryzen 9 8945HS"),
        "pipeline scale qualification is local-development-machine only"
    );
    ensure!(
        (1..=1000000).contains(&args.files),
        "files must be in 1..=1000000"
    );
    let required = (args.files as u64)
        .saturating_mul(14000)
        .saturating_add(512 * 1024 * 1024);
    ensure!(
        free_bytes(&args.scratch_dir)? >= required,
        "insufficient scratch space: reserve at least {required} bytes"
    );
    let temporary = tempfile::Builder::new()
        .prefix("ree-pipeline-scale-")
        .tempdir_in(&args.scratch_dir)?;
    let workspace = temporary.path().to_path_buf();
    // Retention is explicit, including failures; default RAII cleanup never
    // touches any path outside this newly created private workspace.
    let _cleanup = if args.keep {
        let _ = temporary.keep();
        None
    } else {
        Some(temporary)
    };
    let docs = workspace.join("documents");
    let db_path = workspace.join("ree.db");
    let config_path = workspace.join("config.toml");
    std::fs::write(&config_path, "")?;
    let options = Options {
        db: Some(db_path.clone()),
        config: Some(config_path),
        device: Some("cpu".into()),
        ..Default::default()
    };
    let mut config = Config::load(&options)?;
    config.metadata = json!({"synthetic_vectors":true,"purpose":"pipeline scale; not retrieval"});
    let mut engine = SyntheticEngine {
        tokenizer: LocalEngine::new(config.clone()),
        embedded: 0,
    };
    let mut events = Events::new(true, false, false);
    // Provision/tokenize before measured stages. No inference model is loaded.
    engine.chunk("tokenizer warm-up", &mut events)?;
    println!(
        "{}",
        json!({"type":"pipeline_scale_started","files":args.files,"workspace":workspace,
        "keep":args.keep,"host":telemetry::host_snapshot(None),"warning":"real tokenizer; synthetic vectors, NOT model inference throughput; prewarmed tokenizer"})
    );
    measure("create_files", || {
        for i in 0..args.files {
            let path = file(&docs, i);
            if i % 1000 == 0 {
                std::fs::create_dir_all(path.parent().unwrap())?;
            }
            std::fs::write(
                path,
                format!("Synthetic pipeline document {i}. Configuration, code and prose.\n"),
            )?;
        }
        Ok(())
    })?;
    let _lock = WriterLock::acquire(&db_path)?;
    let mut db = Database::open(&db_path, true)?;
    db.conn.execute(
        "INSERT INTO schema_metadata VALUES('synthetic_vectors','true')",
        [],
    )?;
    let inputs = [docs.to_str().unwrap().to_string()];
    measure("ingest", || {
        ensure!(
            pipeline::ingest(
                &mut db,
                &mut engine,
                &mut events,
                &config,
                &options,
                &inputs
            )? == 0,
            "initial ingestion failed"
        );
        Ok(())
    })?;
    ensure!(
        db.status()?["documents"] == args.files,
        "initial coverage mismatch"
    );
    let embedded = engine.embedded;
    measure("hash_noop", || {
        ensure!(
            pipeline::ingest(
                &mut db,
                &mut engine,
                &mut events,
                &config,
                &options,
                &inputs
            )? == 0,
            "no-op ingestion failed"
        );
        ensure!(engine.embedded == embedded, "no-op generated vectors");
        Ok(())
    })?;
    let mut deleted = 0;
    for i in (0..args.files).step_by(100) {
        if i % 200 == 0 {
            std::fs::remove_file(file(&docs, i))?;
            deleted += 1;
        } else {
            std::fs::write(
                file(&docs, i),
                format!("Changed synthetic pipeline document {i}.\n"),
            )?;
        }
    }
    measure("change_delete_sync", || {
        ensure!(
            pipeline::ingest(
                &mut db,
                &mut engine,
                &mut events,
                &config,
                &options,
                &inputs
            )? == 0,
            "changed ingestion failed"
        );
        ensure!(
            db.status()?["documents"] == args.files - deleted,
            "deletion coverage mismatch"
        );
        Ok(())
    })?;
    // Free source blocks before creating shadow vectors and prove rebuild never
    // reopens the sources. Do not run ingestion after this deliberate removal.
    std::fs::remove_dir_all(&docs)?;
    db.conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")?;
    let old = db.active_generation()?;
    measure("stored_input_rebuild", || {
        ensure!(
            pipeline::rebuild::rebuild(&mut db, &mut engine, &mut events)? == 0,
            "rebuild failed"
        );
        ensure!(
            db.active_generation()? != old,
            "generation did not activate"
        );
        Ok(())
    })?;
    db.conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")?;
    let integrity: String = db.conn.query_row("PRAGMA quick_check", [], |r| r.get(0))?;
    ensure!(integrity == "ok", "database integrity: {integrity}");
    let status = db.status()?;
    ensure!(
        status["chunks"] == status["embeddings"],
        "vector coverage mismatch"
    );
    println!(
        "{}",
        json!({"type":"pipeline_scale_completed","files":args.files,"deleted":deleted,"status":status,
        "database_bytes":db_path.metadata()?.len(),"free_scratch_bytes":free_bytes(&args.scratch_dir)?,
        "host":telemetry::host_snapshot(None),"warning":"synthetic vectors; not real-model throughput or SSD qualification"})
    );
    Ok(())
}
