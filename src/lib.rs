pub mod chunk;
pub mod cli;
pub mod config;
pub mod error;
pub mod events;
pub mod extract;
pub mod input;
pub mod model;
pub mod pipeline;
pub mod storage;
pub mod util;

use anyhow::{Context, Result};
use cli::{Cli, Command};
use config::Config;
use error::AppError;
use events::Events;
use model::LocalEngine;
use serde_json::json;
use storage::{Database, WriterLock};

pub fn run(cli: Cli, events: &mut Events) -> Result<u8> {
    let config = Config::load(&cli.options).map_err(|e| AppError::new(2, format!("{e:#}")))?;
    if cli.command.is_some() && !cli.inputs.is_empty() {
        return Err(AppError::new(
            2,
            "input paths cannot be combined with a maintenance command",
        )
        .into());
    }
    if cli.command.is_none() {
        if cli.inputs.is_empty() {
            return Err(AppError::new(2, "provide at least one INPUT; see ree --help").into());
        }
        if cli.inputs.iter().filter(|s| s.as_str() == "-").count() > 1 {
            return Err(AppError::new(2, "stdin may only be ingested once per invocation").into());
        }
        if cli.options.source.is_some() && (cli.inputs.len() != 1 || cli.inputs[0] != "-") {
            return Err(AppError::new(2, "--source requires exactly one stdin input (-)").into());
        }
        for input in &cli.inputs {
            if input::classify(input) == input::Kind::Glob {
                glob::Pattern::new(input)
                    .map_err(|e| AppError::new(2, format!("invalid glob: {e}")))?;
            }
        }
    }
    match cli.command {
        Some(Command::Doctor) => doctor(&config, events),
        Some(Command::Status) => {
            let db = Database::open(&config.db, false)?;
            events.emit(db.status()?)?;
            Ok(0)
        }
        Some(Command::Sources) => {
            let db = Database::open(&config.db, false)?;
            for source in db.sources()? {
                events.emit(source)?;
            }
            Ok(0)
        }
        command => {
            let _lock = WriterLock::acquire(&config.db)?;
            let mut db = Database::open(&config.db, true)
                .with_context(|| format!("open database {}", config.db.display()))?;
            let mut engine = LocalEngine::new(config.clone());
            match command {
                Some(Command::Remove { source }) => {
                    let identity = std::path::Path::new(&source)
                        .canonicalize()
                        .ok()
                        .and_then(|p| p.to_str().map(str::to_string))
                        .unwrap_or(source);
                    let removed = db.remove(&identity)?;
                    events.emit(json!({"type":"removed","source":identity,"roots":removed}))?;
                    Ok(0)
                }
                Some(Command::Rebuild) => pipeline::rebuild::rebuild(&mut db, &mut engine, events),
                None => pipeline::ingest(
                    &mut db,
                    &mut engine,
                    events,
                    &config,
                    &cli.options,
                    &cli.inputs,
                ),
                _ => unreachable!(),
            }
        }
    }
}

fn doctor(config: &Config, events: &mut Events) -> Result<u8> {
    let database = if config.db.exists() {
        match Database::open(&config.db, false) {
            Ok(db) => {
                let integrity: String =
                    db.conn.query_row("PRAGMA quick_check", [], |r| r.get(0))?;
                let vec: String = db
                    .conn
                    .query_row("SELECT vec_version()", [], |r| r.get(0))?;
                json!({"path":config.db,"integrity":integrity,"sqlite_vec":vec,"status":db.status()?})
            }
            Err(e) => json!({"path":config.db,"error":format!("{e:#}")}),
        }
    } else {
        json!({"path":config.db,"exists":false})
    };
    let models:Vec<_>=[model::download::TOKENIZER,model::download::CPU,model::download::CUDA].iter().map(|&a| {
        let path=model::download::path(&config.cache,a);
        json!({"artifact":a.name,"path":path,"present":path.exists(),"verified":model::download::verified(&path,a).unwrap_or(false)})
    }).collect();
    let gpu = match model::device::discover() {
        Ok(gpus) => json!({"devices":gpus}),
        Err(e) => json!({"unavailable":e.to_string()}),
    };
    use ort::ep::ExecutionProvider;
    let provider = std::panic::catch_unwind(|| ort::ep::CUDA::default().is_available());
    let cuda = match provider {
        Ok(Ok(v)) => json!(v),
        Ok(Err(e)) => json!({"error":e.to_string()}),
        Err(_) => json!({"error":"ONNX Runtime could not initialize"}),
    };
    let tools: Vec<_> = ["pdftotext", "pandoc", "libreoffice", "tesseract", "git"]
        .iter()
        .map(|p| json!({"program":p,"available":extract::external::available(p)}))
        .collect();
    let mut lock_path = config.db.as_os_str().to_owned();
    lock_path.push(".lock");
    let locked = std::fs::File::open(lock_path)
        .ok()
        .map(|file| fs2::FileExt::try_lock_shared(&file).is_err())
        .unwrap_or(false);
    events.emit(json!({"type":"doctor","database":database,"cache":config.cache,"models":models,"gpu":gpu,"cuda_provider_compiled":cuda,"runtime":model::RUNTIME_VERSION,"cuda_compatibility":"CUDA 13.x; actual CUDA graph execution is verified at warmup","writer_locked":locked,"extractors":tools}))?;
    Ok(if database.get("error").is_some() {
        1
    } else {
        0
    })
}
