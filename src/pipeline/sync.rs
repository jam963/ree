use crate::{
    cli::Options,
    config::Config,
    events::Events,
    extract::{self, Extracted},
    input::{self, Content, Item, Root, RootContent},
    model::{self, Engine},
    storage::{Database, Document, Store},
    util::{document_id, hash, path_string, read_limited},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    os::unix::fs::OpenOptionsExt,
    sync::atomic::{AtomicBool, Ordering},
    time::{Instant, UNIX_EPOCH},
};

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Counts {
    pub documents: usize,
    pub chunks: usize,
    pub unchanged: usize,
    pub deleted: usize,
    pub failed: usize,
}
// Only two entries per queue; keep the common document inline rather than add
// a heap allocation for every file to optimize rare small diagnostic events.
#[allow(clippy::large_enum_variant)]
enum Prepared {
    Document {
        uri: String,
        result: Result<Payload>,
    },
    Event(Value),
}

struct PreparationEvents {
    sender: crossbeam_channel::Sender<Prepared>,
    line: Vec<u8>,
}
impl std::io::Write for PreparationEvents {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        for &byte in bytes {
            if self.line.len() >= 128 * 1024 {
                return Err(std::io::Error::other("preparation event exceeds limit"));
            }
            self.line.push(byte);
            if byte == b'\n' {
                let value = serde_json::from_slice(&self.line).map_err(std::io::Error::other)?;
                self.sender
                    .send(Prepared::Event(value))
                    .map_err(std::io::Error::other)?;
                self.line.clear();
            }
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
struct Payload {
    content_hash: String,
    size: u64,
    modified_ns: Option<String>,
    extracted: Option<Extracted>,
    chunks: Option<Vec<crate::chunk::Chunk>>,
    metadata: Value,
}

#[allow(clippy::too_many_arguments)]
fn prepare(
    item: Item,
    config: &Config,
    forced: Option<&str>,
    cache: Option<&Database>,
    root: &str,
    recipe: &str,
    mut chunker: Option<&mut (dyn model::passage::Chunker + 'static)>,
    events: &mut Events,
) -> Prepared {
    let _span = crate::metrics::Span::new("preparation_service");
    let result = (|| -> Result<Payload> {
        let (bytes, extension, modified_ns) = match item.content {
            Content::Bytes(bytes, ext) => (bytes, ext, None),
            Content::File(path) => {
                // Nonblocking open plus fstat prevents a raced-in FIFO/device from
                // hanging the pipeline after discovery checked a regular file.
                let mut file = std::fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_NONBLOCK)
                    .open(&path)?;
                let before = file.metadata()?;
                ensure!(before.is_file(), "unsupported: not a regular file");
                ensure!(
                    before.len() <= config.max_file_size,
                    "size_limit: file exceeds {} bytes",
                    config.max_file_size
                );
                let bytes = read_limited(&mut file, config.max_file_size)?;
                let after = file.metadata()?;
                ensure!(
                    before.len() == after.len() && before.modified().ok() == after.modified().ok(),
                    "changed_while_reading: source changed during read"
                );
                let modified = before
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_nanos().to_string());
                let ext = path
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                (bytes, ext, modified)
            }
        };
        let size = bytes.len() as u64;
        let content_hash = hash(&bytes);
        let unchanged = cache
            .map(|db| db.unchanged_content(&document_id(root, &item.uri), &content_hash, recipe))
            .transpose()?
            .unwrap_or(false);
        let extracted = if unchanged {
            None
        } else {
            Some(extract::extract(&bytes, &extension, config, forced)?)
        };
        // Bound extra token/chunk staging to small extracted documents. Large
        // documents keep the original single-consumer tokenization path.
        let chunks = match (&extracted, &mut chunker) {
            (Some(text), Some(chunker)) if text.text.len() <= 16 * 1024 => {
                let chunks = chunker.chunk(&text.text, events)?;
                let bytes: usize = chunks
                    .iter()
                    .map(|c| c.text.len() + c.input_ids.len() * 8)
                    .sum();
                (bytes <= 256 * 1024).then_some(chunks)
            }
            _ => None,
        };
        Ok(Payload {
            content_hash,
            size,
            modified_ns,
            extracted,
            chunks,
            metadata: item.metadata,
        })
    })();
    Prepared::Document {
        uri: item.uri,
        result,
    }
}
fn error_code(e: &anyhow::Error) -> &'static str {
    let s = format!("{e:#}");
    for code in [
        "size_limit",
        "private_network",
        "missing_extractor",
        "extractor_timeout",
        "binary_input",
        "unsupported",
        "unsafe_symlink",
        "changed_while_reading",
    ] {
        if s.contains(code) {
            return code;
        }
    }
    "input_failed"
}
fn failure(
    db: &Database,
    events: &mut Events,
    run: &str,
    source: &str,
    error: &anyhow::Error,
    counts: &mut Counts,
) -> Result<()> {
    if let Some(e) = error.downcast_ref::<crate::error::AppError>() {
        return Err(crate::error::AppError::new(e.code, e.message.clone()).into());
    }
    counts.failed += 1;
    let code = error_code(error);
    let message = format!("{error:#}");
    db.failure(run, source, code, &message)?;
    events.failure(source, code, &message)
}

struct Pending {
    document: Document,
    chunks: Vec<crate::chunk::Chunk>,
}

#[allow(clippy::too_many_arguments)]
fn process(
    db: &mut Database,
    engine: &mut dyn Engine,
    events: &mut Events,
    config: &Config,
    options: &Options,
    run: &str,
    root: &str,
    recipe: &str,
    p: Prepared,
    counts: &mut Counts,
    pending: &mut Vec<Pending>,
) -> Result<()> {
    let (uri, result) = match p {
        Prepared::Event(value) => return events.emit(value),
        Prepared::Document { uri, result } => (uri, result),
    };
    let id = document_id(root, &uri);
    // Seen is recorded independently of extraction/inference success, ensuring
    // a failed document keeps its old valid chunks during deletion reconciliation.
    db.seen(&id, run)?;
    let payload = match result {
        Ok(payload) => payload,
        Err(e) => return failure(db, events, run, &uri, &e, counts),
    };
    let Some(extracted) = payload.extracted else {
        counts.unchanged += 1;
        return Ok(());
    };
    let mut metadata = extracted.metadata.as_object().cloned().unwrap_or_default();
    metadata.extend(payload.metadata.as_object().cloned().unwrap_or_default());
    metadata.extend(config.metadata.as_object().cloned().unwrap_or_default());
    let metadata = Value::Object(metadata);
    if !options.force && db.unchanged(&id, &payload.content_hash, recipe, &metadata)? {
        counts.unchanged += 1;
        return Ok(());
    }
    let chunks = match payload
        .chunks
        .map(Ok)
        .unwrap_or_else(|| engine.chunk(&extracted.text, events))
    {
        Ok(v) => v,
        Err(e) => {
            if e.downcast_ref::<crate::error::AppError>().is_some() {
                return Err(e);
            }
            return failure(db, events, run, &uri, &e, counts);
        }
    };
    let document = Document {
        id,
        root: root.into(),
        uri,
        content_hash: payload.content_hash,
        text_hash: hash(extracted.text.as_bytes()),
        media_type: extracted.media_type,
        extractor: extracted.extractor,
        recipe: recipe.into(),
        size: payload.size,
        modified_ns: payload.modified_ns,
        metadata,
        run: run.into(),
    };
    pending.push(Pending { document, chunks });
    Ok(())
}

fn flush_documents(
    db: &mut Database,
    engine: &mut dyn Engine,
    events: &mut Events,
    counts: &mut Counts,
    pending: &mut Vec<Pending>,
) -> Result<()> {
    if pending.is_empty() {
        return Ok(());
    }
    let inputs: Vec<_> = pending
        .iter()
        .flat_map(|p| p.chunks.iter().map(|c| c.input_ids.clone()))
        .collect();
    let vectors = engine.embed(&inputs, events);
    if let Err(e) = &vectors
        && e.downcast_ref::<crate::error::AppError>().is_some()
    {
        return Err(vectors.unwrap_err());
    }
    if let Ok(v) = &vectors {
        ensure!(
            v.len() == inputs.len(),
            "runtime returned wrong vector count"
        );
    }
    let mut offset = 0;
    for p in pending.drain(..) {
        // A failed mixed-document CPU batch is retried per document to isolate
        // bad inputs. No output is committed until that document is complete.
        let result = match &vectors {
            Ok(v) => Ok(std::borrow::Cow::Borrowed(
                &v[offset..offset + p.chunks.len()],
            )),
            Err(_) => engine
                .embed(
                    &p.chunks
                        .iter()
                        .map(|c| c.input_ids.clone())
                        .collect::<Vec<_>>(),
                    events,
                )
                .map(std::borrow::Cow::Owned),
        };
        offset += p.chunks.len();
        match result {
            Ok(v) => {
                db.replace_document(&p.document, &p.chunks, &v)?;
                counts.documents += 1;
                counts.chunks += p.chunks.len();
                if events.verbose {
                    events.emit(
                        json!({"type":"document","source":p.document.uri,"chunks":p.chunks.len()}),
                    )?;
                }
            }
            Err(e) => {
                if e.downcast_ref::<crate::error::AppError>().is_some() {
                    return Err(e);
                }
                failure(db, events, &p.document.run, &p.document.uri, &e, counts)?;
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn ingest_root(
    db: &mut Database,
    engine: &mut dyn Engine,
    events: &mut Events,
    config: &Config,
    options: &Options,
    run: &str,
    mut root: Root,
    counts: &mut Counts,
) -> Result<()> {
    // Hold a shared-cache checkout lease through extraction/inference, not merely
    // discovery, including writers targeting different databases.
    let _repository_lease = root.lease.take();
    let id = db.root(root.kind, &root.identity, &config.metadata, run)?;
    db.conn.execute(
        "UPDATE source_roots SET options_json=?2 WHERE id=?1",
        rusqlite::params![id, root.metadata.to_string()],
    )?;
    let recipe = hash(
        format!(
            "{}:{}:{}:{}:{:?}:{:?}:{:?}:{}:{}",
            model::MODEL_KEY,
            extract::VERSION,
            config.chunk_size,
            config.overlap,
            options.extractor,
            config.extractors,
            config.pdf,
            config.metadata,
            root.metadata
        )
        .as_bytes(),
    );
    let before = counts.failed;
    let directory = matches!(root.content, RootContent::Directory { .. });
    let stop = AtomicBool::new(false);
    let traversal_ok = AtomicBool::new(true);
    std::thread::scope(|scope| -> Result<()> {
        let (jobs_tx, jobs_rx) = crossbeam_channel::bounded::<Item>(2);
        let (results_tx, results_rx) = crossbeam_channel::bounded::<Prepared>(2);
        let discovery_results = results_tx.clone();
        let stop_ref = &stop;
        let traversal_ref = &traversal_ok;
        scope.spawn(move || match root.content {
            RootContent::One(item) => {
                let _ = jobs_tx.send(item);
            }
            RootContent::Directory { .. } => input::discover(&root, |item| {
                if stop_ref.load(Ordering::Relaxed) {
                    return false;
                }
                match item {
                    Ok(item) => jobs_tx.send(item).is_ok(),
                    Err(e) => {
                        traversal_ref.store(false, Ordering::Relaxed);
                        discovery_results
                            .send(Prepared::Document {
                                uri: root.identity.clone(),
                                result: Err(e),
                            })
                            .is_ok()
                    }
                }
            }),
        });
        for _ in 0..2 {
            let jobs = jobs_rx.clone();
            let results = results_tx.clone();
            let root_id = &id;
            let recipe_ref = &recipe;
            let mut chunker = engine.chunker();
            scope.spawn(move || {
                let mut preparation_events = Events::with_writer(PreparationEvents { sender: results.clone(), line: vec![] }, false, false);
                // A read-only connection per extractor avoids a million-entry
                // in-memory hash map and skips extraction for content-hash hits.
                let cache = if options.force {
                    None
                } else {
                    Database::open(&config.db, false).ok()
                };
                for item in jobs {
                    if stop_ref.load(Ordering::Relaxed) {
                        break;
                    }
                    if results
                        .send(prepare(
                            item,
                            config,
                            options.extractor.as_deref(),
                            cache.as_ref(),
                            root_id,
                            recipe_ref,
                            chunker.as_deref_mut(),
                            &mut preparation_events,
                        ))
                        .is_err()
                    {
                        break;
                    }
                    if crate::metrics::enabled() {
                        let _ = preparation_events.emit(json!({"type":"stage_metrics","scope":"preparation_worker","overlapping":true,"stages":crate::metrics::take()}));
                    }
                    let _ = preparation_events.flush();
                }
            });
        }
        drop(jobs_rx);
        drop(results_tx);
        let mut fatal = None;
        let mut pending = Vec::new();
        for p in results_rx {
            if fatal.is_some() || stop.load(Ordering::Relaxed) {
                continue;
            }
            if let Err(e) = process(
                db,
                engine,
                events,
                config,
                options,
                run,
                &id,
                &recipe,
                p,
                counts,
                &mut pending,
            ) {
                fatal = Some(e);
                stop.store(true, Ordering::Relaxed);
            }
            let bytes: usize = pending
                .iter()
                .flat_map(|p: &Pending| &p.chunks)
                .map(|c| c.text.len() + c.input_ids.len() * 8)
                .sum();
            if fatal.is_none()
                && (pending.len() >= 32 || bytes >= 8 * 1024 * 1024)
                && let Err(e) = flush_documents(db, engine, events, counts, &mut pending)
            {
                fatal = Some(e);
                stop.store(true, Ordering::Relaxed);
            }
            if options.fail_fast && counts.failed > before {
                stop.store(true, Ordering::Relaxed);
            }
        }
        if let Some(e) = fatal {
            return Err(e);
        }
        flush_documents(db, engine, events, counts, &mut pending)?;
        if options.fail_fast && counts.failed > before {
            stop.store(true, Ordering::Relaxed);
        }
        Ok(())
    })?;
    let complete = traversal_ok.load(Ordering::Relaxed) && !stop.load(Ordering::Relaxed);
    counts.deleted += db.finish_root(
        &id,
        run,
        directory && complete,
        complete && counts.failed == before,
    )?;
    Ok(())
}

pub fn ingest(
    db: &mut Database,
    engine: &mut dyn Engine,
    events: &mut Events,
    config: &Config,
    options: &Options,
    inputs: &[String],
) -> Result<u8> {
    db.register_model()?;
    let active_model: String = db.conn.query_row(
        "SELECT model_id FROM embedding_generations WHERE status='active'",
        [],
        |r| r.get(0),
    )?;
    ensure!(
        active_model == model::MODEL_KEY,
        "active embedding recipe differs from this binary; rebuild/reingest explicitly before ingestion"
    );
    let run = db.start_run("ingest")?;
    let start = Instant::now();
    let mut counts = Counts::default();
    events.emit(json!({"type":"started","run_id":run,"inputs":inputs}))?;
    events.flush()?;
    for input in inputs {
        if options.fail_fast && counts.failed > 0 {
            break;
        }
        if input::classify(input) == input::Kind::Glob {
            let mut matched = false;
            for entry in glob::glob(input).context("invalid glob")? {
                if options.fail_fast && counts.failed > 0 {
                    break;
                }
                matched = true;
                let prepared = entry
                    .map_err(anyhow::Error::from)
                    .and_then(|p| path_string(&p))
                    .and_then(|p| input::prepare(&p, None, config));
                match prepared {
                    Ok(root) => {
                        ingest_root(db, engine, events, config, options, &run, root, &mut counts)?
                    }
                    Err(e) => failure(db, events, &run, input, &e, &mut counts)?,
                }
            }
            if !matched {
                failure(
                    db,
                    events,
                    &run,
                    input,
                    &anyhow::anyhow!("glob matched no inputs"),
                    &mut counts,
                )?;
            }
        } else {
            match input::prepare(input, options.source.as_deref(), config) {
                Ok(root) => {
                    ingest_root(db, engine, events, config, options, &run, root, &mut counts)?
                }
                Err(e) => failure(db, events, &run, input, &e, &mut counts)?,
            }
        }
    }
    let mut event = serde_json::to_value(&counts)?;
    event["type"] = json!("completed");
    event["run_id"] = json!(run);
    event["elapsed_ms"] = json!(start.elapsed().as_millis());
    event["runtime_metrics"] = engine.metrics();
    db.finish_run(&run, counts.failed, &event)?;
    events.emit(event)?;
    Ok(if counts.failed > 0 { 1 } else { 0 })
}
