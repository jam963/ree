//! Read-only latency harness for an existing REAL database. No synthetic vectors,
//! source ingestion, model comparisons, or database writes. First use may download
//! the pinned artifacts, exactly as `ree search` does (lexical mode never does).
use anyhow::{Result, ensure};
use clap::Parser;
use ree::{
    cli::Options,
    config::Config,
    events::Events,
    model::LocalEngine,
    retrieval::{self, SearchMode, SearchRequest},
    storage::Database,
};
use serde_json::json;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[command(flatten)]
    options: Options,
    #[arg(long)]
    query: String,
    #[arg(long, value_enum, default_value_t = SearchMode::Semantic)]
    mode: SearchMode,
    #[arg(long, default_value_t = 10)]
    limit: usize,
    #[arg(long)]
    root: Option<String>,
    #[arg(long)]
    media_type: Option<String>,
    #[arg(long, default_value_t = 5)]
    repeat: usize,
    /// Recreate the model session each iteration; OS/disk caches remain uncontrolled.
    #[arg(long)]
    fresh_engine: bool,
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        (1..=1000).contains(&args.repeat),
        "repeat must be in 1..=1000"
    );
    let config = Config::load(&args.options)?;
    let request = SearchRequest {
        query: args.query,
        limit: args.limit,
        mode: args.mode,
        root: args.root,
        media_type: args.media_type,
    };
    request.validate()?;
    let mut engine = LocalEngine::new(config.clone());
    let mut events = Events::new(
        args.options.quiet,
        args.options.verbose,
        args.options.progress,
    );
    for iteration in 0..args.repeat {
        let start = Instant::now();
        if args.fresh_engine {
            engine = LocalEngine::new(config.clone());
        }
        let open_start = Instant::now();
        let mut db = Database::open(&config.db, false)?;
        let open_ms = open_start.elapsed().as_millis();
        let report = retrieval::search(&mut db, &mut engine, &request, &mut events)?;
        events.emit(json!({"type":"retrieval_benchmark","iteration":iteration,
            "session":if iteration==0 || args.fresh_engine {"new"} else {"reused"},
            "mode":request.mode,"results":report.results.len(),
            "generation":report.generation,"model_key":report.model_key,
            "query_provider":report.query_provider,"query_tokens":report.query_tokens,
            "candidate_limit":report.candidate_limit,"dense_candidates":report.dense_candidates,
            "lexical_candidates":report.lexical_candidates,"open_ms":open_ms,
            "timings":report.timings,"total_ms":start.elapsed().as_millis(),
            "warning":"session warmth is labeled; OS/disk cache warmth is uncontrolled; not a quality benchmark"}))?;
        events.flush()?;
    }
    Ok(())
}
