use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Index and search documents locally with SQLite",
    subcommand_precedence_over_arg = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
    #[arg(value_name = "INPUT", required = false)]
    pub inputs: Vec<String>,
    #[command(flatten)]
    pub options: Options,
}

#[derive(Debug, clap::Args, Default)]
pub struct Options {
    #[arg(long, global = true, env = "REE_DB")]
    pub db: Option<PathBuf>,
    #[arg(long, global = true, env = "REE_CONFIG")]
    pub config: Option<PathBuf>,
    #[arg(long, global = true, env = "REE_DEVICE")]
    pub device: Option<String>,
    #[arg(long, global = true, env = "REE_BATCH_SIZE")]
    pub batch_size: Option<usize>,
    #[arg(long, global = true, env = "REE_GPU_MEMORY_FRACTION")]
    pub gpu_memory_fraction: Option<f64>,
    #[arg(long, env = "REE_CHUNK_SIZE")]
    pub chunk_size: Option<usize>,
    #[arg(long, env = "REE_OVERLAP")]
    pub overlap: Option<usize>,
    #[arg(long, env = "REE_MAX_FILE_SIZE")]
    pub max_file_size: Option<String>,
    #[arg(long)]
    pub source: Option<String>,
    #[arg(long, value_name = "KEY=VALUE")]
    pub metadata: Vec<String>,
    #[arg(long)]
    pub metadata_json: Option<String>,
    #[arg(long)]
    pub extractor: Option<String>,
    #[arg(long)]
    pub force: bool,
    #[arg(long)]
    pub fail_fast: bool,
    #[arg(long, global = true)]
    pub quiet: bool,
    #[arg(long, global = true)]
    pub verbose: bool,
    #[arg(long, global = true)]
    pub progress: bool,
    #[arg(long)]
    pub allow_private_network: bool,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Search stored chunks without modifying the database or reopening sources
    Search(SearchArgs),
    /// Serve reusable queries on an explicitly started local Unix socket
    Worker(WorkerArgs),
    #[command(name = "__query-engine", hide = true)]
    QueryEngine,
    /// Upgrade/initialize the database, including the lexical index; no model required
    Migrate,
    /// Show database counts and the active embedding generation
    Status,
    /// List synchronization roots
    Sources,
    /// Remove only this source root (ID or canonical identity)
    Remove { source: String },
    /// Re-embed stored chunks; does not rechunk or reopen sources
    Rebuild,
    /// Diagnose storage, runtime, model cache, and optional extractors
    Doctor,
}

#[derive(Debug, Clone, clap::Args)]
pub struct SearchArgs {
    /// Literal query text (quote multiple words); no FTS query syntax
    #[arg(required_unless_present = "stream", conflicts_with = "stream")]
    pub query: Option<String>,
    /// Read versioned JSONL requests from stdin and reuse a query engine
    #[arg(long, conflicts_with_all = ["query", "socket", "mode", "limit", "root", "media_type"])]
    pub stream: bool,
    /// Route this query to an explicitly started worker (never auto-started)
    #[arg(long)]
    pub socket: Option<PathBuf>,
    /// Release the engine process after this many idle seconds (streaming only)
    #[arg(long, default_value = "300s", value_parser = parse_idle, requires = "stream")]
    pub idle_timeout: u64,
    #[arg(long, default_value_t = 10)]
    pub limit: usize,
    #[arg(long, value_enum, default_value_t = crate::retrieval::SearchMode::Semantic)]
    pub mode: crate::retrieval::SearchMode,
    /// Exact root ID/identity, or a local path resolved to its canonical identity
    #[arg(long)]
    pub root: Option<String>,
    /// Exact stored document media type
    #[arg(long)]
    pub media_type: Option<String>,
}
#[derive(Debug, clap::Args)]
pub struct WorkerArgs {
    #[arg(long)]
    pub socket: Option<PathBuf>,
    #[arg(long, default_value = "300s", value_parser = parse_idle)]
    pub idle_timeout: u64,
}
fn parse_idle(s: &str) -> Result<u64, String> {
    let n = s
        .strip_suffix('s')
        .unwrap_or(s)
        .parse::<u64>()
        .map_err(|_| "idle timeout must be seconds, e.g. 300s".to_owned())?;
    if !(1..=86400).contains(&n) {
        return Err("idle timeout must be 1..86400 seconds".into());
    }
    Ok(n)
}
impl TryFrom<SearchArgs> for crate::retrieval::SearchRequest {
    type Error = anyhow::Error;
    fn try_from(args: SearchArgs) -> Result<Self, Self::Error> {
        Ok(Self {
            query: args
                .query
                .ok_or_else(|| crate::error::AppError::new(2, "search requires query text"))?,
            limit: args.limit,
            mode: args.mode,
            root: args.root,
            media_type: args.media_type,
        })
    }
}
