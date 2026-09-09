use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Embed files into a local SQLite database",
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
