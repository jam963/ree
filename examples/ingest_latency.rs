//! Development-host latency harness: production ingestion with a reused engine.
//! Explicit --force redoes the same documents; commits retain normal durability.
use anyhow::{Result, ensure};
use clap::Parser;
use ree::{
    cli::Options,
    config::Config,
    events::Events,
    model::{Engine, LocalEngine},
    pipeline,
    storage::{Database, WriterLock},
};
use serde_json::json;
use std::time::Instant;
#[derive(Parser)]
struct Args {
    #[command(flatten)]
    options: Options,
    #[arg(long)]
    input: String,
    #[arg(long, default_value_t = 4)]
    repeat: usize,
}
fn main() -> Result<()> {
    let mut args = Args::parse();
    ensure!(
        std::env::var_os("CI").is_none(),
        "real-model measurements are development-host only"
    );
    ensure!(
        std::fs::read_to_string("/proc/cpuinfo")?.contains("AMD Ryzen 9 8945HS"),
        "development host required"
    );
    ensure!((2..=10).contains(&args.repeat), "repeat must be 2..10");
    args.options.force = true;
    let config = Config::load(&args.options)?;
    ensure!(!config.db.exists(), "choose a fresh disposable --db path");
    let _lock = WriterLock::acquire(&config.db)?;
    let mut db = Database::open(&config.db, true)?;
    let mut engine = LocalEngine::new(config.clone());
    let mut events = Events::new(true, false, false);
    for iteration in 0..args.repeat {
        let start = Instant::now();
        let exit = pipeline::ingest(
            &mut db,
            &mut engine,
            &mut events,
            &config,
            &args.options,
            std::slice::from_ref(&args.input),
        )?;
        let seconds = start.elapsed().as_secs_f64();
        events.flush()?;
        println!(
            "{}",
            json!({"type":"ingestion_benchmark","iteration":iteration,"session":if iteration==0{"new"}else{"reused"},"seconds":seconds,"exit":exit,"runtime_metrics_cumulative":engine.metrics(),"stages":ree::metrics::take()})
        );
        ensure!(exit == 0, "ingestion failed");
    }
    Ok(())
}
