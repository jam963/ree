//! Bounded development-host-only parity/timing check for the CLS derivative.
//! No downloads, database writes, or model comparisons. Run GPU trials serially.
use anyhow::{Result, ensure};
use clap::Parser;
use ree::{
    events::Events,
    model::{
        Runtime, device, download,
        onnx::{Onnx, cls},
    },
};
use serde_json::json;
use std::{path::PathBuf, time::Instant};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    cache: PathBuf,
    #[arg(long)]
    derived_cache: PathBuf,
    #[arg(long, default_value = "cuda")]
    device: String,
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        std::env::var_os("CI").is_none(),
        "real-model qualification is development-host only"
    );
    ensure!(
        std::fs::read_to_string("/proc/cpuinfo")?.contains("AMD Ryzen 9 8945HS"),
        "development host required"
    );
    ensure!(
        matches!(args.device.as_str(), "cpu" | "cuda"),
        "device must be cpu or cuda"
    );
    let gpu = if args.device == "cuda" {
        let gpu = device::select(&device::discover()?, "cuda", 0.85)?;
        ensure!(
            gpu.name.contains("RTX 4070 Laptop"),
            "development GPU required"
        );
        Some(gpu)
    } else {
        None
    };
    let artifact = if gpu.is_some() {
        download::CUDA
    } else {
        download::CPU
    };
    let path = download::path(&args.cache, artifact);
    ensure!(
        download::verified(&path, artifact)?,
        "verified parent must already be cached; no downloads"
    );
    let derived = cls::ensure_artifact(
        &path,
        &args.derived_cache,
        artifact,
        &mut Events::new(true, false, false),
    )?;
    let gpu_options = gpu.as_ref().map(|g| (g, device::budget(g, 0.85)));
    let shapes: Vec<_> = [1, 8, 16]
        .into_iter()
        .flat_map(|b| [64, 256, 512].into_iter().map(move |l| (b, l)))
        .collect();
    let inputs: Vec<Vec<Vec<i64>>> = shapes
        .iter()
        .map(|&(b, l)| {
            (0..b)
                .map(|i| {
                    let mut ids = vec![42 + i as i64; l - i % 3];
                    ids[0] = 0;
                    *ids.last_mut().unwrap() = 2;
                    ids
                })
                .collect()
        })
        .collect();
    let mut references = vec![];
    let mut times = vec![];
    {
        let mut runtime = Onnx::load(&path, gpu_options, 512)?;
        println!(
            "{}",
            json!({"type":"cls_verification","variant":"token_output","device":args.device,"profile":runtime.verification()})
        );
        for batch in &inputs {
            runtime.infer(batch)?;
            let mut measurements = vec![];
            let mut vectors = vec![];
            for _ in 0..3 {
                let start = Instant::now();
                vectors = runtime.infer(batch)?;
                measurements.push(start.elapsed().as_secs_f64());
            }
            references.push(vectors);
            times.push(measurements);
        }
    }
    let mut runtime = Onnx::load_cls(&derived, gpu_options, 512)?;
    println!(
        "{}",
        json!({"type":"cls_verification","variant":cls::VERSION,"device":args.device,"profile":runtime.verification()})
    );
    for (i, batch) in inputs.iter().enumerate() {
        runtime.infer(batch)?;
        let mut measurements = vec![];
        let mut min_cosine = 1.0f64;
        let mut max_abs = 0.0f64;
        let mut bitwise = true;
        for _ in 0..3 {
            let start = Instant::now();
            let actual = runtime.infer(batch)?;
            measurements.push(start.elapsed().as_secs_f64());
            for (a, b) in references[i].iter().zip(&actual) {
                let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
                let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum();
                let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum();
                min_cosine = min_cosine.min(dot / (na * nb).sqrt());
                for (&x, &y) in a.iter().zip(b) {
                    max_abs = max_abs.max((x as f64 - y as f64).abs());
                    bitwise &= x.to_bits() == y.to_bits();
                }
            }
        }
        println!(
            "{}",
            json!({"type":"cls_shape","device":args.device,"batch":shapes[i].0,"length":shapes[i].1,"baseline_seconds":times[i],"cls_seconds":measurements,"min_cosine":min_cosine,"max_abs":max_abs,"bitwise":bitwise})
        );
        ensure!(
            min_cosine >= 0.99999 && max_abs <= 0.0005,
            "CLS parity gate failed"
        );
    }
    println!(
        "{}",
        json!({"type":"cls_qualification_completed","device":args.device,"parent_sha256":artifact.sha256,"derivative_sha256":ree::util::hash_file(&derived)?,"scope":"single process, fixed shapes only; not end-to-end speedup or broad retrieval qualification"})
    );
    Ok(())
}
