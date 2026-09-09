//! Local-only qualification through exactly the ONNX runtime used by ingestion.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use ree::{
    config::xdg,
    events::Events,
    model::{
        self, Runtime, device, download,
        onnx::{InferenceRecipe, Onnx},
    },
    util::hash_file,
};
use serde::Deserialize;
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf, time::Instant};
use tokenizers::Tokenizer;

#[derive(Parser)]
struct Args {
    /// Candidate manifest. Omit for the pinned Arctic model.
    #[arg(long)]
    manifest: Option<PathBuf>,
    /// JSON documents/queries/qrels corpus; omitted for throughput-only runs.
    #[arg(long)]
    corpus: Option<PathBuf>,
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,8,16")]
    batches: Vec<usize>,
    #[arg(long, default_value_t = 5)]
    repeats: usize,
    /// Record the actual power mode; run while the development machine is idle.
    #[arg(long)]
    power_mode: String,
}
#[derive(Deserialize)]
struct Artifact {
    path: PathBuf,
    sha256: String,
}
#[derive(Deserialize)]
struct Candidate {
    name: String,
    revision: String,
    precision: String,
    model: Artifact,
    tokenizer: Artifact,
    recipe: InferenceRecipe,
    document_prompt: String,
    query_prompt: String,
}
#[derive(Deserialize)]
struct Document {
    id: String,
    text: String,
}
#[derive(Deserialize)]
struct Query {
    id: String,
    text: String,
    relevant: Vec<String>,
    domain: String,
}
#[derive(Deserialize)]
struct Corpus {
    documents: Vec<Document>,
    queries: Vec<Query>,
}
fn tokens(tokenizer: &Tokenizer, text: &str) -> Result<Vec<i64>> {
    Ok(tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!(e))?
        .get_ids()
        .iter()
        .map(|&id| i64::from(id))
        .collect())
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        std::env::var_os("CI").is_none(),
        "qualification must not run in CI"
    );
    let cpu = std::fs::read_to_string("/proc/cpuinfo")?;
    ensure!(
        cpu.contains("AMD Ryzen 9 8945HS"),
        "qualification is restricted to the development Ryzen 9 8945HS"
    );
    let devices = device::discover()?;
    ensure!(
        devices.iter().any(|g| g.name.contains("RTX 4070 Laptop")),
        "qualification is restricted to the development RTX 4070 Laptop host"
    );
    ensure!(
        args.repeats >= 2 && args.batches.iter().all(|&n| n > 0 && n <= 256),
        "use at least two repeats and batches in 1..=256"
    );
    let gpu = if args.device == "cpu" {
        None
    } else {
        ensure!(
            args.device == "cuda"
                || args
                    .device
                    .strip_prefix("cuda:")
                    .is_some_and(|n| n.parse::<u32>().is_ok()),
            "device must be cpu or cuda[:N]"
        );
        Some(device::select(&devices, &args.device, 0.85)?)
    };
    let candidate = if let Some(path) = args.manifest {
        serde_json::from_reader(std::fs::File::open(path)?)?
    } else {
        let cache = xdg("XDG_CACHE_HOME", ".cache")?;
        let artifact = if gpu.is_some() {
            download::CUDA
        } else {
            download::CPU
        };
        if let Some(g) = &gpu {
            Onnx::check_cuda(g, device::budget(g, 0.85))?;
        }
        let mut events = Events::new(true, false, false);
        Candidate {
            name: model::MODEL_ID.into(),
            revision: model::REVISION.into(),
            precision: if gpu.is_some() { "fp16" } else { "int8" }.into(),
            model: Artifact {
                path: download::ensure_artifact(&cache, artifact, &mut events)?,
                sha256: artifact.sha256.into(),
            },
            tokenizer: Artifact {
                path: download::ensure_artifact(&cache, download::TOKENIZER, &mut events)?,
                sha256: download::TOKENIZER.sha256.into(),
            },
            recipe: InferenceRecipe::default(),
            document_prompt: "".into(),
            query_prompt: "query: ".into(),
        }
    };
    ensure!(
        candidate.revision.len() == 40 && candidate.revision.chars().all(|c| c.is_ascii_hexdigit()),
        "candidate requires an exact 40-character model revision"
    );
    for artifact in [&candidate.model, &candidate.tokenizer] {
        ensure!(
            hash_file(&artifact.path)? == artifact.sha256,
            "artifact checksum mismatch: {}",
            artifact.path.display()
        );
    }
    let mut tokenizer =
        Tokenizer::from_file(&candidate.tokenizer.path).map_err(|e| anyhow::anyhow!(e))?;
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length: 512,
            ..Default::default()
        }))
        .map_err(|e| anyhow::anyhow!(e))?;
    let start = Instant::now();
    let mut runtime = Onnx::load_for_qualification(
        &candidate.model.path,
        gpu.as_ref().map(|g| (g, device::budget(g, 0.85))),
        512,
        candidate.recipe.clone(),
    )?;
    println!(
        "{}",
        json!({"type":"qualification_started","model":candidate.name,"revision":candidate.revision,"model_sha256":candidate.model.sha256,"tokenizer_sha256":candidate.tokenizer.sha256,"precision":candidate.precision,"recipe":candidate.recipe,"runtime":model::RUNTIME_VERSION,"cpu":"AMD Ryzen 9 8945HS","gpu":gpu,"power_mode":args.power_mode,"threads":8,"load_and_warmup_ms":start.elapsed().as_millis(),"artifact_bytes":candidate.model.path.metadata()?.len(),"scope":"this development machine only"})
    );
    for length in [64, 256, 512] {
        for &batch in &args.batches {
            let mut input = vec![42; length];
            input[0] = candidate.recipe.cls_id;
            input[length - 1] = candidate.recipe.sep_id;
            let inputs = vec![input; batch];
            if let Err(e) = runtime.infer(&inputs) {
                println!(
                    "{}",
                    json!({"type":"probe_failed","tokens":length,"batch":batch,"error":format!("{e:#}")})
                );
                break;
            }
            let mut samples = Vec::new();
            for _ in 0..args.repeats {
                let start = Instant::now();
                runtime.infer(&inputs)?;
                samples.push(start.elapsed().as_secs_f64());
            }
            let mean = samples.iter().sum::<f64>() / samples.len() as f64;
            let variance =
                samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
            println!(
                "{}",
                json!({"type":"throughput","tokens":length,"batch":batch,"samples_seconds":samples,"mean_seconds":mean,"variance_seconds_squared":variance,"embeddings_per_second":batch as f64/mean,"tokens_per_second":(batch*length) as f64/mean,"free_vram_after":gpu.as_ref().and_then(|g|device::free_memory(g.index).ok())})
            );
        }
    }
    if let Some(path) = args.corpus {
        let corpus: Corpus = serde_json::from_reader(std::fs::File::open(path)?)?;
        ensure!(
            !corpus.documents.is_empty()
                && corpus.documents.len() <= 10000
                && !corpus.queries.is_empty(),
            "corpus must contain 1..=10000 documents and at least one query"
        );
        let mut vectors = Vec::new();
        for docs in corpus.documents.chunks(8) {
            let inputs = docs
                .iter()
                .map(|d| {
                    tokens(
                        &tokenizer,
                        &format!("{}{}", candidate.document_prompt, d.text),
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            vectors.extend(runtime.infer(&inputs)?);
        }
        let mut domains: BTreeMap<String, Vec<(f64, f64, f64)>> = BTreeMap::new();
        for q in corpus.queries {
            ensure!(
                !q.relevant.is_empty()
                    && q.relevant
                        .iter()
                        .all(|id| corpus.documents.iter().any(|d| &d.id == id)),
                "query {} has missing/empty relevance judgments",
                q.id
            );
            let embedding = runtime
                .infer(&[tokens(
                    &tokenizer,
                    &format!("{}{}", candidate.query_prompt, q.text),
                )?])?
                .pop()
                .context("missing query vector")?;
            let mut rankings: Vec<_> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, v.iter().zip(&embedding).map(|(a, b)| a * b).sum::<f32>()))
                .collect();
            rankings.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let mut hits = 0;
            let mut reciprocal = 0.;
            let mut dcg = 0.;
            for (rank, (i, _)) in rankings.iter().take(10).enumerate() {
                if q.relevant.contains(&corpus.documents[*i].id) {
                    hits += 1;
                    if reciprocal == 0. {
                        reciprocal = 1. / (rank + 1) as f64;
                    }
                    dcg += 1. / ((rank + 2) as f64).log2();
                }
            }
            let ideal = (0..q.relevant.len().min(10))
                .map(|r| 1. / ((r + 2) as f64).log2())
                .sum::<f64>();
            domains.entry(q.domain).or_default().push((
                hits as f64 / q.relevant.len() as f64,
                reciprocal,
                dcg / ideal,
            ));
        }
        for (domain, scores) in domains {
            let n = scores.len() as f64;
            println!(
                "{}",
                json!({"type":"quality","domain":domain,"queries":scores.len(),"recall_at_10":scores.iter().map(|s|s.0).sum::<f64>()/n,"mrr_at_10":scores.iter().map(|s|s.1).sum::<f64>()/n,"ndcg_at_10":scores.iter().map(|s|s.2).sum::<f64>()/n})
            );
        }
    }
    Ok(())
}
