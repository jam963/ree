//! Local-only qualification through exactly the ONNX runtime used by ingestion.
#[path = "support/quality.rs"]
mod quality;
#[path = "support/telemetry.rs"]
mod telemetry;
#[path = "support/vector_export.rs"]
mod vector_export;
use anyhow::{Result, ensure};
use clap::Parser;
use quality::Corpus;
use ree::{
    config::xdg,
    events::Events,
    model::{
        self, Runtime, device, download,
        onnx::{InferenceRecipe, Onnx},
    },
    util::{hash_file, read_limited},
};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Write,
    path::PathBuf,
    time::Instant,
};
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
    /// Recreate the session repeatedly; OS file caches are NOT flushed.
    #[arg(long, default_value_t = 3)]
    load_repeats: usize,
    /// Skip synthetic throughput probes, not model load/warm-up.
    #[arg(long, requires = "corpus")]
    quality_only: bool,
    /// New local directory for full native vectors; completion metadata is written last.
    #[arg(long, requires = "quality_only", conflicts_with_all = ["dimensions", "validate_only"])]
    export_vectors: Option<PathBuf>,
    /// Validate the tokenizer/graph/provider and mixed-length shapes, without quality scoring.
    #[arg(long, conflicts_with_all = ["corpus", "quality_only", "dimensions"])]
    validate_only: bool,
    /// Truncate and renormalize for quality evaluation (e.g. Arctic Matryoshka 256).
    /// Inference always produces the full native dimension.
    #[arg(long)]
    dimensions: Option<usize>,
    /// Record the actual power mode; run while the development machine is idle.
    #[arg(long)]
    power_mode: String,
}
#[derive(Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    path: PathBuf,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Candidate {
    name: String,
    revision: String,
    precision: String,
    model: Artifact,
    tokenizer: Artifact,
    /// All reviewed ONNX external-data dependencies, verified before session creation.
    #[serde(default)]
    sidecars: Vec<Artifact>,
    recipe: InferenceRecipe,
    document_prompt: String,
    query_prompt: String,
}
impl Candidate {
    fn verify_artifacts(&self) -> Result<()> {
        for artifact in [&self.model, &self.tokenizer]
            .into_iter()
            .chain(&self.sidecars)
        {
            ensure!(
                hash_file(&artifact.path)? == artifact.sha256,
                "artifact checksum mismatch: {}",
                artifact.path.display()
            );
        }
        Ok(())
    }
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
#[derive(Default, serde::Serialize)]
struct TokenStatistics {
    items: usize,
    truncated: usize,
    original_tokens: usize,
    encoded_tokens: usize,
    longest_original: usize,
}
fn audited_tokens(
    limited: &Tokenizer,
    full: &Tokenizer,
    text: &str,
    statistics: &mut TokenStatistics,
    digest: &mut Sha256,
) -> Result<Vec<i64>> {
    let original = full
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!(e))?
        .len();
    let ids = tokens(limited, text)?;
    statistics.items += 1;
    statistics.truncated += usize::from(original > ids.len());
    statistics.original_tokens += original;
    statistics.encoded_tokens += ids.len();
    statistics.longest_original = statistics.longest_original.max(original);
    digest.update((ids.len() as u64).to_le_bytes());
    for id in &ids {
        digest.update(id.to_le_bytes());
    }
    Ok(ids)
}
fn compare(reference: &[Vec<f32>], output: &[Vec<f32>]) -> (f64, f32) {
    let mut cosine = 1f64;
    let mut difference = 0f32;
    for (a, b) in reference.iter().zip(output) {
        let dot = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum::<f64>();
        let norm = |v: &[f32]| v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
        cosine = cosine.min(dot / (norm(a) * norm(b)));
        for (&x, &y) in a.iter().zip(b) {
            difference = difference.max((x - y).abs());
        }
    }
    (cosine, difference)
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
        args.repeats >= 2
            && (1..=20).contains(&args.load_repeats)
            && !args.batches.is_empty()
            && args.batches.iter().all(|&n| n > 0 && n <= 256)
            && args.batches.windows(2).all(|w| w[0] < w[1]),
        "use at least two repeats, 1..=20 loads, and strictly increasing batches in 1..=256"
    );
    let corpus = args
        .corpus
        .as_ref()
        .map(|path| -> Result<(Corpus, String)> {
            let bytes = read_limited(std::fs::File::open(path)?, 128 * 1024 * 1024)?;
            let c: Corpus = serde_json::from_slice(&bytes)?;
            c.validate()?;
            Ok((c, ree::util::hash(&bytes)))
        })
        .transpose()?;
    let (corpus, corpus_sha256) = match corpus {
        Some((c, hash)) => (Some(c), Some(hash)),
        None => (None, None),
    };
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
    let uuid = gpu.as_ref().map(|g| g.uuid.clone());
    let host_start = telemetry::host_snapshot(uuid.as_deref());
    let candidate: Candidate = if let Some(path) = args.manifest {
        let mut candidate: Candidate =
            serde_json::from_slice(&read_limited(std::fs::File::open(&path)?, 1024 * 1024)?)?;
        let base = path.parent().unwrap_or(std::path::Path::new("."));
        for artifact in [&mut candidate.model, &mut candidate.tokenizer]
            .into_iter()
            .chain(candidate.sidecars.iter_mut())
        {
            if artifact.path.is_relative() {
                artifact.path = base.join(&artifact.path);
            }
        }
        candidate
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
            sidecars: vec![],
            recipe: InferenceRecipe::default(),
            document_prompt: "".into(),
            query_prompt: "query: ".into(),
        }
    };
    ensure!(
        candidate.revision.len() == 40 && candidate.revision.chars().all(|c| c.is_ascii_hexdigit()),
        "candidate requires an exact 40-character model revision"
    );
    candidate.verify_artifacts()?;
    let dimensions = args.dimensions.unwrap_or(candidate.recipe.dimensions);
    ensure!(
        dimensions > 0 && dimensions <= candidate.recipe.dimensions,
        "evaluation dimensions exceed model dimensions"
    );
    let mut tokenizer =
        Tokenizer::from_file(&candidate.tokenizer.path).map_err(|e| anyhow::anyhow!(e))?;
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length: 512,
            ..Default::default()
        }))
        .map_err(|e| anyhow::anyhow!(e))?;
    let empty_ids = tokens(&tokenizer, "")?;
    ensure!(
        empty_ids == [candidate.recipe.cls_id, candidate.recipe.sep_id],
        "tokenizer special IDs do not match candidate recipe"
    );
    let artifact_bytes = std::iter::once(&candidate.model)
        .chain(&candidate.sidecars)
        .try_fold(0u64, |total, artifact| -> Result<u64> {
            total
                .checked_add(artifact.path.metadata()?.len())
                .ok_or_else(|| anyhow::anyhow!("artifact size overflow"))
        })?;
    if let Some(path) = &args.export_vectors {
        let corpus = corpus
            .as_ref()
            .expect("clap requires a corpus for vector exports");
        vector_export::bytes(corpus.documents.len(), dimensions)?;
        vector_export::bytes(corpus.queries.len(), dimensions)?;
        std::fs::create_dir(path).map_err(|e| {
            anyhow::anyhow!("export directory must be new: {}: {e}", path.display())
        })?;
    }
    let load = || {
        Onnx::load_for_qualification(
            &candidate.model.path,
            gpu.as_ref().map(|g| (g, device::budget(g, 0.85))),
            512,
            candidate.recipe.clone(),
        )
    };
    let information = json!({"type":"qualification_started","model":candidate.name,"revision":candidate.revision,
        "model_sha256":candidate.model.sha256,"tokenizer_sha256":candidate.tokenizer.sha256,"precision":candidate.precision,
        "recipe":candidate.recipe,"document_prompt":candidate.document_prompt,"query_prompt":candidate.query_prompt,
        "runtime":model::RUNTIME_VERSION,"cpu":"AMD Ryzen 9 8945HS","gpu":gpu,"power_mode":args.power_mode,
        "threads":model::telemetry::inference_threads(),"artifact_bytes":artifact_bytes,
        "sidecars":candidate.sidecars,"tokenizer_bytes":candidate.tokenizer.path.metadata()?.len(),
        "corpus_sha256":corpus_sha256,"provenance":corpus.as_ref().map(|c| &c.provenance),"evaluation_dimensions":dimensions,
        "metrics":{"version":2,"cutoff":10,"ndcg_gain":"linear","positive_threshold":0,"tie_break":"document_id_ascending",
        "exclude_query_id":corpus.as_ref().map(|c| c.exclude_query_id)},
        "host":host_start,"scope":"this development machine only; resource sampling enabled"});
    println!("{information}");
    let mut runtime = None;
    let mut loads = Vec::new();
    for repeat in 0..args.load_repeats {
        drop(runtime.take());
        let sampler = telemetry::Sampler::start(uuid.clone());
        let start = Instant::now();
        runtime = Some(load()?);
        let seconds = start.elapsed().as_secs_f64();
        loads.push(seconds);
        println!(
            "{}",
            json!({"type":"model_load","repeat":repeat,"load_and_warmup_seconds":seconds,
            "cache_state":"OS caches not flushed; artifact hashing precedes first load","resources":sampler.finish()})
        );
    }
    println!(
        "{}",
        json!({"type":"runtime_versions","versions":model::telemetry::versions(gpu.is_some()),"load_samples_seconds":loads})
    );
    if args.validate_only {
        let inputs = [
            "A short document.",
            "日本語の検索と العربية",
            "fn main() { println!(\"hello\"); }",
            "",
        ]
        .iter()
        .map(|text| tokens(&tokenizer, text))
        .collect::<Result<Vec<_>>>()?;
        let r = runtime.as_mut().unwrap();
        let batched = r.infer(&inputs)?;
        let individual = inputs
            .iter()
            .map(|input| -> Result<Vec<f32>> {
                Ok(r.infer(std::slice::from_ref(input))?.remove(0))
            })
            .collect::<Result<Vec<_>>>()?;
        let (cosine, difference) = compare(&individual, &batched);
        println!(
            "{}",
            json!({"type":"candidate_validation","token_ids":inputs,
            "dimensions":candidate.recipe.dimensions,"provider":r.provider(),
            "mixed_vs_individual_min_cosine":cosine,"mixed_vs_individual_max_abs_difference":difference,
            "scope":"load/shapes/finite unit vectors; batch drift diagnostic, not retrieval quality qualification"})
        );
    }
    if !args.quality_only && !args.validate_only {
        for length in [64, 256, 512] {
            if runtime.is_none() {
                runtime = Some(load()?);
            }
            for &batch in &args.batches {
                let mut input = vec![42; length];
                input[0] = candidate.recipe.cls_id;
                input[length - 1] = candidate.recipe.sep_id;
                let inputs = vec![input; batch];
                let sampler = telemetry::Sampler::start(uuid.clone());
                let probe = (|| -> Result<_> {
                    let r = runtime.as_mut().unwrap();
                    let reference = r.infer(&inputs)?;
                    let mut samples = Vec::new();
                    let mut min_cosine = 1f64;
                    let mut max_difference = 0f32;
                    for _ in 0..args.repeats {
                        let start = Instant::now();
                        let output = r.infer(&inputs)?;
                        samples.push(start.elapsed().as_secs_f64());
                        let (cosine, diff) = compare(&reference, &output);
                        min_cosine = min_cosine.min(cosine);
                        max_difference = max_difference.max(diff);
                    }
                    Ok((samples, min_cosine, max_difference))
                })();
                let resources = sampler.finish();
                let (samples, cosine, difference) = match probe {
                    Ok(v) => v,
                    Err(e) => {
                        println!(
                            "{}",
                            json!({"type":"probe_failed","tokens":length,"batch":batch,"error":format!("{e:#}"),"resources":resources})
                        );
                        // An OOM is a measured bound, not a fatal qualification
                        // error. Recreate before the next length after any OOM.
                        if device::classify(&format!("{e:#}")) != device::Failure::Oom {
                            return Err(e);
                        }
                        drop(runtime.take());
                        break;
                    }
                };
                let mean = samples.iter().sum::<f64>() / samples.len() as f64;
                let variance =
                    samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
                println!(
                    "{}",
                    json!({"type":"throughput","tokens":length,"batch":batch,"samples_seconds":samples,
                    "mean_seconds":mean,"variance_seconds_squared":variance,"embeddings_per_second":batch as f64/mean,
                    "tokens_per_second":(batch*length) as f64/mean,"repeat_min_cosine":cosine,"repeat_max_abs_difference":difference,
                    "resources":resources,"free_vram_after":gpu.as_ref().and_then(|g| device::free_memory(g.index).ok())})
                );
            }
        }
    }
    if let Some(corpus) = corpus {
        if runtime.is_none() {
            runtime = Some(load()?);
        }
        let runtime = runtime.as_mut().unwrap();
        let sampler = telemetry::Sampler::start(uuid.clone());
        let start = Instant::now();
        let mut full_tokenizer = tokenizer.clone();
        full_tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!(e))?;
        let mut document_statistics = TokenStatistics::default();
        let mut query_statistics = TokenStatistics::default();
        let mut document_digest = Sha256::new();
        let mut query_digest = Sha256::new();
        let mut sample: Vec<_> = (0..corpus.documents.len()).collect();
        sample.sort_by_cached_key(|&i| {
            (
                ree::util::hash(
                    format!(
                        "ree-batch-drift-v1\0{}\0{}",
                        "20260909", corpus.documents[i].id
                    )
                    .as_bytes(),
                ),
                i,
            )
        });
        sample.truncate(if args.export_vectors.is_some() { 32 } else { 0 });
        let sample: BTreeSet<_> = sample.into_iter().collect();
        let mut sampled_inputs = BTreeMap::new();
        let mut vectors = Vec::new();
        for (batch, docs) in corpus.documents.chunks(8).enumerate() {
            let inputs = docs
                .iter()
                .map(|d| {
                    audited_tokens(
                        &tokenizer,
                        &full_tokenizer,
                        &format!("{}{}", candidate.document_prompt, d.text),
                        &mut document_statistics,
                        &mut document_digest,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            for (offset, ids) in inputs.iter().enumerate() {
                if sample.contains(&(batch * 8 + offset)) {
                    sampled_inputs.insert(batch * 8 + offset, ids.clone());
                }
            }
            for vector in runtime.infer(&inputs)? {
                vectors.push(quality::project(vector, dimensions)?);
            }
            if vectors.len() % 128 == 0 || vectors.len() == corpus.documents.len() {
                println!(
                    "{}",
                    json!({"type":"quality_progress","documents_encoded":vectors.len(),
                    "documents_total":corpus.documents.len(),"elapsed_seconds":start.elapsed().as_secs_f64()})
                );
                std::io::stdout().flush()?;
            }
        }
        let mut query_vectors = Vec::new();
        let mut domains: BTreeMap<String, Vec<(f64, f64, f64)>> = BTreeMap::new();
        for q in &corpus.queries {
            let embedding = quality::project(
                runtime
                    .infer(&[audited_tokens(
                        &tokenizer,
                        &full_tokenizer,
                        &format!("{}{}", candidate.query_prompt, q.text),
                        &mut query_statistics,
                        &mut query_digest,
                    )?])?
                    .remove(0),
                dimensions,
            )?;
            if args.export_vectors.is_some() {
                query_vectors.push(embedding.clone());
            }
            let scores: Vec<_> = vectors
                .iter()
                .map(|v| v.iter().zip(&embedding).map(|(a, b)| a * b).sum::<f32>())
                .collect();
            let top = quality::top_ids(&corpus.documents, &scores, &q.id, corpus.exclude_query_id);
            let (recall, mrr, ndcg) = quality::score(&top, &q.relevant, &q.relevance);
            domains
                .entry(q.domain.clone())
                .or_default()
                .push((recall, mrr, ndcg));
            println!(
                "{}",
                json!({"type":"quality_query","query":q.id,"domain":q.domain,"top_10":top,
                "recall_at_10":recall,"mrr_at_10":mrr,"ndcg_at_10":ndcg})
            );
        }
        for (domain, scores) in domains {
            let n = scores.len() as f64;
            println!(
                "{}",
                json!({"type":"quality","domain":domain,"queries":scores.len(),
                "recall_at_10":scores.iter().map(|s|s.0).sum::<f64>()/n,"mrr_at_10":scores.iter().map(|s|s.1).sum::<f64>()/n,
                "ndcg_at_10":scores.iter().map(|s|s.2).sum::<f64>()/n})
            );
        }
        let statistics = json!({"documents":document_statistics,"queries":query_statistics});
        if let Some(path) = &args.export_vectors {
            let mut individual_vectors = Vec::new();
            for ids in sampled_inputs.values() {
                individual_vectors.push(quality::project(
                    runtime.infer(std::slice::from_ref(ids))?.remove(0),
                    dimensions,
                )?);
            }
            let matrix = |name: &str,
                          ids: Vec<String>,
                          values: &[Vec<f32>]|
             -> Result<vector_export::Matrix> {
                Ok(vector_export::Matrix {
                    ids,
                    sha256: vector_export::write_matrix(path, name, values, dimensions)?,
                })
            };
            let metadata = vector_export::Metadata {
                version: 1,
                dimensions,
                information: information.clone(),
                documents: matrix(
                    "documents.f32",
                    corpus.documents.iter().map(|d| d.id.clone()).collect(),
                    &vectors,
                )?,
                queries: matrix(
                    "queries.f32",
                    corpus.queries.iter().map(|q| q.id.clone()).collect(),
                    &query_vectors,
                )?,
                single_documents: matrix(
                    "single-documents.f32",
                    sampled_inputs
                        .keys()
                        .map(|&i| corpus.documents[i].id.clone())
                        .collect(),
                    &individual_vectors,
                )?,
                document_tokens_sha256: format!("{:x}", document_digest.finalize()),
                query_tokens_sha256: format!("{:x}", query_digest.finalize()),
                token_statistics: statistics.clone(),
                document_batch: 8,
                query_batch: 1,
            };
            vector_export::complete(path, &metadata)?;
            println!(
                "{}",
                json!({"type":"vector_export_completed","path":path,
                "metadata_sha256":hash_file(&path.join("metadata.json"))?,"single_document_samples":individual_vectors.len()})
            );
        }
        println!(
            "{}",
            json!({"type":"quality_completed","documents":corpus.documents.len(),"queries":corpus.queries.len(),
            "elapsed_seconds":start.elapsed().as_secs_f64(),"token_statistics":statistics,"resources":sampler.finish()})
        );
    }
    println!(
        "{}",
        json!({"type":"qualification_completed","host":telemetry::host_snapshot(uuid.as_deref())})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn manifest_sidecars_are_verified_and_unknown_fields_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("artifact");
        std::fs::write(&file, b"fixture").unwrap();
        let artifact = json!({"path":file,"sha256":ree::util::hash(b"fixture")});
        let mut value = json!({"name":"fixture","revision":"0".repeat(40),"precision":"fp32",
            "model":artifact,"tokenizer":artifact,"sidecars":[artifact],
            "recipe":{"dimensions":384,"pooling":"cls","pad_id":1,"cls_id":0,"sep_id":2,"token_output":"logits"},
            "document_prompt":"","query_prompt":""});
        let mut candidate: Candidate = serde_json::from_value(value.clone()).unwrap();
        candidate.verify_artifacts().unwrap();
        assert_eq!(candidate.recipe.token_output.as_deref(), Some("logits"));
        candidate.sidecars[0].sha256 = "0".repeat(64);
        assert!(candidate.verify_artifacts().is_err());
        value["sidecar"] = json!([]);
        assert!(serde_json::from_value::<Candidate>(value).is_err());
    }
    #[test]
    fn legacy_recipes_keep_production_output_selection() {
        let recipe: InferenceRecipe = serde_json::from_value(json!({
            "dimensions":768,"pooling":"cls","pad_id":1,"cls_id":0,"sep_id":2
        }))
        .unwrap();
        assert!(recipe.token_output.is_none());
        assert!(
            serde_json::to_value(recipe)
                .unwrap()
                .get("token_output")
                .is_none()
        );
    }
    #[test]
    fn repeatability_metrics_detect_drift() {
        assert_eq!(compare(&[vec![1., 0.]], &[vec![1., 0.]]), (1., 0.));
        assert_eq!(compare(&[vec![1., 0.]], &[vec![0., 1.]]), (0., 1.));
    }
}
