//! Offline retrieval over hash-verified exports from ree's actual Rust runtime.
#[path = "support/quality.rs"]
mod quality;
#[path = "support/vector_export.rs"]
mod vector_export;
use anyhow::{Result, ensure};
use clap::Parser;
use quality::Corpus;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf};
use vector_export::Export;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    corpus: PathBuf,
    #[arg(long)]
    cpu_vectors: PathBuf,
    #[arg(long)]
    cuda_vectors: PathBuf,
    /// Only the explicitly planned Arctic 768 -> 256 projection is supported.
    #[arg(long)]
    dimensions: Option<usize>,
}
fn validate_pair(corpus: &Corpus, hash: &str, cpu: &Export, cuda: &Export) -> Result<()> {
    let docs: Vec<_> = corpus.documents.iter().map(|d| &d.id).collect();
    let queries: Vec<_> = corpus.queries.iter().map(|q| &q.id).collect();
    for export in [cpu, cuda] {
        let m = &export.metadata;
        ensure!(
            m.information["corpus_sha256"] == hash
                && m.documents.ids.iter().collect::<Vec<_>>() == docs
                && m.queries.ids.iter().collect::<Vec<_>>() == queries,
            "export/corpus IDs or hash mismatch"
        );
        ensure!(
            m.dimensions == m.information["evaluation_dimensions"].as_u64().unwrap_or(0) as usize,
            "export dimension metadata mismatch"
        );
        ensure!(
            m.information["metrics"]["version"] == 2
                && m.information["metrics"]["exclude_query_id"] == corpus.exclude_query_id,
            "export metric policy mismatch"
        );
        ensure!(
            m.single_documents.ids.iter().all(|id| docs.contains(&id)),
            "sample references unknown document"
        );
    }
    ensure!(
        cpu.metadata.information["gpu"].is_null() && cuda.metadata.information["gpu"].is_object(),
        "provider labels do not match supplied CPU/CUDA exports"
    );
    for field in [
        "model",
        "revision",
        "tokenizer_sha256",
        "recipe",
        "document_prompt",
        "query_prompt",
        "runtime",
        "threads",
        "metrics",
    ] {
        ensure!(
            !cpu.metadata.information[field].is_null()
                && cpu.metadata.information[field] == cuda.metadata.information[field],
            "incompatible semantic metadata: {field}"
        );
    }
    ensure!(
        cpu.metadata.dimensions == cuda.metadata.dimensions
            && cpu.metadata.document_tokens_sha256 == cuda.metadata.document_tokens_sha256
            && cpu.metadata.query_tokens_sha256 == cuda.metadata.query_tokens_sha256
            && cpu.metadata.single_documents.ids == cuda.metadata.single_documents.ids,
        "token inputs, dimensions or sample selection differ across providers"
    );
    Ok(())
}
fn project(export: &mut Export, dimensions: usize) -> Result<()> {
    if dimensions != export.metadata.dimensions {
        for v in export
            .documents
            .iter_mut()
            .chain(&mut export.queries)
            .chain(&mut export.single_documents)
        {
            *v = quality::project(std::mem::take(v), dimensions)?;
        }
    }
    Ok(())
}
fn cosines(a: &[Vec<f32>], b: &[Vec<f32>]) -> Value {
    assert_eq!(a.len(), b.len());
    let mut values: Vec<_> = a
        .iter()
        .zip(b)
        .map(|(a, b)| {
            let dot = a
                .iter()
                .zip(b)
                .map(|(&x, &y)| f64::from(x) * f64::from(y))
                .sum::<f64>();
            let norm = |v: &[f32]| v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
            (dot / (norm(a) * norm(b))).clamp(-1., 1.)
        })
        .collect();
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        return Value::Null;
    }
    json!({"count":values.len(),"min":values[0],"median":(values[(values.len()-1)/2]+values[values.len()/2])/2.,
        "mean":values.iter().sum::<f64>()/values.len() as f64,"max":values[values.len()-1]})
}
fn document_scores(
    cpu: &[Vec<f32>],
    cuda: &[Vec<f32>],
    mixed: &[bool],
    provider: &str,
    query: &[f32],
) -> Vec<f32> {
    assert_eq!(cpu.len(), cuda.len());
    assert_eq!(cpu.len(), mixed.len());
    cpu.iter()
        .zip(cuda)
        .enumerate()
        .map(|(i, (c, g))| {
            let v = match provider {
                "cpu" => c,
                "cuda" => g,
                "mixed" => {
                    if mixed[i] {
                        g
                    } else {
                        c
                    }
                }
                _ => panic!("unknown index provider"),
            };
            v.iter().zip(query).map(|(a, b)| a * b).sum::<f32>()
        })
        .collect()
}
fn mixed_uses_cuda(id: &str) -> bool {
    // First hex digit's low bit is NOT the first byte's low bit; decode the byte.
    let hash = ree::util::hash(format!("ree-mixed-index-v1\0{}\0{id}", "20260909").as_bytes());
    u8::from_str_radix(&hash[..2], 16).unwrap() & 1 == 1
}
fn main() -> Result<()> {
    let args = Args::parse();
    let bytes = ree::util::read_limited(std::fs::File::open(&args.corpus)?, 128 * 1024 * 1024)?;
    let corpus: Corpus = serde_json::from_slice(&bytes)?;
    corpus.validate()?;
    let hash = ree::util::hash(&bytes);
    let mut cpu = vector_export::load(&args.cpu_vectors)?;
    let mut cuda = vector_export::load(&args.cuda_vectors)?;
    validate_pair(&corpus, &hash, &cpu, &cuda)?;
    let dimensions = args.dimensions.unwrap_or(cpu.metadata.dimensions);
    ensure!(
        dimensions == cpu.metadata.dimensions
            || (dimensions == 256
                && cpu.metadata.dimensions == 768
                && cpu.metadata.information["model"] == ree::model::MODEL_ID),
        "unsupported dimension projection"
    );
    project(&mut cpu, dimensions)?;
    project(&mut cuda, dimensions)?;
    let mixed: Vec<_> = corpus
        .documents
        .iter()
        .map(|d| mixed_uses_cuda(&d.id))
        .collect();
    println!(
        "{}",
        json!({"type":"compatibility_started","corpus_sha256":hash,"dimensions":dimensions,
        "model":cpu.metadata.information["model"],"revision":cpu.metadata.information["revision"],
        "cpu_precision":cpu.metadata.information["precision"],"cuda_precision":cuda.metadata.information["precision"],
        "cpu_export_sha256":ree::util::hash_file(&args.cpu_vectors.join("metadata.json"))?,
        "cuda_export_sha256":ree::util::hash_file(&args.cuda_vectors.join("metadata.json"))?,
        "mixed_cuda_documents":mixed.iter().filter(|&&x|x).count(),"total_documents":mixed.len(),
        "token_statistics":cpu.metadata.token_statistics,
        "scope":"two-corpus exploratory retrieval; query provider / index provider; not full model-selection qualification"})
    );
    let single_baseline = |export: &Export| -> Vec<Vec<f32>> {
        export
            .metadata
            .single_documents
            .ids
            .iter()
            .map(|id| {
                let index = export
                    .metadata
                    .documents
                    .ids
                    .iter()
                    .position(|d| d == id)
                    .unwrap();
                export.documents[index].clone()
            })
            .collect()
    };
    println!(
        "{}",
        json!({"type":"vector_drift","dimensions":dimensions,
        "cpu_cuda_documents":cosines(&cpu.documents,&cuda.documents),
        "cpu_cuda_queries":cosines(&cpu.queries,&cuda.queries),
        "cpu_batch8_single_documents":cosines(&single_baseline(&cpu),&cpu.single_documents),
        "cuda_batch8_single_documents":cosines(&single_baseline(&cuda),&cuda.single_documents),
        "sample_ids":cpu.metadata.single_documents.ids})
    );
    for combination in [
        "cpu/cpu",
        "cuda/cuda",
        "cpu/cuda",
        "cuda/cpu",
        "cpu/mixed",
        "cuda/mixed",
    ] {
        let (query_provider, index_provider) = combination.split_once('/').unwrap();
        let queries = if query_provider == "cpu" {
            &cpu.queries
        } else {
            &cuda.queries
        };
        let mut domains: BTreeMap<&str, Vec<(f64, f64, f64)>> = BTreeMap::new();
        for (q, embedding) in corpus.queries.iter().zip(queries) {
            let scores = document_scores(
                &cpu.documents,
                &cuda.documents,
                &mixed,
                index_provider,
                embedding,
            );
            let top = quality::top_ids(&corpus.documents, &scores, &q.id, corpus.exclude_query_id);
            let (recall, mrr, ndcg) = quality::score(&top, &q.relevant, &q.relevance);
            domains
                .entry(&q.domain)
                .or_default()
                .push((recall, mrr, ndcg));
            println!(
                "{}",
                json!({"type":"compatibility_query","combination":combination,"query":q.id,
                "domain":q.domain,"top_10":top,"recall_at_10":recall,"mrr_at_10":mrr,"ndcg_at_10":ndcg})
            );
        }
        for (domain, values) in domains {
            let n = values.len() as f64;
            println!(
                "{}",
                json!({"type":"compatibility_quality","combination":combination,"domain":domain,
                "queries":values.len(),"recall_at_10":values.iter().map(|v|v.0).sum::<f64>()/n,
                "mrr_at_10":values.iter().map(|v|v.1).sum::<f64>()/n,"ndcg_at_10":values.iter().map(|v|v.2).sum::<f64>()/n})
            );
        }
    }
    println!("{}", json!({"type":"compatibility_completed"}));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Corpus, Export, Export) {
        let corpus: Corpus = serde_json::from_value(json!({"documents":[{"id":"d","text":"doc"}],
            "queries":[{"id":"q","text":"query","relevant":["d"],"domain":"test"}]}))
        .unwrap();
        let metadata = |gpu| {
            serde_json::from_value(json!({"version":1,"dimensions":2,
            "information":{"model":"fixture","revision":"r","tokenizer_sha256":"t","recipe":{},
                "document_prompt":"","query_prompt":"","runtime":"r","threads":1,"gpu":gpu,
                "corpus_sha256":"hash","evaluation_dimensions":2,"metrics":{"version":2,"exclude_query_id":false}},
            "documents":{"ids":["d"],"sha256":"d"},"queries":{"ids":["q"],"sha256":"q"},
            "single_documents":{"ids":[],"sha256":"s"},"document_tokens_sha256":"d",
            "query_tokens_sha256":"q","token_statistics":{},"document_batch":8,"query_batch":1})).unwrap()
        };
        let make = |gpu| Export {
            metadata: metadata(gpu),
            documents: vec![vec![1., 0.]],
            queries: vec![vec![1., 0.]],
            single_documents: vec![],
        };
        (corpus, make(Value::Null), make(json!({})))
    }
    #[test]
    fn incompatible_exports_are_rejected_before_scoring() {
        let (corpus, cpu, mut cuda) = fixture();
        validate_pair(&corpus, "hash", &cpu, &cuda).unwrap();
        cuda.metadata.query_tokens_sha256 = "changed".into();
        assert!(validate_pair(&corpus, "hash", &cpu, &cuda).is_err());
        cuda.metadata.query_tokens_sha256 = "q".into();
        cuda.metadata.information["query_prompt"] = json!("different");
        assert!(validate_pair(&corpus, "hash", &cpu, &cuda).is_err());
        assert!(validate_pair(&corpus, "other-corpus", &cpu, &cpu).is_err());
    }
    #[test]
    fn homogeneous_and_mixed_indexes_use_the_correct_vectors() {
        // A rotated artifact can preserve homogeneous retrieval and break mixed retrieval.
        let cpu = vec![vec![1., 0.], vec![0., 1.]];
        let cuda = vec![vec![0., 1.], vec![1., 0.]];
        assert_eq!(
            document_scores(&cpu, &cuda, &[false, true], "cpu", &[1., 0.]),
            vec![1., 0.]
        );
        assert_eq!(
            document_scores(&cpu, &cuda, &[false, true], "cuda", &[0., 1.]),
            vec![1., 0.]
        );
        assert_eq!(
            document_scores(&cpu, &cuda, &[false, true], "cuda", &[1., 0.]),
            vec![0., 1.]
        );
        assert_eq!(
            document_scores(&cpu, &cuda, &[false, true], "cpu", &[0., 1.]),
            vec![0., 1.]
        );
        assert_eq!(
            document_scores(&cpu, &cuda, &[false, true], "mixed", &[1., 0.]),
            vec![1., 1.]
        );
    }
    #[test]
    fn cosine_summary_and_projection_do_not_hide_drift() {
        let (_, mut cpu, _) = fixture();
        assert_eq!(cosines(&[vec![1., 0.]], &[vec![0., 1.]])["mean"], 0.);
        assert!(cosines(&[], &[]).is_null());
        project(&mut cpu, 1).unwrap();
        assert_eq!(cpu.documents, vec![vec![1.]]);
        assert_eq!(mixed_uses_cuda("d"), mixed_uses_cuda("d"));
    }
}
