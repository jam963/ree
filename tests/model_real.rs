//! Deliberately excluded from ordinary CI. These download real model artifacts.
use anyhow::{Result, ensure};
use ree::{
    chunk,
    config::xdg,
    events::Events,
    model::{self, Runtime, device, download, onnx::Onnx},
};
use tokenizers::Tokenizer;
fn local_host() -> Result<Vec<device::Gpu>> {
    ensure!(
        std::env::var_os("CI").is_none(),
        "real model tests must not run in CI"
    );
    ensure!(
        std::fs::read_to_string("/proc/cpuinfo")?.contains("AMD Ryzen 9 8945HS"),
        "development CPU required"
    );
    let devices = device::discover()?;
    ensure!(
        devices.iter().any(|d| d.name.contains("RTX 4070 Laptop")),
        "development GPU host required"
    );
    Ok(devices)
}
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
#[test]
#[ignore = "downloads model; run only on the development machine"]
fn cpu_token_windows_and_repeatability() -> Result<()> {
    local_host()?;
    let cache = xdg("XDG_CACHE_HOME", ".cache")?;
    let mut events = Events::new(true, false, false);
    let tokenizer = Tokenizer::from_file(download::ensure_artifact(
        &cache,
        download::TOKENIZER,
        &mut events,
    )?)
    .map_err(|e| anyhow::anyhow!(e))?;
    let text = "Rust: fn main() {} 日本語 文書 مرحبا بالعالم.\n".repeat(120);
    let chunks = chunk::windows(&tokenizer, &text, 512, 64)?;
    assert!(chunks.len() > 1);
    for c in &chunks {
        assert_eq!(&text[c.byte_start..c.byte_end], c.text);
        assert!(c.input_ids.len() <= 512);
    }
    for pair in chunks.windows(2) {
        assert_eq!(pair[0].token_end - pair[1].token_start, 64);
    }
    let mut runtime = Onnx::load(
        &download::ensure_artifact(&cache, download::CPU, &mut events)?,
        None,
        512,
    )?;
    let inputs: Vec<_> = chunks.iter().map(|c| c.input_ids.clone()).collect();
    let a = runtime.infer(&inputs)?;
    let b = runtime.infer(&inputs)?;
    for (a, b) in a.iter().zip(b) {
        model::validate_vector(a)?;
        assert!(cosine(a, &b) > 0.99999);
    }
    Ok(())
}
#[test]
#[ignore = "downloads model; query encoding and retrieval on the development machine only"]
fn cpu_query_encoder_and_search_end_to_end() -> Result<()> {
    use ree::{
        cli::Options,
        config::Config,
        model::{
            LocalEngine,
            query::{self, QueryEmbedder},
        },
        pipeline,
        retrieval::{self, SearchMode, SearchRequest},
        storage::Database,
    };
    local_host()?;
    let temp = tempfile::tempdir()?;
    let config_path = temp.path().join("config.toml");
    std::fs::write(&config_path, "")?;
    let options = Options {
        config: Some(config_path),
        db: Some(temp.path().join("ree.db")),
        device: Some("cpu".into()),
        ..Default::default()
    };
    let config = Config::load(&options)?;
    let mut events = Events::new(true, false, false);
    let path = download::ensure_artifact(&config.cache, download::TOKENIZER, &mut events)?;
    let mut tokenizer = Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!(e))?;
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(None)
        .map_err(|e| anyhow::anyhow!(e))?;
    let text = "Where are embeddings stored?";
    let ids = query::encode(&tokenizer, text)?;
    let reference = tokenizer
        .encode(format!("query: {text}"), true)
        .map_err(|e| anyhow::anyhow!(e))?;
    assert_eq!(
        ids,
        reference
            .get_ids()
            .iter()
            .map(|&i| i64::from(i))
            .collect::<Vec<_>>()
    );
    let mut engine = LocalEngine::new(config.clone());
    let embedded = engine.embed_query(text, &mut events)?;
    model::validate_vector(&embedded.vector)?;
    assert_eq!(embedded.provider, "cpu");
    assert_eq!(embedded.token_count, ids.len());
    let docs = temp.path().join("docs");
    std::fs::create_dir(&docs)?;
    std::fs::write(
        docs.join("storage.txt"),
        "SQLite stores embeddings in a local database.",
    )?;
    std::fs::write(
        docs.join("ocean.txt"),
        "The ocean is blue and contains salt water.",
    )?;
    let mut db = Database::open(&config.db, true)?;
    // A separate engine ensures this test also exercises the ordinary ingestion
    // scheduler; query initialization must not change ingestion defaults.
    pipeline::ingest(
        &mut db,
        &mut LocalEngine::new(config.clone()),
        &mut events,
        &config,
        &options,
        &[docs.to_string_lossy().into_owned()],
    )?;
    let mut reader = Database::open(&config.db, false)?;
    let report = retrieval::search(
        &mut reader,
        &mut engine,
        &SearchRequest {
            query: text.into(),
            mode: SearchMode::Semantic,
            limit: 2,
            root: None,
            media_type: None,
        },
        &mut events,
    )?;
    assert_eq!(report.results.len(), 2);
    assert!(report.results[0].chunk.uri.ends_with("storage.txt"));
    assert_eq!(report.query_provider.as_deref(), Some("cpu"));
    Ok(())
}

#[test]
#[ignore = "requires installed CUDA 13 and both real artifacts on development GPU"]
fn cpu_int8_cuda_fp16_cosine_and_ranking_parity() -> Result<()> {
    let devices = local_host()?;
    let gpu = device::select(&devices, "cuda", 0.85)?;
    Onnx::check_cuda(&gpu, device::budget(&gpu, 0.85))?;
    let cache = xdg("XDG_CACHE_HOME", ".cache")?;
    let mut events = Events::new(true, false, false);
    let tokenizer = Tokenizer::from_file(download::ensure_artifact(
        &cache,
        download::TOKENIZER,
        &mut events,
    )?)
    .map_err(|e| anyhow::anyhow!(e))?;
    let texts = [
        "query: Where are embeddings stored?",
        "SQLite stores vectors in a local database.",
        "The ocean is blue.",
        "日本語の文書を検索します。",
        "fn main() { println!(\"hello\"); }",
        "query: Rust source code",
    ];
    let mut inputs = texts
        .iter()
        .map(|s| {
            tokenizer
                .encode(*s, true)
                .map(|e| e.get_ids().iter().map(|&i| i64::from(i)).collect())
                .map_err(|e| anyhow::anyhow!(e))
        })
        .collect::<Result<Vec<_>>>()?;
    for length in [64, 256, 512] {
        let mut ids = vec![42; length];
        ids[0] = 0;
        ids[length - 1] = 2;
        inputs.push(ids);
    }
    let mut cpu = Onnx::load(
        &download::ensure_artifact(&cache, download::CPU, &mut events)?,
        None,
        512,
    )?;
    let a = cpu.infer(&inputs)?;
    drop(cpu);
    let fp16_path = download::ensure_artifact(&cache, download::CUDA, &mut events)?;
    let mut fp16_cpu = Onnx::load(&fp16_path, None, 512)?;
    let reference = fp16_cpu.infer(&inputs)?;
    drop(fp16_cpu);
    let mut cuda = Onnx::load(&fp16_path, Some((&gpu, device::budget(&gpu, 0.85))), 512)?;
    let b = cuda.infer(&inputs)?;
    let minimum_cosine = a
        .iter()
        .zip(&b)
        .map(|(a, b)| cosine(a, b))
        .fold(1.0f32, f32::min);
    println!("CPU INT8/CUDA FP16 minimum cosine: {minimum_cosine}");
    for (i, (r, g)) in reference.iter().zip(&b).enumerate() {
        assert!(
            cosine(r, g) >= 0.999,
            "same FP16 artifact disagrees between CPU and CUDA"
        );
        println!(
            "FP16 CPU/reference vs GPU sample {i}: {}; INT8 vs FP16 CPU: {}",
            cosine(r, g),
            cosine(&a[i], r)
        );
    }
    // Reusing the same session with different padded shapes must remain stable.
    for batch in [1, 2, 4] {
        for (offset, inputs) in inputs.chunks(batch).enumerate() {
            for (j, vector) in cuda.infer(inputs)?.iter().enumerate() {
                assert!(cosine(vector, &b[offset * batch + j]) > 0.999);
            }
        }
    }
    for (index, (a, b)) in a.iter().zip(&b).enumerate() {
        println!("sample {index}: cosine {}", cosine(a, b));
    }
    for (a, b) in a.iter().zip(&b) {
        assert!(
            cosine(a, b) >= 0.94,
            "INT8/FP16 quantization compatibility fell below provisional 0.94 bound (CUDA accuracy is checked separately against CPU FP16)"
        );
    }
    for query in [0, 5] {
        let nearest = |v: &[Vec<f32>]| {
            (1..5)
                .max_by(|&i, &j| cosine(&v[query], &v[i]).total_cmp(&cosine(&v[query], &v[j])))
                .unwrap()
        };
        assert_eq!(nearest(&a), nearest(&b));
    }
    Ok(())
}
