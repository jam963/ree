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
