pub mod batch;
pub mod device;
pub mod download;
pub mod onnx;
pub mod passage;
pub mod query;
pub mod telemetry;

use crate::{
    chunk::{self, Chunk},
    config::Config,
    events::Events,
    util::hash,
};
use anyhow::{Result, anyhow, ensure};
use batch::Limits;
use device::{Failure, Gpu};
use serde_json::{Value, json};
use std::{path::PathBuf, time::Instant};
use tokenizers::Tokenizer;

pub const MODEL_ID: &str = "Snowflake/snowflake-arctic-embed-m-v2.0";
pub const REVISION: &str = "95c2741480856aa9666782eb4afe11959938017f";
pub const MODEL_KEY: &str = "arctic-m-v2-95c27414-cls-l2-768-ort128-strict-v1";
pub const RUNTIME_VERSION: &str = "onnxruntime-1.28.0";
pub fn recipe() -> Value {
    json!({"model_id":MODEL_ID,"revision":REVISION,"dimensions":768,"distance":"cosine","document_prompt":"","query_prompt":"query: ","pooling":"cls","normalization":"l2","max_tokens":8192,"tokenizer_sha256":download::TOKENIZER.sha256,"cpu":{"precision":"int8","sha256":download::CPU.sha256},"cuda":{"precision":"fp16","sha256":download::CUDA.sha256},"runtime":RUNTIME_VERSION,"cuda_options":{"use_tf32":false,"enable_skip_layer_norm_strict_mode":true},"qualification":"provisional"})
}
pub fn normalize(vector: &mut [f32]) -> Result<()> {
    ensure!(vector.len() == 768, "embedding must contain 768 values");
    normalize_dense(vector)
}
pub fn normalize_dense(vector: &mut [f32]) -> Result<()> {
    ensure!(
        !vector.is_empty() && vector.iter().all(|v| v.is_finite()),
        "embedding must contain finite values"
    );
    let norm = vector
        .iter()
        .map(|v| (*v as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    ensure!(
        norm > 0.0 && norm.is_finite(),
        "zero or non-finite vector norm"
    );
    for v in vector {
        *v = (*v as f64 / norm) as f32;
    }
    Ok(())
}
pub fn validate_vector(vector: &[f32]) -> Result<()> {
    ensure!(
        vector.len() == 768 && vector.iter().all(|v| v.is_finite()),
        "embedding must contain 768 finite values"
    );
    let norm = vector.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
    ensure!(
        (norm - 1.0).abs() < 0.001,
        "embedding must be L2 normalized"
    );
    Ok(())
}

/// All runtime-specific types are kept behind this boundary.
pub trait Runtime {
    fn provider(&self) -> &'static str;
    fn infer(&mut self, inputs: &[Vec<i64>]) -> Result<Vec<Vec<f32>>>;
}
pub trait RuntimeFactory {
    fn create(&mut self, gpu: Option<&Gpu>, events: &mut Events) -> Result<Box<dyn Runtime>>;
}
fn provision_cpu(
    config: &Config,
    events: &mut Events,
    mut ensure_artifact: impl FnMut(
        &std::path::Path,
        download::Artifact,
        &mut Events,
    ) -> Result<PathBuf>,
) -> Result<Option<PathBuf>> {
    if matches!(config.device.as_str(), "cpu" | "auto") {
        Ok(Some(ensure_artifact(&config.cache, download::CPU, events)?))
    } else {
        Ok(None)
    }
}

struct OnnxFactory {
    config: Config,
    cpu_path: Option<PathBuf>,
}
impl RuntimeFactory for OnnxFactory {
    fn create(&mut self, gpu: Option<&Gpu>, events: &mut Events) -> Result<Box<dyn Runtime>> {
        if let Some(gpu) = gpu {
            // Auto rescue was provisioned, but a later fallback must reverify it
            // rather than trust a path retained throughout a GPU session.
            self.cpu_path = None;
            onnx::Onnx::check_cuda(gpu, device::budget(gpu, self.config.gpu_memory_fraction))?;
        }
        // Reverify files before each session recreation; loaded sessions alone
        // may reuse verified state. Explicit CUDA never provisions rescue weights.
        let path = if gpu.is_some() {
            download::ensure_artifact(&self.config.cache, download::CUDA, events)?
        } else if let Some(path) = self.cpu_path.take() {
            path
        } else {
            download::ensure_artifact(&self.config.cache, download::CPU, events)?
        };
        let gpu_options = gpu.map(|g| (g, device::budget(g, self.config.gpu_memory_fraction)));
        if std::env::var("REE_CLS_OUTPUT").is_ok_and(|v| v == "1") {
            let artifact = if gpu.is_some() {
                download::CUDA
            } else {
                download::CPU
            };
            let path = onnx::cls::ensure_artifact(&path, &self.config.cache, artifact, events)?;
            Ok(Box::new(onnx::Onnx::load_cls(
                &path,
                gpu_options,
                self.config.chunk_size,
            )?))
        } else {
            Ok(Box::new(onnx::Onnx::load(
                &path,
                gpu_options,
                self.config.chunk_size,
            )?))
        }
    }
}

pub struct Scheduler {
    runtime: Option<Box<dyn Runtime>>,
    factory: Box<dyn RuntimeFactory>,
    probe: Box<dyn device::DeviceProbe>,
    gpu: Option<Gpu>,
    auto: bool,
    pub limits: Limits,
    pub workload_ns: u128,
    pub calibration_ns: u128,
    pub workload_calls: usize,
    pub calibration_calls: usize,
    pub real_tokens: usize,
    pub padded_tokens: usize,
    last_successful_items: usize,
    pub oom_reductions: usize,
    pub fallbacks: usize,
    pub inference_ms: u128,
}
impl Scheduler {
    pub fn initialize(
        config: &Config,
        factory: Box<dyn RuntimeFactory>,
        probe: Box<dyn device::DeviceProbe>,
        events: &mut Events,
    ) -> Result<Self> {
        Self::initialize_policy(config, factory, probe, events, false)
    }
    /// Singleton queries retain provider verification/fallback, but never run
    /// throughput calibration or read/write ingestion batch profiles.
    pub fn initialize_interactive(
        config: &Config,
        factory: Box<dyn RuntimeFactory>,
        probe: Box<dyn device::DeviceProbe>,
        events: &mut Events,
    ) -> Result<Self> {
        Self::initialize_policy(config, factory, probe, events, true)
    }
    fn initialize_policy(
        config: &Config,
        mut factory: Box<dyn RuntimeFactory>,
        mut probe: Box<dyn device::DeviceProbe>,
        events: &mut Events,
        interactive: bool,
    ) -> Result<Self> {
        let discovery_span = crate::metrics::Span::new("device_discovery");
        let mut fallbacks = 0;
        let gpu = if config.device == "cpu" {
            None
        } else {
            match probe
                .devices()
                .and_then(|ds| device::select(&ds, &config.device, config.gpu_memory_fraction))
            {
                Ok(g) => Some(g),
                Err(e) if config.device == "auto" => {
                    events.emit(json!({"type":"device_fallback","from":"cuda","to":"cpu","reason":e.to_string()}))?;
                    fallbacks = 1;
                    None
                }
                Err(e) => return Err(e),
            }
        };
        drop(discovery_span);
        let runtime = create_runtime(factory.as_mut(), gpu.as_ref(), events);
        let items = if interactive {
            1
        } else {
            config
                .batch_size
                .unwrap_or(if gpu.is_some() { 2 } else { 8 })
        };
        let mut s = Self {
            runtime: None,
            factory,
            probe,
            gpu,
            auto: config.device == "auto",
            limits: Limits {
                max_items: items,
                max_padded_tokens: items.saturating_mul(config.chunk_size),
            },
            workload_ns: 0,
            calibration_ns: 0,
            workload_calls: 0,
            calibration_calls: 0,
            real_tokens: 0,
            padded_tokens: 0,
            last_successful_items: 0,
            oom_reductions: 0,
            fallbacks,
            inference_ms: 0,
        };
        match runtime {
            Ok(r) => s.runtime = Some(r),
            Err(e) if s.gpu.is_some() => s.fallback(&format!("{e:#}"), events)?,
            Err(e) => return Err(e),
        }
        if !interactive {
            s.calibrate(config, events)?;
        }
        events.emit(json!({"type":"runtime","provider":s.runtime.as_ref().unwrap().provider(),"device":s.gpu,"precision":if s.gpu.is_some(){"fp16"}else{"int8"},"runtime":RUNTIME_VERSION,"runtime_versions":telemetry::versions(s.gpu.is_some()),"threads":telemetry::inference_threads(),"batch_limits":s.limits}))?;
        Ok(s)
    }
    pub fn provider(&self) -> Option<&'static str> {
        self.runtime.as_ref().map(|r| r.provider())
    }
    fn fallback(&mut self, error: &str, events: &mut Events) -> Result<()> {
        if !self.auto {
            return Err(crate::error::AppError::new(
                3,
                format!("CUDA failed: {error}; explicit CUDA requested, CPU fallback disabled"),
            )
            .into());
        }
        let from = self
            .gpu
            .as_ref()
            .map_or("cuda".into(), |g| format!("cuda:{}", g.index));
        self.runtime.take();
        self.gpu = None;
        events.emit(json!({"type":"device_fallback","from":from,"to":"cpu","reason":error,"retry_batch_items":self.limits.max_items}))?;
        self.runtime = Some(
            create_runtime(self.factory.as_mut(), None, events).map_err(|e| {
                crate::error::AppError::new(3, format!("CPU fallback initialization failed: {e:#}"))
            })?,
        );
        self.limits.max_items = self.limits.max_items.min(8);
        self.limits.max_padded_tokens = self.limits.max_padded_tokens.min(4096);
        self.fallbacks += 1;
        Ok(())
    }
    pub fn embed(&mut self, inputs: &[Vec<i64>], events: &mut Events) -> Result<Vec<Vec<f32>>> {
        self.embed_inner(inputs, events, false)
    }
    fn embed_inner(
        &mut self,
        inputs: &[Vec<i64>],
        events: &mut Events,
        calibration: bool,
    ) -> Result<Vec<Vec<f32>>> {
        let _span = crate::metrics::Span::new(if calibration {
            "calibration_embed"
        } else {
            "workload_embed"
        });
        let mut order: Vec<_> = (0..inputs.len()).collect();
        order.sort_by_key(|&i| inputs[i].len());
        let mut output = vec![Vec::new(); inputs.len()];
        let mut pos = 0;
        while pos < order.len() {
            if let Some(g) = &self.gpu
                && let Ok(free) = self.probe.free_memory(g.index)
                && free < (g.total as f64 * 0.15) as u64
            {
                self.limits.reduce();
            }
            let count = self.limits.pack(&order[pos..], inputs);
            let batch: Vec<_> = order[pos..pos + count]
                .iter()
                .map(|&i| inputs[i].clone())
                .collect();
            let start = Instant::now();
            let result = self
                .runtime
                .as_mut()
                .ok_or_else(|| anyhow!("runtime unavailable"))?
                .infer(&batch);
            let elapsed = start.elapsed();
            // Keep the legacy inclusive/truncated counter for compatibility.
            self.inference_ms += elapsed.as_millis();
            if calibration {
                self.calibration_ns += elapsed.as_nanos();
                self.calibration_calls += 1;
            } else {
                self.workload_ns += elapsed.as_nanos();
                self.workload_calls += 1;
                self.real_tokens += batch.iter().map(Vec::len).sum::<usize>();
                self.padded_tokens += count * batch.iter().map(Vec::len).max().unwrap_or(0);
            }
            match result {
                Ok(vectors) => {
                    ensure!(
                        vectors.len() == count,
                        "runtime returned wrong vector count"
                    );
                    for (&index, vector) in order[pos..pos + count].iter().zip(vectors) {
                        validate_vector(&vector)?;
                        output[index] = vector;
                    }
                    pos += count;
                    self.last_successful_items = count;
                    // Success (especially underfilled success) is not throughput
                    // evidence. Limits grow only during bounded calibration.
                }
                Err(e) if self.gpu.is_some() => {
                    if device::classify(&e.to_string()) == Failure::Oom && count > 1 {
                        self.oom_reductions += 1;
                        self.limits.reduce();
                        self.limits.max_items = self.limits.max_items.min((count / 2).max(1));
                        events.emit(json!({"type":"batch_reduced","reason":"cuda_out_of_memory","max_items":self.limits.max_items,"max_padded_tokens":self.limits.max_padded_tokens}))?;
                        self.runtime.take();
                        match create_runtime(self.factory.as_mut(), self.gpu.as_ref(), events) {
                            Ok(runtime) => self.runtime = Some(runtime),
                            Err(error) => self.fallback(&format!("{error:#}"), events)?,
                        }
                    } else {
                        self.fallback(&format!("{e:#}"), events)?;
                    }
                }
                Err(e) => return Err(e),
            }
        }
        Ok(output)
    }
    fn calibrate(&mut self, config: &Config, events: &mut Events) -> Result<()> {
        let _span = crate::metrics::Span::new("scheduler_calibration");
        let Some(gpu) = self.gpu.clone() else {
            return Ok(());
        };
        let key = hash(
            format!(
                "bounded-calibration-v2:{MODEL_KEY}:fp16:{}:{}:{RUNTIME_VERSION}:{}:{}",
                gpu.uuid,
                gpu.driver,
                config.chunk_size,
                if std::env::var("REE_CLS_OUTPUT").is_ok_and(|v| v == "1") {
                    onnx::cls::VERSION
                } else {
                    "token-output"
                }
            )
            .as_bytes(),
        );
        let path = config.cache.join("profiles").join(format!("{key}.json"));
        if config.batch_size.is_none()
            && let Ok(bytes) = std::fs::read(&path)
            && let Ok(limits) = serde_json::from_slice::<Limits>(&bytes)
        {
            self.limits = Limits {
                max_items: limits.max_items.clamp(1, 64),
                max_padded_tokens: limits.max_padded_tokens.clamp(config.chunk_size, 32768),
            };
        }
        let mut input = vec![42; config.chunk_size];
        input[0] = 0;
        input[config.chunk_size - 1] = 2;
        let mut last_rate = 0.0;
        let cap = config.batch_size.unwrap_or(16).min(64);
        let mut n = self
            .limits
            .max_items
            .min(self.limits.max_padded_tokens / config.chunk_size)
            .clamp(1, cap);
        let calibration_start = Instant::now();
        let mut best = None;
        loop {
            // Probe precisely the retained envelope, not a nominal request split
            // into unrelated shapes. Old cache entries are hints, never limits.
            self.limits = Limits {
                max_items: n,
                max_padded_tokens: n * config.chunk_size,
            };
            let reductions = self.oom_reductions;
            let calls = self.calibration_calls;
            let start = Instant::now();
            self.embed_inner(&vec![input.clone(); n], events, true)?;
            if self.gpu.is_none() {
                return Ok(());
            }
            self.limits.max_items = self.limits.max_items.min(self.last_successful_items).min(n);
            self.limits.max_padded_tokens = self.limits.max_items * config.chunk_size;
            // Never restore a formerly fast but now unsafe shape after memory
            // reduction/recreation. A split call cannot validate the nominal n.
            if self.oom_reductions != reductions
                || self.calibration_calls != calls + 1
                || self.limits.max_items != n
            {
                break;
            }
            let rate = n as f64 / start.elapsed().as_secs_f64();
            if rate < last_rate * 1.03
                && let Some(previous) = best
            {
                self.limits = previous;
                break;
            }
            best = Some(self.limits.clone());
            last_rate = rate;
            let free = self.probe.free_memory(gpu.index).unwrap_or(0);
            if config.batch_size.is_some()
                || n >= cap
                || free < (gpu.total as f64 * 0.25) as u64
                || calibration_start.elapsed() >= std::time::Duration::from_secs(5)
            {
                break;
            }
            n = (n * 2).min(cap);
        }
        save_profile(path, &self.limits)?;
        Ok(())
    }
}
fn create_runtime(
    factory: &mut dyn RuntimeFactory,
    gpu: Option<&Gpu>,
    events: &mut Events,
) -> Result<Box<dyn Runtime>> {
    let _span = crate::metrics::Span::new("runtime_creation");
    let runtime = factory.create(gpu, events)?;
    ensure!(
        runtime.provider() == if gpu.is_some() { "cuda" } else { "cpu" },
        "execution provider activation mismatch"
    );
    Ok(runtime)
}
fn save_profile(path: PathBuf, limits: &Limits) -> Result<()> {
    let parent = path.parent().unwrap();
    std::fs::create_dir_all(parent)?;
    let temp = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer(temp.as_file(), limits)?;
    temp.as_file().sync_all()?;
    temp.persist(path).map_err(|e| e.error)?;
    Ok(())
}

/// Injected in pipeline tests; production always uses the pinned tokenizer/model.
pub trait Engine {
    /// Optional independent tokenizer for bounded preparation overlap. Injected
    /// engines retain the serial path unless they explicitly provide one.
    fn chunker(&self) -> Option<Box<dyn passage::Chunker>> {
        None
    }
    fn chunk(&mut self, text: &str, events: &mut Events) -> Result<Vec<Chunk>>;
    fn embed(&mut self, inputs: &[Vec<i64>], events: &mut Events) -> Result<Vec<Vec<f32>>>;
    fn metrics(&self) -> Value {
        json!({})
    }
}
pub struct LocalEngine {
    config: Config,
    tokenizer: Option<std::sync::Arc<Tokenizer>>,
    tokenizer_source: std::sync::Arc<passage::TokenizerSource>,
    scheduler: Option<Scheduler>,
}
impl LocalEngine {
    pub fn new(config: Config) -> Self {
        Self {
            tokenizer_source: std::sync::Arc::new(passage::TokenizerSource::new(
                config.cache.clone(),
            )),
            config,
            tokenizer: None,
            scheduler: None,
        }
    }
    fn init(&mut self, events: &mut Events) -> Result<()> {
        self.init_policy(events, false)
    }
    fn init_policy(&mut self, events: &mut Events, interactive: bool) -> Result<()> {
        if self.scheduler.is_some() {
            return Ok(());
        }
        let cpu_path = provision_cpu(&self.config, events, download::ensure_artifact)?;
        let mut config = self.config.clone();
        if interactive {
            config.chunk_size = query::MAX_QUERY_TOKENS;
        }
        let factory = Box::new(OnnxFactory {
            config: config.clone(),
            cpu_path,
        });
        self.scheduler = Some(Scheduler::initialize_policy(
            &config,
            factory,
            Box::new(device::NvmlProbe::default()),
            events,
            interactive,
        )?);
        Ok(())
    }
    fn tokenizer(&mut self, events: &mut Events) -> Result<&Tokenizer> {
        if self.tokenizer.is_none() {
            self.tokenizer = Some(self.tokenizer_source.get(events)?);
        }
        Ok(self.tokenizer.as_deref().unwrap())
    }
}
impl Engine for LocalEngine {
    fn chunker(&self) -> Option<Box<dyn passage::Chunker>> {
        // Opt-in until sustained-ingestion and memory gates pass. Keep extreme
        // window/overlap configurations on the original materialization path.
        (std::env::var("REE_PIPELINE_OVERLAP").is_ok_and(|v| v == "1")
            // INT8 regrouping failed the equivalence gate. Auto can fall back
            // to INT8, so only strict CUDA may enable this experiment.
            && passage::supports_overlap(&self.config.device, self.config.chunk_size, self.config.overlap))
            .then(|| {
                Box::new(passage::PassageChunker {
                    source: self.tokenizer_source.clone(),
                    size: self.config.chunk_size,
                    overlap: self.config.overlap,
                }) as Box<dyn passage::Chunker>
            })
    }
    fn chunk(&mut self, text: &str, events: &mut Events) -> Result<Vec<Chunk>> {
        if text.trim().is_empty() {
            return Ok(vec![]);
        }
        let (size, overlap) = (self.config.chunk_size, self.config.overlap);
        let tokenizer = self.tokenizer(events)?;
        let _span = crate::metrics::Span::new("document_tokenize");
        chunk::windows(tokenizer, text, size, overlap)
    }
    fn embed(&mut self, inputs: &[Vec<i64>], events: &mut Events) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(vec![]);
        }
        self.init(events).map_err(|e| {
            crate::error::AppError::new(3, format!("model/runtime initialization: {e:#}"))
        })?;
        self.scheduler.as_mut().unwrap().embed(inputs, events)
    }
    fn metrics(&self) -> Value {
        self.scheduler.as_ref().map_or(json!({}),|s|json!({"provider":s.provider(),"inference_ms":s.inference_ms,"workload_ns":s.workload_ns,"calibration_ns":s.calibration_ns,"workload_calls":s.workload_calls,"calibration_calls":s.calibration_calls,"attempted_real_tokens":s.real_tokens,"attempted_padded_tokens":s.padded_tokens,"oom_reductions":s.oom_reductions,"provider_fallbacks":s.fallbacks,"final_batch_limits":s.limits,"batch_policy":"bounded-calibration-v2"}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Mock {
        gpu: bool,
        fail: bool,
    }
    impl Runtime for Mock {
        fn provider(&self) -> &'static str {
            if self.gpu { "cuda" } else { "cpu" }
        }
        fn infer(&mut self, i: &[Vec<i64>]) -> Result<Vec<Vec<f32>>> {
            if self.gpu && (self.fail || i.len() > 1) {
                anyhow::bail!("CUDA out of memory");
            }
            let mut v = vec![0.; 768];
            v[0] = 1.;
            Ok(vec![v; i.len()])
        }
    }
    struct Factory {
        fail: bool,
    }
    impl RuntimeFactory for Factory {
        fn create(&mut self, g: Option<&Gpu>, _: &mut Events) -> Result<Box<dyn Runtime>> {
            Ok(Box::new(Mock {
                gpu: g.is_some(),
                fail: self.fail,
            }))
        }
    }
    fn scheduler(auto: bool, fail: bool) -> Scheduler {
        Scheduler {
            runtime: Some(Box::new(Mock { gpu: true, fail })),
            factory: Box::new(Factory { fail }),
            probe: Box::new(device::NvmlProbe::default()),
            gpu: Some(Gpu {
                index: 999,
                name: "mock".into(),
                uuid: "mock".into(),
                free: 8 << 30,
                total: 8 << 30,
                driver: "mock".into(),
                compute_major: 8,
                compute_minor: 9,
            }),
            auto,
            limits: Limits {
                max_items: 4,
                max_padded_tokens: 2048,
            },
            workload_ns: 0,
            calibration_ns: 0,
            workload_calls: 0,
            calibration_calls: 0,
            real_tokens: 0,
            padded_tokens: 0,
            last_successful_items: 0,
            oom_reductions: 0,
            fallbacks: 0,
            inference_ms: 0,
        }
    }
    #[test]
    fn reduced_retry_keeps_order() {
        let mut s = scheduler(true, false);
        let out = s
            .embed(
                &vec![vec![0, 42, 2]; 5],
                &mut Events::new(true, false, false),
            )
            .unwrap();
        assert_eq!(out.len(), 5);
        assert!(s.oom_reductions > 0);
        assert_eq!(s.fallbacks, 0);
    }
    #[test]
    fn one_item_oom_falls_back() {
        let mut s = scheduler(true, true);
        assert!(
            s.embed(&[vec![0, 2]], &mut Events::new(true, false, false))
                .is_ok()
        );
        assert_eq!(s.fallbacks, 1);
    }
    #[test]
    fn explicit_cuda_never_falls_back() {
        let mut s = scheduler(false, true);
        assert!(
            s.embed(&[vec![0, 2]], &mut Events::new(true, false, false))
                .is_err()
        );
    }
    #[test]
    fn cpu_provisioning_respects_strict_cuda_and_auto_rescue() {
        let mut config = Config::load(&crate::cli::Options::default()).unwrap();
        for device in ["cpu", "auto", "cuda", "cuda:0"] {
            config.device = device.into();
            let mut calls = 0;
            let result = provision_cpu(
                &config,
                &mut Events::new(true, false, false),
                |_, artifact, _| {
                    calls += 1;
                    assert_eq!(artifact.sha256, download::CPU.sha256);
                    anyhow::bail!("missing or corrupt rescue artifact")
                },
            );
            if matches!(device, "cpu" | "auto") {
                assert!(result.is_err());
                assert_eq!(calls, 1);
            } else {
                assert!(result.unwrap().is_none());
                assert_eq!(calls, 0);
            }
        }
    }
    #[test]
    fn invalid_vectors_rejected() {
        assert!(validate_vector(&[0.; 768]).is_err());
        assert!(normalize(&mut [f32::NAN; 768]).is_err());
    }
}
