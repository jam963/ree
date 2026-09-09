pub mod batch;
pub mod device;
pub mod download;
pub mod onnx;

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
struct OnnxFactory {
    config: Config,
    cpu_path: PathBuf,
    cuda_path: Option<PathBuf>,
}
impl RuntimeFactory for OnnxFactory {
    fn create(&mut self, gpu: Option<&Gpu>, events: &mut Events) -> Result<Box<dyn Runtime>> {
        if let Some(gpu) = gpu {
            onnx::Onnx::check_cuda(gpu, device::budget(gpu, self.config.gpu_memory_fraction))?;
        }
        let path = if gpu.is_some() {
            if self.cuda_path.is_none() {
                self.cuda_path = Some(download::ensure_artifact(
                    &self.config.cache,
                    download::CUDA,
                    events,
                )?);
            }
            self.cuda_path.as_ref().unwrap().clone()
        } else {
            self.cpu_path.clone()
        };
        Ok(Box::new(onnx::Onnx::load(
            &path,
            gpu.map(|g| (g, device::budget(g, self.config.gpu_memory_fraction))),
            self.config.chunk_size,
        )?))
    }
}

pub struct Scheduler {
    runtime: Option<Box<dyn Runtime>>,
    factory: Box<dyn RuntimeFactory>,
    probe: Box<dyn device::DeviceProbe>,
    gpu: Option<Gpu>,
    auto: bool,
    pub limits: Limits,
    growth: bool,
    stable: usize,
    pub oom_reductions: usize,
    pub fallbacks: usize,
    pub inference_ms: u128,
}
impl Scheduler {
    pub fn initialize(
        config: &Config,
        mut factory: Box<dyn RuntimeFactory>,
        mut probe: Box<dyn device::DeviceProbe>,
        events: &mut Events,
    ) -> Result<Self> {
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
        let runtime = create_runtime(factory.as_mut(), gpu.as_ref(), events);
        let items = config
            .batch_size
            .unwrap_or(if gpu.is_some() { 2 } else { 8 });
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
            growth: config.batch_size.is_none(),
            stable: 0,
            oom_reductions: 0,
            fallbacks,
            inference_ms: 0,
        };
        match runtime {
            Ok(r) => s.runtime = Some(r),
            Err(e) if s.gpu.is_some() => s.fallback(&format!("{e:#}"), events)?,
            Err(e) => return Err(e),
        }
        s.calibrate(config, events)?;
        events.emit(json!({"type":"runtime","provider":s.runtime.as_ref().unwrap().provider(),"device":s.gpu,"precision":if s.gpu.is_some(){"fp16"}else{"int8"},"runtime":RUNTIME_VERSION,"batch_limits":s.limits}))?;
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
        self.growth = false;
        self.fallbacks += 1;
        Ok(())
    }
    pub fn embed(&mut self, inputs: &[Vec<i64>], events: &mut Events) -> Result<Vec<Vec<f32>>> {
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
                self.stable = 0;
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
            self.inference_ms += start.elapsed().as_millis();
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
                    self.stable += 1;
                    if self.growth && self.stable >= 32 {
                        self.limits.max_items = (self.limits.max_items + 1).min(128);
                        self.limits.max_padded_tokens = (self.limits.max_padded_tokens
                            + self.limits.max_padded_tokens / 16)
                            .min(65536);
                        self.stable = 0;
                    }
                }
                Err(e) if self.gpu.is_some() => {
                    self.stable = 0;
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
        let Some(gpu) = self.gpu.clone() else {
            return Ok(());
        };
        let key = hash(
            format!(
                "{MODEL_KEY}:fp16:{}:{}:{RUNTIME_VERSION}:{}",
                gpu.uuid, gpu.driver, config.chunk_size
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
        // Probe real padded workloads without publishing their output. Cached limits
        // are hints, not guarantees: embed() applies the same OOM recovery here.
        let mut last_rate = 0.0;
        let mut best = self.limits.clone();
        let cap = config.batch_size.unwrap_or(16).min(64);
        let mut n = self.limits.max_items.min(cap).max(1);
        loop {
            let start = Instant::now();
            self.embed(&vec![input.clone(); n], events)?;
            if self.gpu.is_none() {
                return Ok(());
            }
            let rate = n as f64 / start.elapsed().as_secs_f64();
            if rate < last_rate * 1.03 && last_rate > 0.0 {
                self.limits = best;
                break;
            }
            best = self.limits.clone();
            last_rate = rate;
            let free = self.probe.free_memory(gpu.index).unwrap_or(0);
            if config.batch_size.is_some()
                || n >= cap
                || free < (gpu.total as f64 * 0.25) as u64
                || self.oom_reductions > 0
            {
                break;
            }
            n = (n * 2).min(cap);
            self.limits.max_items = n;
            self.limits.max_padded_tokens = n * config.chunk_size;
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
    fn chunk(&mut self, text: &str, events: &mut Events) -> Result<Vec<Chunk>>;
    fn embed(&mut self, inputs: &[Vec<i64>], events: &mut Events) -> Result<Vec<Vec<f32>>>;
    fn metrics(&self) -> Value {
        json!({})
    }
}
pub struct LocalEngine {
    config: Config,
    tokenizer: Option<Tokenizer>,
    scheduler: Option<Scheduler>,
}
impl LocalEngine {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            tokenizer: None,
            scheduler: None,
        }
    }
    fn init(&mut self, events: &mut Events) -> Result<()> {
        if self.scheduler.is_some() {
            return Ok(());
        }
        // Always provision the CPU rescue artifact before trying CUDA.
        let cpu_path = download::ensure_artifact(&self.config.cache, download::CPU, events)?;
        let factory = Box::new(OnnxFactory {
            config: self.config.clone(),
            cpu_path,
            cuda_path: None,
        });
        self.scheduler = Some(Scheduler::initialize(
            &self.config,
            factory,
            Box::new(device::NvmlProbe::default()),
            events,
        )?);
        Ok(())
    }
}
impl Engine for LocalEngine {
    fn chunk(&mut self, text: &str, events: &mut Events) -> Result<Vec<Chunk>> {
        if text.trim().is_empty() {
            return Ok(vec![]);
        }
        if self.tokenizer.is_none() {
            let path = download::ensure_artifact(&self.config.cache, download::TOKENIZER, events)
                .map_err(|e| {
                crate::error::AppError::new(3, format!("tokenizer initialization: {e:#}"))
            })?;
            let mut tokenizer = Tokenizer::from_file(path).map_err(|e| anyhow!(e))?;
            tokenizer.with_padding(None);
            tokenizer.with_truncation(None).map_err(|e| anyhow!(e))?;
            ensure!(
                tokenizer.token_to_id("<s>") == Some(0) && tokenizer.token_to_id("</s>") == Some(2),
                "unexpected tokenizer special tokens"
            );
            self.tokenizer = Some(tokenizer);
        }
        chunk::windows(
            self.tokenizer.as_ref().unwrap(),
            text,
            self.config.chunk_size,
            self.config.overlap,
        )
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
        self.scheduler.as_ref().map_or(json!({}),|s|json!({"inference_ms":s.inference_ms,"oom_reductions":s.oom_reductions,"provider_fallbacks":s.fallbacks,"final_batch_limits":s.limits}))
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
            growth: false,
            stable: 0,
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
    fn invalid_vectors_rejected() {
        assert!(validate_vector(&[0.; 768]).is_err());
        assert!(normalize(&mut [f32::NAN; 768]).is_err());
    }
}
