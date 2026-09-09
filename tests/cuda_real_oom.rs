//! Real arena-limited OOM tests: never reset the GPU or exhaust display VRAM.
use anyhow::{Result, ensure};
use ree::{
    cli::Options,
    config::{Config, xdg},
    events::Events,
    model::{
        self, Runtime, RuntimeFactory, Scheduler,
        device::{self, Gpu, NvmlProbe},
        download,
        onnx::Onnx,
    },
};
use std::path::PathBuf;
struct LimitedFactory {
    cpu: PathBuf,
    cuda: PathBuf,
    limit: u64,
}
impl RuntimeFactory for LimitedFactory {
    fn create(&mut self, gpu: Option<&Gpu>, _: &mut Events) -> Result<Box<dyn Runtime>> {
        Ok(Box::new(Onnx::load(
            if gpu.is_some() { &self.cuda } else { &self.cpu },
            gpu.map(|g| (g, self.limit)),
            512,
        )?))
    }
}
#[test]
#[ignore = "real CUDA 13: local RTX 4070 only; enforces small ONNX arena, no GPU reset"]
fn actual_cuda_oom_retries_and_initialization_fallback() -> Result<()> {
    ensure!(
        std::env::var_os("CI").is_none(),
        "never run real GPU qualification in CI"
    );
    ensure!(
        std::fs::read_to_string("/proc/cpuinfo")?.contains("AMD Ryzen 9 8945HS"),
        "development CPU required"
    );
    let devices = device::discover()?;
    ensure!(
        devices.iter().any(|g| g.name.contains("RTX 4070 Laptop")),
        "development GPU required"
    );
    let gpu = device::select(&devices, "cuda", 0.85)?;
    Onnx::check_cuda(&gpu, device::budget(&gpu, 0.85))?;
    let cache = xdg("XDG_CACHE_HOME", ".cache")?;
    let mut events = Events::with_writer(std::io::stdout(), false, false);
    let cpu = download::ensure_artifact(&cache, download::CPU, &mut events)?;
    let cuda = download::ensure_artifact(&cache, download::CUDA, &mut events)?;
    let temp = tempfile::tempdir()?;
    let config_path = temp.path().join("config.toml");
    std::fs::write(&config_path, "")?;
    let mut config = Config::load(&Options {
        config: Some(config_path),
        device: Some(format!("cuda:{}", gpu.index)),
        batch_size: Some(1),
        ..Default::default()
    })?;
    config.cache = temp.path().join("cache");
    let factory = |limit| {
        Box::new(LimitedFactory {
            cpu: cpu.clone(),
            cuda: cuda.clone(),
            limit,
        })
    };
    let mut scheduler = Scheduler::initialize(
        &config,
        factory(1280 << 20),
        Box::new(NvmlProbe::default()),
        &mut events,
    )?;
    assert_eq!(scheduler.provider(), Some("cuda"));
    // Initialization uses one item. The first real large batch should exceed the
    // 1.25-GiB arena without approaching this GPU's 8-GiB physical memory limit.
    scheduler.limits.max_items = 256;
    scheduler.limits.max_padded_tokens = 256 * 512;
    let mut input = vec![42; 512];
    input[0] = 0;
    input[511] = 2;
    let vectors = scheduler.embed(&vec![input; 256], &mut events)?;
    assert_eq!(vectors.len(), 256);
    for vector in &vectors {
        model::validate_vector(vector)?;
    }
    assert!(
        scheduler.oom_reductions > 0,
        "expected a real arena OOM and smaller-batch retry"
    );
    assert_eq!(scheduler.fallbacks, 0);
    assert_eq!(scheduler.provider(), Some("cuda"));
    println!(
        "Real CUDA OOM reductions: {}; final batch items: {}",
        scheduler.oom_reductions, scheduler.limits.max_items
    );
    drop(scheduler);
    // A deliberately impossible arena tests actual failed session initialization,
    // both strict CUDA rejection and auto CPU continuation, using real models.
    assert!(
        Scheduler::initialize(
            &config,
            factory(1024),
            Box::new(NvmlProbe::default()),
            &mut events
        )
        .is_err()
    );
    config.device = "auto".into();
    let mut scheduler = Scheduler::initialize(
        &config,
        factory(1024),
        Box::new(NvmlProbe::default()),
        &mut events,
    )?;
    assert_eq!(scheduler.provider(), Some("cpu"));
    assert_eq!(scheduler.fallbacks, 1);
    let vectors = scheduler.embed(&[vec![0, 42, 2]], &mut events)?;
    model::validate_vector(&vectors[0])?;
    events.flush()?;
    Ok(())
}
