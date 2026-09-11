//! Historical offline behavior replay, not a desired-behavior regression test.
//! No model/session creation; runtime and device probes are injected.
use anyhow::Result;
use ree::{
    cli::Options,
    config::Config,
    events::Events,
    model::{
        Runtime, RuntimeFactory, Scheduler,
        device::{DeviceProbe, Gpu},
    },
};
use serde_json::json;
use std::sync::{Arc, Mutex};

type Calls = Arc<Mutex<Vec<(usize, usize)>>>;
struct FakeRuntime {
    calls: Calls,
    cuda: bool,
}
impl Runtime for FakeRuntime {
    fn provider(&self) -> &'static str {
        if self.cuda { "cuda" } else { "cpu" }
    }
    fn infer(&mut self, inputs: &[Vec<i64>]) -> Result<Vec<Vec<f32>>> {
        self.calls
            .lock()
            .unwrap()
            .push((inputs.len(), inputs.iter().map(Vec::len).max().unwrap_or(0)));
        let mut vector = vec![0.; 768];
        vector[0] = 1.;
        Ok(vec![vector; inputs.len()])
    }
}
struct Factory(Calls);
impl RuntimeFactory for Factory {
    fn create(&mut self, gpu: Option<&Gpu>, _: &mut Events) -> Result<Box<dyn Runtime>> {
        Ok(Box::new(FakeRuntime {
            calls: self.0.clone(),
            cuda: gpu.is_some(),
        }))
    }
}
struct Probe;
fn gpu() -> Gpu {
    Gpu {
        index: 0,
        name: "injected-audit-gpu".into(),
        uuid: "injected-audit-uuid".into(),
        driver: "injected-driver".into(),
        total: 8 << 30,
        free: 7 << 30,
        compute_major: 8,
        compute_minor: 9,
    }
}
impl DeviceProbe for Probe {
    fn devices(&mut self) -> Result<Vec<Gpu>> {
        Ok(vec![gpu()])
    }
    fn free_memory(&mut self, _: u32) -> Result<u64> {
        Ok(7 << 30)
    }
}
fn config(device: &str, batch: Option<usize>) -> Result<(tempfile::TempDir, Config)> {
    let temp = tempfile::tempdir()?;
    let path = temp.path().join("config.toml");
    std::fs::write(&path, "")?;
    let mut config = Config::load(&Options {
        config: Some(path),
        device: Some(device.into()),
        batch_size: batch,
        ..Default::default()
    })?;
    config.cache = temp.path().join("cache");
    config.db = temp.path().join("unused.db");
    assert_eq!(config.chunk_size, 512);
    Ok((temp, config))
}
fn start(config: &Config, calls: Calls, events: &mut Events) -> Result<Scheduler> {
    Scheduler::initialize(config, Box::new(Factory(calls)), Box::new(Probe), events)
}
fn cached(config: &Config, items: usize, tokens: usize) -> Result<()> {
    let gpu = gpu();
    let key = ree::util::hash(
        format!(
            "{}:fp16:{}:{}:{}:{}",
            ree::model::MODEL_KEY,
            gpu.uuid,
            gpu.driver,
            ree::model::RUNTIME_VERSION,
            config.chunk_size
        )
        .as_bytes(),
    );
    let directory = config.cache.join("profiles");
    std::fs::create_dir_all(&directory)?;
    std::fs::write(
        directory.join(format!("{key}.json")),
        json!({"max_items":items,"max_padded_tokens":tokens}).to_string(),
    )?;
    Ok(())
}
fn main() -> Result<()> {
    let mut events = Events::new(true, false, false);
    for batch in [None, Some(8)] {
        let (_temp, config) = config("cpu", batch)?;
        let calls = Calls::default();
        let mut scheduler = start(&config, calls.clone(), &mut events)?;
        let before = scheduler.limits.clone();
        for _ in 0..32 {
            scheduler.embed(&[vec![0, 42, 2]], &mut events)?;
        }
        assert!(calls.lock().unwrap().iter().all(|&(n, _)| n == 1));
        assert_eq!(
            scheduler.limits.max_items,
            if batch.is_none() { 9 } else { 8 }
        );
        println!(
            "{}",
            json!({"case":"cpu_underfilled_growth","explicit_batch":batch,
            "single_item_calls":32,"before":before,"after":scheduler.limits})
        );
    }
    {
        let (_temp, config) = config("cuda", None)?;
        cached(&config, 64, 32768)?;
        let calls = Calls::default();
        let mut scheduler = start(&config, calls.clone(), &mut events)?;
        let calibration = calls.lock().unwrap().clone();
        assert_eq!(calibration, vec![(16, 512)]);
        assert_eq!(scheduler.limits.max_items, 64);
        scheduler.embed(&vec![vec![0; 512]; 64], &mut events)?;
        assert_eq!(calls.lock().unwrap().last(), Some(&(64, 512)));
        println!(
            "{}",
            json!({"case":"cached_64_validated_at_16","scheduler_calibration_calls":calibration,
            "retained_limits":scheduler.limits,"first_workload_call":[64,512]})
        );
    }
    {
        let (_temp, config) = config("cuda", None)?;
        cached(&config, 16, 512)?;
        let calls = Calls::default();
        let scheduler = start(&config, calls.clone(), &mut events)?;
        let calibration = calls.lock().unwrap().clone();
        assert_eq!(calibration, vec![(1, 512); 16]);
        println!(
            "{}",
            json!({"case":"nominal_16_probe_is_16_singletons","nominal_items":16,
            "scheduler_calibration_calls":calibration,"retained_limits":scheduler.limits})
        );
    }
    Ok(())
}
