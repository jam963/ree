//! Complete CUDA control-flow coverage without any NVIDIA libraries in CI.
use anyhow::{Result, bail};
use ree::{
    cli::Options,
    config::Config,
    events::Events,
    model::{
        Runtime, RuntimeFactory, Scheduler,
        device::{DeviceProbe, Gpu},
    },
};
use serde_json::Value;
use std::{
    collections::VecDeque,
    io::Write,
    sync::{Arc, Mutex},
};

#[derive(Default)]
struct State {
    creates: Vec<Option<u32>>,
    calls: Vec<(bool, Vec<i64>)>,
    gpu_load_error: Option<&'static str>,
    cpu_load_error: Option<&'static str>,
    gpu_errors: VecDeque<Option<&'static str>>,
    max_gpu_items: Option<usize>,
    misreport: bool,
    invalid_vector: bool,
}
struct Factory(Arc<Mutex<State>>);
struct FakeRuntime {
    state: Arc<Mutex<State>>,
    cuda: bool,
}
impl RuntimeFactory for Factory {
    fn create(&mut self, gpu: Option<&Gpu>, _: &mut Events) -> Result<Box<dyn Runtime>> {
        let mut s = self.0.lock().unwrap();
        s.creates.push(gpu.map(|g| g.index));
        if let Some(error) = if gpu.is_some() {
            s.gpu_load_error
        } else {
            s.cpu_load_error
        } {
            bail!(error);
        }
        Ok(Box::new(FakeRuntime {
            state: self.0.clone(),
            cuda: gpu.is_some() && !s.misreport,
        }))
    }
}
impl Runtime for FakeRuntime {
    fn provider(&self) -> &'static str {
        if self.cuda { "cuda" } else { "cpu" }
    }
    fn infer(&mut self, inputs: &[Vec<i64>]) -> Result<Vec<Vec<f32>>> {
        let mut s = self.state.lock().unwrap();
        s.calls
            .push((self.cuda, inputs.iter().map(|i| i[1]).collect()));
        if self.cuda {
            if let Some(Some(error)) = s.gpu_errors.pop_front() {
                bail!(error);
            }
            if s.max_gpu_items.is_some_and(|n| inputs.len() > n) {
                bail!("CUDA out of memory");
            }
        }
        Ok(inputs
            .iter()
            .map(|i| {
                let mut v = vec![0.; 768];
                v[i[1] as usize % 768] = if s.invalid_vector { f32::NAN } else { 1. };
                v
            })
            .collect())
    }
}
struct Probe {
    devices: Vec<Gpu>,
    error: Option<&'static str>,
    free: Arc<Mutex<VecDeque<u64>>>,
}
impl DeviceProbe for Probe {
    fn devices(&mut self) -> Result<Vec<Gpu>> {
        if let Some(e) = self.error {
            bail!(e);
        }
        Ok(self.devices.clone())
    }
    fn free_memory(&mut self, _: u32) -> Result<u64> {
        Ok(self.free.lock().unwrap().pop_front().unwrap_or(7 << 30))
    }
}
#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);
impl Write for Capture {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn gpu(index: u32, free: u64) -> Gpu {
    Gpu {
        index,
        name: format!("mock-{index}"),
        uuid: format!("mock-{index}"),
        total: 8 << 30,
        free,
        driver: "mock-driver".into(),
        compute_major: 8,
        compute_minor: 9,
    }
}
struct Fixture {
    _temp: tempfile::TempDir,
    config: Config,
    state: Arc<Mutex<State>>,
    capture: Capture,
    events: Events,
    free: Arc<Mutex<VecDeque<u64>>>,
}
impl Fixture {
    fn new(device: &str) -> Self {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("config.toml");
        std::fs::write(&path, "").unwrap();
        let mut config = Config::load(&Options {
            config: Some(path),
            device: Some(device.into()),
            batch_size: Some(4),
            ..Default::default()
        })
        .unwrap();
        config.cache = temp.path().join("cache");
        config.db = temp.path().join("ree.db");
        config.chunk_size = 8;
        config.overlap = 1;
        let capture = Capture::default();
        let events = Events::with_writer(capture.clone(), false, false);
        Self {
            _temp: temp,
            config,
            state: Arc::default(),
            capture,
            events,
            free: Arc::default(),
        }
    }
    fn start(&mut self, devices: Vec<Gpu>, error: Option<&'static str>) -> Result<Scheduler> {
        Scheduler::initialize(
            &self.config,
            Box::new(Factory(self.state.clone())),
            Box::new(Probe {
                devices,
                error,
                free: self.free.clone(),
            }),
            &mut self.events,
        )
    }
    fn events(&mut self) -> Vec<Value> {
        self.events.flush().unwrap();
        String::from_utf8(self.capture.0.lock().unwrap().clone())
            .unwrap()
            .lines()
            .map(|s| serde_json::from_str(s).unwrap())
            .collect()
    }
}
fn inputs(n: usize) -> Vec<Vec<i64>> {
    (1..=n)
        .map(|i| {
            let mut v = vec![0, i as i64];
            v.extend(vec![42; i % 5]);
            v.push(2);
            v
        })
        .collect()
}
fn assert_order(v: &[Vec<f32>], n: usize) {
    assert_eq!(v.len(), n);
    for (i, v) in v.iter().enumerate() {
        assert_eq!(v[i + 1], 1.);
    }
}

#[test]
fn no_driver_and_no_devices_use_cpu() {
    for error in [Some("NVML library unavailable"), None] {
        let mut f = Fixture::new("auto");
        let mut s = f.start(vec![], error).unwrap();
        assert_eq!(s.provider(), Some("cpu"));
        assert_eq!(s.fallbacks, 1);
        assert_eq!(f.state.lock().unwrap().creates, vec![None]);
        assert_order(&s.embed(&inputs(5), &mut f.events).unwrap(), 5);
        assert!(f.events().iter().any(|e| e["type"] == "device_fallback"));
    }
}
#[test]
fn cpu_request_does_not_require_nvml() {
    let mut f = Fixture::new("cpu");
    let s = f.start(vec![], Some("must not need NVML")).unwrap();
    assert_eq!(s.provider(), Some("cpu"));
    assert_eq!(s.fallbacks, 0);
}
#[test]
fn most_free_gpu_is_selected_and_explicit_index_is_respected() {
    for (request, index) in [("auto", 1), ("cuda:0", 0)] {
        let mut f = Fixture::new(request);
        let s = f
            .start(vec![gpu(0, 3 << 30), gpu(1, 7 << 30)], None)
            .unwrap();
        assert_eq!(s.provider(), Some("cuda"));
        assert_eq!(f.state.lock().unwrap().creates, vec![Some(index)]);
    }
}
#[test]
fn init_provider_model_and_warmup_errors_fall_back() {
    for error in [
        "CUDA execution provider unavailable",
        "FP16 invalid graph",
        "warm-up CUDA out of memory",
        "cuDNN incompatible",
        "warm-up profile contains no CUDA nodes",
    ] {
        let mut f = Fixture::new("auto");
        f.state.lock().unwrap().gpu_load_error = Some(error);
        let s = f.start(vec![gpu(0, 7 << 30)], None).unwrap();
        assert_eq!(s.provider(), Some("cpu"));
        assert_eq!(s.fallbacks, 1);
        assert_eq!(f.state.lock().unwrap().creates, vec![Some(0), None]);
    }
}
#[test]
fn silent_provider_fallback_is_rejected() {
    let mut f = Fixture::new("auto");
    f.state.lock().unwrap().misreport = true;
    let s = f.start(vec![gpu(0, 7 << 30)], None).unwrap();
    assert_eq!(s.provider(), Some("cpu"));
    assert_eq!(s.fallbacks, 1);
}
#[test]
fn explicit_cuda_initialization_failure_never_creates_cpu() {
    let mut f = Fixture::new("cuda");
    f.state.lock().unwrap().gpu_load_error = Some("FP16 model load failed");
    assert!(f.start(vec![gpu(0, 7 << 30)], None).is_err());
    assert_eq!(f.state.lock().unwrap().creates, vec![Some(0)]);
    let mut f = Fixture::new("cuda:99");
    assert!(f.start(vec![gpu(0, 7 << 30)], None).is_err());
    assert!(f.state.lock().unwrap().creates.is_empty());
}
#[test]
fn calibration_oom_reduces_recreates_and_recovers() {
    let mut f = Fixture::new("auto");
    f.state.lock().unwrap().max_gpu_items = Some(1);
    let mut s = f.start(vec![gpu(0, 7 << 30)], None).unwrap();
    assert_eq!(s.oom_reductions, 2);
    assert_eq!(s.fallbacks, 0);
    assert_eq!(s.limits.max_items, 1);
    assert_order(&s.embed(&inputs(7), &mut f.events).unwrap(), 7);
}
#[test]
fn midrun_oom_preserves_order_and_bounds_user_batch() {
    let mut f = Fixture::new("auto");
    let mut s = f.start(vec![gpu(0, 7 << 30)], None).unwrap();
    f.state.lock().unwrap().max_gpu_items = Some(2);
    assert_order(&s.embed(&inputs(19), &mut f.events).unwrap(), 19);
    assert_eq!(s.oom_reductions, 1);
    assert_eq!(s.fallbacks, 0);
    assert_eq!(s.limits.max_items, 2);
    s.embed(&inputs(140), &mut f.events).unwrap();
    assert_eq!(s.limits.max_items, 2);
    assert!(f.events().iter().any(|e| e["type"] == "batch_reduced"));
}
#[test]
fn device_loss_and_unknown_errors_resume_at_first_uncommitted_chunk() {
    for error in [
        "device lost/reset",
        "unknown ONNX Runtime failure",
        "CUDA driver unavailable",
    ] {
        let mut f = Fixture::new("auto");
        let mut s = f.start(vec![gpu(0, 7 << 30)], None).unwrap();
        {
            let mut state = f.state.lock().unwrap();
            state.calls.clear();
            state.gpu_errors = VecDeque::from([None, Some(error)]);
        }
        let input = vec![
            vec![0, 1, 2],
            vec![0, 2, 2],
            vec![0, 3, 2],
            vec![0, 4, 2],
            vec![0, 5, 2],
            vec![0, 6, 2],
            vec![0, 7, 2],
            vec![0, 8, 2],
            vec![0, 9, 2],
        ];
        assert_order(&s.embed(&input, &mut f.events).unwrap(), 9);
        assert_eq!(s.provider(), Some("cpu"));
        assert_eq!(s.fallbacks, 1);
        let state = f.state.lock().unwrap();
        let cpu_ids: Vec<_> = state
            .calls
            .iter()
            .filter(|(cuda, _)| !cuda)
            .flat_map(|(_, ids)| ids.clone())
            .collect();
        assert_eq!(cpu_ids, vec![5, 6, 7, 8, 9]);
        assert_eq!(state.creates, vec![Some(0), None]);
    }
}
#[test]
fn one_item_oom_falls_back_but_explicit_cuda_fails() {
    for request in ["auto", "cuda"] {
        let mut f = Fixture::new(request);
        let mut s = f.start(vec![gpu(0, 7 << 30)], None).unwrap();
        f.state.lock().unwrap().max_gpu_items = Some(0);
        let result = s.embed(&[vec![0, 1, 2]], &mut f.events);
        if request == "auto" {
            assert_order(&result.unwrap(), 1);
            assert_eq!(s.fallbacks, 1);
        } else {
            assert_eq!(ree::error::exit_code(&result.unwrap_err()), 3);
            assert_eq!(f.state.lock().unwrap().creates, vec![Some(0)]);
        }
    }
}
#[test]
fn reduced_session_failure_falls_back_without_retry_loop() {
    let mut f = Fixture::new("auto");
    let mut s = f.start(vec![gpu(0, 7 << 30)], None).unwrap();
    {
        let mut state = f.state.lock().unwrap();
        state.max_gpu_items = Some(1);
        state.gpu_load_error = Some("device lost during recreation");
    }
    assert_order(&s.embed(&inputs(8), &mut f.events).unwrap(), 8);
    assert_eq!(s.fallbacks, 1);
    assert_eq!(s.oom_reductions, 1);
}
#[test]
fn cpu_fallback_initialization_failure_is_fatal() {
    let mut f = Fixture::new("auto");
    let mut s = f.start(vec![gpu(0, 7 << 30)], None).unwrap();
    {
        let mut state = f.state.lock().unwrap();
        state.gpu_errors.push_back(Some("device lost"));
        state.cpu_load_error = Some("CPU graph load failed");
    }
    let error = s.embed(&inputs(1), &mut f.events).unwrap_err();
    assert_eq!(ree::error::exit_code(&error), 3);
    assert_eq!(f.state.lock().unwrap().creates, vec![Some(0), None]);
}
#[test]
fn shrinking_vram_reduces_limits_and_invalid_vectors_are_not_returned() {
    let mut f = Fixture::new("auto");
    let mut s = f.start(vec![gpu(0, 7 << 30)], None).unwrap();
    f.free.lock().unwrap().push_back(256 << 20);
    assert_order(&s.embed(&inputs(5), &mut f.events).unwrap(), 5);
    assert_eq!(s.limits.max_items, 2);
    f.state.lock().unwrap().invalid_vector = true;
    assert!(s.embed(&inputs(1), &mut f.events).is_err());
}
#[test]
fn oom_and_device_loss_writes_are_idempotent() -> Result<()> {
    use ree::{chunk::Chunk, model::Engine, pipeline, storage::Database};
    struct Adapter(Scheduler);
    impl Engine for Adapter {
        fn chunk(&mut self, text: &str, _: &mut Events) -> Result<Vec<Chunk>> {
            Ok(text
                .lines()
                .enumerate()
                .map(|(i, line)| Chunk {
                    text: line.into(),
                    token_start: i,
                    token_end: i + 1,
                    byte_start: 0,
                    byte_end: line.len(),
                    input_ids: vec![0, (i + 1) as i64, 2],
                })
                .collect())
        }
        fn embed(&mut self, i: &[Vec<i64>], e: &mut Events) -> Result<Vec<Vec<f32>>> {
            self.0.embed(i, e)
        }
    }
    let mut f = Fixture::new("auto");
    let mut engine = Adapter(f.start(vec![gpu(0, 7 << 30)], None)?);
    let file = f._temp.path().join("document.txt");
    std::fs::write(
        &file,
        (1..=17).map(|i| format!("line {i}\n")).collect::<String>(),
    )?;
    let mut db = Database::open(&f.config.db, true)?;
    let input = [file.to_string_lossy().into()];
    f.state.lock().unwrap().max_gpu_items = Some(2);
    assert_eq!(
        pipeline::ingest(
            &mut db,
            &mut engine,
            &mut f.events,
            &f.config,
            &Options::default(),
            &input
        )?,
        0
    );
    let ids = |db: &Database| -> Vec<String> {
        db.conn
            .prepare("SELECT id FROM chunks ORDER BY id")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap()
    };
    let before = ids(&db);
    f.state
        .lock()
        .unwrap()
        .gpu_errors
        .push_back(Some("device lost"));
    assert_eq!(
        pipeline::ingest(
            &mut db,
            &mut engine,
            &mut f.events,
            &f.config,
            &Options {
                force: true,
                ..Default::default()
            },
            &input
        )?,
        0
    );
    assert_eq!(ids(&db), before);
    assert_eq!(db.status()?["embeddings"], 17);
    assert_eq!(db.status()?["chunks"], 17);
    assert_eq!(engine.0.provider(), Some("cpu"));
    Ok(())
}
