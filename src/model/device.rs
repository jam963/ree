use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gpu {
    pub index: u32,
    pub name: String,
    pub uuid: String,
    pub total: u64,
    pub free: u64,
    pub driver: String,
    pub compute_major: i32,
    pub compute_minor: i32,
}
/// Injected by scheduler tests; no driver or GPU is needed in ordinary CI.
pub trait DeviceProbe {
    fn devices(&mut self) -> Result<Vec<Gpu>>;
    fn free_memory(&mut self, index: u32) -> Result<u64>;
}
#[derive(Default)]
pub struct NvmlProbe {
    handle: Option<nvml_wrapper::Nvml>,
    visible: Vec<Gpu>,
}
impl NvmlProbe {
    fn handle(&mut self) -> Result<&nvml_wrapper::Nvml> {
        if self.handle.is_none() {
            self.handle = Some(nvml_wrapper::Nvml::init()?);
        }
        Ok(self.handle.as_ref().unwrap())
    }
}
impl DeviceProbe for NvmlProbe {
    fn devices(&mut self) -> Result<Vec<Gpu>> {
        self.visible = remap_visible(&discover_with(self.handle()?)?, &cuda_visible_uuids()?);
        Ok(self.visible.clone())
    }
    fn free_memory(&mut self, index: u32) -> Result<u64> {
        let uuid = self
            .visible
            .iter()
            .find(|g| g.index == index)
            .context("CUDA device is not visible")?
            .uuid
            .clone();
        Ok(self
            .handle()?
            .device_by_uuid(uuid.as_str())?
            .memory_info()?
            .free)
    }
}
pub fn discover() -> Result<Vec<Gpu>> {
    NvmlProbe::default().devices()
}
// CUDA ordinals need not equal NVML indices (CUDA_VISIBLE_DEVICES, UUID masks,
// PCI ordering). Join by driver UUID and sample NVML memory using that UUID.
fn remap_visible(physical: &[Gpu], uuids: &[String]) -> Vec<Gpu> {
    uuids
        .iter()
        .enumerate()
        .filter_map(|(index, uuid)| {
            physical
                .iter()
                .find(|g| g.uuid.eq_ignore_ascii_case(uuid))
                .map(|g| {
                    let mut g = g.clone();
                    g.index = index as u32;
                    g
                })
        })
        .collect()
}
fn cuda_visible_uuids() -> Result<Vec<String>> {
    use std::sync::OnceLock;
    static DRIVER: OnceLock<std::result::Result<libloading::Library, String>> = OnceLock::new();
    // Keep the driver loaded for process lifetime; use only the documented C
    // driver API and keep every typed function pointer within the library lifetime.
    let library = DRIVER
        .get_or_init(|| {
            unsafe { libloading::Library::new("libcuda.so.1") }.map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|e| anyhow::anyhow!("CUDA driver unavailable: {e}"))?;
    unsafe {
        let init = library.get::<unsafe extern "C" fn(u32) -> i32>(b"cuInit\0")?;
        let count_fn =
            library.get::<unsafe extern "C" fn(*mut i32) -> i32>(b"cuDeviceGetCount\0")?;
        let device_fn =
            library.get::<unsafe extern "C" fn(*mut i32, i32) -> i32>(b"cuDeviceGet\0")?;
        let uuid_fn =
            library.get::<unsafe extern "C" fn(*mut [u8; 16], i32) -> i32>(b"cuDeviceGetUuid\0")?;
        let status = init(0);
        ensure!(status == 0, "CUDA driver initialization failed ({status})");
        let mut count = 0;
        ensure!(count_fn(&mut count) == 0, "CUDA device enumeration failed");
        ensure!((0..=1024).contains(&count), "invalid CUDA device count");
        let mut uuids = Vec::new();
        for ordinal in 0..count {
            let mut device = 0;
            let mut bytes = [0; 16];
            ensure!(
                device_fn(&mut device, ordinal) == 0 && uuid_fn(&mut bytes, device) == 0,
                "CUDA UUID lookup failed for ordinal {ordinal}"
            );
            let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
            uuids.push(format!(
                "GPU-{}-{}-{}-{}-{}",
                &hex[..8],
                &hex[8..12],
                &hex[12..16],
                &hex[16..20],
                &hex[20..]
            ));
        }
        Ok(uuids)
    }
}
fn discover_with(nvml: &nvml_wrapper::Nvml) -> Result<Vec<Gpu>> {
    let driver = nvml.sys_driver_version()?;
    let mut devices = Vec::new();
    for index in 0..nvml.device_count()? {
        // One inaccessible GPU must not hide the remaining usable devices.
        let probe = || -> Result<Gpu> {
            let d = nvml.device_by_index(index)?;
            let m = d.memory_info()?;
            let c = d.cuda_compute_capability()?;
            Ok(Gpu {
                index,
                name: d.name()?,
                uuid: d.uuid()?,
                total: m.total,
                free: m.free,
                driver: driver.clone(),
                compute_major: c.major,
                compute_minor: c.minor,
            })
        };
        if let Ok(d) = probe() {
            devices.push(d);
        }
    }
    Ok(devices)
}
pub fn budget(gpu: &Gpu, fraction: f64) -> u64 {
    let headroom = (512 * 1024 * 1024).max((gpu.total as f64 * 0.15) as u64);
    gpu.free
        .saturating_sub(headroom)
        .min((gpu.total as f64 * fraction) as u64)
}
pub fn select(devices: &[Gpu], request: &str, fraction: f64) -> Result<Gpu> {
    let index = request
        .strip_prefix("cuda:")
        .and_then(|s| s.parse::<u32>().ok());
    devices
        .iter()
        .filter(|d| {
            index.is_none_or(|i| d.index == i)
                && (d.compute_major, d.compute_minor) >= (7, 5)
                && budget(d, fraction) > 1024 * 1024 * 1024
        })
        .max_by_key(|d| budget(d, fraction))
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no compatible NVIDIA GPU with sufficient free VRAM"))
}
pub fn free_memory(index: u32) -> Result<u64> {
    let mut probe = NvmlProbe::default();
    probe.devices()?;
    probe.free_memory(index)
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Failure {
    Oom,
    DeviceLost,
    Setup,
    Graph,
    Unknown,
}
pub fn classify(message: &str) -> Failure {
    let m = message.to_ascii_lowercase();
    if [
        "out of memory",
        "out_of_memory",
        "failed to allocate",
        "available memory of",
        "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed",
        "cuda failure 2:",
        "resource exhausted",
    ]
    .iter()
    .any(|s| m.contains(s))
    {
        Failure::Oom
    } else if [
        "device lost",
        "device unavailable",
        "device-side assert",
        "cuda failure 719",
        "cuda failure 700",
    ]
    .iter()
    .any(|s| m.contains(s))
    {
        Failure::DeviceLost
    } else if ["cudnn", "libcuda", "execution provider", "cuda driver"]
        .iter()
        .any(|s| m.contains(s))
    {
        Failure::Setup
    } else if ["invalid graph", "invalidgraph", "not implemented"]
        .iter()
        .any(|s| m.contains(s))
    {
        Failure::Graph
    } else {
        Failure::Unknown
    }
}
pub fn require_auto(auto: bool) -> Result<()> {
    if !auto {
        bail!("explicit CUDA requested; CPU fallback disabled");
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    fn gpu(index: u32, free: u64) -> Gpu {
        Gpu {
            index,
            name: "test".into(),
            uuid: index.to_string(),
            total: 8 << 30,
            free,
            driver: "mock".into(),
            compute_major: 8,
            compute_minor: 9,
        }
    }
    #[test]
    fn most_free_not_zero() {
        let ds = [gpu(0, 3 << 30), gpu(1, 7 << 30)];
        assert_eq!(select(&ds, "auto", 0.85).unwrap().index, 1);
        assert_eq!(select(&ds, "cuda:0", 0.85).unwrap().index, 0);
    }
    #[test]
    fn cuda_ordinals_are_matched_to_nvml_by_uuid() {
        let ds = [gpu(0, 3 << 30), gpu(1, 7 << 30)];
        let visible = remap_visible(&ds, &["1".into(), "0".into()]);
        assert_eq!(visible[0].uuid, "1");
        assert_eq!(select(&visible, "auto", 0.85).unwrap().index, 0);
        let visible = remap_visible(&ds, &["unknown-mig".into(), "0".into()]);
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].index, 1);
        assert!(remap_visible(&ds, &[]).is_empty());
    }
    #[test]
    fn cuda13_unsupported_architectures_are_not_selected() {
        let mut old = gpu(0, 7 << 30);
        old.compute_major = 7;
        old.compute_minor = 0;
        assert!(select(&[old.clone()], "cuda", 0.85).is_err());
        old.compute_minor = 5;
        assert!(select(&[old], "cuda", 0.85).is_ok());
        assert_eq!(
            classify("Available memory of 0 is smaller than requested bytes of 1024"),
            Failure::Oom
        );
    }
    #[test]
    fn headroom_saturates() {
        assert_eq!(budget(&gpu(0, 1), 0.85), 0);
        assert!(select(&[], "auto", 0.85).is_err());
    }
    #[test]
    fn failures() {
        assert_eq!(classify("CUDA out of memory"), Failure::Oom);
        assert_eq!(classify("device lost"), Failure::DeviceLost);
        assert!(require_auto(false).is_err());
    }
}
