//! Sampled resource telemetry for local benchmarks, not production scheduling.
use nvml_wrapper::{Nvml, enums::device::UsedGpuMemory};
use serde_json::{Value, json};
use std::{sync::mpsc, thread::JoinHandle, time::Duration};

#[derive(Default)]
struct Peaks {
    samples: u64,
    rss: Option<u64>,
    lifetime_hwm: Option<u64>,
    process_vram: Option<u64>,
    device_used: Option<u64>,
    gpu_errors: u64,
}
fn maximum(old: &mut Option<u64>, new: u64) {
    *old = Some(old.unwrap_or(0).max(new));
}
fn status_bytes(status: &str, key: &str) -> Option<u64> {
    status
        .lines()
        .find_map(|line| line.strip_prefix(key))?
        .split_whitespace()
        .next()?
        .parse::<u64>()
        .ok()?
        .checked_mul(1024)
}
impl Peaks {
    fn sample(&mut self, nvml: Option<&Nvml>, uuid: Option<&str>) {
        self.samples += 1;
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            if let Some(n) = status_bytes(&status, "VmRSS:") {
                maximum(&mut self.rss, n);
            }
            if let Some(n) = status_bytes(&status, "VmHWM:") {
                maximum(&mut self.lifetime_hwm, n);
            }
        }
        if let Some(uuid) = uuid {
            let sample = || -> anyhow::Result<(u64, Option<u64>)> {
                let device = nvml
                    .ok_or_else(|| anyhow::anyhow!("NVML unavailable"))?
                    .device_by_uuid(uuid)?;
                let memory = device.memory_info()?;
                let processes = device.running_compute_processes()?;
                let used = match processes.iter().find(|p| p.pid == std::process::id()) {
                    Some(p) => match p.used_gpu_memory {
                        UsedGpuMemory::Used(n) => Some(n),
                        _ => None,
                    },
                    None => Some(0),
                };
                Ok((memory.used, used))
            };
            match sample() {
                Ok((device, process)) => {
                    maximum(&mut self.device_used, device);
                    if let Some(n) = process {
                        maximum(&mut self.process_vram, n);
                    }
                }
                Err(_) => self.gpu_errors += 1,
            }
        }
    }
    fn report(self) -> Value {
        json!({"sample_interval_ms":10,"samples":self.samples,"sampled_peak_rss_bytes":self.rss,
            "process_lifetime_peak_rss_bytes":self.lifetime_hwm,"sampled_peak_process_vram_bytes":self.process_vram,
            "sampled_peak_device_used_bytes":self.device_used,"gpu_sample_errors":self.gpu_errors,
            "scope":"sampled peaks are lower bounds; device usage includes other processes; lifetime RSS includes earlier phases"})
    }
}
pub struct Sampler {
    stop: mpsc::Sender<()>,
    worker: Option<JoinHandle<Value>>,
}
impl Sampler {
    pub fn start(uuid: Option<String>) -> Self {
        let (stop, recv) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let nvml = uuid.as_ref().and_then(|_| Nvml::init().ok());
            let mut peaks = Peaks::default();
            loop {
                peaks.sample(nvml.as_ref(), uuid.as_deref());
                if recv.recv_timeout(Duration::from_millis(10))
                    != Err(mpsc::RecvTimeoutError::Timeout)
                {
                    peaks.sample(nvml.as_ref(), uuid.as_deref());
                    return peaks.report();
                }
            }
        });
        Self {
            stop,
            worker: Some(worker),
        }
    }
    pub fn finish(mut self) -> Value {
        let _ = self.stop.send(());
        self.worker
            .take()
            .unwrap()
            .join()
            .unwrap_or_else(|_| json!({"error":"resource sampler panicked"}))
    }
}
impl Drop for Sampler {
    fn drop(&mut self) {
        let _ = self.stop.send(());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

pub fn host_snapshot(uuid: Option<&str>) -> Value {
    let text = |path| {
        std::fs::read_to_string(path)
            .ok()
            .map(|s| s.trim().to_string())
    };
    let processes = uuid.and_then(|uuid| {
        let nvml = Nvml::init().ok()?;
        let device = nvml.device_by_uuid(uuid).ok()?;
        Some(device.running_compute_processes().ok()?.iter().map(|p| {
            json!({"pid":p.pid,"used_gpu_memory":match p.used_gpu_memory { UsedGpuMemory::Used(n) => Some(n), _ => None }})
        }).collect::<Vec<_>>())
    });
    json!({"kernel":text("/proc/sys/kernel/osrelease"),"loadavg":text("/proc/loadavg"),
        "platform_profile":text("/sys/firmware/acpi/platform_profile"),
        "cpu0_governor":text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "gpu_compute_processes":processes})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn memory_units_and_sampling_without_gpu() {
        assert_eq!(status_bytes("VmRSS:\t123 kB\n", "VmRSS:"), Some(123 * 1024));
        assert_eq!(status_bytes("VmHWM: unknown", "VmHWM:"), None);
        let report = Sampler::start(None).finish();
        assert!(report["samples"].as_u64().unwrap() >= 2);
        assert!(report["sampled_peak_rss_bytes"].as_u64().unwrap() > 0);
        assert!(report["sampled_peak_process_vram_bytes"].is_null());
    }
}
