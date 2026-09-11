//! Best-effort runtime telemetry. Missing GPU libraries never affect CPU use.
use serde_json::{Value, json};

pub fn inference_threads() -> usize {
    std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .saturating_sub(2)
        .clamp(1, 8)
}

pub fn versions(cuda: bool) -> Value {
    if !cuda {
        return json!({"cuda_runtime_version":null,"cuda_driver_api_version":null,"cudnn_version":null});
    }
    // These are documented C ABI version queries. Resolve symbols only while
    // their library handle is live; do not retain pointers after dlclose.
    unsafe fn integer_version(library: &str, symbol: &[u8]) -> Option<i32> {
        let lib = unsafe { libloading::Library::new(library) }.ok()?;
        let query = unsafe { lib.get::<unsafe extern "C" fn(*mut i32) -> i32>(symbol) }.ok()?;
        let mut version = 0;
        (unsafe { query(&mut version) } == 0).then_some(version)
    }
    unsafe fn cudnn_version() -> Option<usize> {
        let lib = unsafe { libloading::Library::new("libcudnn.so.9") }.ok()?;
        let query =
            unsafe { lib.get::<unsafe extern "C" fn() -> usize>(b"cudnnGetVersion\0") }.ok()?;
        Some(unsafe { query() })
    }
    json!({
        "cuda_runtime_version":unsafe { integer_version("libcudart.so.13", b"cudaRuntimeGetVersion\0") },
        "cuda_driver_api_version":unsafe { integer_version("libcuda.so.1", b"cuDriverGetVersion\0") },
        "cudnn_version":unsafe { cudnn_version() },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cpu_telemetry_does_not_require_gpu_libraries() {
        assert!(versions(false)["cuda_runtime_version"].is_null());
        assert!((1..=8).contains(&inference_threads()));
    }
}
