fn main() {
    // ONNX Runtime's core is static. Its optional CUDA provider is dlopened;
    // find packaged provider libraries beside the executable without requiring
    // LD_LIBRARY_PATH. Missing CUDA libraries still permit CPU startup.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    }
}
