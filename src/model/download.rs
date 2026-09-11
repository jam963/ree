use crate::{events::Events, util::hash_file};
use anyhow::{Context, Result, ensure};
use serde_json::json;
use std::{
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Clone, Copy)]
pub struct Artifact {
    pub name: &'static str,
    pub sha256: &'static str,
    pub size: u64,
}
pub const TOKENIZER: Artifact = Artifact {
    name: "tokenizer.json",
    sha256: "f1cc44ad7faaeec47241864835473fd5403f2da94673f3f764a77ebcb0a803ec",
    size: 17083009,
};
pub const CPU: Artifact = Artifact {
    name: "onnx/model_int8.onnx",
    sha256: "03d923bb1850ebdccb068e2f3abd8aa43fe81c50d07d037ef103fe3d0fb78e3b",
    size: 310916060,
};
pub const CUDA: Artifact = Artifact {
    name: "onnx/model_fp16.onnx",
    sha256: "f27ab40ab6e230265ba49a202a37f1ad031556256cbbc105d0ca9c0bdc7ec42e",
    size: 613266244,
};
pub fn path(cache: &Path, a: Artifact) -> PathBuf {
    cache.join("models").join(super::REVISION).join(a.name)
}
pub fn verified(path: &Path, a: Artifact) -> Result<bool> {
    let _span = crate::metrics::Span::new(match a.name {
        "tokenizer.json" => "verify_tokenizer",
        "onnx/model_int8.onnx" => "verify_cpu",
        _ => "verify_cuda",
    });
    if !path.is_file() {
        return Ok(false);
    }
    Ok(path.metadata()?.len() == a.size && hash_file(path)? == a.sha256)
}
pub fn ensure_artifact(cache: &Path, a: Artifact, events: &mut Events) -> Result<PathBuf> {
    let dest = path(cache, a);
    if verified(&dest, a)? {
        return Ok(dest);
    }
    let _span = crate::metrics::Span::new("artifact_download");
    let parent = dest.parent().context("artifact parent")?;
    std::fs::create_dir_all(parent)?;
    events.emit(
        json!({"type":"download","artifact":a.name,"bytes":a.size,"revision":super::REVISION}),
    )?;
    events.flush()?;
    let url = format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        super::MODEL_ID,
        super::REVISION,
        a.name
    );
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(20))
        .timeout(Duration::from_secs(1800))
        .https_only(true)
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()?;
    for attempt in 1..=3 {
        let result = (|| -> Result<tempfile::NamedTempFile> {
            let mut response = client.get(&url).send()?.error_for_status()?;
            let mut temp = tempfile::NamedTempFile::new_in(parent)?;
            let n = std::io::copy(&mut (&mut response).take(a.size + 1), temp.as_file_mut())?;
            ensure!(n == a.size, "model artifact size mismatch for {}", a.name);
            temp.flush()?;
            temp.as_file().sync_all()?;
            Ok(temp)
        })();
        match result {
            Ok(temp) => {
                // A checksum mismatch is not retried and never activated.
                ensure!(
                    hash_file(temp.path())? == a.sha256,
                    "model checksum mismatch for {}",
                    a.name
                );
                temp.persist(&dest).map_err(|e| e.error)?;
                std::fs::File::open(parent)?.sync_all()?;
                return Ok(dest);
            }
            Err(e) if attempt < 3 => {
                events.emit(json!({"type":"download_retry","artifact":a.name,"attempt":attempt,"message":format!("{e:#}")}))?;
                events.flush()?;
                std::thread::sleep(Duration::from_secs(attempt));
            }
            Err(e) => return Err(e),
        }
    }
    unreachable!()
}
