use crate::cli::Options;
use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use serde_json::{Map, Value};
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExtractorConfig {
    pub extensions: Vec<String>,
    pub command: Vec<String>,
    pub timeout_seconds: u64,
    pub max_output_bytes: u64,
}
impl Default for ExtractorConfig {
    fn default() -> Self {
        Self {
            extensions: vec![],
            command: vec![],
            timeout_seconds: 60,
            max_output_bytes: 64 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FileConfig {
    db: Option<PathBuf>,
    device: Option<String>,
    batch_size: Option<usize>,
    gpu_memory_fraction: Option<f64>,
    chunk_size: Option<usize>,
    overlap: Option<usize>,
    max_file_size: Option<String>,
    metadata: Map<String, Value>,
    extractors: BTreeMap<String, ExtractorConfig>,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub db: PathBuf,
    pub cache: PathBuf,
    pub device: String,
    pub batch_size: Option<usize>,
    pub gpu_memory_fraction: f64,
    pub chunk_size: usize,
    pub overlap: usize,
    pub max_file_size: u64,
    pub metadata: Value,
    pub extractors: BTreeMap<String, ExtractorConfig>,
    pub allow_private_network: bool,
}

pub fn xdg(variable: &str, fallback: &str) -> Result<PathBuf> {
    if let Some(path) = std::env::var_os(variable)
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
    {
        return Ok(path.join("ree"));
    }
    let home =
        std::env::var_os("HOME").context("HOME is unset and no absolute XDG path is configured")?;
    Ok(PathBuf::from(home).join(fallback).join("ree"))
}

impl Config {
    pub fn load(o: &Options) -> Result<Self> {
        let path = o
            .config
            .clone()
            .unwrap_or(xdg("XDG_CONFIG_HOME", ".config")?.join("config.toml"));
        let f: FileConfig = match std::fs::read_to_string(&path) {
            Ok(s) => {
                toml::from_str(&s).with_context(|| format!("invalid config {}", path.display()))?
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && o.config.is_none() => {
                FileConfig::default()
            }
            Err(e) => return Err(e).with_context(|| format!("read config {}", path.display())),
        };
        let mut metadata = f.metadata;
        if let Some(json) = &o.metadata_json {
            let v: Value = serde_json::from_str(json).context("invalid --metadata-json")?;
            metadata.extend(
                v.as_object()
                    .context("--metadata-json must be an object")?
                    .clone(),
            );
        }
        for m in &o.metadata {
            let (k, v) = m.split_once('=').context("--metadata requires KEY=VALUE")?;
            ensure!(!k.is_empty(), "metadata key cannot be empty");
            metadata.insert(k.into(), Value::String(v.into()));
        }
        let c = Self {
            db: o
                .db
                .clone()
                .or(f.db)
                .unwrap_or(xdg("XDG_DATA_HOME", ".local/share")?.join("ree.db")),
            cache: xdg("XDG_CACHE_HOME", ".cache")?,
            device: o.device.clone().or(f.device).unwrap_or("auto".into()),
            batch_size: o.batch_size.or(f.batch_size),
            gpu_memory_fraction: o
                .gpu_memory_fraction
                .or(f.gpu_memory_fraction)
                .unwrap_or(0.85),
            chunk_size: o.chunk_size.or(f.chunk_size).unwrap_or(512),
            overlap: o.overlap.or(f.overlap).unwrap_or(64),
            max_file_size: parse_size(
                o.max_file_size
                    .as_deref()
                    .or(f.max_file_size.as_deref())
                    .unwrap_or("64MiB"),
            )?,
            metadata: Value::Object(metadata),
            extractors: f.extractors,
            allow_private_network: o.allow_private_network,
        };
        ensure!(
            (3..=8192).contains(&c.chunk_size),
            "chunk-size must be between 3 and 8192 (including two special tokens)"
        );
        ensure!(
            c.overlap < c.chunk_size - 2,
            "overlap must be smaller than chunk-size minus two special tokens"
        );
        ensure!(c.batch_size != Some(0), "batch-size must be positive");
        ensure!(
            c.gpu_memory_fraction.is_finite()
                && c.gpu_memory_fraction > 0.0
                && c.gpu_memory_fraction <= 0.95,
            "gpu-memory-fraction must be in (0, 0.95]"
        );
        ensure!(
            matches!(c.device.as_str(), "auto" | "cpu" | "cuda")
                || c.device
                    .strip_prefix("cuda:")
                    .is_some_and(|n| n.parse::<u32>().is_ok()),
            "device must be auto, cpu, cuda, or cuda:N"
        );
        for (name, e) in &c.extractors {
            ensure!(
                !e.command.is_empty() && e.timeout_seconds > 0 && e.max_output_bytes > 0,
                "invalid extractor {name}"
            );
        }
        Ok(c)
    }
}

pub fn parse_size(s: &str) -> Result<u64> {
    let pos = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    let n: u64 = s[..pos]
        .parse()
        .context("size requires a positive integer")?;
    let scale = match s[pos..].to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "k" | "kb" | "kib" => 1024,
        "m" | "mb" | "mib" => 1024 * 1024,
        "g" | "gb" | "gib" => 1024 * 1024 * 1024,
        _ => bail!("unknown size suffix in {s}"),
    };
    let bytes = n.checked_mul(scale).context("size overflow")?;
    ensure!(bytes > 0, "size must be positive");
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sizes() {
        assert_eq!(parse_size("64MiB").unwrap(), 67108864);
        assert!(parse_size("0").is_err());
        assert!(parse_size("2TB").is_err());
    }
    #[test]
    fn metadata_precedence() {
        let c = Config::load(&Options {
            metadata_json: Some(r#"{"a":1,"b":true}"#.into()),
            metadata: vec!["a=two".into()],
            ..Default::default()
        })
        .unwrap();
        assert_eq!(c.metadata["a"], "two");
        assert_eq!(c.metadata["b"], true);
    }
}
