//! Shared lazy tokenizer, independent of runtime/device ownership. Preparation
//! workers only obtain it for changed, small documents; hash no-ops stay offline.
use super::download;
use crate::{
    chunk::{self, Chunk},
    events::Events,
};
use anyhow::{Result, anyhow, ensure};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};
use tokenizers::Tokenizer;

pub(crate) struct TokenizerSource {
    cache: PathBuf,
    loaded: Mutex<Option<Arc<Tokenizer>>>,
}
impl TokenizerSource {
    pub fn new(cache: PathBuf) -> Self {
        Self {
            cache,
            loaded: Mutex::new(None),
        }
    }
    pub fn get(&self, events: &mut Events) -> Result<Arc<Tokenizer>> {
        let mut loaded = self
            .loaded
            .lock()
            .map_err(|_| anyhow!("tokenizer initialization poisoned"))?;
        if let Some(tokenizer) = &*loaded {
            return Ok(tokenizer.clone());
        }
        let path =
            download::ensure_artifact(&self.cache, download::TOKENIZER, events).map_err(|e| {
                crate::error::AppError::new(3, format!("tokenizer initialization: {e:#}"))
            })?;
        let _span = crate::metrics::Span::new("tokenizer_load");
        let mut tokenizer = Tokenizer::from_file(path).map_err(|e| anyhow!(e))?;
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).map_err(|e| anyhow!(e))?;
        ensure!(
            tokenizer.token_to_id("<s>") == Some(0) && tokenizer.token_to_id("</s>") == Some(2),
            "unexpected tokenizer special tokens"
        );
        let tokenizer = Arc::new(tokenizer);
        *loaded = Some(tokenizer.clone());
        Ok(tokenizer)
    }
}
pub(crate) fn supports_overlap(device: &str, size: usize, overlap: usize) -> bool {
    device.starts_with("cuda") && size == 512 && overlap <= 64
}

pub trait Chunker: Send {
    fn chunk(&mut self, text: &str, events: &mut Events) -> Result<Vec<Chunk>>;
}
pub(crate) struct PassageChunker {
    pub source: Arc<TokenizerSource>,
    pub size: usize,
    pub overlap: usize,
}
impl Chunker for PassageChunker {
    fn chunk(&mut self, text: &str, events: &mut Events) -> Result<Vec<Chunk>> {
        if text.trim().is_empty() {
            return Ok(vec![]);
        }
        let tokenizer = self.source.get(events)?;
        let _span = crate::metrics::Span::new("document_tokenize");
        chunk::windows(&tokenizer, text, self.size, self.overlap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn failed_int8_equivalence_gate_excludes_cpu_and_auto() {
        assert!(!supports_overlap("cpu", 512, 64));
        assert!(!supports_overlap("auto", 512, 64));
        assert!(supports_overlap("cuda", 512, 64));
        assert!(supports_overlap("cuda:0", 512, 0));
        assert!(!supports_overlap("cuda", 8192, 64));
        assert!(!supports_overlap("cuda", 512, 128));
    }
}
