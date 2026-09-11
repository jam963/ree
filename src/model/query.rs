//! Query encoding is deliberately separate from passage windowing.
use super::{LocalEngine, validate_vector};
use crate::{error::AppError, events::Events};
use anyhow::{Result, anyhow, ensure};
use std::time::Instant;
use tokenizers::Tokenizer;

pub const MAX_QUERY_TOKENS: usize = 512;
pub const MAX_QUERY_BYTES: usize = 64 * 1024;
pub const QUERY_PREFIX: &str = "query: ";

pub fn validate_text(text: &str) -> Result<()> {
    if text.trim().is_empty() || text.len() > MAX_QUERY_BYTES || text.contains('\0') {
        return Err(AppError::new(
            2,
            format!("query must be nonblank, contain no NUL, and be at most {MAX_QUERY_BYTES} UTF-8 bytes"),
        ).into());
    }
    Ok(())
}

/// The tokenizer must have padding and truncation disabled. The prefix is always
/// added exactly once by ree; user text is not stripped or interpreted as a prompt.
pub fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<i64>> {
    validate_text(text)?;
    ensure!(
        tokenizer.get_truncation().is_none() && tokenizer.get_padding().is_none(),
        "query tokenizer must not truncate or pad"
    );
    let _span = crate::metrics::Span::new("query_tokenize");
    let encoding = tokenizer
        .encode(format!("{QUERY_PREFIX}{text}"), true)
        .map_err(|e| anyhow!(e))?;
    let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| i64::from(id)).collect();
    if ids.len() > MAX_QUERY_TOKENS {
        return Err(AppError::new(2, format!(
            "query contains {} model tokens; maximum is {MAX_QUERY_TOKENS}, including prefix and special tokens; shorten the query",
            ids.len()
        )).into());
    }
    ensure!(
        ids.first() == Some(&0) && ids.last() == Some(&2),
        "unexpected query special tokens"
    );
    Ok(ids)
}

#[derive(Debug)]
pub struct QueryEmbedding {
    pub vector: Vec<f32>,
    pub provider: String,
    pub token_count: usize,
    pub tokenization_ms: u128,
    pub initialization_ms: u128,
    pub embedding_ms: u128,
}

/// Injectable without introducing a synthetic runtime into the production CLI.
pub trait QueryEmbedder {
    fn embed_query(&mut self, text: &str, events: &mut Events) -> Result<QueryEmbedding>;
}

impl QueryEmbedder for LocalEngine {
    fn embed_query(&mut self, text: &str, events: &mut Events) -> Result<QueryEmbedding> {
        validate_text(text)?;
        let start = Instant::now();
        let ids = encode(self.tokenizer(events)?, text)?;
        let tokenization_ms = start.elapsed().as_millis();
        let token_count = ids.len();
        let start = Instant::now();
        self.init_policy(events, true)
            .map_err(|e| AppError::new(3, format!("query runtime initialization: {e:#}")))?;
        let initialization_ms = start.elapsed().as_millis();
        let start = Instant::now();
        let scheduler = self.scheduler.as_mut().unwrap();
        let mut vectors = match scheduler.embed(&[ids], events) {
            Ok(vectors) => vectors,
            Err(error) => {
                // A failed/recreated runtime must not poison a retained service.
                self.scheduler.take();
                return Err(error);
            }
        };
        ensure!(
            vectors.len() == 1,
            "query runtime returned wrong vector count"
        );
        let vector = vectors.remove(0);
        validate_vector(&vector)?;
        Ok(QueryEmbedding {
            vector,
            provider: scheduler.provider().unwrap().into(),
            token_count,
            tokenization_ms,
            initialization_ms,
            embedding_ms: start.elapsed().as_millis(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::{
        models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
        processors::template::TemplateProcessing,
    };

    fn tokenizer() -> Tokenizer {
        let vocab = [
            ("<s>", 0),
            ("<pad>", 1),
            ("</s>", 2),
            ("[UNK]", 3),
            ("query", 4),
            (":", 5),
            ("hello", 6),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect();
        let mut t = Tokenizer::new(
            WordLevel::builder()
                .vocab(vocab)
                .unk_token("[UNK]".into())
                .build()
                .unwrap(),
        );
        t.with_pre_tokenizer(Some(Whitespace));
        t.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single("<s> $A </s>")
                .unwrap()
                .special_tokens(vec![("<s>", 0), ("</s>", 2)])
                .build()
                .unwrap(),
        ));
        t
    }
    #[test]
    fn prefix_special_tokens_and_limits() {
        let t = tokenizer();
        assert_eq!(encode(&t, "hello").unwrap(), [0, 4, 5, 6, 2]);
        assert_eq!(encode(&t, "query: hello").unwrap(), [0, 4, 5, 4, 5, 6, 2]);
        assert_eq!(encode(&t, &"hello ".repeat(508)).unwrap().len(), 512);
        assert_eq!(
            crate::error::exit_code(&encode(&t, &"hello ".repeat(509)).unwrap_err()),
            2
        );
        for text in ["", " \n\t", "hello\0world"] {
            assert!(encode(&t, text).is_err());
        }
        assert!(validate_text(&"a".repeat(MAX_QUERY_BYTES + 1)).is_err());
        assert!(encode(&t, "こんにちは 🌍").is_ok());
    }
}
