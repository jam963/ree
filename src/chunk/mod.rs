use anyhow::{Result, anyhow, ensure};
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct Chunk {
    pub text: String,
    pub token_start: usize,
    pub token_end: usize,
    pub byte_start: usize,
    pub byte_end: usize,
    /// Exact passage model input, including CLS/SEP. Persisted for rebuild parity.
    pub input_ids: Vec<i64>,
}

pub fn windows(
    tokenizer: &Tokenizer,
    text: &str,
    size: usize,
    overlap: usize,
) -> Result<Vec<Chunk>> {
    ensure!(size >= 3 && overlap < size - 2, "invalid token window");
    let encoding = tokenizer.encode(text, false).map_err(|e| anyhow!(e))?;
    from_offsets(
        text,
        encoding.get_ids(),
        encoding.get_offsets(),
        size - 2,
        overlap,
    )
}

fn from_offsets(
    text: &str,
    ids: &[u32],
    offsets: &[(usize, usize)],
    capacity: usize,
    overlap: usize,
) -> Result<Vec<Chunk>> {
    ensure!(
        capacity > overlap && ids.len() == offsets.len(),
        "invalid offsets/window"
    );
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < ids.len() {
        let end = (start + capacity).min(ids.len());
        let byte_start = if start == 0 { 0 } else { offsets[start].0 };
        let byte_end = if end == ids.len() {
            text.len()
        } else {
            offsets[end - 1].1.max(offsets[end].0)
        };
        let exact = text
            .get(byte_start..byte_end)
            .ok_or_else(|| anyhow!("tokenizer returned invalid UTF-8 offsets"))?;
        let mut input_ids = Vec::with_capacity(end - start + 2);
        input_ids.push(0);
        input_ids.extend(ids[start..end].iter().map(|id| i64::from(*id)));
        input_ids.push(2);
        chunks.push(Chunk {
            text: exact.into(),
            token_start: start,
            token_end: end,
            byte_start,
            byte_end,
            input_ids,
        });
        if end == ids.len() {
            break;
        }
        start = end - overlap;
    }
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unicode_exact_overlap() {
        let text = "  α β γ δ  ";
        let c = from_offsets(
            text,
            &[5, 6, 7, 8],
            &[(2, 4), (5, 7), (8, 10), (11, 13)],
            3,
            1,
        )
        .unwrap();
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].text, "  α β γ ");
        assert_eq!(c[1].text, "γ δ  ");
        assert_eq!(c[1].token_start, 2);
        assert_eq!(c[0].input_ids, vec![0, 5, 6, 7, 2]);
    }
    #[test]
    fn empty_and_invalid() {
        assert!(from_offsets("", &[], &[], 2, 1).unwrap().is_empty());
        assert!(from_offsets("x", &[1], &[(0, 1)], 1, 1).is_err());
    }
}
