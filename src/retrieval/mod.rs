//! Retrieval orchestration: no writes, source reads, or answer generation.
pub mod protocol;
pub mod service;
pub mod transport;
use crate::{
    error::AppError,
    events::Events,
    model::query::{self, QueryEmbedder},
    storage::{
        Database,
        search::{Candidate, ChunkResult, Filters},
    },
};
use anyhow::Result;
use serde::Serialize;
use serde_json::json;
use std::{collections::BTreeMap, time::Instant};

pub const MAX_RESULTS: usize = 1000;
const MAX_LEXICAL_TERMS: usize = 512;
const RRF_K: f64 = 60.0;

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum, Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    #[default]
    Semantic,
    Lexical,
    Hybrid,
}
impl SearchMode {
    pub fn uses_dense(self) -> bool {
        self != Self::Lexical
    }
    pub fn uses_lexical(self) -> bool {
        self != Self::Semantic
    }
}

#[derive(Debug, Clone)]
pub struct SearchRequest {
    pub query: String,
    pub limit: usize,
    pub mode: SearchMode,
    pub root: Option<String>,
    pub media_type: Option<String>,
}
impl SearchRequest {
    pub fn validate(&self) -> Result<()> {
        query::validate_text(&self.query)?;
        if !(1..=MAX_RESULTS).contains(&self.limit) {
            return Err(
                AppError::new(2, format!("search limit must be in 1..={MAX_RESULTS}")).into(),
            );
        }
        if [&self.root, &self.media_type].iter().any(|s| {
            s.as_ref()
                .is_some_and(|s| s.trim().is_empty() || s.contains('\0'))
        }) {
            return Err(
                AppError::new(2, "search filters must be nonblank and contain no NUL").into(),
            );
        }
        if self.mode.uses_lexical() {
            lexical_expression(&self.query)?;
        }
        Ok(())
    }
}

/// Whitespace-separated literal phrases joined by OR, for candidate recall.
/// Quote escaping prevents FTS operators/column names from becoming syntax.
/// FTS unicode61 still tokenizes punctuation inside each quoted phrase.
pub fn lexical_expression(text: &str) -> Result<String> {
    query::validate_text(text)?;
    let terms: Vec<_> = text.split_whitespace().collect();
    if terms.len() > MAX_LEXICAL_TERMS {
        return Err(AppError::new(
            2,
            format!("lexical query exceeds {MAX_LEXICAL_TERMS} whitespace-separated terms"),
        )
        .into());
    }
    Ok(terms
        .into_iter()
        .map(|s| format!("\"{}\"", s.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" OR "))
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    #[serde(rename = "type")]
    kind: &'static str,
    pub rank: usize,
    #[serde(flatten)]
    pub chunk: ChunkResult,
    pub distance: Option<f64>,
    pub bm25: Option<f64>,
    pub fusion_score: Option<f64>,
}
#[derive(Debug, Default, Serialize)]
pub struct Timings {
    pub tokenization_ms: u128,
    pub initialization_ms: u128,
    pub embedding_ms: u128,
    pub sql_ms: u128,
    pub hydration_ms: u128,
    pub elapsed_ms: u128,
}
#[derive(Debug)]
pub struct SearchReport {
    pub results: Vec<SearchResult>,
    pub mode: SearchMode,
    pub generation: Option<i64>,
    pub model_key: Option<String>,
    pub query_provider: Option<String>,
    pub query_tokens: Option<usize>,
    pub candidate_limit: usize,
    pub dense_candidates: usize,
    pub lexical_candidates: usize,
    pub timings: Timings,
}
impl SearchReport {
    pub fn emit(&self, events: &mut Events) -> Result<()> {
        for result in &self.results {
            events.emit(serde_json::to_value(result)?)?;
        }
        events.emit(json!({"type":"search_completed", "mode":self.mode,
            "generation":self.generation,"model_key":self.model_key,
            "query_provider":self.query_provider,"query_tokens":self.query_tokens,
            "results":self.results.len(),"candidate_limit":self.candidate_limit,
            "dense_candidates":self.dense_candidates,"lexical_candidates":self.lexical_candidates,
            "timings":self.timings,
            "warnings":if self.query_provider.is_some() {
                vec!["mixed_int8_fp16_compatibility_unqualified"]
            } else { vec![] }
        }))
    }
}

#[derive(Debug)]
struct Ranked {
    id: String,
    distance: Option<f64>,
    bm25: Option<f64>,
    fusion: Option<f64>,
}
fn rank(
    mode: SearchMode,
    dense: Vec<Candidate>,
    lexical: Vec<Candidate>,
    limit: usize,
) -> Vec<Ranked> {
    if mode != SearchMode::Hybrid {
        let candidates = if mode == SearchMode::Semantic {
            dense
        } else {
            lexical
        };
        return candidates
            .into_iter()
            .take(limit)
            .map(|c| Ranked {
                id: c.chunk_id,
                distance: (mode == SearchMode::Semantic).then_some(c.value),
                bm25: (mode == SearchMode::Lexical).then_some(c.value),
                fusion: None,
            })
            .collect();
    }
    let mut union: BTreeMap<String, Ranked> = BTreeMap::new();
    for (is_dense, candidates) in [(true, dense), (false, lexical)] {
        for (i, c) in candidates.into_iter().enumerate() {
            let entry = union.entry(c.chunk_id.clone()).or_insert(Ranked {
                id: c.chunk_id,
                distance: None,
                bm25: None,
                fusion: Some(0.0),
            });
            if is_dense {
                entry.distance = Some(c.value);
            } else {
                entry.bm25 = Some(c.value);
            }
            *entry.fusion.as_mut().unwrap() += 1.0 / (RRF_K + (i + 1) as f64);
        }
    }
    let mut ranked: Vec<_> = union.into_values().collect();
    ranked.sort_by(|a, b| {
        b.fusion
            .unwrap()
            .total_cmp(&a.fusion.unwrap())
            .then_with(|| a.id.cmp(&b.id))
    });
    ranked.truncate(limit);
    ranked
}

/// The caller opens the database read-only. Dropping snapshots before model
/// initialization/output avoids retaining WAL history during slow downloads or IO.
pub fn search(
    db: &mut Database,
    embedder: &mut dyn QueryEmbedder,
    request: &SearchRequest,
    events: &mut Events,
) -> Result<SearchReport> {
    search_bounded(db, embedder, request, events, usize::MAX)
}

pub fn search_bounded(
    db: &mut Database,
    embedder: &mut dyn QueryEmbedder,
    request: &SearchRequest,
    events: &mut Events,
    mut hydration_budget: usize,
) -> Result<SearchReport> {
    let _span = crate::metrics::Span::new("retrieval");
    let start = Instant::now();
    request.validate()?;
    let expression = if request.mode.uses_lexical() {
        Some(lexical_expression(&request.query)?)
    } else {
        None
    };
    let filters = Filters::new(request.root.clone(), request.media_type.clone());
    let candidate_limit = if request.mode == SearchMode::Hybrid {
        (request.limit * 4).clamp(100, 4096)
    } else {
        request.limit
    };
    let mut report = SearchReport {
        results: vec![],
        mode: request.mode,
        generation: None,
        model_key: None,
        query_provider: None,
        query_tokens: None,
        candidate_limit,
        dense_candidates: 0,
        lexical_candidates: 0,
        timings: Timings::default(),
    };
    let sql_start = Instant::now();
    let initial = db.search_snapshot(request.mode.uses_lexical())?;
    if let Some(active) = &initial.active {
        if request.mode.uses_dense() {
            active.validate_dense()?;
        }
        report.generation = Some(active.generation);
        report.model_key = Some(active.model_key.clone());
    }
    let eligible = initial.has_candidates(&filters)?;
    drop(initial);
    report.timings.sql_ms += sql_start.elapsed().as_millis();
    // Empty scopes are successful offline queries. No tokenizer/model download
    // is necessary; token-count validation applies only when encoding occurs.
    if !eligible {
        report.timings.elapsed_ms = start.elapsed().as_millis();
        return Ok(report);
    }
    let embedding = if request.mode.uses_dense() {
        let embedding = embedder.embed_query(&request.query, events)?;
        crate::model::validate_vector(&embedding.vector)?;
        report.query_provider = Some(embedding.provider.clone());
        report.query_tokens = Some(embedding.token_count);
        report.timings.tokenization_ms = embedding.tokenization_ms;
        report.timings.initialization_ms = embedding.initialization_ms;
        report.timings.embedding_ms = embedding.embedding_ms;
        Some(embedding)
    } else {
        None
    };
    let sql_start = Instant::now();
    let snapshot = db.search_snapshot(request.mode.uses_lexical())?;
    // Activation during embedding is fine only if the now-active recipe is
    // still supported. Generation IDs alone are not recipe compatibility.
    if let Some(active) = &snapshot.active
        && request.mode.uses_dense()
    {
        active.validate_dense()?;
    }
    report.generation = snapshot.active.as_ref().map(|a| a.generation);
    report.model_key = snapshot.active.as_ref().map(|a| a.model_key.clone());
    let dense = if let Some(embedding) = &embedding {
        snapshot.dense(&embedding.vector, &filters, candidate_limit)?
    } else {
        vec![]
    };
    let lexical = if let Some(expression) = &expression {
        snapshot.lexical(expression, &filters, candidate_limit)?
    } else {
        vec![]
    };
    report.dense_candidates = dense.len();
    report.lexical_candidates = lexical.len();
    report.timings.sql_ms += sql_start.elapsed().as_millis();
    let ranked = rank(request.mode, dense, lexical, request.limit);
    let hydration_start = Instant::now();
    for (i, c) in ranked.into_iter().enumerate() {
        report.results.push(SearchResult {
            kind: "search_result",
            rank: i + 1,
            chunk: snapshot.hydrate_bounded(&c.id, &mut hydration_budget)?,
            distance: c.distance,
            bm25: c.bm25,
            fusion_score: c.fusion,
        });
    }
    drop(snapshot);
    report.timings.hydration_ms = hydration_start.elapsed().as_millis();
    report.timings.elapsed_ms = start.elapsed().as_millis();
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fusion_uses_ranks_not_incomparable_scores_and_stable_ties() {
        let dense = vec![
            Candidate {
                chunk_id: "a".into(),
                value: 0.0,
            },
            Candidate {
                chunk_id: "b".into(),
                value: 0.9,
            },
        ];
        let lexical = vec![
            Candidate {
                chunk_id: "b".into(),
                value: -10000.0,
            },
            Candidate {
                chunk_id: "a".into(),
                value: -0.001,
            },
        ];
        let ranked = rank(SearchMode::Hybrid, dense, lexical, 10);
        assert_eq!(
            ranked.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
            ["a", "b"]
        );
        assert_eq!(ranked[0].fusion, ranked[1].fusion);
        assert!((ranked[0].fusion.unwrap() - (1.0 / 61.0 + 1.0 / 62.0)).abs() < 1e-12);
    }
    #[test]
    fn plain_query_cannot_inject_fts_syntax() {
        assert_eq!(
            lexical_expression("text:hello OR \"x\"").unwrap(),
            "\"text:hello\" OR \"OR\" OR \"\"\"x\"\"\""
        );
        assert!(lexical_expression(&"x ".repeat(513)).is_err());
    }
}
