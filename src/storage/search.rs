//! Read-only retrieval. A snapshot covers generation selection, ranking and text
//! hydration, so concurrent activation/deletion cannot mix generations or rows.
use super::Database;
use crate::model;
use anyhow::{Context, Result, ensure};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct ActiveRecipe {
    pub generation: i64,
    pub model_key: String,
    pub model_id: String,
    pub revision: String,
    pub tokenizer_revision: String,
    pub dimensions: usize,
    pub recipe: Value,
}
impl ActiveRecipe {
    pub fn validate_dense(&self) -> Result<()> {
        ensure!(
            self.model_key == model::MODEL_KEY
                && self.model_id == model::MODEL_ID
                && self.revision == model::REVISION
                && self.tokenizer_revision == model::REVISION
                && self.dimensions == 768
                && self.recipe == model::recipe(),
            "active embedding recipe is not supported by this binary; use a compatible ree binary or explicitly rebuild/reingest; search will not modify the database"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct Filters {
    pub root: Option<String>,
    pub media_type: Option<String>,
    canonical_root: Option<String>,
}
impl Filters {
    pub fn new(root: Option<String>, media_type: Option<String>) -> Self {
        let canonical_root = root
            .as_ref()
            .and_then(|s| std::path::Path::new(s).canonicalize().ok())
            .and_then(|p| p.to_str().map(str::to_owned));
        Self {
            root,
            media_type,
            canonical_root,
        }
    }
    fn is_empty(&self) -> bool {
        self.root.is_none() && self.media_type.is_none()
    }
}

// This predicate is evaluated over eligible rows BEFORE LIMIT, never after a
// global KNN. Unknown roots match no rows. Prefer an exact ID over an identity.
const FILTER_SQL: &str = "
 AND (?3 IS NULL OR d.source_root_id=(
   SELECT id FROM source_roots WHERE id=?3 OR identity=?3 OR identity=?4
   ORDER BY (id=?3) DESC, (identity=?3) DESC LIMIT 1))
 AND (?5 IS NULL OR d.media_type=?5)";

#[derive(Debug, Clone)]
pub struct Candidate {
    pub chunk_id: String,
    /// Cosine distance or BM25; both sort ascending in their respective lists.
    pub value: f64,
}

#[derive(Debug, Serialize)]
pub struct ChunkResult {
    pub chunk_id: String,
    pub document_id: String,
    pub ordinal: usize,
    pub text: String,
    pub uri: String,
    pub source_id: String,
    pub source: String,
    pub media_type: String,
    pub byte_start: usize,
    pub byte_end: usize,
    pub token_start: usize,
    pub token_end: usize,
    pub location: Value,
    pub document_metadata: Value,
    pub source_metadata: Value,
    /// Overlapping extracted-document location ranges, never original-file bytes.
    pub locations: Vec<Value>,
}

pub struct SearchSnapshot<'a> {
    tx: Transaction<'a>,
    pub active: Option<ActiveRecipe>,
}
impl Database {
    pub fn schema_version(&self) -> Result<i64> {
        Ok(self
            .conn
            .pragma_query_value(None, "user_version", |r| r.get(0))?)
    }
    pub fn search_snapshot(&mut self, lexical: bool) -> Result<SearchSnapshot<'_>> {
        let tx = self.conn.transaction()?;
        let synthetic: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM schema_metadata WHERE key='synthetic_vectors' AND value='true')",
            [], |r| r.get(0))?;
        ensure!(
            !synthetic,
            "synthetic benchmark database: search requires real indexed documents"
        );
        let version: i64 = tx.pragma_query_value(None, "user_version", |r| r.get(0))?;
        ensure!(
            !lexical || version >= 2,
            "lexical/hybrid search needs schema 2; run ree migrate with the same --db (no model download or re-embedding)"
        );
        let active = active_recipe(&tx)?;
        Ok(SearchSnapshot { tx, active })
    }
}
fn active_recipe(conn: &Connection) -> Result<Option<ActiveRecipe>> {
    let row = conn
        .query_row(
            "SELECT g.id,m.id,m.model_id,m.revision,m.tokenizer_revision,m.dimensions,m.recipe_json
         FROM embedding_generations g JOIN models m ON m.id=g.model_id WHERE g.status='active'",
            [],
            |r| {
                Ok((
                    r.get(0)?,
                    r.get(1)?,
                    r.get(2)?,
                    r.get(3)?,
                    r.get(4)?,
                    r.get(5)?,
                    r.get::<_, String>(6)?,
                ))
            },
        )
        .optional()?;
    row.map(
        |(generation, model_key, model_id, revision, tokenizer_revision, dimensions, recipe)| {
            Ok(ActiveRecipe {
                generation,
                model_key,
                model_id,
                revision,
                tokenizer_revision,
                dimensions,
                recipe: serde_json::from_str(&recipe)
                    .context("invalid active model recipe JSON")?,
            })
        },
    )
    .transpose()
}
impl SearchSnapshot<'_> {
    pub fn has_candidates(&self, filters: &Filters) -> Result<bool> {
        let Some(active) = &self.active else {
            return Ok(false);
        };
        let sql = format!(
            "SELECT EXISTS(SELECT 1 FROM embedding_records r
            JOIN chunks c ON c.id=r.chunk_id JOIN documents d ON d.id=c.document_id
            WHERE r.generation_id=?2 {FILTER_SQL})"
        );
        Ok(self.tx.query_row(
            &sql,
            params![
                rusqlite::types::Null,
                active.generation,
                filters.root,
                filters.canonical_root,
                filters.media_type
            ],
            |r| r.get(0),
        )?)
    }
    pub fn dense(&self, vector: &[f32], filters: &Filters, limit: usize) -> Result<Vec<Candidate>> {
        model::validate_vector(vector)?;
        ensure!(
            (1..=4096).contains(&limit),
            "dense candidate limit must be in 1..=4096"
        );
        let Some(active) = &self.active else {
            return Ok(vec![]);
        };
        let blob: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
        if filters.is_empty() {
            // MATERIALIZED keeps the outer tie-sort away from vec0's single
            // ORDER BY restriction. Boundary ties are selected by vec0 itself.
            let mut stmt = self.tx.prepare_cached(
                "WITH nearest AS MATERIALIZED (
                SELECT rowid,distance FROM embeddings
                WHERE embedding MATCH ?1 AND generation_id=?2 AND k=?3 ORDER BY distance)
                SELECT r.chunk_id,n.distance FROM nearest n
                JOIN embedding_records r ON r.vector_id=n.rowid
                ORDER BY n.distance,r.chunk_id",
            )?;
            candidates(&mut stmt, params![blob, active.generation, limit])
        } else {
            let sql = format!(
                "SELECT c.id,vec_distance_cosine(e.embedding,?1) AS distance
                FROM embedding_records r JOIN embeddings e ON e.rowid=r.vector_id
                JOIN chunks c ON c.id=r.chunk_id JOIN documents d ON d.id=c.document_id
                WHERE r.generation_id=?2 {FILTER_SQL} ORDER BY distance,c.id LIMIT ?6"
            );
            let mut stmt = self.tx.prepare_cached(&sql)?;
            candidates(
                &mut stmt,
                params![
                    blob,
                    active.generation,
                    filters.root,
                    filters.canonical_root,
                    filters.media_type,
                    limit
                ],
            )
        }
    }
    /// `expression` is generated by retrieval::lexical_expression, not raw user
    /// syntax. Both branches use the identical active generation/filter scope.
    pub fn lexical(
        &self,
        expression: &str,
        filters: &Filters,
        limit: usize,
    ) -> Result<Vec<Candidate>> {
        ensure!(
            (1..=4096).contains(&limit),
            "lexical candidate limit must be in 1..=4096"
        );
        let Some(active) = &self.active else {
            return Ok(vec![]);
        };
        let sql = format!(
            "SELECT c.id,bm25(chunk_fts) AS score FROM chunk_fts
            JOIN chunk_search_ids m ON m.id=chunk_fts.rowid
            JOIN chunks c ON c.id=m.chunk_id JOIN documents d ON d.id=c.document_id
            JOIN embedding_records r ON r.chunk_id=c.id
            WHERE chunk_fts MATCH ?1 AND r.generation_id=?2 {FILTER_SQL}
            ORDER BY score,c.id LIMIT ?6"
        );
        let mut stmt = self.tx.prepare_cached(&sql)?;
        candidates(
            &mut stmt,
            params![
                expression,
                active.generation,
                filters.root,
                filters.canonical_root,
                filters.media_type,
                limit
            ],
        )
    }
    pub fn hydrate(&self, chunk_id: &str) -> Result<ChunkResult> {
        let mut budget = usize::MAX;
        self.hydrate_bounded(chunk_id, &mut budget)
    }
    pub fn hydrate_bounded(&self, chunk_id: &str, budget: &mut usize) -> Result<ChunkResult> {
        let _span = crate::metrics::Span::new("result_hydration");
        let mut stmt = self.tx.prepare_cached(
            "SELECT c.id,c.document_id,c.ordinal,c.text,
            d.uri,s.id,s.identity,d.media_type,c.byte_start,c.byte_end,c.token_start,c.token_end,
            c.location_json,d.metadata_json,s.metadata_json FROM chunks c
            JOIN documents d ON d.id=c.document_id JOIN source_roots s ON s.id=d.source_root_id
            WHERE c.id=?1",
        )?;
        let (mut result, location, document_metadata, source_metadata) =
            stmt.query_row([chunk_id], |r| {
                // Inspect SQLite-owned bytes before copying text or decoding JSON.
                let mut bytes = 0usize;
                for i in [0, 1, 3, 4, 5, 6, 7, 12, 13, 14] {
                    bytes = bytes.saturating_add(r.get_ref(i)?.as_bytes()?.len());
                }
                if bytes > *budget {
                    return Err(rusqlite::Error::FromSqlConversionFailure(
                        3,
                        rusqlite::types::Type::Text,
                        Box::new(std::io::Error::other(
                            "response_limit: narrow the query or reduce --limit",
                        )),
                    ));
                }
                *budget -= bytes;
                Ok((
                    ChunkResult {
                        chunk_id: r.get(0)?,
                        document_id: r.get(1)?,
                        ordinal: r.get(2)?,
                        text: r.get(3)?,
                        uri: r.get(4)?,
                        source_id: r.get(5)?,
                        source: r.get(6)?,
                        media_type: r.get(7)?,
                        byte_start: r.get(8)?,
                        byte_end: r.get(9)?,
                        token_start: r.get(10)?,
                        token_end: r.get(11)?,
                        location: Value::Null,
                        document_metadata: Value::Null,
                        source_metadata: Value::Null,
                        locations: vec![],
                    },
                    r.get::<_, String>(12)?,
                    r.get::<_, String>(13)?,
                    r.get::<_, String>(14)?,
                ))
            })?;
        result.location = serde_json::from_str(&location).context("invalid chunk location JSON")?;
        result.document_metadata =
            serde_json::from_str(&document_metadata).context("invalid document metadata JSON")?;
        result.source_metadata =
            serde_json::from_str(&source_metadata).context("invalid source metadata JSON")?;
        // PDF pages and notebook cells live in document metadata, not necessarily
        // in chunk.location_json. Return every overlapping range without inventing
        // a page/line mapping for formats that do not have one.
        for key in ["pages", "cells", "locations"] {
            if let Some(ranges) = result.document_metadata.get(key).and_then(Value::as_array) {
                for range in ranges {
                    if let (Some(start), Some(end)) =
                        (range["byte_start"].as_u64(), range["byte_end"].as_u64())
                        && start < result.byte_end as u64
                        && end > result.byte_start as u64
                    {
                        result.locations.push(range.clone());
                    }
                }
            }
        }
        Ok(result)
    }
}
fn candidates(
    stmt: &mut rusqlite::CachedStatement<'_>,
    params: impl rusqlite::Params,
) -> Result<Vec<Candidate>> {
    let rows = stmt.query_map(params, |r| {
        Ok(Candidate {
            chunk_id: r.get(0)?,
            value: r.get(1)?,
        })
    })?;
    let result = rows.collect::<rusqlite::Result<Vec<_>>>()?;
    ensure!(
        result.iter().all(|c| c.value.is_finite()),
        "non-finite retrieval score"
    );
    Ok(result)
}
