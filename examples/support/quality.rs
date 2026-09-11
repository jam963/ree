use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Deserialize, Serialize)]
pub struct Document {
    pub id: String,
    pub text: String,
}
#[derive(Deserialize, Serialize)]
pub struct Query {
    pub id: String,
    pub text: String,
    pub relevant: Vec<String>,
    /// Optional positive integer grades. Empty means binary relevance.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub relevance: BTreeMap<String, u32>,
    pub domain: String,
}
#[derive(Deserialize, Serialize)]
pub struct Corpus {
    pub documents: Vec<Document>,
    pub queries: Vec<Query>,
    /// BEIR-style self-match exclusion, explicitly pinned by prepared corpora.
    #[serde(default)]
    pub exclude_query_id: bool,
    /// Dataset URLs/revisions/licenses, selection seed and original hashes.
    #[serde(default)]
    pub provenance: Value,
}
impl Corpus {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.documents.is_empty()
                && self.documents.len() <= 10000
                && !self.queries.is_empty()
                && self.queries.len() <= 100000,
            "corpus requires 1..=10000 documents and 1..=100000 queries"
        );
        let ids: BTreeSet<_> = self.documents.iter().map(|d| &d.id).collect();
        ensure!(
            ids.len() == self.documents.len() && ids.iter().all(|s| !s.is_empty()),
            "empty/duplicate document IDs"
        );
        let mut queries = BTreeSet::new();
        for q in &self.queries {
            ensure!(
                !q.id.is_empty() && queries.insert(&q.id),
                "empty/duplicate query ID: {}",
                q.id
            );
            ensure!(
                !self.exclude_query_id || !q.relevant.contains(&q.id),
                "query {} has a positive self-match but self-matches are excluded",
                q.id
            );
            let relevant: BTreeSet<_> = q.relevant.iter().collect();
            ensure!(
                q.relevance.is_empty()
                    || (q.relevance.keys().collect::<BTreeSet<_>>() == relevant
                        && q.relevance.values().all(|&grade| grade > 0)),
                "query {} has inconsistent relevance grades",
                q.id
            );
            ensure!(
                !q.domain.is_empty()
                    && !relevant.is_empty()
                    && relevant.len() == q.relevant.len()
                    && relevant.iter().all(|id| ids.contains(id)),
                "query {} has empty domain or missing/duplicate/empty relevance judgments",
                q.id
            );
        }
        Ok(())
    }
}

/// Score descending, ID ascending; exclusion happens before applying the cutoff.
#[allow(dead_code)]
pub fn top_ids<'a>(
    documents: &'a [Document],
    scores: &[f32],
    query_id: &str,
    exclude_query_id: bool,
) -> Vec<&'a str> {
    assert_eq!(documents.len(), scores.len());
    let mut ranked: Vec<_> = documents
        .iter()
        .zip(scores)
        .filter(|(doc, _)| !exclude_query_id || doc.id != query_id)
        .collect();
    ranked.sort_by(|(a, x), (b, y)| y.total_cmp(x).then_with(|| a.id.cmp(&b.id)));
    ranked
        .into_iter()
        .take(10)
        .map(|(doc, _)| doc.id.as_str())
        .collect()
}

/// Fixed cutoff 10; linear graded gains for nDCG (binary if grades are absent).
/// Recall/MRR treat every positive grade as relevant. Call after corpus validation.
#[allow(dead_code)] // Shared with the preparation example, which does not score.
pub fn score(
    ranked_ids: &[&str],
    relevant: &[String],
    relevance: &BTreeMap<String, u32>,
) -> (f64, f64, f64) {
    let mut hits = 0;
    let mut reciprocal = 0.;
    let mut dcg = 0.;
    for (rank, id) in ranked_ids.iter().take(10).enumerate() {
        if relevant.iter().any(|r| r == id) {
            hits += 1;
            if reciprocal == 0. {
                reciprocal = 1. / (rank + 1) as f64;
            }
            dcg += f64::from(*relevance.get(*id).unwrap_or(&1)) / ((rank + 2) as f64).log2();
        }
    }
    let mut grades: Vec<_> = relevant
        .iter()
        .map(|id| *relevance.get(id).unwrap_or(&1))
        .collect();
    grades.sort_unstable_by(|a, b| b.cmp(a));
    let ideal = grades
        .iter()
        .take(10)
        .enumerate()
        .map(|(rank, &grade)| f64::from(grade) / ((rank + 2) as f64).log2())
        .sum::<f64>();
    (hits as f64 / relevant.len() as f64, reciprocal, dcg / ideal)
}

#[allow(dead_code)] // Shared with the preparation example, which does not infer.
pub fn project(mut vector: Vec<f32>, dimensions: usize) -> Result<Vec<f32>> {
    ensure!(
        dimensions > 0 && dimensions <= vector.len(),
        "invalid evaluation dimensions"
    );
    vector.truncate(dimensions);
    ree::model::normalize_dense(&mut vector)?;
    Ok(vector)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_ambiguous_corpora_before_inference() {
        let mut corpus: Corpus = serde_json::from_value(serde_json::json!({
            "documents":[{"id":"a","text":"doc"}],
            "queries":[{"id":"q","text":"query","relevant":["a"],"domain":"test"}]
        }))
        .unwrap();
        corpus.validate().unwrap();
        corpus.queries[0].relevant.push("a".into());
        assert!(corpus.validate().is_err());
        corpus.queries[0].relevant = vec!["missing".into()];
        assert!(corpus.validate().is_err());
        corpus.queries[0].relevant = vec!["a".into()];
        corpus.documents.push(Document {
            id: "a".into(),
            text: "ambiguous".into(),
        });
        assert!(corpus.validate().is_err());
    }
    #[test]
    fn self_matches_and_score_ties_are_explicit() {
        let mut corpus: Corpus = serde_json::from_value(serde_json::json!({
            "documents":[{"id":"q","text":"self"},{"id":"b","text":"b"},{"id":"a","text":"a"}],
            "queries":[{"id":"q","text":"query","relevant":["a"],"domain":"test"}],
            "exclude_query_id":true
        }))
        .unwrap();
        corpus.validate().unwrap();
        assert_eq!(
            top_ids(&corpus.documents, &[1., 0.5, 0.5], "q", true),
            ["a", "b"]
        );
        assert_eq!(
            top_ids(&corpus.documents, &[1., 0.5, 0.5], "q", false),
            ["q", "a", "b"]
        );
        corpus.queries[0].relevant.push("q".into());
        assert!(corpus.validate().is_err());
    }
    #[test]
    fn graded_ndcg_preserves_judgments() {
        let grades = BTreeMap::from([("a".into(), 1), ("b".into(), 3)]);
        let (recall, mrr, ndcg) = score(&["a", "b"], &["a".into(), "b".into()], &grades);
        assert_eq!((recall, mrr), (1., 1.));
        assert!((ndcg - (1. + 3. / 3f64.log2()) / (3. + 1. / 3f64.log2())).abs() < 1e-12);
        let mut corpus: Corpus = serde_json::from_value(serde_json::json!({
            "documents":[{"id":"a","text":"doc"}],
            "queries":[{"id":"q","text":"query","relevant":["a"],
                "relevance":{"a":0},"domain":"test"}]
        }))
        .unwrap();
        assert!(corpus.validate().is_err());
        corpus.queries[0].relevance.insert("a".into(), 2);
        corpus.validate().unwrap();
        corpus.queries[0].relevance.insert("missing".into(), 1);
        assert!(corpus.validate().is_err());
    }
    #[test]
    fn binary_metrics_have_known_answers() {
        let (recall, mrr, ndcg) = score(
            &["x", "a", "b"],
            &["a".into(), "b".into()],
            &BTreeMap::new(),
        );
        assert_eq!(recall, 1.);
        assert_eq!(mrr, 0.5);
        assert!((ndcg - ((1. / 3f64.log2() + 0.5) / (1. + 1. / 3f64.log2()))).abs() < 1e-12);
        assert_eq!(score(&["x"], &["a".into()], &BTreeMap::new()), (0., 0., 0.));
        assert_eq!(project(vec![3., 4., 99.], 2).unwrap(), vec![0.6, 0.8]);
        assert!(project(vec![0., 1.], 1).is_err());
    }
}
