use anyhow::Result;
use ree::{
    chunk::Chunk,
    events::Events,
    model::{
        self,
        query::{QueryEmbedder, QueryEmbedding},
    },
    retrieval::{self, SearchMode, SearchRequest},
    storage::{Database, Document, Store, WriterLock, migrations, search::Filters},
    util::hash,
};
use serde_json::{Value, json};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

fn vector(x: f32, y: f32) -> Vec<f32> {
    let mut v = vec![0.; 768];
    v[0] = x;
    v[1] = y;
    model::normalize(&mut v).unwrap();
    v
}
struct Fake {
    calls: usize,
    vector: Vec<f32>,
}
impl Default for Fake {
    fn default() -> Self {
        Self {
            calls: 0,
            vector: vector(1., 0.),
        }
    }
}
impl QueryEmbedder for Fake {
    fn embed_query(&mut self, _: &str, _: &mut Events) -> Result<QueryEmbedding> {
        self.calls += 1;
        Ok(QueryEmbedding {
            vector: self.vector.clone(),
            provider: "cpu".into(),
            token_count: 7,
            tokenization_ms: 1,
            initialization_ms: 2,
            embedding_ms: 3,
        })
    }
}
struct OnEmbed<F>(F);
impl<F: FnMut()> QueryEmbedder for OnEmbed<F> {
    fn embed_query(&mut self, _: &str, _: &mut Events) -> Result<QueryEmbedding> {
        (self.0)();
        Fake::default().embed_query("", &mut quiet())
    }
}
fn quiet() -> Events {
    Events::new(true, false, false)
}
fn request(mode: SearchMode) -> SearchRequest {
    SearchRequest {
        query: "needle".into(),
        limit: 10,
        mode,
        root: None,
        media_type: None,
    }
}
struct Fixture {
    _temp: tempfile::TempDir,
    path: PathBuf,
    db: Database,
    run: String,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("ree.db");
        let db = Database::open(&path, true).unwrap();
        db.register_model().unwrap();
        let run = db.start_run("ingest").unwrap();
        Self {
            _temp: temp,
            path,
            db,
            run,
        }
    }
    fn legacy() -> Self {
        // Register the bundled extension, then create genuine v1 DDL separately.
        let mut f = Self::new();
        let path = f._temp.path().join("legacy.db");
        let conn = rusqlite::Connection::open(&path).unwrap();
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
            .unwrap();
        conn.execute_batch(migrations::INITIAL).unwrap();
        f.db = Database { conn };
        f.path = path;
        f.db.register_model().unwrap();
        f.run = f.db.start_run("ingest").unwrap();
        f
    }
    fn put(&mut self, root: &str, uri: &str, text: &str, v: Vec<f32>) -> String {
        self.put_metadata(root, uri, text, v, "text/plain", json!({}))
    }
    fn put_metadata(
        &mut self,
        root: &str,
        uri: &str,
        text: &str,
        v: Vec<f32>,
        media: &str,
        metadata: Value,
    ) -> String {
        let root = self
            .db
            .root(
                "directory",
                root,
                &json!({"root_note":"retained"}),
                &self.run,
            )
            .unwrap();
        let document_id = hash(format!("{root}\0{uri}").as_bytes());
        let d = Document {
            id: document_id.clone(),
            root,
            uri: uri.into(),
            content_hash: hash(text.as_bytes()),
            text_hash: hash(text.as_bytes()),
            media_type: media.into(),
            extractor: "fixture".into(),
            recipe: "test-v1".into(),
            size: text.len() as u64,
            modified_ns: None,
            metadata,
            run: self.run.clone(),
        };
        let c = Chunk {
            text: text.into(),
            token_start: 0,
            token_end: 3,
            byte_start: 0,
            byte_end: text.len(),
            input_ids: vec![0, 42, 2],
        };
        self.db.replace_document(&d, &[c], &[v]).unwrap();
        self.db
            .conn
            .query_row(
                "SELECT id FROM chunks WHERE document_id=?1",
                [document_id],
                |r| r.get(0),
            )
            .unwrap()
    }
    fn search(&self, req: &SearchRequest, fake: &mut dyn QueryEmbedder) -> retrieval::SearchReport {
        let mut reader = Database::open(&self.path, false).unwrap();
        retrieval::search(&mut reader, fake, req, &mut quiet()).unwrap()
    }
    fn integrity(&self) {
        self.db
            .conn
            .execute(
                "INSERT INTO chunk_fts(chunk_fts,rank) VALUES('integrity-check',1)",
                [],
            )
            .unwrap();
        let dangling: i64 = self.db.conn.query_row(
            "SELECT count(*) FROM chunk_search_ids m LEFT JOIN chunks c ON c.id=m.chunk_id WHERE c.id IS NULL",
            [], |r| r.get(0)).unwrap();
        assert_eq!(dangling, 0);
    }
}

#[test]
fn reusable_service_observes_commits_and_bounds_hydration() {
    let mut f = Fixture::new();
    f.put("/a", "a.txt", "needle old", vector(1., 0.));
    let path = f._temp.path().join("config.toml");
    std::fs::write(&path, "").unwrap();
    let config = ree::config::Config::load(&ree::cli::Options {
        config: Some(path),
        db: Some(f.path.clone()),
        ..Default::default()
    })
    .unwrap();
    let mut service = retrieval::service::QueryService::with_embedder(config, Fake::default());
    assert_eq!(
        service
            .search(&request(SearchMode::Semantic), &mut quiet())
            .unwrap()
            .results[0]
            .chunk
            .text,
        "needle old"
    );
    f.put("/a", "a.txt", "needle new", vector(1., 0.));
    assert_eq!(
        service
            .search(&request(SearchMode::Semantic), &mut quiet())
            .unwrap()
            .results[0]
            .chunk
            .text,
        "needle new"
    );
    let key = f.db.active_generation().unwrap();
    f.db.conn.execute("UPDATE models SET recipe_json='{}' WHERE id=(SELECT model_id FROM embedding_generations WHERE id=?1)", [key]).unwrap();
    assert!(
        service
            .search(&request(SearchMode::Semantic), &mut quiet())
            .is_err()
    );
    f.db.conn
        .execute(
            "UPDATE models SET recipe_json=?1",
            [model::recipe().to_string()],
        )
        .unwrap();
    f.put_metadata(
        "/a",
        "a.txt",
        "needle",
        vector(1., 0.),
        "text/plain",
        json!({"oversized":"x".repeat(1024*1024)}),
    );
    let error = service
        .search(&request(SearchMode::Lexical), &mut quiet())
        .unwrap_err();
    assert!(format!("{error:#}").contains("response_limit"));
    // The new transport cap is not retroactively imposed on standalone search.
    assert_eq!(
        f.search(&request(SearchMode::Lexical), &mut Fake::default())
            .results
            .len(),
        1
    );
}

#[test]
fn dense_knn_and_filtered_exact_scan_agree_with_reference() {
    let mut f = Fixture::new();
    let a = f.put("/a", "a.txt", "nearest", vector(1., 0.));
    let b = f.put("/a", "b.txt", "second", vector(1., 1.));
    let c = f.put("/a", "c.txt", "third", vector(0., 1.));
    let mut fake = Fake::default();
    let req = request(SearchMode::Semantic);
    let report = f.search(&req, &mut fake);
    assert_eq!(
        report
            .results
            .iter()
            .map(|r| &r.chunk.chunk_id)
            .collect::<Vec<_>>(),
        [&a, &b, &c]
    );
    for (r, expected) in report.results.iter().zip([0., 1. - 0.5f64.sqrt(), 1.]) {
        assert!((r.distance.unwrap() - expected).abs() < 1e-6);
        assert!(r.bm25.is_none() && r.fusion_score.is_none());
    }
    let filtered = f.search(
        &SearchRequest {
            root: Some("/a".into()),
            ..req
        },
        &mut fake,
    );
    assert_eq!(
        filtered
            .results
            .iter()
            .map(|r| r.distance)
            .collect::<Vec<_>>(),
        report
            .results
            .iter()
            .map(|r| r.distance)
            .collect::<Vec<_>>()
    );
    assert_eq!(fake.calls, 2);
    assert_eq!(report.query_provider.as_deref(), Some("cpu"));
    assert_eq!(report.timings.initialization_ms, 2);
}
#[test]
fn selective_root_and_media_filters_precede_top_k_in_all_modes() {
    let mut f = Fixture::new();
    for i in 0..20 {
        f.put("/outside", &format!("{i}.txt"), "needle", vector(1., 0.));
    }
    f.put("/inside", "wrong-type", "needle", vector(1., 0.));
    let expected = f.put_metadata(
        "/inside",
        "wanted",
        "needle plus surrounding text",
        vector(0., 1.),
        "text/markdown",
        json!({}),
    );
    for mode in [
        SearchMode::Semantic,
        SearchMode::Lexical,
        SearchMode::Hybrid,
    ] {
        let req = SearchRequest {
            mode,
            limit: 1,
            root: Some("/inside".into()),
            media_type: Some("text/markdown".into()),
            ..request(mode)
        };
        let mut fake = Fake::default();
        let report = f.search(&req, &mut fake);
        assert_eq!(report.results.len(), 1);
        assert_eq!(report.results[0].chunk.chunk_id, expected);
        assert_eq!(fake.calls, usize::from(mode.uses_dense()));
        let by_id = f.search(
            &SearchRequest {
                root: Some(hash(b"/inside")),
                ..req.clone()
            },
            &mut Fake::default(),
        );
        assert_eq!(by_id.results[0].chunk.chunk_id, expected);
        for req in [
            SearchRequest {
                root: Some("/absent".into()),
                ..req.clone()
            },
            SearchRequest {
                media_type: Some("text/plain' OR 1=1 --".into()),
                ..req
            },
        ] {
            let mut fake = Fake::default();
            assert!(f.search(&req, &mut fake).results.is_empty());
            assert_eq!(fake.calls, 0);
        }
    }
}
#[test]
fn root_paths_canonicalize_and_unknown_roots_do_not_match() {
    let mut f = Fixture::new();
    let root = f._temp.path().join("source");
    std::fs::create_dir(&root).unwrap();
    f.put(root.to_str().unwrap(), "file", "needle", vector(1., 0.));
    let req = SearchRequest {
        root: Some(root.join(".").to_str().unwrap().into()),
        ..request(SearchMode::Lexical)
    };
    assert_eq!(f.search(&req, &mut Fake::default()).results.len(), 1);
}
#[test]
fn fts_stays_consistent_on_replacement_update_delete_rollback_and_vacuum() {
    let mut f = Fixture::new();
    let old = f.put("/a", "file", "oldterm", vector(1., 0.));
    let id = f.put("/a", "file", "newterm", vector(1., 0.));
    assert_ne!(old, id);
    let mut req = request(SearchMode::Lexical);
    req.query = "oldterm".into();
    assert!(f.search(&req, &mut Fake::default()).results.is_empty());
    req.query = "newterm".into();
    assert_eq!(f.search(&req, &mut Fake::default()).results.len(), 1);
    f.db.conn
        .execute_batch("BEGIN; DELETE FROM source_roots; ROLLBACK;")
        .unwrap();
    assert_eq!(f.search(&req, &mut Fake::default()).results.len(), 1);
    f.integrity();
    let mapping: i64 =
        f.db.conn
            .query_row(
                "SELECT id FROM chunk_search_ids WHERE chunk_id=?1",
                [&id],
                |r| r.get(0),
            )
            .unwrap();
    f.db.conn.execute_batch("VACUUM").unwrap();
    assert_eq!(
        f.db.conn
            .query_row(
                "SELECT id FROM chunk_search_ids WHERE chunk_id=?1",
                [&id],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        mapping
    );
    assert_eq!(f.search(&req, &mut Fake::default()).results.len(), 1);
    f.db.conn
        .execute("UPDATE chunks SET text='updatedterm' WHERE id=?1", [&id])
        .unwrap();
    assert!(f.search(&req, &mut Fake::default()).results.is_empty());
    req.query = "updatedterm".into();
    assert_eq!(f.search(&req, &mut Fake::default()).results.len(), 1);
    f.integrity();
    f.db.remove("/a").unwrap();
    assert!(f.search(&req, &mut Fake::default()).results.is_empty());
    f.integrity();
    assert_eq!(
        f.db.conn
            .query_row("SELECT count(*) FROM chunk_search_ids", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
}
#[test]
fn plain_fts_queries_handle_operators_quotes_unicode_and_punctuation() {
    let mut f = Fixture::new();
    f.put(
        "/a",
        "file",
        "SQLITE_BUSY café こんにちは OR quoted text",
        vector(1., 0.),
    );
    for query in [
        "SQLITE_BUSY",
        "café",
        "こんにちは",
        "OR",
        "\"quoted\"",
        "text:quoted",
        "( )",
        "\"",
        "*",
        "-",
        "NEAR(a,b)",
        "x' OR 1=1 --",
    ] {
        let req = SearchRequest {
            query: query.into(),
            ..request(SearchMode::Lexical)
        };
        let mut fake = Fake::default();
        let _ = f.search(&req, &mut fake);
        assert_eq!(fake.calls, 0);
    }
    let req = SearchRequest {
        query: "unfindable OR unfindable".into(),
        ..request(SearchMode::Lexical)
    };
    // OR is a literal word and matches the document, not a user-supplied operator.
    assert_eq!(f.search(&req, &mut Fake::default()).results.len(), 1);
    let req = SearchRequest {
        query: "\"quoted\"".into(),
        ..request(SearchMode::Lexical)
    };
    assert_eq!(f.search(&req, &mut Fake::default()).results.len(), 1);
}
#[test]
fn hybrid_unions_candidates_and_deduplicates_chunk_ids() {
    let mut f = Fixture::new();
    let common = f.put("/a", "both", "needle", vector(1., 0.));
    f.put("/a", "dense", "otherword", vector(1., 1.));
    let mut req = request(SearchMode::Hybrid);
    req.limit = 1;
    let r = f.search(&req, &mut Fake::default());
    assert_eq!(r.results[0].chunk.chunk_id, common);
    assert!(r.results[0].distance.is_some() && r.results[0].bm25.is_some());
    assert_eq!(r.results[0].fusion_score, Some(2. / 61.));
    assert_eq!(r.candidate_limit, 100);
    assert_eq!(r.dense_candidates, 2);
    assert_eq!(r.lexical_candidates, 1);
    req.query = "absent lexical phrase".into();
    assert_eq!(f.search(&req, &mut Fake::default()).results.len(), 1);
}
#[test]
fn shadow_generation_is_excluded_from_dense_and_lexical_rankings() {
    let mut f = Fixture::new();
    let a = f.put("/a", "a", "needle", vector(0., 1.));
    let b = f.put("/a", "b", "needle more words", vector(1., 0.));
    let building = f.db.rebuild_generation().unwrap();
    f.db.write_vectors(
        building,
        &[a.clone(), b.clone()],
        &[vector(1., 0.), vector(0., 1.)],
    )
    .unwrap();
    let report = f.search(
        &SearchRequest {
            limit: 1,
            ..request(SearchMode::Semantic)
        },
        &mut Fake::default(),
    );
    assert_eq!(report.results[0].chunk.chunk_id, b);
    assert_ne!(report.generation, Some(building));
    let lexical = f.search(&request(SearchMode::Lexical), &mut Fake::default());
    assert_eq!(lexical.results.len(), 2);
}
#[test]
fn activation_during_embedding_uses_new_compatible_generation() {
    let mut f = Fixture::new();
    let a = f.put("/a", "a", "needle", vector(0., 1.));
    let b = f.put("/a", "b", "needle", vector(1., 0.));
    let building = f.db.rebuild_generation().unwrap();
    f.db.write_vectors(building, &[a.clone(), b], &[vector(1., 0.), vector(0., 1.)])
        .unwrap();
    let mut reader = Database::open(&f.path, false).unwrap();
    let mut embedder = OnEmbed(|| f.db.activate(building).unwrap());
    let report = retrieval::search(
        &mut reader,
        &mut embedder,
        &SearchRequest {
            limit: 1,
            ..request(SearchMode::Semantic)
        },
        &mut quiet(),
    )
    .unwrap();
    assert_eq!(report.generation, Some(building));
    assert_eq!(report.results[0].chunk.chunk_id, a);
    f.integrity();
}
#[test]
fn snapshot_keeps_old_text_vectors_and_fts_during_replacement_and_deletion() {
    let mut f = Fixture::new();
    let old = f.put("/a", "file", "oldword", vector(1., 0.));
    let mut reader = Database::open(&f.path, false).unwrap();
    let snapshot = reader.search_snapshot(true).unwrap();
    let new = f.put("/a", "file", "newword", vector(0., 1.));
    assert_eq!(
        snapshot
            .dense(&vector(1., 0.), &Filters::default(), 1)
            .unwrap()[0]
            .chunk_id,
        old
    );
    assert_eq!(
        snapshot
            .lexical("\"oldword\"", &Filters::default(), 1)
            .unwrap()[0]
            .chunk_id,
        old
    );
    assert_eq!(snapshot.hydrate(&old).unwrap().text, "oldword");
    f.db.remove("/a").unwrap();
    assert_eq!(snapshot.hydrate(&old).unwrap().text, "oldword");
    assert!(snapshot.hydrate(&new).is_err());
    drop(snapshot);
    assert!(
        reader
            .search_snapshot(true)
            .unwrap()
            .dense(&vector(1., 0.), &Filters::default(), 1)
            .unwrap()
            .is_empty()
    );
}
#[test]
fn search_does_not_write_or_take_writer_lock_and_returns_locations() {
    let mut f = Fixture::new();
    let id = f.put_metadata(
        "/a",
        "file.pdf",
        "needle on two pages",
        vector(1., 0.),
        "application/pdf",
        json!({
            "pages":[{"page":1,"byte_start":0,"byte_end":6,"ocr":false},
                     {"page":2,"byte_start":7,"byte_end":19,"ocr":true},
                     {"page":3,"byte_start":20,"byte_end":30}],
            "cells":[{"cell":0,"byte_start":0,"byte_end":6}]
        }),
    );
    let _lock = WriterLock::acquire(&f.path).unwrap();
    let changes = f.db.conn.total_changes();
    let report = f.search(&request(SearchMode::Hybrid), &mut Fake::default());
    assert_eq!(f.db.conn.total_changes(), changes);
    assert_eq!(report.results[0].chunk.chunk_id, id);
    assert_eq!(report.results[0].chunk.locations.len(), 3);
    assert_eq!(
        report.results[0].chunk.source_metadata["root_note"],
        "retained"
    );
    let reader = Database::open(&f.path, false).unwrap();
    assert!(reader.conn.execute("DELETE FROM chunks", []).is_err());
    let status: String =
        f.db.conn
            .query_row("SELECT status FROM runs WHERE id=?1", [&f.run], |r| {
                r.get(0)
            })
            .unwrap();
    assert_eq!(status, "running");
}
#[test]
fn validates_before_inference_and_revalidates_after_inference() {
    let mut f = Fixture::new();
    f.put("/a", "file", "needle", vector(1., 0.));
    let mut reader = Database::open(&f.path, false).unwrap();
    let mut fake = Fake::default();
    for req in [
        SearchRequest {
            query: " ".into(),
            ..request(SearchMode::Semantic)
        },
        SearchRequest {
            limit: 0,
            ..request(SearchMode::Semantic)
        },
        SearchRequest {
            limit: 1001,
            ..request(SearchMode::Semantic)
        },
        SearchRequest {
            root: Some("".into()),
            ..request(SearchMode::Semantic)
        },
    ] {
        let e = retrieval::search(&mut reader, &mut fake, &req, &mut quiet()).unwrap_err();
        assert_eq!(ree::error::exit_code(&e), 2);
    }
    assert_eq!(fake.calls, 0);
    f.db.conn
        .execute("UPDATE models SET recipe_json='{}'", [])
        .unwrap();
    assert!(
        retrieval::search(
            &mut reader,
            &mut fake,
            &request(SearchMode::Semantic),
            &mut quiet()
        )
        .is_err()
    );
    assert_eq!(fake.calls, 0);
    // Lexical retrieval does not require a supported model recipe.
    assert_eq!(
        f.search(&request(SearchMode::Lexical), &mut fake)
            .results
            .len(),
        1
    );
    f.db.conn
        .execute(
            "UPDATE models SET recipe_json=?1",
            [model::recipe().to_string()],
        )
        .unwrap();
    let mut embedder = OnEmbed(|| {
        f.db.conn
            .execute("UPDATE models SET recipe_json='{}'", [])
            .unwrap();
    });
    assert!(
        retrieval::search(
            &mut reader,
            &mut embedder,
            &request(SearchMode::Semantic),
            &mut quiet()
        )
        .is_err()
    );
}
#[test]
fn empty_scopes_skip_models_and_invalid_vectors_and_synthetic_databases_fail() {
    let mut f = Fixture::new();
    let mut fake = Fake::default();
    for mode in [
        SearchMode::Semantic,
        SearchMode::Lexical,
        SearchMode::Hybrid,
    ] {
        assert!(f.search(&request(mode), &mut fake).results.is_empty());
    }
    assert_eq!(fake.calls, 0);
    f.put("/a", "file", "needle", vector(1., 0.));
    let mut reader = Database::open(&f.path, false).unwrap();
    for vector in [vec![0.; 768], vec![f32::NAN; 768], vec![1.; 2]] {
        let mut fake = Fake { calls: 0, vector };
        assert!(
            retrieval::search(
                &mut reader,
                &mut fake,
                &request(SearchMode::Semantic),
                &mut quiet()
            )
            .is_err()
        );
    }
    f.db.conn
        .execute(
            "INSERT INTO schema_metadata VALUES('synthetic_vectors','true')",
            [],
        )
        .unwrap();
    for mode in [
        SearchMode::Semantic,
        SearchMode::Lexical,
        SearchMode::Hybrid,
    ] {
        assert!(retrieval::search(&mut reader, &mut fake, &request(mode), &mut quiet()).is_err());
    }
}
#[test]
fn genuine_v1_semantic_read_and_transactional_fts_backfill() {
    let mut f = Fixture::legacy();
    let id = f.put("/a", "file", "needle legacy", vector(1., 0.));
    let before: Vec<u8> =
        f.db.conn
            .query_row("SELECT embedding FROM embeddings LIMIT 1", [], |r| r.get(0))
            .unwrap();
    let report = f.search(&request(SearchMode::Semantic), &mut Fake::default());
    assert_eq!(report.results[0].chunk.chunk_id, id);
    assert_eq!(f.db.schema_version().unwrap(), 1);
    let mut reader = Database::open(&f.path, false).unwrap();
    let err = retrieval::search(
        &mut reader,
        &mut Fake::default(),
        &request(SearchMode::Lexical),
        &mut quiet(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("ree migrate"));
    // Inject a migration conflict after earlier DDL to prove full rollback.
    f.db.conn
        .execute_batch("CREATE TABLE chunk_search_content(conflict);")
        .unwrap();
    assert!(Database::open(&f.path, true).is_err());
    assert_eq!(f.db.schema_version().unwrap(), 1);
    assert_eq!(
        f.db.conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE name='chunk_search_ids'",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        0
    );
    f.db.conn
        .execute_batch("DROP TABLE chunk_search_content;")
        .unwrap();
    f.db = Database::open(&f.path, true).unwrap();
    assert_eq!(f.db.schema_version().unwrap(), 2);
    assert_eq!(
        f.db.conn
            .query_row("SELECT embedding FROM embeddings LIMIT 1", [], |r| r
                .get::<_, Vec<u8>>(0))
            .unwrap(),
        before
    );
    assert_eq!(
        f.search(&request(SearchMode::Lexical), &mut Fake::default())
            .results[0]
            .chunk
            .chunk_id,
        id
    );
    f.integrity();
    let changes = f.db.conn.total_changes();
    let _again = Database::open(&f.path, true).unwrap();
    assert_eq!(f.db.conn.total_changes(), changes);
}
#[test]
fn ties_are_stably_sorted_within_candidates_and_output_obeys_quiet() {
    let mut f = Fixture::new();
    let a = f.put("/a", "a", "needle", vector(1., 0.));
    let b = f.put("/a", "b", "needle", vector(1., 0.));
    let mut ids = vec![a, b];
    ids.sort();
    for mode in [
        SearchMode::Semantic,
        SearchMode::Lexical,
        SearchMode::Hybrid,
    ] {
        let report = f.search(&request(mode), &mut Fake::default());
        assert_eq!(
            report
                .results
                .iter()
                .map(|r| r.chunk.chunk_id.clone())
                .collect::<Vec<_>>(),
            ids
        );
        for is_quiet in [false, true] {
            let output = Arc::new(Mutex::new(Vec::new()));
            struct Sink(Arc<Mutex<Vec<u8>>>);
            impl std::io::Write for Sink {
                fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                    self.0.lock().unwrap().extend(b);
                    Ok(b.len())
                }
                fn flush(&mut self) -> std::io::Result<()> {
                    Ok(())
                }
            }
            let mut events = Events::with_writer(Sink(output.clone()), is_quiet, false);
            report.emit(&mut events).unwrap();
            events.flush().unwrap();
            let bytes = output.lock().unwrap();
            if is_quiet {
                assert!(bytes.is_empty());
                continue;
            }
            let values: Vec<Value> = String::from_utf8_lossy(&bytes)
                .lines()
                .map(|l| serde_json::from_str(l).unwrap())
                .collect();
            assert_eq!(values.len(), 3);
            assert_eq!(values[0]["type"], "search_result");
            assert_eq!(values[0]["rank"], 1);
            assert_eq!(values[2]["type"], "search_completed");
            assert_eq!(values[2]["results"], 2);
            assert!(values[2].get("query").is_none());
        }
    }
}
