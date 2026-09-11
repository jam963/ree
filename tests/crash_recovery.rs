//! Real subprocess SIGKILL at SQLite's pre-commit hook; no model required.
use ree::{
    chunk::Chunk,
    storage::{Database, Document, Store, WriterLock},
};
use serde_json::json;
use std::{
    path::Path,
    process::{Child, Command},
    time::{Duration, Instant},
};

fn chunk(text: &str) -> Chunk {
    Chunk {
        text: text.into(),
        token_start: 0,
        token_end: 1,
        byte_start: 0,
        byte_end: text.len(),
        input_ids: vec![0, 42, 2],
    }
}
fn vector(axis: usize) -> Vec<f32> {
    let mut v = vec![0.; 768];
    v[axis] = 1.;
    v
}
fn document(db: &Database, run: &str) -> Document {
    let root = db
        .root("benchmark", "synthetic://crash-test", &json!({}), run)
        .unwrap();
    Document {
        id: "test-document".into(),
        root,
        uri: "synthetic://crash-test/document".into(),
        content_hash: "test".into(),
        text_hash: "test".into(),
        media_type: "text/plain".into(),
        extractor: "test".into(),
        recipe: "test".into(),
        size: 1,
        modified_ns: None,
        metadata: json!({}),
        run: run.into(),
    }
}

// Called only in a disposable subprocess. Block before SQLite's commit so the
// parent kills actual in-flight writes (including sqlite-vec and triggers).
unsafe extern "C" fn block_commit(data: *mut std::ffi::c_void) -> i32 {
    let ready = unsafe { &*(data as *const std::path::PathBuf) };
    if std::fs::write(ready, b"pre-commit").is_err() {
        return 1;
    }
    loop {
        std::thread::sleep(Duration::from_secs(1));
    }
}

#[test]
fn crash_worker() {
    let Some(path) = std::env::var_os("REE_CRASH_TEST_DB") else {
        return;
    };
    let path = std::path::PathBuf::from(path);
    let ready = std::path::PathBuf::from(std::env::var_os("REE_CRASH_TEST_READY").unwrap());
    let operation = std::env::var("REE_CRASH_TEST_OPERATION").unwrap();
    let _lock = WriterLock::acquire(&path).unwrap();
    let mut db = Database::open(&path, true).unwrap();
    let run = db.start_run("crash-test").unwrap();
    let doc = document(&db, &run);
    let generation = if operation == "activate" {
        let g = db.rebuild_generation().unwrap();
        let ids: Vec<_> = db
            .pending_chunks(g, 64)
            .unwrap()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        db.write_vectors(g, &ids, &vec![vector(1); ids.len()])
            .unwrap();
        Some(g)
    } else {
        None
    };
    // The pointer stays live while the callback blocks; the process is killed
    // before the connection or its callback data can be dropped.
    unsafe {
        rusqlite::ffi::sqlite3_commit_hook(
            db.conn.handle(),
            Some(block_commit),
            (&ready as *const std::path::PathBuf).cast_mut().cast(),
        );
    }
    if let Some(g) = generation {
        db.activate(g).unwrap();
    } else {
        db.replace_document(&doc, &[chunk("replacement")], &[vector(1)])
            .unwrap();
    }
    panic!("parent should kill worker at the pre-commit hook");
}

struct Worker(Child);
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn assert_old(db: &Database, active: i64) {
    assert_eq!(db.active_generation().unwrap(), active);
    let text: String = db
        .conn
        .query_row("SELECT text FROM chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(text, "original");
    let n: i64 = db
        .conn
        .query_row("SELECT count(*) FROM active_embeddings", [], |r| r.get(0))
        .unwrap();
    assert_eq!(n, 1);
    let integrity: String = db
        .conn
        .query_row("PRAGMA integrity_check", [], |r| r.get(0))
        .unwrap();
    assert_eq!(integrity, "ok");
    let orphans: i64 = db.conn.query_row("SELECT count(*) FROM embeddings e LEFT JOIN embedding_records r ON r.vector_id=e.rowid WHERE r.vector_id IS NULL", [], |r| r.get(0)).unwrap();
    assert_eq!(orphans, 0);
}
fn crash_case(operation: &str) {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("ree.db");
    let ready = temp.path().join("ready");
    let active = {
        let _lock = WriterLock::acquire(&path).unwrap();
        let mut db = Database::open(&path, true).unwrap();
        db.register_model().unwrap();
        let run = db.start_run("setup").unwrap();
        let doc = document(&db, &run);
        db.replace_document(&doc, &[chunk("original")], &[vector(0)])
            .unwrap();
        db.finish_run(&run, 0, &json!({})).unwrap();
        db.active_generation().unwrap()
    };
    let mut worker = Worker(
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "crash_worker", "--nocapture"])
            .env("REE_CRASH_TEST_DB", &path)
            .env("REE_CRASH_TEST_READY", &ready)
            .env("REE_CRASH_TEST_OPERATION", operation)
            .spawn()
            .unwrap(),
    );
    let start = Instant::now();
    while !Path::new(&ready).exists() {
        assert!(
            worker.0.try_wait().unwrap().is_none(),
            "worker exited before hook"
        );
        assert!(
            start.elapsed() < Duration::from_secs(15),
            "worker never reached commit hook"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    // WAL readers see the old complete generation even while a writer is
    // suspended after all replacement/activation mutations have run.
    assert_old(&Database::open(&path, false).unwrap(), active);
    worker.0.kill().unwrap();
    assert!(!worker.0.wait().unwrap().success());
    let _lock = WriterLock::acquire(&path).unwrap();
    let mut db = Database::open(&path, true).unwrap();
    assert_old(&db, active);
    let interrupted: i64 = db
        .conn
        .query_row(
            "SELECT count(*) FROM runs WHERE status='interrupted'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(interrupted, 1);
    if operation == "activate" {
        let g = db.rebuild_generation().unwrap();
        // Checkpointed shadow vectors survive; only activation rolled back.
        assert!(db.pending_chunks(g, 64).unwrap().is_empty());
        db.activate(g).unwrap();
        assert_eq!(db.active_generation().unwrap(), g);
        let n: i64 = db
            .conn
            .query_row("SELECT count(*) FROM embeddings", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 1);
    } else {
        let run = db.start_run("retry").unwrap();
        let doc = document(&db, &run);
        db.replace_document(&doc, &[chunk("replacement")], &[vector(1)])
            .unwrap();
        assert_eq!(db.status().unwrap()["chunks"], 1);
        assert_eq!(db.status().unwrap()["embeddings"], 1);
    }
}
#[test]
fn killed_document_commit_rolls_back_and_retry_is_idempotent() {
    crash_case("replace");
}
#[test]
fn killed_activation_keeps_old_generation_and_resumes_shadow() {
    crash_case("activate");
}
