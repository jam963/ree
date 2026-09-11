use assert_cmd::Command;
use serde_json::Value;
use tempfile::TempDir;
fn command(temp: &TempDir) -> Command {
    let mut cmd = Command::cargo_bin("ree").unwrap();
    cmd.env("HOME", temp.path())
        .env("XDG_CONFIG_HOME", temp.path().join("config"))
        .env("XDG_DATA_HOME", temp.path().join("data"))
        .env("XDG_CACHE_HOME", temp.path().join("cache"));
    for name in [
        "REE_DB",
        "REE_CONFIG",
        "REE_DEVICE",
        "REE_BATCH_SIZE",
        "REE_GPU_MEMORY_FRACTION",
        "REE_CHUNK_SIZE",
        "REE_OVERLAP",
        "REE_MAX_FILE_SIZE",
    ] {
        cmd.env_remove(name);
    }
    cmd
}
#[test]
fn invalid_args_are_json_and_code_two() {
    let t = tempfile::tempdir().unwrap();
    for args in [
        vec!["--overlap", "999", "x"],
        vec!["--device", "bogus", "x"],
        vec!["--batch-size", "0", "x"],
        vec!["--unknown"],
        vec![],
        vec!["-", "--source", "x", "other"],
        vec!["search"],
        vec!["search", " "],
        vec!["search", "needle", "--limit", "0"],
        vec!["search", "needle", "--limit", "1001"],
        vec!["search", "needle", "--limit", "-1"],
        vec!["search", "needle", "--mode", "bogus"],
        vec!["search", "needle", "--root", ""],
        vec!["file.txt", "search", "needle"],
    ] {
        let output = command(&t)
            .args(args)
            .assert()
            .code(2)
            .get_output()
            .stdout
            .clone();
        let event: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(event["type"], "fatal");
    }
}
#[test]
fn empty_stdin_dedup_named_source_and_maintenance_offline() {
    let t = tempfile::tempdir().unwrap();
    command(&t).arg("-").write_stdin("").assert().success();
    command(&t).arg("-").write_stdin("").assert().success();
    let output = command(&t)
        .arg("status")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let status: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(status["documents"], 1);
    assert_eq!(status["chunks"], 0);
    command(&t)
        .args(["-", "--source", "reports/current"])
        .write_stdin("")
        .assert()
        .success();
    command(&t)
        .args(["-", "--source", "reports/current"])
        .write_stdin("\n\t")
        .assert()
        .success();
    let output = command(&t)
        .arg("sources")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    assert_eq!(String::from_utf8(output).unwrap().lines().count(), 2);
    command(&t).arg("rebuild").assert().success();
    command(&t)
        .args(["remove", "stdin://source/reports/current"])
        .assert()
        .success();
    command(&t)
        .args(["status", "--quiet"])
        .assert()
        .success()
        .stdout("");
    assert!(!t.path().join("cache/ree/models").exists());
}
#[test]
fn global_options_before_subcommands_are_not_ingestion_inputs() {
    let t = tempfile::tempdir().unwrap();
    let db = t.path().join("custom.db");
    let db = db.to_str().unwrap();
    command(&t)
        .args(["--db", db, "-"])
        .write_stdin("")
        .assert()
        .success();
    for subcommand in ["status", "sources", "rebuild"] {
        let output = command(&t)
            .args(["--db", db, subcommand])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        for line in String::from_utf8(output).unwrap().lines() {
            let event: Value = serde_json::from_str(line).unwrap();
            assert!(
                event.get("inputs").is_none(),
                "maintenance parsed as ingestion: {event}"
            );
        }
    }
    command(&t)
        .args(["--db", db, "file.txt", "status"])
        .assert()
        .code(2);
}
#[test]
fn synthetic_database_cannot_be_mixed_with_real_embeddings() {
    let t = tempfile::tempdir().unwrap();
    let path = t.path().join("synthetic.db");
    let db = ree::storage::Database::open(&path, true).unwrap();
    db.conn
        .execute(
            "INSERT INTO schema_metadata VALUES('synthetic_vectors','true')",
            [],
        )
        .unwrap();
    drop(db);
    for operation in ["-", "rebuild"] {
        let output = command(&t)
            .args(["--db", path.to_str().unwrap(), operation])
            .write_stdin("")
            .assert()
            .code(3)
            .get_output()
            .stdout
            .clone();
        let event: Value = serde_json::from_slice(&output).unwrap();
        assert!(
            event["message"]
                .as_str()
                .unwrap()
                .contains("synthetic benchmark database")
        );
    }
    command(&t)
        .args(["--db", path.to_str().unwrap(), "status"])
        .assert()
        .success();
    assert!(!t.path().join("cache/ree/models").exists());
}
#[test]
fn search_and_migrate_are_offline_on_empty_scopes_and_missing_db_is_not_created() {
    let t = tempfile::tempdir().unwrap();
    command(&t).args(["search", "needle"]).assert().code(3);
    assert!(!t.path().join("data/ree/ree.db").exists());
    let output = command(&t)
        .arg("migrate")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let event: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(event["type"], "migrated");
    assert_eq!(event["schema_version"], 2);
    for mode in ["semantic", "lexical", "hybrid"] {
        let output = command(&t)
            .args(["search", "needle", "--mode", mode])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let event: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(event["type"], "search_completed");
        assert_eq!(event["results"], 0);
        assert!(event["query_provider"].is_null());
    }
    command(&t)
        .args(["search", "needle", "--quiet"])
        .assert()
        .success()
        .stdout("");
    assert!(!t.path().join("cache/ree/models").exists());
    // A new command name must not prevent explicit ingestion of that filename.
    std::fs::write(t.path().join("search"), "").unwrap();
    command(&t)
        .current_dir(t.path())
        .arg("./search")
        .assert()
        .success();
    assert!(!t.path().join("cache/ree/models").exists());
}

#[test]
fn lexical_cli_returns_stored_text_without_writer_lock_or_models() {
    use ree::{
        chunk::Chunk,
        storage::{Database, Document, Store, WriterLock},
    };
    use serde_json::json;
    let t = tempfile::tempdir().unwrap();
    let path = t.path().join("ree.db");
    let mut db = Database::open(&path, true).unwrap();
    db.register_model().unwrap();
    let run = db.start_run("fixture").unwrap();
    let root = db.root("directory", "/offline", &json!({}), &run).unwrap();
    let text = "SQLITE_BUSY writer lock";
    let document = Document {
        id: "document".into(),
        root,
        uri: "/offline/file.txt".into(),
        content_hash: "fixture".into(),
        text_hash: "fixture".into(),
        media_type: "text/plain".into(),
        extractor: "fixture".into(),
        recipe: "fixture".into(),
        size: text.len() as u64,
        modified_ns: None,
        metadata: json!({}),
        run,
    };
    let chunk = Chunk {
        text: text.into(),
        token_start: 0,
        token_end: 4,
        byte_start: 0,
        byte_end: text.len(),
        input_ids: vec![0, 42, 2],
    };
    let mut vector = vec![0.; 768];
    vector[0] = 1.;
    db.replace_document(&document, &[chunk], &[vector]).unwrap();
    let _lock = WriterLock::acquire(&path).unwrap();
    let output = command(&t)
        .args([
            "--db",
            path.to_str().unwrap(),
            "search",
            "SQLITE_BUSY",
            "--mode",
            "lexical",
            "--root",
            "/offline",
            "--limit",
            "1",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let events: Vec<Value> = String::from_utf8(output)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0]["type"], "search_result");
    assert_eq!(events[0]["text"], text);
    assert!(events[0]["bm25"].as_f64().unwrap() < 0.);
    assert!(events[0]["distance"].is_null());
    assert_eq!(events[1]["results"], 1);
    assert!(!t.path().join("cache/ree/models").exists());
    command(&t)
        .args(["--db", path.to_str().unwrap(), "migrate"])
        .assert()
        .code(4);
}

#[test]
fn missing_inputs_return_partial_not_success() {
    let t = tempfile::tempdir().unwrap();
    command(&t)
        .args(["absent-file", "--quiet"])
        .assert()
        .code(1);
}
