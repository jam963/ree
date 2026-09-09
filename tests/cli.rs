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
fn missing_inputs_return_partial_not_success() {
    let t = tempfile::tempdir().unwrap();
    command(&t)
        .args(["absent-file", "--quiet"])
        .assert()
        .code(1);
}
