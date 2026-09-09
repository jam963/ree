//! Real production CLI, no mocked inference or environment/library overrides.
use anyhow::{Result, ensure};
use assert_cmd::Command;
use ree::{model::device, storage::Database};
use serde_json::Value;
use std::path::Path;
fn invoke(config: &Path, db: &Path, args: &[&str], expected: i32) -> Vec<Value> {
    let mut cmd = Command::cargo_bin("ree").unwrap();
    cmd.args([
        "--config",
        config.to_str().unwrap(),
        "--db",
        db.to_str().unwrap(),
    ])
    .args(args)
    .timeout(std::time::Duration::from_secs(240));
    let output = cmd.assert().code(expected).get_output().stdout.clone();
    String::from_utf8(output)
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect()
}
fn used_cuda(events: &[Value]) {
    assert!(
        events
            .iter()
            .any(|e| e["type"] == "runtime" && e["provider"] == "cuda" && e["precision"] == "fp16"),
        "real CLI did not use CUDA: {events:?}"
    );
    assert!(!events.iter().any(|e| e["type"] == "device_fallback"));
}
#[test]
#[ignore = "real CUDA 13 end-to-end ingestion/sync/rebuild on development host only"]
fn cuda_auto_ingestion_sync_failure_preservation_and_rebuild() -> Result<()> {
    ensure!(
        std::env::var_os("CI").is_none(),
        "never run real GPU qualification in CI"
    );
    ensure!(
        std::fs::read_to_string("/proc/cpuinfo")?.contains("AMD Ryzen 9 8945HS"),
        "development CPU required"
    );
    ensure!(
        device::discover()?
            .iter()
            .any(|g| g.name.contains("RTX 4070 Laptop")),
        "development GPU required"
    );
    let temp = tempfile::tempdir()?;
    let docs = temp.path().join("docs");
    std::fs::create_dir(&docs)?;
    let db = temp.path().join("ree.db");
    let config = temp.path().join("config.toml");
    std::fs::write(&config, "")?;
    std::fs::write(
        docs.join("a.md"),
        "ree stores embeddings locally in SQLite.",
    )?;
    std::fs::write(
        docs.join("b.html"),
        "<article>Local text extraction</article><script>never embed</script>",
    )?;
    std::fs::write(
        docs.join("japanese.txt"),
        "日本語の文章をベクトルに変換します。",
    )?;
    std::fs::write(docs.join("code.rs"), "fn main() { println!(\"hello\"); }")?;
    std::fs::write(
        docs.join("long.txt"),
        "English multilingual documentation 日本語の文章.\n".repeat(150),
    )?;
    let input = docs.to_str().unwrap();
    let first = invoke(&config, &db, &[input, "--device", "auto"], 0);
    used_cuda(&first);
    let conn = Database::open(&db, false)?;
    let first_generation = conn.active_generation()?;
    assert_eq!(conn.status()?["documents"], 5);
    assert!(conn.status()?["chunks"].as_u64().unwrap() > 5);
    let old_japanese:Vec<u8>=conn.conn.query_row("SELECT e.embedding FROM embeddings e JOIN embedding_records r ON r.vector_id=e.rowid JOIN chunks c ON c.id=r.chunk_id JOIN documents d ON d.id=c.document_id WHERE d.uri LIKE '%/japanese.txt'",[],|r|r.get(0))?;
    drop(conn);
    let noop = invoke(&config, &db, &[input, "--device", "auto"], 0);
    assert_eq!(noop.last().unwrap()["unchanged"], 5);
    assert!(!noop.iter().any(|e| e["type"] == "runtime"));
    std::fs::write(docs.join("a.md"), "Changed content is replaced atomically.")?;
    std::fs::remove_file(docs.join("b.html"))?;
    std::fs::rename(docs.join("code.rs"), docs.join("renamed.rs"))?;
    std::fs::write(docs.join("new.cfg"), "chunk_size = 512\noverlap = 64\n")?;
    std::fs::write(docs.join("japanese.txt"), b"\0binary replacement")?;
    let changed = invoke(&config, &db, &[input, "--device", "auto"], 1);
    used_cuda(&changed);
    assert_eq!(changed.last().unwrap()["failed"], 1);
    let conn = Database::open(&db, false)?;
    let preserved:Vec<u8>=conn.conn.query_row("SELECT e.embedding FROM embeddings e JOIN embedding_records r ON r.vector_id=e.rowid JOIN chunks c ON c.id=r.chunk_id JOIN documents d ON d.id=c.document_id WHERE d.uri LIKE '%/japanese.txt'",[],|r|r.get(0))?;
    assert_eq!(preserved, old_japanese);
    let chunks = conn.status()?["chunks"].clone();
    drop(conn);
    let rebuild = invoke(&config, &db, &["rebuild", "--device", "cuda"], 0);
    used_cuda(&rebuild);
    let conn = Database::open(&db, false)?;
    assert_ne!(conn.active_generation()?, first_generation);
    assert_eq!(conn.status()?["embeddings"], chunks);
    assert_eq!(conn.status()?["documents"], 5);
    drop(conn);
    invoke(&config, &db, &["remove", input], 0);
    let conn = Database::open(&db, false)?;
    assert_eq!(conn.status()?["embeddings"], 0);
    assert_eq!(conn.status()?["documents"], 0);
    Ok(())
}
