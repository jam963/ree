use anyhow::{Result, bail};
use ree::{
    chunk::Chunk,
    cli::Options,
    config::Config,
    events::Events,
    model::Engine,
    pipeline,
    storage::{Database, WriterLock},
    util::hash,
};
use serde_json::json;
use std::path::Path;
use tempfile::TempDir;

struct TestEngine {
    calls: usize,
    fail: bool,
}
impl Engine for TestEngine {
    fn chunk(&mut self, text: &str, _: &mut Events) -> Result<Vec<Chunk>> {
        if text.is_empty() {
            return Ok(vec![]);
        }
        Ok(vec![Chunk {
            text: text.into(),
            token_start: 0,
            token_end: 1,
            byte_start: 0,
            byte_end: text.len(),
            input_ids: vec![0, 42, 2],
        }])
    }
    fn embed(&mut self, inputs: &[Vec<i64>], _: &mut Events) -> Result<Vec<Vec<f32>>> {
        self.calls += 1;
        if self.fail {
            bail!("injected inference failure");
        }
        let mut v = vec![0.; 768];
        v[0] = 1.;
        Ok(vec![v; inputs.len()])
    }
}
struct Fixture {
    temp: TempDir,
    config: Config,
    db: Database,
    engine: TestEngine,
    events: Events,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("ree.db");
        let config_path = temp.path().join("config.toml");
        std::fs::write(&config_path, "").unwrap();
        let options = Options {
            db: Some(path.clone()),
            config: Some(config_path),
            ..Default::default()
        };
        let mut config = Config::load(&options).unwrap();
        config.cache = temp.path().join("cache");
        let db = Database::open(&path, true).unwrap();
        Self {
            temp,
            config,
            db,
            engine: TestEngine {
                calls: 0,
                fail: false,
            },
            events: Events::new(true, false, false),
        }
    }
    fn dir(&self, name: &str) -> std::path::PathBuf {
        let p = self.temp.path().join(name);
        std::fs::create_dir_all(&p).unwrap();
        p
    }
    fn ingest(&mut self, path: &Path) -> u8 {
        self.ingest_options(path, &Options::default())
    }
    fn ingest_options(&mut self, path: &Path, options: &Options) -> u8 {
        pipeline::ingest(
            &mut self.db,
            &mut self.engine,
            &mut self.events,
            &self.config,
            options,
            &[path.to_str().unwrap().into()],
        )
        .unwrap()
    }
    fn count(&self, table: &str) -> i64 {
        self.db
            .conn
            .query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))
            .unwrap()
    }
    fn texts(&self) -> Vec<String> {
        self.db
            .conn
            .prepare("SELECT text FROM chunks ORDER BY text")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap()
    }
}

#[test]
fn sync_noop_change_delete_rename_and_isolation() {
    let mut f = Fixture::new();
    let a = f.dir("a");
    let b = f.dir("b");
    std::fs::write(a.join("x.txt"), "original").unwrap();
    std::fs::write(b.join("y.txt"), "unrelated").unwrap();
    assert_eq!(f.ingest(&a), 0);
    assert_eq!(f.ingest(&b), 0);
    assert_eq!(f.count("documents"), 2);
    let calls = f.engine.calls;
    assert_eq!(f.ingest(&a), 0);
    assert_eq!(f.engine.calls, calls);
    std::fs::write(a.join("x.txt"), "updated").unwrap();
    assert_eq!(f.ingest(&a), 0);
    assert_eq!(f.count("embeddings"), 2);
    std::fs::rename(a.join("x.txt"), a.join("renamed.txt")).unwrap();
    assert_eq!(f.ingest(&a), 0);
    assert_eq!(f.count("documents"), 2);
    std::fs::remove_file(a.join("renamed.txt")).unwrap();
    assert_eq!(f.ingest(&a), 0);
    assert_eq!(f.texts(), vec!["unrelated"]);
    assert_eq!(f.count("embeddings"), 1);
}
#[test]
fn extraction_and_inference_failures_preserve_old_vectors() {
    let mut f = Fixture::new();
    let dir = f.dir("docs");
    let path = dir.join("x");
    std::fs::write(&path, "old").unwrap();
    assert_eq!(f.ingest(&dir), 0);
    std::fs::write(&path, b"\0binary").unwrap();
    assert_eq!(f.ingest(&dir), 1);
    assert_eq!(f.texts(), vec!["old"]);
    std::fs::write(&path, "new").unwrap();
    f.engine.fail = true;
    assert_eq!(f.ingest(&dir), 1);
    assert_eq!(f.texts(), vec!["old"]);
    assert_eq!(f.count("embeddings"), 1);
    f.engine.fail = false;
    assert_eq!(f.ingest(&dir), 0);
    assert_eq!(f.texts(), vec!["new"]);
}
#[test]
fn incomplete_walk_does_not_prune() {
    use std::os::unix::fs::symlink;
    let mut f = Fixture::new();
    let dir = f.dir("docs");
    let path = dir.join("x");
    std::fs::write(&path, "keep until confirmed").unwrap();
    f.ingest(&dir);
    std::fs::remove_file(&path).unwrap();
    symlink("missing", dir.join("broken-link")).unwrap();
    assert_eq!(f.ingest(&dir), 1);
    assert_eq!(f.count("documents"), 1);
    std::fs::remove_file(dir.join("broken-link")).unwrap();
    assert_eq!(f.ingest(&dir), 0);
    assert_eq!(f.count("documents"), 0);
}
#[test]
fn fail_fast_prevents_reconciliation() {
    let mut f = Fixture::new();
    let dir = f.dir("docs");
    std::fs::write(dir.join("old"), "preserve").unwrap();
    f.ingest(&dir);
    std::fs::remove_file(dir.join("old")).unwrap();
    std::fs::write(dir.join("bad"), b"\0").unwrap();
    assert_eq!(
        f.ingest_options(
            &dir,
            &Options {
                fail_fast: true,
                ..Default::default()
            }
        ),
        1
    );
    assert_eq!(f.count("documents"), 1);
}
#[test]
fn overlapping_roots_are_independent() {
    let mut f = Fixture::new();
    let parent = f.dir("parent");
    let child = parent.join("child");
    std::fs::create_dir(&child).unwrap();
    std::fs::write(child.join("x"), "shared").unwrap();
    f.ingest(&parent);
    f.ingest(&child);
    assert_eq!(f.count("documents"), 2);
    let root = hash(parent.to_str().unwrap().as_bytes());
    f.db.remove(&root).unwrap();
    assert_eq!(f.count("documents"), 1);
    assert_eq!(f.count("embeddings"), 1);
}
#[test]
fn dotfiles_ignored_files_and_symlink_files_are_attempted() {
    use std::os::unix::fs::symlink;
    let mut f = Fixture::new();
    let dir = f.dir("docs");
    std::fs::write(dir.join(".gitignore"), "ignored.txt").unwrap();
    std::fs::write(dir.join("ignored.txt"), "included").unwrap();
    std::fs::create_dir(dir.join(".hidden")).unwrap();
    std::fs::write(dir.join(".hidden/x"), "hidden").unwrap();
    symlink("ignored.txt", dir.join("link")).unwrap();
    symlink(".", dir.join("cycle")).unwrap();
    assert_eq!(f.ingest(&dir), 0);
    assert_eq!(f.count("documents"), 4);
}
#[test]
fn rebuild_failure_resume_activation_and_direct_vector_sql() {
    let mut f = Fixture::new();
    let dir = f.dir("docs");
    std::fs::write(dir.join("x"), "one").unwrap();
    std::fs::write(dir.join("y"), "two").unwrap();
    f.ingest(&dir);
    let active = f.db.active_generation().unwrap();
    let building = f.db.rebuild_generation().unwrap();
    let pending = f.db.pending_chunks(building, 1).unwrap();
    let mut v = vec![0.; 768];
    v[0] = 1.;
    f.db.write_vectors(building, &[pending[0].0.clone()], &[v.clone()])
        .unwrap();
    assert!(f.db.activate(building).is_err());
    assert_eq!(f.db.active_generation().unwrap(), active);
    f.engine.fail = true;
    assert!(pipeline::rebuild::rebuild(&mut f.db, &mut f.engine, &mut f.events).is_err());
    assert_eq!(f.db.active_generation().unwrap(), active);
    assert_eq!(f.db.pending_chunks(building, 64).unwrap().len(), 1);
    f.engine.fail = false;
    assert_eq!(
        pipeline::rebuild::rebuild(&mut f.db, &mut f.engine, &mut f.events).unwrap(),
        0
    );
    assert_eq!(f.db.active_generation().unwrap(), building);
    assert_eq!(f.count("embeddings"), 2);
    let query = serde_json::to_string(&v).unwrap();
    let text:String=f.db.conn.query_row("SELECT c.text FROM embeddings e JOIN active_embeddings a ON a.vector_id=e.rowid JOIN chunks c ON c.id=a.chunk_id WHERE e.embedding MATCH ?1 AND k=1 ORDER BY distance",[query],|r|r.get(0)).unwrap();
    assert!(["one", "two"].contains(&text.as_str()));
}
#[test]
fn invalid_vector_write_is_atomic() {
    let mut f = Fixture::new();
    let dir = f.dir("docs");
    std::fs::write(dir.join("x"), "valid").unwrap();
    f.ingest(&dir);
    let generation = f.db.rebuild_generation().unwrap();
    let pending = f.db.pending_chunks(generation, 1).unwrap();
    assert!(
        f.db.write_vectors(generation, &[pending[0].0.clone()], &[vec![f32::NAN; 768]])
            .is_err()
    );
    assert_eq!(f.count("embeddings"), 1);
    assert_eq!(f.count("embedding_records"), 1);
}
#[test]
fn writer_lock_rejects_concurrency_and_releases_on_drop() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("ree.db");
    let guard = WriterLock::acquire(&path).unwrap();
    let e = WriterLock::acquire(&path).err().unwrap();
    assert_eq!(ree::error::exit_code(&e), 4);
    drop(guard);
    assert!(WriterLock::acquire(&path).is_ok());
}
#[test]
fn migration_reopen_and_newer_schema_rejection() {
    let f = Fixture::new();
    assert_eq!(f.db.status().unwrap()["schema_version"], 1);
    let run = f.db.start_run("ingest").unwrap();
    let db = Database::open(&f.config.db, true).unwrap();
    let status: String = db
        .conn
        .query_row("SELECT status FROM runs WHERE id=?1", [run], |r| r.get(0))
        .unwrap();
    assert_eq!(status, "interrupted");
    db.conn.pragma_update(None, "user_version", 999).unwrap();
    assert!(Database::open(&f.config.db, true).is_err());
}
#[test]
fn metadata_is_json_and_empty_documents_replace() {
    let mut f = Fixture::new();
    f.config.metadata = json!({"key":"'; DROP TABLE chunks; --"});
    let dir = f.dir("docs");
    let path = dir.join("x");
    std::fs::write(&path, "text").unwrap();
    f.ingest(&dir);
    std::fs::write(path, "").unwrap();
    assert_eq!(f.ingest(&dir), 0);
    assert_eq!(f.count("documents"), 1);
    assert_eq!(f.count("chunks"), 0);
    assert_eq!(f.count("embeddings"), 0);
}
