use anyhow::Result;
use ree::{
    chunk::Chunk, cli::Options, config::Config, events::Events, model::Engine, pipeline,
    storage::Database,
};
use std::os::unix::fs::symlink;
struct EngineProbe {
    batches: Vec<usize>,
}
impl Engine for EngineProbe {
    fn chunk(&mut self, text: &str, _: &mut Events) -> Result<Vec<Chunk>> {
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
        self.batches.push(inputs.len());
        let mut v = vec![0.; 768];
        v[0] = 1.;
        Ok(vec![v; inputs.len()])
    }
}
#[test]
fn tiny_documents_share_inference_batches_and_noop_skips_helpers() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let docs = temp.path().join("docs");
    std::fs::create_dir(&docs)?;
    let config_file = temp.path().join("config.toml");
    std::fs::write(&config_file, "")?;
    let options = Options {
        db: Some(temp.path().join("ree.db")),
        config: Some(config_file),
        ..Default::default()
    };
    let mut config = Config::load(&options)?;
    let mut db = Database::open(&config.db, true)?;
    let mut engine = EngineProbe { batches: vec![] };
    let mut events = Events::new(true, false, false);
    for i in 0..33 {
        std::fs::write(docs.join(format!("{i}.txt")), format!("document {i}"))?;
    }
    assert_eq!(
        pipeline::ingest(
            &mut db,
            &mut engine,
            &mut events,
            &config,
            &options,
            &[docs.to_string_lossy().into()]
        )?,
        0
    );
    assert_eq!(engine.batches, vec![32, 1]);
    let helper = temp.path().join("converter");
    symlink("/usr/bin/printf", &helper)?;
    config.extractors.insert(
        "test".into(),
        ree::config::ExtractorConfig {
            extensions: vec!["custom".into()],
            command: vec![helper.to_string_lossy().into(), "converted".into()],
            ..Default::default()
        },
    );
    let file = temp.path().join("document.custom");
    std::fs::write(&file, "original")?;
    let inputs = [file.to_string_lossy().into()];
    assert_eq!(
        pipeline::ingest(
            &mut db,
            &mut engine,
            &mut events,
            &config,
            &options,
            &inputs
        )?,
        0
    );
    std::fs::remove_file(helper)?;
    let calls = engine.batches.len();
    assert_eq!(
        pipeline::ingest(
            &mut db,
            &mut engine,
            &mut events,
            &config,
            &options,
            &inputs
        )?,
        0
    );
    assert_eq!(engine.batches.len(), calls);
    Ok(())
}
