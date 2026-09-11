use crate::{
    events::Events,
    model::{self, Engine},
    storage::Database,
};
use anyhow::{Result, ensure};
use serde_json::json;
use std::time::Instant;

pub fn rebuild(db: &mut Database, engine: &mut dyn Engine, events: &mut Events) -> Result<u8> {
    db.register_model()?;
    let tokenizer:String=db.conn.query_row("SELECT m.tokenizer_revision FROM models m JOIN embedding_generations g ON g.model_id=m.id WHERE g.status='active'",[],|r|r.get(0))?;
    ensure!(
        tokenizer == model::REVISION,
        "tokenizer revision changed: rebuild cannot rechunk; explicitly reingest original sources with --force"
    );
    let run = db.start_run("rebuild")?;
    let generation = db.rebuild_generation()?;
    let start = Instant::now();
    let mut count = 0;
    events.emit(
        json!({"type":"started","run_id":run,"operation":"rebuild","generation":generation}),
    )?;
    events.flush()?;
    let mut cursor = String::new();
    loop {
        let chunks = db.pending_chunks_after(generation, &cursor, 64)?;
        if chunks.is_empty() {
            break;
        }
        let (ids, inputs): (Vec<_>, Vec<_>) = chunks.into_iter().unzip();
        let vectors = match engine.embed(&inputs, events) {
            Ok(v) => v,
            Err(e) => {
                db.failure(&run, "rebuild", "inference_failed", &format!("{e:#}"))?;
                db.finish_run(
                    &run,
                    1,
                    &json!({"embedded":count,"generation":generation,"resumable":true}),
                )?;
                return Err(e);
            }
        };
        db.write_vectors(generation, &ids, &vectors)?;
        cursor = ids.last().unwrap().clone();
        count += ids.len();
        if events.verbose {
            events.emit(
                json!({"type":"rebuild_checkpoint","generation":generation,"embedded":count}),
            )?;
        }
    }
    db.activate(generation)?;
    let result = json!({"type":"completed","run_id":run,"operation":"rebuild","generation":generation,"chunks":count,"elapsed_ms":start.elapsed().as_millis(),"runtime_metrics":engine.metrics()});
    db.finish_run(&run, 0, &result)?;
    events.emit(result)?;
    Ok(0)
}
