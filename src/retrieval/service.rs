//! A query-only engine with a fresh read-only database connection per request.
use super::{SearchReport, SearchRequest};
use crate::{
    config::Config,
    events::Events,
    model::{LocalEngine, query::QueryEmbedder},
    storage::Database,
};
use anyhow::{Context, Result};

pub struct QueryService<E = LocalEngine> {
    config: Config,
    embedder: E,
}
impl QueryService {
    pub fn new(mut config: Config) -> Result<Self> {
        config.db = std::path::absolute(&config.db)?;
        config.cache = std::path::absolute(&config.cache)?;
        Ok(Self {
            embedder: LocalEngine::new(config.clone()),
            config,
        })
    }
}
impl<E: QueryEmbedder> QueryService<E> {
    pub fn with_embedder(config: Config, embedder: E) -> Self {
        Self { config, embedder }
    }
    pub fn search(&mut self, request: &SearchRequest, events: &mut Events) -> Result<SearchReport> {
        request.validate()?;
        let mut db = Database::open(&self.config.db, false).with_context(|| {
            format!(
                "open search database {}; ingest documents or run ree migrate first",
                self.config.db.display()
            )
        })?;
        // Limit raw hydrated text/JSON before allocating/decoding it. Serialized
        // protocol output has its own 8-MiB cap. Standalone search stays unchanged.
        super::search_bounded(&mut db, &mut self.embedder, request, events, 1024 * 1024)
    }
}
