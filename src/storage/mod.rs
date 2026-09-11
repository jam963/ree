pub mod migrations;
pub mod search;
pub mod sqlite;
use crate::chunk::Chunk;
use anyhow::Result;
pub use sqlite::{Database, Document, WriterLock};

/// The single-writer persistence boundary. Implementations must atomically replace
/// complete documents, never publish partially embedded replacements.
pub trait Store {
    fn replace_document(
        &mut self,
        document: &Document,
        chunks: &[Chunk],
        vectors: &[Vec<f32>],
    ) -> Result<()>;
}
