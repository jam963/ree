//! Local qualification cache only. A final metadata file is the completion marker.
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufWriter, Read, Write},
    path::Path,
};

const MAX_BYTES: usize = 256 * 1024 * 1024;
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Matrix {
    pub ids: Vec<String>,
    pub sha256: String,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Metadata {
    pub version: u32,
    pub dimensions: usize,
    pub information: Value,
    pub documents: Matrix,
    pub queries: Matrix,
    pub single_documents: Matrix,
    pub document_tokens_sha256: String,
    pub query_tokens_sha256: String,
    pub token_statistics: Value,
    pub document_batch: usize,
    pub query_batch: usize,
}
#[allow(dead_code)]
pub struct Export {
    pub metadata: Metadata,
    pub documents: Vec<Vec<f32>>,
    pub queries: Vec<Vec<f32>>,
    pub single_documents: Vec<Vec<f32>>,
}
fn validate(vector: &[f32], dimensions: usize) -> Result<()> {
    ensure!(
        vector.len() == dimensions && vector.iter().all(|v| v.is_finite()),
        "invalid exported vector"
    );
    let squared = vector.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
    ensure!(
        (squared - 1.).abs() < 0.001,
        "exported vector is not unit normalized"
    );
    Ok(())
}
pub fn bytes(rows: usize, dimensions: usize) -> Result<usize> {
    ensure!(
        (1..=4096).contains(&dimensions),
        "invalid exported dimensions"
    );
    let size = rows
        .checked_mul(dimensions)
        .and_then(|n| n.checked_mul(4))
        .context("matrix size overflow")?;
    ensure!(size <= MAX_BYTES, "matrix exceeds 256 MiB");
    Ok(size)
}
#[allow(dead_code)]
pub fn write_matrix(
    dir: &Path,
    name: &str,
    vectors: &[Vec<f32>],
    dimensions: usize,
) -> Result<String> {
    bytes(vectors.len(), dimensions)?;
    let path = dir.join(name);
    let mut output = BufWriter::new(File::options().write(true).create_new(true).open(path)?);
    let mut digest = Sha256::new();
    for vector in vectors {
        validate(vector, dimensions)?;
        for v in vector {
            let data = v.to_le_bytes();
            output.write_all(&data)?;
            digest.update(data);
        }
    }
    output.flush()?;
    output.get_ref().sync_all()?;
    Ok(format!("{:x}", digest.finalize()))
}
#[allow(dead_code)]
pub fn complete(dir: &Path, metadata: &Metadata) -> Result<()> {
    let mut temp = tempfile::NamedTempFile::new_in(dir)?;
    serde_json::to_writer(&mut temp, metadata)?;
    temp.write_all(b"\n")?;
    temp.as_file().sync_all()?;
    temp.persist_noclobber(dir.join("metadata.json"))?;
    File::open(dir)?.sync_all()?;
    Ok(())
}
fn read_matrix(
    dir: &Path,
    name: &str,
    matrix: &Matrix,
    dimensions: usize,
) -> Result<Vec<Vec<f32>>> {
    let expected = bytes(matrix.ids.len(), dimensions)?;
    let unique: std::collections::BTreeSet<_> = matrix.ids.iter().collect();
    ensure!(
        unique.len() == matrix.ids.len() && matrix.ids.iter().all(|s| !s.is_empty()),
        "invalid matrix IDs"
    );
    let mut file = File::open(dir.join(name))?;
    ensure!(
        file.metadata()?.len() == expected as u64,
        "matrix size does not match declared shape"
    );
    let mut data = Vec::with_capacity(expected);
    (&mut file)
        .take(expected as u64 + 1)
        .read_to_end(&mut data)?;
    ensure!(
        data.len() == expected && ree::util::hash(&data) == matrix.sha256,
        "matrix checksum mismatch"
    );
    let vectors = data
        .chunks_exact(dimensions * 4)
        .map(|row| {
            row.chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for vector in &vectors {
        validate(vector, dimensions)?;
    }
    Ok(vectors)
}
#[allow(dead_code)]
pub fn load(dir: &Path) -> Result<Export> {
    let metadata: Metadata = serde_json::from_slice(&ree::util::read_limited(
        File::open(dir.join("metadata.json"))?,
        16 * 1024 * 1024,
    )?)?;
    ensure!(
        metadata.version == 1 && metadata.query_batch == 1 && metadata.document_batch == 8,
        "unsupported export version/batching policy"
    );
    let documents = read_matrix(
        dir,
        "documents.f32",
        &metadata.documents,
        metadata.dimensions,
    )?;
    let queries = read_matrix(dir, "queries.f32", &metadata.queries, metadata.dimensions)?;
    let single_documents = read_matrix(
        dir,
        "single-documents.f32",
        &metadata.single_documents,
        metadata.dimensions,
    )?;
    Ok(Export {
        metadata,
        documents,
        queries,
        single_documents,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn round_trip_requires_complete_hash_valid_unit_vectors() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load(dir.path()).is_err());
        let matrix = |name: &str, ids: Vec<String>, vectors: &[Vec<f32>]| Matrix {
            ids,
            sha256: write_matrix(dir.path(), name, vectors, 2).unwrap(),
        };
        let metadata = Metadata {
            version: 1,
            dimensions: 2,
            information: serde_json::json!({}),
            documents: matrix("documents.f32", vec!["d".into()], &[vec![1., 0.]]),
            queries: matrix("queries.f32", vec!["q".into()], &[vec![0., 1.]]),
            single_documents: matrix("single-documents.f32", vec![], &[]),
            document_tokens_sha256: "d".into(),
            query_tokens_sha256: "q".into(),
            token_statistics: serde_json::json!({}),
            document_batch: 8,
            query_batch: 1,
        };
        complete(dir.path(), &metadata).unwrap();
        assert_eq!(load(dir.path()).unwrap().documents, vec![vec![1., 0.]]);
        assert!(complete(dir.path(), &metadata).is_err());
        std::fs::write(dir.path().join("documents.f32"), [0; 8]).unwrap();
        assert!(load(dir.path()).is_err());
        assert!(write_matrix(dir.path(), "bad.f32", &[vec![0., 0.]], 2).is_err());
        assert!(write_matrix(dir.path(), "nan.f32", &[vec![f32::NAN, 1.]], 2).is_err());
        assert!(bytes(usize::MAX, 4096).is_err());
    }
}
