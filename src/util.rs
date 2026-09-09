use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{io::Read, path::Path};
pub fn hash(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}
pub fn hash_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        digest.update(&buf[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
pub fn read_limited(reader: impl Read, max: u64) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader.take(max.saturating_add(1)).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= max,
        "size_limit: input exceeds {max} bytes"
    );
    Ok(bytes)
}
pub fn path_string(path: &Path) -> Result<String> {
    Ok(path
        .to_str()
        .context("non-UTF-8 paths are not supported as public identities")?
        .into())
}
pub fn document_id(root: &str, uri: &str) -> String {
    hash(format!("{root}\0{uri}").as_bytes())
}
