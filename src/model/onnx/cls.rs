//! Versioned derivative of the checksum-pinned Arctic graphs, not a generic ONNX
//! editor. Copy all original graph fields except outputs, append a scalar-zero
//! Gather of token_embeddings, and expose only unnormalized [batch,768] f32 CLS.
//! Wire schema: onnx/onnx.proto; a narrow parser preserves unknown fields verbatim.
use crate::{events::Events, model::download, util::hash_file};
use anyhow::{Context, Result, ensure};
use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

pub const OUTPUT: &str = "ree_cls_embedding_v1";
pub const VERSION: &str = "arctic-cls-gather-v1";
// Independently reproduced by benchmarks/derive-cls.py from the pinned parents.
const CPU_SHA: &str = "27b7e290193aa20738c81753b1f82a0c8d44ac03fe4bd9cab019eb52218be366";
const CUDA_SHA: &str = "e02d9d888ac67d028720b904afd49f04c54d9d0bcc8a88c7ae95a0412ab113c8";

fn varint(mut n: u64, out: &mut Vec<u8>) {
    while n >= 128 {
        out.push((n as u8 & 127) | 128);
        n >>= 7;
    }
    out.push(n as u8);
}
fn int(field: u64, n: u64) -> Vec<u8> {
    let mut v = vec![];
    varint(field << 3, &mut v);
    varint(n, &mut v);
    v
}
fn bytes(field: u64, data: &[u8]) -> Vec<u8> {
    let mut v = vec![];
    varint(field << 3 | 2, &mut v);
    varint(data.len() as u64, &mut v);
    v.extend(data);
    v
}
fn text(field: u64, s: &str) -> Vec<u8> {
    bytes(field, s.as_bytes())
}
fn attribute(name: &str, value: u64) -> Vec<u8> {
    [text(1, name), int(3, value), int(20, 2)].concat()
}
fn additions() -> Vec<u8> {
    let constant = [
        text(2, "ree_cls_index_v1"),
        text(3, "ree/cls-index-v1"),
        text(4, "Constant"),
        bytes(5, &attribute("value_int", 0)),
    ]
    .concat();
    let gather = [
        text(1, "token_embeddings"),
        text(1, "ree_cls_index_v1"),
        text(2, OUTPUT),
        text(3, "ree/cls-gather-v1"),
        text(4, "Gather"),
        bytes(5, &attribute("axis", 1)),
    ]
    .concat();
    let shape = [bytes(1, &text(2, "batch")), bytes(1, &int(1, 768))].concat();
    let tensor = [int(1, 1), bytes(2, &shape)].concat();
    let output = [text(1, OUTPUT), bytes(2, &bytes(1, &tensor))].concat();
    [bytes(1, &constant), bytes(1, &gather), bytes(12, &output)].concat()
}
#[derive(Clone, Copy)]
struct Field {
    number: u64,
    start: u64,
    data: u64,
    end: u64,
}
fn read_varint(r: &mut (impl Read + Seek), end: u64) -> Result<u64> {
    let mut value = 0;
    for shift in (0..70).step_by(7) {
        ensure!(r.stream_position()? < end, "truncated protobuf varint");
        let mut b = [0];
        r.read_exact(&mut b)?;
        ensure!(shift != 63 || b[0] <= 1, "protobuf varint overflow");
        value |= ((b[0] & 127) as u64) << shift;
        if b[0] < 128 {
            return Ok(value);
        }
    }
    anyhow::bail!("overlong protobuf varint")
}
fn fields(r: &mut (impl Read + Seek), start: u64, end: u64) -> Result<Vec<Field>> {
    r.seek(SeekFrom::Start(start))?;
    let mut result = vec![];
    while r.stream_position()? < end {
        ensure!(result.len() < 100_000, "protobuf field limit");
        let start = r.stream_position()?;
        let key = read_varint(r, end)?;
        ensure!(key >> 3 > 0, "invalid protobuf field zero");
        let size = match key & 7 {
            0 => {
                read_varint(r, end)?;
                0
            }
            1 => 8,
            2 => read_varint(r, end)?,
            5 => 4,
            _ => anyhow::bail!("unsupported protobuf wire type"),
        };
        let data = r.stream_position()?;
        let stop = data.checked_add(size).context("protobuf size overflow")?;
        ensure!(stop <= end, "truncated protobuf field");
        r.seek(SeekFrom::Start(stop))?;
        result.push(Field {
            number: key >> 3,
            start,
            data,
            end: stop,
        });
    }
    Ok(result)
}
fn copy(r: &mut (impl Read + Seek), w: &mut impl Write, start: u64, end: u64) -> Result<()> {
    r.seek(SeekFrom::Start(start))?;
    ensure!(
        std::io::copy(&mut r.take(end - start), w)? == end - start,
        "truncated graph during copy"
    );
    Ok(())
}
fn rewrite(r: &mut (impl Read + Seek), w: &mut impl Write, size: u64) -> Result<()> {
    let root = fields(r, 0, size)?;
    let graphs: Vec<_> = root.iter().filter(|f| f.number == 7).collect();
    ensure!(graphs.len() == 1, "expected one model graph");
    let graph = *graphs[0];
    let contents = fields(r, graph.data, graph.end)?;
    let outputs: Vec<_> = contents.iter().filter(|f| f.number == 12).collect();
    ensure!(outputs.len() == 2, "expected pinned Arctic's two outputs");
    let extra = additions();
    let length = graph.end - graph.data - outputs.iter().map(|f| f.end - f.start).sum::<u64>()
        + extra.len() as u64;
    for f in root {
        if f.number != 7 {
            copy(r, w, f.start, f.end)?;
            continue;
        }
        let mut header = vec![];
        varint(7 << 3 | 2, &mut header);
        varint(length, &mut header);
        w.write_all(&header)?;
        for f in &contents {
            if f.number != 12 {
                copy(r, w, f.start, f.end)?;
            }
        }
        w.write_all(&extra)?;
    }
    Ok(())
}
/// The parent must be freshly verified by ensure_artifact before calling here.
/// Cached derivatives are independently checked on every session creation.
pub fn ensure_artifact(
    parent: &Path,
    cache: &Path,
    artifact: download::Artifact,
    events: &mut Events,
) -> Result<PathBuf> {
    let expected = match artifact.sha256 {
        s if s == download::CPU.sha256 => CPU_SHA,
        s if s == download::CUDA.sha256 => CUDA_SHA,
        _ => anyhow::bail!("CLS derivative requires a pinned Arctic parent"),
    };
    let dir = cache.join("derived").join(VERSION).join(artifact.sha256);
    std::fs::create_dir_all(&dir)?;
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.join("build.lock"))?;
    fs2::FileExt::lock_exclusive(&lock)?;
    let path = dir.join("model.onnx");
    if path.is_file() && hash_file(&path)? == expected {
        events.emit(serde_json::json!({"type":"runtime_artifact","transform":VERSION,"parent_sha256":artifact.sha256,"sha256":expected}))?;
        return Ok(path);
    }
    let _span = crate::metrics::Span::new("cls_derivative_build");
    // Recheck the parent at the transformation trust boundary, independently of
    // factory callers; mutation during copy also fails the fixed output hash.
    ensure!(
        download::verified(parent, artifact)?,
        "CLS parent checksum mismatch"
    );
    let file = File::open(parent)?;
    let size = file.metadata()?.len();
    let mut temp = tempfile::NamedTempFile::new_in(&dir)?;
    rewrite(&mut BufReader::new(file), temp.as_file_mut(), size)?;
    temp.flush()?;
    temp.as_file().sync_all()?;
    let actual = hash_file(temp.path())?;
    ensure!(actual == expected, "CLS derivative hash mismatch: {actual}");
    temp.persist(&path).map_err(|e| e.error)?;
    let manifest = serde_json::json!({"transform":VERSION,"parent_sha256":artifact.sha256,"sha256":expected,"output":OUTPUT,"normalization":"unchanged_host_l2","dimensions":768});
    let mut temp = tempfile::NamedTempFile::new_in(&dir)?;
    serde_json::to_writer(temp.as_file_mut(), &manifest)?;
    temp.as_file().sync_all()?;
    temp.persist(dir.join("manifest.json"))
        .map_err(|e| e.error)?;
    File::open(&dir)?.sync_all()?;
    events.emit(serde_json::json!({"type":"runtime_artifact","transform":VERSION,"parent_sha256":artifact.sha256,"sha256":expected}))?;
    Ok(path)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rewriting_preserves_other_fields_and_has_one_new_output() {
        let original = [
            text(2, "fixture"),
            bytes(
                7,
                &[text(2, "graph"), bytes(12, b"one"), bytes(12, b"two")].concat(),
            ),
            int(1, 8),
        ]
        .concat();
        let mut result = vec![];
        rewrite(
            &mut std::io::Cursor::new(&original),
            &mut result,
            original.len() as u64,
        )
        .unwrap();
        let mut r = std::io::Cursor::new(&result);
        let root = fields(&mut r, 0, result.len() as u64).unwrap();
        assert_eq!(root.iter().map(|f| f.number).collect::<Vec<_>>(), [2, 7, 1]);
        let g = root[1];
        let fields = fields(&mut r, g.data, g.end).unwrap();
        assert_eq!(fields.iter().filter(|f| f.number == 12).count(), 1);
        assert_eq!(fields.iter().filter(|f| f.number == 1).count(), 2);
        assert!(result.windows(OUTPUT.len()).any(|s| s == OUTPUT.as_bytes()));
    }
    #[test]
    fn malformed_graphs_fail_closed() {
        for data in [vec![0], vec![0x3a, 0x7f], vec![0x80; 12], bytes(7, b"")] {
            assert!(
                rewrite(
                    &mut std::io::Cursor::new(&data),
                    &mut vec![],
                    data.len() as u64
                )
                .is_err()
            );
        }
    }
}
