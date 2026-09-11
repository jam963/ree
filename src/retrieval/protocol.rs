//! Version 1 JSONL framing shared by stdin, Unix sockets and the private engine.
use super::{SearchMode, SearchRequest, service::QueryService};
use crate::{config::Config, events::Events};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::{BufRead, Write};

pub const MAX_FRAME: usize = 128 * 1024;
pub const MAX_RESPONSE: usize = 8 * 1024 * 1024;
pub const MAX_ID: usize = 128;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Frame {
    pub v: u8,
    pub id: String,
    #[serde(flatten)]
    pub operation: Operation,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    Search {
        query: String,
        #[serde(default)]
        mode: SearchMode,
        #[serde(default = "default_limit")]
        limit: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        root: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
    },
    Cancel {
        target: String,
    },
}
fn default_limit() -> usize {
    10
}
pub fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= MAX_ID && !id.chars().any(char::is_control)
}
impl Frame {
    pub fn search(id: String, r: SearchRequest) -> Self {
        Self {
            v: 1,
            id,
            operation: Operation::Search {
                query: r.query,
                mode: r.mode,
                limit: r.limit,
                root: r.root,
                media_type: r.media_type,
            },
        }
    }
    pub fn request(&self) -> Option<SearchRequest> {
        match &self.operation {
            Operation::Search {
                query,
                mode,
                limit,
                root,
                media_type,
            } => Some(SearchRequest {
                query: query.clone(),
                mode: *mode,
                limit: *limit,
                root: root.clone(),
                media_type: media_type.clone(),
            }),
            _ => None,
        }
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(self.v == 1, "unsupported protocol version; expected v=1");
        ensure!(
            valid_id(&self.id),
            "id must be 1..128 bytes without control characters"
        );
        match &self.operation {
            Operation::Search {
                root, media_type, ..
            } => {
                ensure!(
                    [root, media_type]
                        .iter()
                        .all(|s| s.as_ref().is_none_or(|s| s.len() <= 4096)),
                    "filter exceeds 4096 bytes"
                );
                self.request().unwrap().validate()?;
            }
            Operation::Cancel { target } => {
                ensure!(valid_id(target), "invalid cancellation target")
            }
        }
        Ok(())
    }
    pub fn dense(&self) -> bool {
        matches!(self.operation, Operation::Search { mode, .. } if mode.uses_dense())
    }
}
/// Read at most cap bytes including LF; never use an unbounded read_line.
pub fn read_frame(reader: &mut impl BufRead, cap: usize) -> Result<Option<Vec<u8>>> {
    let mut frame = Vec::new();
    loop {
        let bytes = reader.fill_buf()?;
        if bytes.is_empty() {
            ensure!(
                frame.is_empty(),
                "partial final JSONL frame (newline required)"
            );
            return Ok(None);
        }
        let n = bytes
            .iter()
            .position(|&b| b == b'\n')
            .map_or(bytes.len(), |n| n + 1);
        ensure!(
            n <= cap.saturating_sub(frame.len()),
            "JSONL frame exceeds byte limit"
        );
        frame.extend_from_slice(&bytes[..n]);
        reader.consume(n);
        if frame.last() == Some(&b'\n') {
            return Ok(Some(frame));
        }
    }
}
pub fn bytes(value: &impl Serialize) -> Result<Vec<u8>> {
    let mut out = serde_json::to_vec(value)?;
    out.push(b'\n');
    Ok(out)
}
pub fn failure(id: Option<&str>, code: &str, message: &str, exit: u8) -> Value {
    json!({"v":1,"id":id,"type":if id.is_some(){"request_failed"}else{"connection_failed"},
        "error_code":code,"message":message,"exit_code":exit,"retryable":code=="busy"})
}

#[derive(Clone, Default)]
struct Capture(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
impl Write for Capture {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        let mut v = self.0.lock().unwrap();
        if b.len() > MAX_RESPONSE.saturating_sub(v.len()) {
            return Err(std::io::Error::other(
                "response_limit: narrow the query or reduce --limit",
            ));
        }
        v.extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Private, single-owner process. The supervisor sends a frozen Config first.
/// Exiting this child releases the entire CUDA context, not just an ORT session.
pub fn engine_main() -> Result<u8> {
    let input = std::io::stdin();
    let mut input = input.lock();
    let config: Config = serde_json::from_slice(
        &read_frame(&mut input, 1024 * 1024)?.context("missing engine configuration")?,
    )?;
    let mut service = QueryService::new(config)?;
    let mut output = std::io::stdout().lock();
    while let Some(line) = read_frame(&mut input, MAX_FRAME)? {
        let frame: Frame = serde_json::from_slice(&line)?;
        frame.validate()?;
        let request = frame.request().context("engine accepts search only")?;
        let capture = Capture::default();
        let mut events = Events::with_writer(capture.clone(), false, false);
        events.set_request_id(&frame.id);
        let report = service.search(&request, &mut events);
        let result = (|| -> Result<()> {
            if crate::metrics::enabled() {
                events.emit(json!({"type":"stage_metrics","scope":"request_thread","overlapping":true,"stages":crate::metrics::take()}))?;
            }
            report?.emit(&mut events)?;
            events.flush()
        })();
        // Do not publish partial hits followed by success after any output error.
        drop(events);
        let response = match result {
            Ok(()) => std::mem::take(&mut *capture.0.lock().unwrap()),
            Err(e) => {
                let message = format!("{e:#}");
                let exit = crate::error::exit_code(&e);
                bytes(&failure(
                    Some(&frame.id),
                    if message.contains("response_limit") {
                        "response_limit"
                    } else if exit == 2 {
                        "invalid_request"
                    } else {
                        "search_failed"
                    },
                    &message,
                    exit,
                ))?
            }
        };
        output.write_all(&response)?;
        output.flush()?;
    }
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_framing_and_escaped_newlines() {
        let mut input = &b"{\"query\":\"a\\nb\"}\n{}\n"[..];
        assert_eq!(
            read_frame(&mut input, 100).unwrap().unwrap(),
            b"{\"query\":\"a\\nb\"}\n"
        );
        assert_eq!(read_frame(&mut input, 100).unwrap().unwrap(), b"{}\n");
        assert!(read_frame(&mut input, 100).unwrap().is_none());
        assert!(read_frame(&mut &b"{}"[..], 100).is_err());
        assert!(read_frame(&mut &b"abc\n"[..], 3).is_err());
    }
    #[test]
    fn defaults_versions_ids_and_bounds() {
        let f: Frame =
            serde_json::from_str(r#"{"v":1,"id":"a","op":"search","query":"hello"}"#).unwrap();
        f.validate().unwrap();
        assert_eq!(f.request().unwrap().limit, 10);
        for id in ["", "\n", &"x".repeat(129)] {
            assert!(!valid_id(id));
        }
        let mut bad = f.clone();
        bad.v = 2;
        assert!(bad.validate().is_err());
        assert!(
            serde_json::from_str::<Frame>(
                r#"{"v":1,"id":"a","op":"search","query":"hello","db":"/tmp/other"}"#
            )
            .is_err()
        );
    }
    #[test]
    fn capture_is_bounded() {
        let mut c = Capture::default();
        c.write_all(&vec![0; MAX_RESPONSE]).unwrap();
        assert!(c.write_all(b"x").is_err());
    }
}
