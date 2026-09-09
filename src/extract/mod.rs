pub mod external;
pub mod html;
pub mod notebook;
pub mod text;
use crate::config::{Config, ExtractorConfig};
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

pub const VERSION: &str = "ree-extract-v1";
#[derive(Debug)]
pub struct Extracted {
    pub text: String,
    pub media_type: String,
    pub extractor: String,
    pub metadata: Value,
}
fn builtin(extension: &str) -> Option<ExtractorConfig> {
    let command: &[&str] = match extension {
        "pdf" => &["pdftotext", "-layout", "-enc", "UTF-8", "{path}", "-"],
        "docx" | "odt" | "epub" => &["pandoc", "{path}", "-t", "plain"],
        "png" | "jpg" | "jpeg" | "tif" | "tiff" | "bmp" | "webp" => {
            &["tesseract", "{path}", "stdout"]
        }
        _ => return None,
    };
    Some(ExtractorConfig {
        command: command.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    })
}
pub fn extract(
    bytes: &[u8],
    extension: &str,
    config: &Config,
    forced: Option<&str>,
) -> Result<Extracted> {
    let extension = extension.to_ascii_lowercase();
    let custom = if let Some(name) = forced {
        if !matches!(name, "text" | "html" | "notebook") {
            Some((
                name.to_string(),
                config
                    .extractors
                    .get(name)
                    .cloned()
                    .or_else(|| builtin(name))
                    .with_context(|| format!("unknown extractor {name}"))?,
            ))
        } else {
            None
        }
    } else {
        config
            .extractors
            .iter()
            .find(|(_, c)| {
                c.extensions
                    .iter()
                    .any(|e| e.eq_ignore_ascii_case(&extension))
            })
            .map(|(n, c)| (n.clone(), c.clone()))
    };
    if let Some((name, c)) = custom {
        return Ok(Extracted {
            text: external::extract(bytes, &extension, &c)?,
            media_type: "text/plain".into(),
            extractor: format!("external:{name}:{VERSION}"),
            metadata: json!({}),
        });
    }
    if forced.is_none() {
        let hint = if bytes.starts_with(b"%PDF-") {
            "pdf"
        } else {
            &extension
        };
        if let Some(c) = builtin(hint) {
            let text = external::extract(bytes, hint, &c)?;
            if hint == "pdf" && text.trim().is_empty() {
                bail!(
                    "unsupported: PDF has no extractable text; scanned-PDF OCR is not yet supported"
                );
            }
            return Ok(Extracted {
                text,
                media_type: if hint == "pdf" {
                    "application/pdf"
                } else {
                    "application/octet-stream"
                }
                .into(),
                extractor: format!("external:{hint}:{VERSION}"),
                metadata: json!({}),
            });
        }
        if bytes.starts_with(b"PK\x03\x04")
            || bytes.starts_with(b"\x7fELF")
            || bytes.starts_with(b"\x1f\x8b")
            || bytes.starts_with(b"SQLite format 3\0")
        {
            bail!("unsupported: binary/archive format");
        }
    }
    let source = text::decode(bytes)?;
    let format = forced.unwrap_or(&extension);
    let (text, media, extractor, metadata) = match format {
        "html" | "htm" => {
            let (t, title) = html::extract(&source);
            (t, "text/html", "html", json!({"title":title}))
        }
        "ipynb" | "notebook" => {
            let (t, m) = notebook::extract(&source)?;
            (t, "application/x-ipynb+json", "notebook", m)
        }
        "json" => (source, "application/json", "text", json!({})),
        "xml" => (source, "application/xml", "text", json!({})),
        "md" | "markdown" => (source, "text/markdown", "text", json!({})),
        _ => (source, "text/plain", "text", json!({})),
    };
    Ok(Extracted {
        text,
        media_type: media.into(),
        extractor: format!("{extractor}:{VERSION}"),
        metadata,
    })
}
