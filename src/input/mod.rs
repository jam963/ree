pub mod git;
pub mod url;
use crate::{
    config::Config,
    util::{hash, path_string, read_limited},
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Stdin,
    Url,
    Git,
    Glob,
    Local,
}
pub fn classify(input: &str) -> Kind {
    if input == "-" {
        Kind::Stdin
    } else if input
        .get(..7)
        .is_some_and(|s| s.eq_ignore_ascii_case("http://"))
        || input
            .get(..8)
            .is_some_and(|s| s.eq_ignore_ascii_case("https://"))
    {
        if url::normalize(input).is_ok_and(|u| u.path().ends_with(".git")) {
            Kind::Git
        } else {
            Kind::Url
        }
    } else if Path::new(input).exists() {
        Kind::Local
    } else if input.contains(['*', '?', '[']) {
        Kind::Glob
    } else {
        Kind::Local
    }
}
#[derive(Debug)]
pub enum Content {
    File(PathBuf),
    Bytes(Vec<u8>, String),
}
#[derive(Debug)]
pub struct Item {
    pub uri: String,
    pub content: Content,
    pub metadata: Value,
}
#[derive(Debug)]
pub enum RootContent {
    Directory { path: PathBuf, remote: bool },
    One(Item),
}
#[derive(Debug)]
pub struct Root {
    pub identity: String,
    pub kind: &'static str,
    pub content: RootContent,
    pub metadata: Value,
    pub lease: Option<crate::storage::WriterLock>,
}

pub fn prepare(input: &str, source: Option<&str>, config: &Config) -> Result<Root> {
    match classify(input) {
        Kind::Stdin => {
            let bytes = read_limited(std::io::stdin().lock(), config.max_file_size)?;
            let identity = source.map_or_else(
                || format!("stdin://sha256/{}", hash(&bytes)),
                |s| format!("stdin://source/{s}"),
            );
            Ok(Root {
                identity: identity.clone(),
                kind: "stdin",
                lease: None,
                content: RootContent::One(Item {
                    uri: identity,
                    content: Content::Bytes(bytes, "txt".into()),
                    metadata: json!({}),
                }),
                metadata: json!({}),
            })
        }
        Kind::Url => {
            let identity = url::normalize(input)?.to_string();
            let response = url::fetch(&identity, config)?;
            Ok(Root {
                identity: identity.clone(),
                kind: "url",
                lease: None,
                content: RootContent::One(Item {
                    uri: identity,
                    content: Content::Bytes(response.bytes, response.extension),
                    metadata: json!({"final_url":response.final_url}),
                }),
                metadata: json!({}),
            })
        }
        Kind::Git => {
            let identity = url::normalize(input)?.to_string();
            let (path, metadata, lease) = git::checkout(&identity, config)?;
            Ok(Root {
                identity,
                kind: "git",
                lease: Some(lease),
                content: RootContent::Directory { path, remote: true },
                metadata,
            })
        }
        Kind::Local => {
            let path = Path::new(input)
                .canonicalize()
                .with_context(|| format!("cannot access {input}"))?;
            let metadata = path.metadata()?;
            let identity = path_string(&path)?;
            if metadata.is_dir() {
                Ok(Root {
                    identity,
                    kind: "directory",
                    lease: None,
                    content: RootContent::Directory {
                        path,
                        remote: false,
                    },
                    metadata: json!({}),
                })
            } else {
                ensure!(
                    metadata.is_file(),
                    "unsupported: input is not a regular file"
                );
                Ok(Root {
                    identity: identity.clone(),
                    kind: "file",
                    lease: None,
                    content: RootContent::One(Item {
                        uri: identity,
                        content: Content::File(path),
                        metadata: json!({}),
                    }),
                    metadata: json!({}),
                })
            }
        }
        Kind::Glob => anyhow::bail!("globs must be expanded before preparation"),
    }
}

/// Streams directory entries, including dotfiles and ignored directories. A
/// traversal error is surfaced so the caller can disable deletion reconciliation.
pub fn discover(root: &Root, mut visit: impl FnMut(Result<Item>) -> bool) {
    let RootContent::Directory { path, remote } = &root.content else {
        return;
    };
    let iter = walkdir::WalkDir::new(path)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| !(*remote && e.file_name() == ".git"));
    for entry in iter {
        let item = (|| -> Result<Option<Item>> {
            let e = entry?;
            if e.file_type().is_dir() {
                return Ok(None);
            }
            if e.file_type().is_symlink() {
                let target = e.path().canonicalize()?;
                if target.is_dir() {
                    return Ok(None);
                }
                ensure!(
                    !remote || target.starts_with(path),
                    "unsafe_symlink: remote checkout points outside repository"
                );
            }
            let metadata = std::fs::metadata(e.path())?;
            ensure!(
                metadata.is_file(),
                "unsupported: non-regular file {}",
                e.path().display()
            );
            let uri = if *remote {
                format!(
                    "{}#{}",
                    root.identity,
                    path_string(e.path().strip_prefix(path)?)?
                )
            } else {
                path_string(e.path())?
            };
            Ok(Some(Item {
                uri,
                content: Content::File(e.path().to_path_buf()),
                metadata: root.metadata.clone(),
            }))
        })();
        match item {
            Ok(Some(item)) => {
                if !visit(Ok(item)) {
                    break;
                }
            }
            Err(e) => {
                if !visit(Err(e)) {
                    break;
                }
            }
            Ok(None) => {}
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn kinds() {
        assert_eq!(classify("-"), Kind::Stdin);
        assert_eq!(classify("https://example.com/r.git?x"), Kind::Git);
        assert_eq!(classify("https://example.com/?q=*"), Kind::Url);
        assert_eq!(classify("src/**/*.rs"), Kind::Glob);
    }
}
