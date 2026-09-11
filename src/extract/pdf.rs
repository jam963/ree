//! Text-first PDF extraction with bounded, sequential OCR of textless pages.
//! Helpers operate only on a private snapshot. This is not an OS sandbox.
use super::{Extracted, VERSION, external};
use crate::config::PdfConfig;
use anyhow::{Context, Result, ensure};
use serde_json::json;
use std::{
    os::unix::fs::OpenOptionsExt,
    path::Path,
    time::{Duration, Instant},
};

type Runner = fn(&[String], Option<&Path>, Duration, u64, Option<(&Path, u64)>) -> Result<Vec<u8>>;

pub fn extract(bytes: &[u8], config: &PdfConfig) -> Result<Extracted> {
    let mut result = extract_with(bytes, config, external::run_bounded as Runner)?;
    let mut versions = serde_json::Map::new();
    for program in ["pdfinfo", "pdftotext"].into_iter().chain(
        (result.metadata["ocr_pages"].as_u64().unwrap_or(0) > 0)
            .then_some(["pdftoppm", "tesseract"])
            .into_iter()
            .flatten(),
    ) {
        versions.insert(program.into(), json!(external::version(program)));
    }
    result.metadata["helper_versions"] = json!(versions);
    Ok(result)
}

fn extract_with(
    bytes: &[u8],
    config: &PdfConfig,
    mut runner: impl FnMut(
        &[String],
        Option<&Path>,
        Duration,
        u64,
        Option<(&Path, u64)>,
    ) -> Result<Vec<u8>>,
) -> Result<Extracted> {
    config.validate()?;
    ensure!(
        bytes.len() as u64 <= config.max_temp_bytes,
        "size_limit: PDF snapshot exceeds temporary-space limit"
    );
    let directory = tempfile::tempdir()?;
    let cwd = directory.path();
    std::fs::write(cwd.join("input.pdf"), bytes)?;
    let start = Instant::now();
    let mut run = |args: &[&str], cap: u64| -> Result<Vec<u8>> {
        let remaining = Duration::from_secs(config.timeout_seconds)
            .checked_sub(start.elapsed())
            .filter(|d| !d.is_zero())
            .context("extractor_timeout: PDF workflow deadline exceeded")?;
        runner(
            &args.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            Some(cwd),
            remaining,
            cap,
            Some((cwd, config.max_temp_bytes)),
        )
    };
    let info = run(&["pdfinfo", "-enc", "UTF-8", "input.pdf"], 1024 * 1024)?;
    let pages = page_count(&info)?;
    ensure!(
        pages <= config.max_pages,
        "size_limit: PDF has {pages} pages; limit is {}",
        config.max_pages
    );
    let raw = run(
        &["pdftotext", "-layout", "-enc", "UTF-8", "input.pdf", "-"],
        config.max_output_bytes,
    )?;
    let raw = String::from_utf8(raw).context("PDF text is not UTF-8")?;
    // Poppler emits a form feed after each page, including textless pages.
    // Reject inconsistent output instead of attaching OCR to the wrong page.
    let texts: Vec<&str> = if raw.is_empty() {
        vec![""; pages]
    } else {
        raw.split_terminator('\x0c').collect()
    };
    ensure!(
        texts.len() == pages,
        "PDF page count disagrees with extracted page boundaries"
    );
    let mut text = String::new();
    let mut locations = Vec::new();
    let mut ocr_pages = 0;
    let mut warnings = Vec::new();
    for (i, page) in texts.into_iter().enumerate() {
        let number = i + 1;
        let blank = page.trim().is_empty();
        let ocr = blank && config.ocr;
        let page_text = if ocr {
            let page_number = number.to_string();
            let side = config.max_image_side.to_string();
            run(
                &[
                    "pdftoppm",
                    "-f",
                    &page_number,
                    "-l",
                    &page_number,
                    "-singlefile",
                    "-scale-to",
                    &side,
                    "-png",
                    "input.pdf",
                    "page",
                ],
                16 * 1024,
            )?;
            // Never follow converter-created links or open FIFOs/devices. The
            // converter is finished before the output is inspected and consumed.
            let image = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
                .open(cwd.join("page.png"))?;
            let metadata = image.metadata()?;
            ensure!(
                metadata.is_file() && metadata.len() <= config.max_temp_bytes,
                "size_limit: invalid rendered PDF page"
            );
            drop(image);
            let remaining = config.max_output_bytes.saturating_sub(text.len() as u64);
            let output = run(&["tesseract", "page.png", "stdout"], remaining)?;
            std::fs::remove_file(cwd.join("page.png"))?;
            ocr_pages += 1;
            String::from_utf8(output).context("PDF OCR output is not UTF-8")?
        } else {
            if blank {
                warnings.push(format!("page {number} has no text; OCR is disabled"));
            }
            page.to_string()
        };
        ensure!(
            (text.len() as u64)
                .saturating_add(page_text.len() as u64)
                .saturating_add(1)
                <= config.max_output_bytes,
            "size_limit: combined PDF text exceeds output limit"
        );
        let byte_start = text.len();
        text.push_str(&page_text);
        locations
            .push(json!({"page":number,"byte_start":byte_start,"byte_end":text.len(),"ocr":ocr}));
        text.push('\x0c');
    }
    ensure!(
        !text.trim().is_empty(),
        "unsupported: PDF has no meaningful text after extraction/OCR"
    );
    Ok(Extracted {
        text,
        media_type: "application/pdf".into(),
        extractor: format!("pdf:{VERSION}"),
        metadata: json!({"pages":locations,"ocr_pages":ocr_pages,"warnings":warnings}),
    })
}

fn page_count(info: &[u8]) -> Result<usize> {
    // run_bounded fixes LC_ALL=C so Poppler labels and numbers are stable.
    let info = std::str::from_utf8(info).context("pdfinfo output is not UTF-8")?;
    let pages: usize = info
        .lines()
        .find_map(|line| line.strip_prefix("Pages:"))
        .context("pdfinfo did not report a page count")?
        .trim()
        .parse()?;
    ensure!(pages > 0, "unsupported: PDF contains no pages");
    Ok(pages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    #[test]
    fn mixed_pdf_preserves_text_and_records_unicode_page_offsets() {
        let mut calls = Vec::new();
        let result = extract_with(
            b"%PDF-mock",
            &PdfConfig::default(),
            |args, cwd, timeout, _, disk| {
                calls.push(args.to_vec());
                assert!(timeout <= Duration::from_secs(120));
                assert!(disk.is_some());
                match args[0].as_str() {
                    "pdfinfo" => Ok(b"Pages: 3\n".to_vec()),
                    "pdftotext" => Ok("Hello\n\x0c\x0c日本語\n\x0c".as_bytes().to_vec()),
                    "pdftoppm" => {
                        assert_eq!(&args[1..5], &["-f", "2", "-l", "2"]);
                        std::fs::write(cwd.unwrap().join("page.png"), b"mock image")?;
                        Ok(vec![])
                    }
                    "tesseract" => Ok("Français\n".as_bytes().to_vec()),
                    _ => unreachable!(),
                }
            },
        )
        .unwrap();
        assert_eq!(result.text, "Hello\n\x0cFrançais\n\x0c日本語\n\x0c");
        assert_eq!(result.metadata["ocr_pages"], 1);
        for page in result.metadata["pages"].as_array().unwrap() {
            let start = page["byte_start"].as_u64().unwrap() as usize;
            let end = page["byte_end"].as_u64().unwrap() as usize;
            assert!(result.text.get(start..end).is_some());
        }
        assert_eq!(calls.len(), 4);
    }

    #[test]
    fn page_limit_checked_before_rendering_or_text_extraction() {
        let mut calls = 0;
        let err = extract_with(
            b"PDF",
            &PdfConfig {
                max_pages: 2,
                ..Default::default()
            },
            |_, _, _, _, _| {
                calls += 1;
                Ok(b"Pages: 3\n".to_vec())
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("size_limit"));
        assert_eq!(calls, 1);
    }

    #[test]
    fn disabled_ocr_and_inconsistent_page_boundaries() {
        let config = PdfConfig {
            ocr: false,
            ..Default::default()
        };
        for raw in ["text\x0c\x0c", "text\x0c"] {
            let result = extract_with(b"PDF", &config, |args, _, _, _, _| {
                Ok(if args[0] == "pdfinfo" {
                    b"Pages: 2\n".to_vec()
                } else {
                    raw.as_bytes().to_vec()
                })
            });
            if raw == "text\x0c\x0c" {
                let result = result.unwrap();
                assert_eq!(result.metadata["ocr_pages"], 0);
                assert_eq!(result.metadata["warnings"].as_array().unwrap().len(), 1);
            } else {
                assert!(result.is_err());
            }
        }
    }

    #[test]
    fn rendered_symlinks_rejected_and_workspace_removed_on_failure() {
        let mut workspace = None;
        let result = extract_with(b"PDF", &PdfConfig::default(), |args, cwd, _, _, _| {
            workspace = Some(cwd.unwrap().to_path_buf());
            match args[0].as_str() {
                "pdfinfo" => Ok(b"Pages: 1\n".to_vec()),
                "pdftotext" => Ok(b"\x0c".to_vec()),
                "pdftoppm" => {
                    symlink("/dev/zero", cwd.unwrap().join("page.png"))?;
                    Ok(vec![])
                }
                _ => panic!("must not OCR a symlink"),
            }
        });
        assert!(result.is_err());
        assert!(!workspace.unwrap().exists());
    }

    #[test]
    fn aggregate_output_is_bounded_and_partial_ocr_is_not_returned() {
        let result = extract_with(
            b"PDF",
            &PdfConfig {
                max_output_bytes: 8,
                ..Default::default()
            },
            |args, cwd, _, _, _| match args[0].as_str() {
                "pdfinfo" => Ok(b"Pages: 2\n".to_vec()),
                "pdftotext" => Ok(b"1234\x0c\x0c".to_vec()),
                "pdftoppm" => {
                    std::fs::write(cwd.unwrap().join("page.png"), b"image")?;
                    Ok(vec![])
                }
                "tesseract" => Ok(b"5678".to_vec()),
                _ => unreachable!(),
            },
        );
        assert!(result.unwrap_err().to_string().contains("size_limit"));
    }

    #[test]
    fn rejects_bad_page_counts_and_limits() {
        for info in ["Pages: 0", "Pages: -1", "Pages: NaN", "no pages"] {
            assert!(page_count(info.as_bytes()).is_err());
        }
        assert!(
            PdfConfig {
                max_image_side: u32::MAX,
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }
}
