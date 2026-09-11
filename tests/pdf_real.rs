//! Optional local helper smoke test; generated fixture, no downloaded documents.
use ree::{
    config::PdfConfig,
    extract::{external, pdf},
};

fn fixture() -> Vec<u8> {
    let text = "BT /F1 18 Tf 72 720 Td (Local PDF extraction smoke test) Tj ET\n";
    let objects = [
        "<< /Type /Catalog /Pages 2 0 R >>".into(),
        "<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>".into(),
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 7 0 R >> >> /Contents 4 0 R >>".into(),
        format!("<< /Length {} >>\nstream\n{text}endstream", text.len()),
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 6 0 R >>".into(),
        "<< /Length 0 >>\nstream\nendstream".into(),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".into(),
    ];
    let mut pdf = String::from("%PDF-1.4\n");
    let mut offsets = Vec::new();
    for (i, object) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.push_str(&format!("{} 0 obj\n{object}\nendobj\n", i + 1));
    }
    let start = pdf.len();
    pdf.push_str("xref\n0 8\n0000000000 65535 f \n");
    for offset in offsets {
        pdf.push_str(&format!("{offset:010} 00000 n \n"));
    }
    pdf.push_str(&format!(
        "trailer\n<< /Size 8 /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n"
    ));
    pdf.into_bytes()
}
#[test]
#[ignore = "requires local Poppler helpers; exercises optional Tesseract or missing-tool failure"]
fn text_pages_and_real_bounded_rendering() {
    for tool in ["pdfinfo", "pdftotext", "pdftoppm"] {
        assert!(external::available(tool), "{tool} missing");
    }
    let bytes = fixture();
    let config = PdfConfig {
        ocr: false,
        ..Default::default()
    };
    let extracted = pdf::extract(&bytes, &config).unwrap();
    assert!(
        extracted
            .text
            .starts_with("Local PDF extraction smoke test")
    );
    assert_eq!(extracted.metadata["pages"].as_array().unwrap().len(), 2);
    assert_eq!(extracted.metadata["warnings"].as_array().unwrap().len(), 1);
    assert!(
        extracted.metadata["helper_versions"]["pdftotext"]
            .as_str()
            .unwrap()
            .contains("pdftotext")
    );
    let result = pdf::extract(&bytes, &PdfConfig::default());
    if external::available("tesseract") {
        let result = result.unwrap();
        assert_eq!(result.metadata["ocr_pages"], 1);
        assert!(result.text.starts_with("Local PDF extraction smoke test"));
    } else {
        let error = format!("{:#}", result.unwrap_err());
        assert!(
            error.contains("missing_extractor") && error.contains("tesseract"),
            "{error}"
        );
    }
    let error = pdf::extract(
        &bytes,
        &PdfConfig {
            max_pages: 1,
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("size_limit"));
}
