use anyhow::{Result, ensure};
pub fn decode(bytes: &[u8]) -> Result<String> {
    if bytes.is_empty() {
        return Ok(String::new());
    }
    let text = if let Some((encoding, skip)) = encoding_rs::Encoding::for_bom(bytes) {
        let (text, errors) = encoding.decode_without_bom_handling(&bytes[skip..]);
        ensure!(!errors, "invalid text encoding");
        text.into_owned()
    } else {
        ensure!(
            !bytes.iter().take(8192).any(|&b| b == 0),
            "binary_input: NUL bytes detected"
        );
        if let Ok(s) = std::str::from_utf8(bytes) {
            s.into()
        } else {
            let mut detector = chardetng::EncodingDetector::new();
            detector.feed(bytes, true);
            let encoding = detector.guess(None, true);
            let (text, errors) = encoding.decode_without_bom_handling(bytes);
            ensure!(!errors, "invalid text encoding");
            text.into_owned()
        }
    };
    let total = text.chars().count();
    let controls = text
        .chars()
        .filter(|c| c.is_control() && !matches!(c, '\n' | '\r' | '\t' | '\u{c}'))
        .count();
    ensure!(
        total == 0 || controls * 100 <= total,
        "binary_input: excessive control characters"
    );
    Ok(text)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unicode_and_boms() {
        assert_eq!(decode("日本語 🌍".as_bytes()).unwrap(), "日本語 🌍");
        assert_eq!(decode(&[255, 254, 65, 0, 66, 0]).unwrap(), "AB");
        assert_eq!(decode(b"\xef\xbb\xbfhello").unwrap(), "hello");
    }
    #[test]
    fn binary() {
        assert!(decode(b"ELF\0\x01\x02").is_err());
        assert!(decode(&[1, 2, 3, 4]).is_err());
    }
    #[test]
    fn legacy() {
        assert!(decode(b"caf\xe9 et d\xe9j\xe0 vu").unwrap().contains('é'));
    }
}
