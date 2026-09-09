use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
pub fn extract(source: &str) -> Result<(String, Value)> {
    let notebook: Value = serde_json::from_str(source).context("invalid notebook JSON")?;
    let cells = notebook["cells"]
        .as_array()
        .context("notebook cells missing")?;
    let mut text = String::new();
    let mut locations = Vec::new();
    for (index, cell) in cells.iter().enumerate() {
        if !matches!(
            cell["cell_type"].as_str(),
            Some("code" | "markdown" | "raw")
        ) {
            continue;
        }
        let start = text.len();
        match &cell["source"] {
            Value::String(s) => text.push_str(s),
            Value::Array(lines) => {
                for line in lines {
                    text.push_str(line.as_str().context("invalid notebook source line")?);
                }
            }
            _ => bail!("invalid notebook cell source"),
        }
        locations.push(json!({"cell":index,"byte_start":start,"byte_end":text.len()}));
        text.push_str("\n\n");
    }
    Ok((text, json!({"cells":locations})))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn no_binary_outputs() {
        let(t,m)=extract(r#"{"cells":[{"cell_type":"code","source":["print(1)"],"outputs":[{"data":{"image/png":"BASE64"}}]}]}"#).unwrap();
        assert_eq!(t, "print(1)\n\n");
        assert_eq!(m["cells"][0]["cell"], 0);
    }
}
