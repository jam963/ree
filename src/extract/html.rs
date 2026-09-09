use scraper::{ElementRef, Html, Node, Selector};
pub fn extract(source: &str) -> (String, Option<String>) {
    let html = Html::parse_document(source);
    let title = html
        .select(&Selector::parse("title").unwrap())
        .next()
        .map(|n| n.text().collect::<String>());
    let selector = Selector::parse("article, main, [role=main]").unwrap();
    let root = html
        .select(&selector)
        .max_by_key(|e| e.text().map(str::len).sum::<usize>())
        .unwrap_or_else(|| html.root_element());
    let mut output = String::new();
    for node in root.descendants() {
        if let Node::Text(text) = node.value() {
            let hidden = node.ancestors().filter_map(ElementRef::wrap).any(|e| {
                matches!(
                    e.value().name(),
                    "script" | "style" | "noscript" | "template" | "head" | "nav" | "footer"
                ) || e.value().attr("hidden").is_some()
                    || e.value().attr("aria-hidden") == Some("true")
            });
            if !hidden && !text.trim().is_empty() {
                output.push_str(text);
                output.push('\n');
            }
        }
    }
    (output, title)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn removes_active_content() {
        let (s, t) = extract(
            "<title>T</title><nav>noise</nav><main>Hello <b>world</b><script>evil()</script><i hidden>secret</i></main>",
        );
        assert_eq!(t.as_deref(), Some("T"));
        assert!(s.contains("Hello"));
        assert!(!s.contains("evil"));
        assert!(!s.contains("secret"));
        assert!(!s.contains("noise"));
    }
}
