//! Opt-in, thread-local service times. Nested spans overlap: never sum them as wall time.
//! REE_TIMING=1 enables bounded aggregate counters, not query/token logging.
use serde::Serialize;
use std::{cell::RefCell, collections::BTreeMap, sync::OnceLock, time::Instant};

#[derive(Default, Serialize)]
struct Counter {
    calls: u64,
    total_ns: u128,
    max_ns: u128,
}
thread_local! {
    static COUNTERS: RefCell<BTreeMap<&'static str, Counter>> = const { RefCell::new(BTreeMap::new()) };
}
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("REE_TIMING").is_ok_and(|v| v == "1"))
}
pub struct Span {
    name: &'static str,
    start: Option<Instant>,
}
impl Span {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            start: enabled().then(Instant::now),
        }
    }
}
impl Drop for Span {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let ns = start.elapsed().as_nanos();
            COUNTERS.with_borrow_mut(|c| {
                let c = c.entry(self.name).or_default();
                c.calls += 1;
                c.total_ns += ns;
                c.max_ns = c.max_ns.max(ns);
            });
        }
    }
}
/// Drain this thread's counters at an operation boundary, including failures.
pub fn take() -> serde_json::Value {
    COUNTERS.with_borrow_mut(|c| serde_json::to_value(std::mem::take(c)).unwrap())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn aggregates_and_drains_without_millisecond_truncation() {
        for _ in 0..2 {
            drop(Span {
                name: "fixture",
                start: Some(Instant::now()),
            });
        }
        let result = take();
        assert_eq!(result["fixture"]["calls"], 2);
        assert!(result["fixture"]["total_ns"].as_u64().unwrap() > 0);
        assert_eq!(take(), serde_json::json!({}));
    }
}
