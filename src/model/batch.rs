use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Limits {
    pub max_items: usize,
    pub max_padded_tokens: usize,
}
impl Limits {
    pub fn reduce(&mut self) {
        self.max_items = (self.max_items / 2).max(1);
        self.max_padded_tokens = (self.max_padded_tokens / 2).max(1);
    }
    pub fn pack(&self, sorted: &[usize], inputs: &[Vec<i64>]) -> usize {
        let mut count = 0;
        let mut longest = 0;
        for &i in sorted.iter().take(self.max_items.max(1)) {
            longest = longest.max(inputs[i].len());
            if count > 0 && longest.saturating_mul(count + 1) > self.max_padded_tokens {
                break;
            }
            count += 1;
        }
        count
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn padded_budget() {
        let l = Limits {
            max_items: 9,
            max_padded_tokens: 100,
        };
        let inputs = vec![vec![0; 10], vec![0; 30], vec![0; 90]];
        assert_eq!(l.pack(&[0, 1, 2], &inputs), 2);
        assert_eq!(l.pack(&[2], &inputs), 1);
    }
    #[test]
    fn reduction_has_floor() {
        let mut l = Limits {
            max_items: 1,
            max_padded_tokens: 1,
        };
        l.reduce();
        assert_eq!(l.max_items, 1);
    }
}
