use std::collections::HashMap;
use rand::seq::SliceRandom;
use rand::thread_rng;

// ============ CharTokenizer ============

pub struct CharTokenizer {
    pub vocab: Vec<char>,
    char_to_id: HashMap<char, usize>,
}

impl CharTokenizer {
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();
        let char_to_id: HashMap<char, usize> =
            chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        CharTokenizer { vocab: chars, char_to_id }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().filter_map(|c| self.char_to_id.get(&c).copied()).collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|&id| self.vocab[id]).collect()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// ============ Dataset Trait ============

pub trait Dataset {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
    fn get(&self, index: usize) -> (Vec<usize>, Vec<usize>);
}

// ============ TextDataset ============

pub struct TextDataset {
    tokens: Vec<usize>,
    seq_len: usize,
}

impl TextDataset {
    pub fn new(tokens: Vec<usize>, seq_len: usize) -> Self {
        TextDataset { tokens, seq_len }
    }
}

impl Dataset for TextDataset {
    fn len(&self) -> usize {
        if self.tokens.len() <= self.seq_len {
            0
        } else {
            self.tokens.len() - self.seq_len
        }
    }

    fn get(&self, index: usize) -> (Vec<usize>, Vec<usize>) {
        let input = self.tokens[index..index + self.seq_len].to_vec();
        let target = self.tokens[index + 1..index + self.seq_len + 1].to_vec();
        (input, target)
    }
}

// ============ DataLoader ============

pub struct DataLoader<'a, D: Dataset> {
    dataset: &'a D,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    pos: usize,
}

impl<'a, D: Dataset> DataLoader<'a, D> {
    pub fn new(dataset: &'a D, batch_size: usize, shuffle: bool) -> Self {
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        if shuffle {
            indices.shuffle(&mut thread_rng());
        }
        DataLoader { dataset, batch_size, shuffle, indices, pos: 0 }
    }

    pub fn reset(&mut self) {
        self.pos = 0;
        if self.shuffle {
            self.indices.shuffle(&mut thread_rng());
        }
    }

    /// Returns (inputs, targets) where each is a flat Vec.
    /// inputs: [batch_size * seq_len], targets: [batch_size * seq_len]
    pub fn next_batch(&mut self) -> Option<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
        if self.pos >= self.indices.len() {
            return None;
        }

        let end = (self.pos + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.pos..end];
        self.pos = end;

        let mut inputs = Vec::new();
        let mut targets = Vec::new();
        for &idx in batch_indices {
            let (inp, tgt) = self.dataset.get(idx);
            inputs.push(inp);
            targets.push(tgt);
        }

        Some((inputs, targets))
    }

    pub fn num_batches(&self) -> usize {
        (self.indices.len() + self.batch_size - 1) / self.batch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_char_tokenizer() {
        let tok = CharTokenizer::from_text("hello");
        assert_eq!(tok.vocab_size(), 4); // e, h, l, o
        let ids = tok.encode("hello");
        assert_eq!(ids.len(), 5);
        assert_eq!(tok.decode(&ids), "hello");
    }

    #[test]
    fn test_char_tokenizer_roundtrip() {
        let text = "the quick brown fox";
        let tok = CharTokenizer::from_text(text);
        let ids = tok.encode(text);
        assert_eq!(tok.decode(&ids), text);
    }

    #[test]
    fn test_text_dataset() {
        let tokens = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let ds = TextDataset::new(tokens, 4);
        assert_eq!(ds.len(), 4); // 8 - 4 = 4 valid positions
        let (inp, tgt) = ds.get(0);
        assert_eq!(inp, vec![0, 1, 2, 3]);
        assert_eq!(tgt, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_dataloader_no_shuffle() {
        let tokens = vec![0, 1, 2, 3, 4, 5];
        let ds = TextDataset::new(tokens, 2);
        // len = 6 - 2 = 4 samples
        let mut dl = DataLoader::new(&ds, 2, false);
        let batch1 = dl.next_batch().unwrap();
        assert_eq!(batch1.0.len(), 2);
        let batch2 = dl.next_batch().unwrap();
        assert_eq!(batch2.0.len(), 2);
        assert!(dl.next_batch().is_none());
    }

    #[test]
    fn test_dataloader_reset() {
        let tokens = vec![0, 1, 2, 3, 4];
        let ds = TextDataset::new(tokens, 2);
        let mut dl = DataLoader::new(&ds, 10, false);
        let _ = dl.next_batch();
        assert!(dl.next_batch().is_none());
        dl.reset();
        assert!(dl.next_batch().is_some());
    }

    #[test]
    fn test_dataloader_num_batches() {
        let tokens = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let ds = TextDataset::new(tokens, 3);
        // len = 10 - 3 = 7 samples, batch_size=3 => ceil(7/3) = 3 batches
        let dl = DataLoader::new(&ds, 3, false);
        assert_eq!(dl.num_batches(), 3);
    }
}
