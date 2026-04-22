use core_tensor::Tensor;
use core_ops::*;
use nn::{Module, Linear, Embedding};
use optim::{AdamW, Optimizer, clip_grad_norm};

// ============ 字符级 Tokenizer ============

struct CharTokenizer {
    vocab: Vec<char>,
    char_to_id: std::collections::HashMap<char, usize>,
}

impl CharTokenizer {
    fn new(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();
        let char_to_id: std::collections::HashMap<char, usize> =
            chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        CharTokenizer { vocab: chars, char_to_id }
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().filter_map(|c| self.char_to_id.get(&c).copied()).collect()
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// ============ 最小 DataLoader ============

struct DataLoader {
    tokens: Vec<usize>,
    seq_len: usize,
    batch_size: usize,
    pos: usize,
}

impl DataLoader {
    fn new(tokens: Vec<usize>, seq_len: usize, batch_size: usize) -> Self {
        DataLoader { tokens, seq_len, batch_size, pos: 0 }
    }

    fn next_batch(&mut self) -> Option<(Vec<usize>, Vec<usize>)> {
        if self.pos + self.seq_len + 1 > self.tokens.len() {
            self.pos = 0;
            return None;
        }

        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        for _ in 0..self.batch_size {
            if self.pos + self.seq_len + 1 > self.tokens.len() {
                break;
            }
            inputs.extend_from_slice(&self.tokens[self.pos..self.pos + self.seq_len]);
            targets.extend_from_slice(&self.tokens[self.pos + 1..self.pos + self.seq_len + 1]);
            self.pos += self.seq_len;
        }

        if inputs.is_empty() {
            None
        } else {
            Some((inputs, targets))
        }
    }
}

// ============ Tiny Language Model ============

struct TinyLM {
    embedding: Embedding,
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl TinyLM {
    fn new(vocab_size: usize, embed_dim: usize, hidden_dim: usize) -> Self {
        TinyLM {
            embedding: Embedding::new(vocab_size, embed_dim),
            fc1: Linear::new(embed_dim, hidden_dim, true),
            fc2: Linear::new(hidden_dim, hidden_dim, true),
            fc3: Linear::new(hidden_dim, vocab_size, true),
        }
    }

    fn forward(&self, input_ids: &[usize]) -> Tensor {
        // input_ids: [seq_len]
        // embedding: [seq_len, embed_dim]
        let x = self.embedding.forward_indices(input_ids);

        // fc1 + relu: [seq_len, hidden_dim]
        let h1 = relu(&self.fc1.forward(&x));

        // fc2 + relu: [seq_len, hidden_dim]
        let h2 = relu(&self.fc2.forward(&h1));

        // fc3: [seq_len, vocab_size]
        self.fc3.forward(&h2)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.embedding.parameters());
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params
    }
}

// ============ 训练循环 ============

fn main() {
    println!("SPTorch Tiny LM Training");
    println!("========================\n");

    // 准备数据
    let text = "hello world this is a tiny language model training example. \
                we will train on this small text corpus to demonstrate the framework. \
                the quick brown fox jumps over the lazy dog. \
                machine learning is fun and exciting. \
                deep learning with rust is powerful. \
                hello world hello world hello world. \
                this is a test this is a test this is a test.";

    let tokenizer = CharTokenizer::new(text);
    let tokens = tokenizer.encode(text);
    let vocab_size = tokenizer.vocab_size();

    println!("Vocab size: {}", vocab_size);
    println!("Total tokens: {}", tokens.len());
    println!();

    // 超参数
    let seq_len = 8;
    let batch_size = 4;
    let embed_dim = 32;
    let hidden_dim = 64;
    let lr = 0.01;
    let max_steps = 1000;
    let log_interval = 100;

    // 模型和优化器
    let model = TinyLM::new(vocab_size, embed_dim, hidden_dim);
    let params = model.parameters();
    let mut optimizer = AdamW::default(params.clone(), lr);

    println!("Model parameters: {}", params.len());
    println!("Training for {} steps...\n", max_steps);

    // 训练
    let mut dataloader = DataLoader::new(tokens.clone(), seq_len, batch_size);
    let mut step = 0;
    let mut total_loss = 0.0;

    while step < max_steps {
        let (inputs, targets) = match dataloader.next_batch() {
            Some(batch) => batch,
            None => {
                dataloader = DataLoader::new(tokens.clone(), seq_len, batch_size);
                continue;
            }
        };

        // Forward
        let logits = model.forward(&inputs);
        let loss = cross_entropy_loss(&logits, &targets);
        let loss_val = loss.data()[0];

        // Backward
        optimizer.zero_grad();
        loss.backward();

        // Clip gradients
        let grad_norm = clip_grad_norm(&params, 1.0);

        // Check for NaN/Inf
        if loss_val.is_nan() || loss_val.is_infinite() {
            eprintln!("Step {}: Loss is NaN/Inf, skipping", step);
            continue;
        }

        // Step
        optimizer.step();

        total_loss += loss_val;
        step += 1;

        if step % log_interval == 0 {
            let avg_loss = total_loss / log_interval as f32;
            println!("Step {}/{}: loss={:.4}, grad_norm={:.4}",
                     step, max_steps, avg_loss, grad_norm);
            total_loss = 0.0;
        }
    }

    println!("\nTraining complete!");

    // 简单生成测试
    println!("\nGenerating sample (greedy):");
    let prompt = "hello";
    let mut gen_ids = tokenizer.encode(prompt);
    print!("{}", prompt);

    for _ in 0..20 {
        let logits = model.forward(&gen_ids[gen_ids.len().saturating_sub(seq_len)..]);
        let logits_data = logits.data();
        let last_logits = &logits_data[logits_data.len() - vocab_size..];

        // Greedy: argmax
        let next_id = last_logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        gen_ids.push(next_id);
        print!("{}", tokenizer.vocab[next_id]);
    }
    println!("\n");
}
