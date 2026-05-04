use std::sync::{Arc, Mutex};

use sptorch::nn::GPT;
use sptorch::optim::{zero_grad, Optimizer, SGD};
use text2sql::schema::TableSchema;
use text2sql::server::{ProductCorrection, ProductInferenceEngine};

#[derive(Debug, Clone)]
pub struct Text2SqlSample {
    pub question: String,
    pub sql: String,
}

pub fn generate_training_data() -> Vec<Text2SqlSample> {
    vec![
        Text2SqlSample {
            question: "How many employees are there?".into(),
            sql: "SELECT COUNT(*) FROM employees;".into(),
        },
        Text2SqlSample {
            question: "What is the average salary?".into(),
            sql: "SELECT AVG(salary) FROM employees;".into(),
        },
        Text2SqlSample {
            question: "total sales amount".into(),
            sql: "SELECT SUM(amount) FROM sales;".into(),
        },
        Text2SqlSample {
            question: "highest budget in departments".into(),
            sql: "SELECT MAX(budget) FROM departments;".into(),
        },
        Text2SqlSample {
            question: "show me all employees".into(),
            sql: "SELECT * FROM employees LIMIT 10;".into(),
        },
    ]
}

#[derive(Clone)]
pub struct SqlTokenizer {
    pub char_to_id: std::collections::HashMap<char, usize>,
    pub id_to_char: Vec<char>,
    pub vocab_size: usize,
}

impl SqlTokenizer {
    pub fn new() -> Self {
        let chars: Vec<char> = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.,;()*><=!'\"?\n"
            .chars()
            .collect();
        let char_to_id: std::collections::HashMap<char, usize> =
            chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let vocab_size = chars.len();
        SqlTokenizer {
            char_to_id,
            id_to_char: chars,
            vocab_size,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().map(|c| *self.char_to_id.get(&c).unwrap_or(&0)).collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|&id| {
                if id < self.id_to_char.len() {
                    self.id_to_char[id]
                } else {
                    '?'
                }
            })
            .collect()
    }
}

pub fn train_text2sql_model(_schemas: &[TableSchema], num_steps: usize, lr: f32) -> (GPT, SqlTokenizer, f32) {
    let tok = SqlTokenizer::new();
    let vocab_size = tok.vocab_size;
    let seq_len = 128;
    let d_model = 64;
    let n_heads = 4;
    let n_layers = 2;
    let ff_dim = 128;

    let model = GPT::new(vocab_size, d_model, n_heads, n_layers, ff_dim, seq_len);
    let params = model.parameters();
    let mut opt = SGD::new(params.clone(), lr, 0.9);

    let data = generate_training_data();
    let separator = " => ";
    let mut last_loss = 0.0f32;

    for step in 0..num_steps {
        zero_grad(&params);
        let sample = &data[step % data.len()];
        let text = format!("{}{}{}", sample.question, separator, sample.sql);
        let ids = tok.encode(&text);
        if ids.len() < 2 {
            continue;
        }
        let ids = if ids.len() > seq_len { &ids[..seq_len] } else { &ids };
        let input = &ids[..ids.len() - 1];
        let targets = &ids[1..];

        let logits = model.forward_ids(input);
        let loss = sptorch::core_ops::cross_entropy_loss(&logits, targets);
        loss.backward();
        opt.step();
        last_loss = loss.data()[0];
    }

    (model, tok, last_loss)
}

pub fn generate_sql(
    model: &GPT,
    tok: &SqlTokenizer,
    question: &str,
    schemas: &[TableSchema],
    max_tokens: usize,
) -> String {
    let prompt_text = format!("{} => ", question);
    let prompt_ids = tok.encode(&prompt_text);

    let mut trie = sptorch::nn::TokenTrie::new();
    for word in &text2sql::sql_constraint::build_sql_vocabulary(schemas) {
        let word_ids = tok.encode(word);
        if !word_ids.is_empty() {
            trie.insert(&word_ids);
        }
    }
    for ch in " .,;()*>=<!'\"0123456789_\n".chars() {
        let id = tok.encode(&ch.to_string());
        if !id.is_empty() {
            trie.insert(&id);
        }
    }

    let result_ids = sptorch::nn::generate_constrained(model, &prompt_ids, max_tokens, tok.vocab_size, 0.7, 10, &trie);
    let generated = &result_ids[prompt_ids.len()..];
    let sql = tok.decode(generated);
    if let Some(pos) = sql.find(';') {
        sql[..=pos].to_string()
    } else {
        sql
    }
}

pub struct NeuralText2SqlEngine {
    model: GPT,
    tokenizer: SqlTokenizer,
    corrections: Mutex<Vec<ProductCorrection>>,
}

impl NeuralText2SqlEngine {
    pub fn new(model: GPT, tokenizer: SqlTokenizer) -> Self {
        Self {
            model,
            tokenizer,
            corrections: Mutex::new(Vec::new()),
        }
    }

    pub fn shared(model: GPT, tokenizer: SqlTokenizer) -> Arc<dyn ProductInferenceEngine> {
        Arc::new(Self::new(model, tokenizer))
    }
}

impl ProductInferenceEngine for NeuralText2SqlEngine {
    fn generate_sql(&self, question: &str, schemas: &[TableSchema], max_tokens: usize) -> Option<String> {
        let sql = generate_sql(&self.model, &self.tokenizer, question, schemas, max_tokens);
        Some(sql)
    }

    fn apply_correction(&self, correction: ProductCorrection) -> Result<String, String> {
        let mut guard = self
            .corrections
            .lock()
            .map_err(|_| "correction lock poisoned".to_string())?;
        guard.push(correction);
        Ok(format!("Correction accepted. queued={}", guard.len()))
    }
}
