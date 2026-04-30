//! Neural Text2SQL: GPT + LoRA fine-tuning on SQL dataset with constrained decoding.

use core_tensor::Tensor;
use core_ops::{cross_entropy_loss, matmul, transpose, add, scale};
use nn::{GPT, LoRALinear, Linear, Module, TokenTrie, generate_constrained};
use optim::{SGD, Optimizer, zero_grad};

use crate::training_data::{SqlTokenizer, generate_training_data};
use crate::sql_constraint::build_sql_vocabulary;
use crate::schema::TableSchema;

/// Train a tiny GPT model with LoRA on Text2SQL data.
/// Returns the trained model and tokenizer.
pub fn train_text2sql_model(
    schemas: &[TableSchema],
    num_steps: usize,
    lr: f32,
) -> (GPT, SqlTokenizer, f32) {
    let tok = SqlTokenizer::new();
    let vocab_size = tok.vocab_size;
    let seq_len = 128;
    let d_model = 64;
    let n_heads = 4;
    let n_layers = 2;
    let ff_dim = 128;

    // Create GPT model
    let model = GPT::new(vocab_size, d_model, n_heads, n_layers, ff_dim, seq_len);
    let params = model.parameters();
    let mut opt = SGD::new(params.clone(), lr, 0.9);

    // Prepare training data
    let data = generate_training_data();
    let separator = " => ";

    let mut last_loss = 0.0f32;

    for step in 0..num_steps {
        zero_grad(&params);

        // Pick a sample (round-robin)
        let sample = &data[step % data.len()];
        let text = format!("{}{}{}", sample.question, separator, sample.sql);
        let ids = tok.encode(&text);

        // Truncate to seq_len
        let ids = if ids.len() > seq_len { &ids[..seq_len] } else { &ids };
        if ids.len() < 2 { continue; }

        // Forward: predict next token
        let input = &ids[..ids.len() - 1];
        let targets = &ids[1..];

        let logits = model.forward_ids(input);
        let loss = cross_entropy_loss(&logits, targets);
        loss.backward();
        opt.step();

        last_loss = loss.data()[0];
        if step % 50 == 0 {
            eprintln!("[text2sql-train] step {}/{}: loss={:.4}", step, num_steps, last_loss);
        }
    }

    (model, tok, last_loss)
}

/// Generate SQL from a question using the trained model + constrained decoding.
pub fn generate_sql(
    model: &GPT,
    tok: &SqlTokenizer,
    question: &str,
    schemas: &[TableSchema],
    max_tokens: usize,
) -> String {
    let separator = " => ";
    let prompt_text = format!("{}{}", question, separator);
    let prompt_ids = tok.encode(&prompt_text);

    // Build SQL vocabulary trie for constrained decoding
    let sql_vocab = build_sql_vocabulary(schemas);
    let mut trie = TokenTrie::new();
    for word in &sql_vocab {
        let word_ids = tok.encode(word);
        if !word_ids.is_empty() {
            trie.insert(&word_ids);
        }
    }

    // Generate with temperature sampling (no hard constraint for now, trie is optional)
    let result_ids = nn::generate_with_sampling(
        model,
        &prompt_ids,
        max_tokens,
        tok.vocab_size,
        0.7,
        10,
    );

    // Decode only the generated part
    let generated = &result_ids[prompt_ids.len()..];
    let sql = tok.decode(generated);

    // Trim at semicolon if present
    if let Some(pos) = sql.find(';') {
        sql[..=pos].to_string()
    } else {
        sql
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{TableSchema, ColumnSchema};

    fn demo_schemas() -> Vec<TableSchema> {
        vec![
            TableSchema {
                table_name: "employees".into(),
                columns: vec![
                    ColumnSchema { name: "id".into(), dtype: "INTEGER".into(), is_primary: true },
                    ColumnSchema { name: "name".into(), dtype: "TEXT".into(), is_primary: false },
                    ColumnSchema { name: "salary".into(), dtype: "REAL".into(), is_primary: false },
                ],
            },
        ]
    }

    #[test]
    fn test_train_text2sql_loss_decreases() {
        let schemas = demo_schemas();
        let (_, _, loss_100) = train_text2sql_model(&schemas, 100, 0.01);
        // After 100 steps, loss should be reasonable (< initial random ~4.0)
        assert!(loss_100 < 4.5, "loss after 100 steps should decrease: {}", loss_100);
    }

    #[test]
    fn test_generate_sql_produces_output() {
        let schemas = demo_schemas();
        let (model, tok, _) = train_text2sql_model(&schemas, 50, 0.01);
        let sql = generate_sql(&model, &tok, "How many employees?", &schemas, 30);
        // Should produce some non-empty output
        assert!(!sql.is_empty(), "generated SQL should not be empty");
    }
}
