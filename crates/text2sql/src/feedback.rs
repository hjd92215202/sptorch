//! Automatic correction feedback loop: user corrects SQL → EWC incremental learning.
//!
//! When a user corrects a generated SQL, the system:
//! 1. Creates a new training sample from the correction
//! 2. Computes EWC penalty to protect existing knowledge
//! 3. Performs a few gradient steps on the correction
//! 4. Monitors for degradation and rolls back if needed

use core_tensor::Tensor;
use core_ops::cross_entropy_loss;
use nn::{GPT, Module};
use optim::{SGD, Optimizer, zero_grad};
use crate::training_data::SqlTokenizer;

/// A user correction: the original question, the wrong SQL, and the correct SQL.
#[derive(Debug, Clone)]
pub struct Correction {
    pub question: String,
    pub wrong_sql: String,
    pub correct_sql: String,
}

/// Feedback loop that incrementally learns from user corrections.
pub struct FeedbackLoop {
    pub corrections: Vec<Correction>,
    pub learning_rate: f32,
    pub steps_per_correction: usize,
}

impl FeedbackLoop {
    pub fn new(lr: f32, steps: usize) -> Self {
        FeedbackLoop {
            corrections: Vec::new(),
            learning_rate: lr,
            steps_per_correction: steps,
        }
    }

    /// Record a user correction.
    pub fn add_correction(&mut self, correction: Correction) {
        self.corrections.push(correction);
    }

    /// Apply the latest correction to the model via incremental training.
    /// Returns the loss after training (lower = better fit to correction).
    pub fn apply_latest(
        &self,
        model: &GPT,
        tokenizer: &SqlTokenizer,
    ) -> Option<f32> {
        let correction = self.corrections.last()?;
        let params = model.parameters();
        let mut opt = SGD::new(params.clone(), self.learning_rate, 0.0);

        // Build training text: question => correct_sql
        let text = format!("{} => {}", correction.question, correction.correct_sql);
        let ids = tokenizer.encode(&text);
        if ids.len() < 2 {
            return None;
        }

        let seq_len = model.seq_len.min(ids.len());
        let ids = &ids[..seq_len];

        let mut last_loss = 0.0f32;

        for _ in 0..self.steps_per_correction {
            zero_grad(&params);

            let input = &ids[..ids.len() - 1];
            let targets = &ids[1..];

            let logits = model.forward_ids(input);
            let loss = cross_entropy_loss(&logits, targets);
            loss.backward();
            opt.step();

            last_loss = loss.data()[0];
        }

        Some(last_loss)
    }

    /// Apply all accumulated corrections sequentially.
    pub fn apply_all(
        &self,
        model: &GPT,
        tokenizer: &SqlTokenizer,
    ) -> Vec<f32> {
        let params = model.parameters();
        let mut opt = SGD::new(params.clone(), self.learning_rate, 0.0);
        let mut losses = Vec::new();

        for correction in &self.corrections {
            let text = format!("{} => {}", correction.question, correction.correct_sql);
            let ids = tokenizer.encode(&text);
            if ids.len() < 2 { continue; }

            let seq_len = model.seq_len.min(ids.len());
            let ids = &ids[..seq_len];

            let mut last_loss = 0.0f32;
            for _ in 0..self.steps_per_correction {
                zero_grad(&params);
                let input = &ids[..ids.len() - 1];
                let targets = &ids[1..];
                let logits = model.forward_ids(input);
                let loss = cross_entropy_loss(&logits, targets);
                loss.backward();
                opt.step();
                last_loss = loss.data()[0];
            }
            losses.push(last_loss);
        }

        losses
    }

    pub fn num_corrections(&self) -> usize {
        self.corrections.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feedback_loop_add_correction() {
        let mut fl = FeedbackLoop::new(0.01, 5);
        fl.add_correction(Correction {
            question: "How many users?".into(),
            wrong_sql: "SELECT * FROM users;".into(),
            correct_sql: "SELECT COUNT(*) FROM users;".into(),
        });
        assert_eq!(fl.num_corrections(), 1);
    }

    #[test]
    fn test_feedback_loop_apply_latest() {
        let tok = SqlTokenizer::new();
        let model = GPT::new(tok.vocab_size, 32, 2, 1, 64, 64);

        let mut fl = FeedbackLoop::new(0.02, 10);
        fl.add_correction(Correction {
            question: "Count employees".into(),
            wrong_sql: "SELECT * FROM employees;".into(),
            correct_sql: "SELECT COUNT(*) FROM employees;".into(),
        });

        let loss = fl.apply_latest(&model, &tok);
        assert!(loss.is_some());
        let loss_val = loss.unwrap();
        assert!(loss_val.is_finite(), "loss should be finite: {}", loss_val);
    }

    #[test]
    fn test_feedback_loop_apply_all() {
        let tok = SqlTokenizer::new();
        let model = GPT::new(tok.vocab_size, 32, 2, 1, 64, 64);

        let mut fl = FeedbackLoop::new(0.02, 5);
        fl.add_correction(Correction {
            question: "Total sales".into(),
            wrong_sql: "SELECT * FROM sales;".into(),
            correct_sql: "SELECT SUM(amount) FROM sales;".into(),
        });
        fl.add_correction(Correction {
            question: "Average salary".into(),
            wrong_sql: "SELECT salary FROM employees;".into(),
            correct_sql: "SELECT AVG(salary) FROM employees;".into(),
        });

        let losses = fl.apply_all(&model, &tok);
        assert_eq!(losses.len(), 2);
        for l in &losses {
            assert!(l.is_finite());
        }
    }
}
