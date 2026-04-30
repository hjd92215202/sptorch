//! Text2SQL product crate — natural language to SQL, end-to-end.
//!
//! - `server`: axum REST API (/query, /health)
//! - `schema`: SQLite schema introspection + DDL generation
//! - `rag`: prompt assembly with table relevance ranking
//! - `sql_constraint`: SQL vocabulary whitelist + template-based SQL generation
//! - `training_data`: synthetic Text2SQL dataset + char-level tokenizer
//! - `neural`: GPT + LoRA fine-tuning + constrained decoding for SQL generation

pub mod rag;
pub mod schema;
pub mod server;
pub mod sql_constraint;
pub mod training_data;
pub mod neural;
