//! Text2SQL product crate — natural language to SQL, end-to-end.
//!
//! - `server`: axum REST API (/query, /health)
//! - `schema`: SQLite schema introspection + DDL generation
//! - `rag`: prompt assembly with table relevance ranking
//! - `sql_constraint`: SQL vocabulary whitelist + template-based SQL generation

pub mod server;
pub mod schema;
pub mod rag;
pub mod sql_constraint;
