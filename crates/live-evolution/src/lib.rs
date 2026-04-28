//! Live evolution engine for sptorch — online continuous learning.
//!
//! - `double_buffer`: atomic swap between train/inference parameter copies
//! - `incremental`: data-flow triggered micro-batch training scheduler
//! - `ewc`: Elastic Weight Consolidation to prevent catastrophic forgetting
//! - `monitor`: rolling-window loss tracking with automatic rollback on degradation

pub mod double_buffer;
pub mod incremental;
pub mod ewc;
pub mod monitor;
