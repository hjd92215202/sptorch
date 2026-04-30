//! Distributed training engine for sptorch.
//!
//! gRPC-based coordinator/worker architecture:
//! - `coordinator`: manages registration, AllReduce gradient averaging, barrier sync
//! - `worker`: connects to coordinator, submits gradients, receives averaged results
//! - `allreduce`: local utilities (average_gradients, ring_allreduce)
//! - `data_parallel`: multi-GPU DataParallel (scatter/allreduce/step)

pub mod proto {
    tonic::include_proto!("sptorch.distributed");
}

pub mod allreduce;
pub mod checkpoint;
pub mod coordinator;
pub mod data_parallel;
pub mod worker;
