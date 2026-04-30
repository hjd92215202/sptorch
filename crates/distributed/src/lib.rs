//! Distributed training engine for sptorch.
//!
//! gRPC-based coordinator/worker architecture:
//! - `coordinator`: manages registration, AllReduce gradient averaging, barrier sync
//! - `worker`: connects to coordinator, submits gradients, receives averaged results
//! - `allreduce`: local utilities (average_gradients, ring_allreduce)

pub mod proto {
    tonic::include_proto!("sptorch.distributed");
}

pub mod allreduce;
pub mod coordinator;
pub mod worker;
