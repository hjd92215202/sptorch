pub mod proto {
    tonic::include_proto!("sptorch.distributed");
}

pub mod coordinator;
pub mod worker;
pub mod allreduce;
