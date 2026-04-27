use crate::proto::*;
use crate::proto::node_service_server::{NodeService, NodeServiceServer};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, Notify};
use tonic::{Request, Response, Status};

struct CoordinatorState {
    peers: Vec<String>,
    world_size: u32,
    global_step: u64,
    // param_index -> accumulated gradient data (f32 LE bytes)
    grad_accum: HashMap<u32, Vec<f32>>,
    grad_count: HashMap<u32, u32>,
    // barrier
    barrier_count: u32,
}

pub struct CoordinatorService {
    state: Arc<Mutex<CoordinatorState>>,
    grad_ready: Arc<Notify>,
    barrier_ready: Arc<Notify>,
}

impl CoordinatorService {
    pub fn new(world_size: u32) -> Self {
        CoordinatorService {
            state: Arc::new(Mutex::new(CoordinatorState {
                peers: Vec::new(),
                world_size,
                global_step: 0,
                grad_accum: HashMap::new(),
                grad_count: HashMap::new(),
                barrier_count: 0,
            })),
            grad_ready: Arc::new(Notify::new()),
            barrier_ready: Arc::new(Notify::new()),
        }
    }

    pub fn into_server(self) -> NodeServiceServer<Self> {
        NodeServiceServer::new(self)
    }
}

#[tonic::async_trait]
impl NodeService for CoordinatorService {
    async fn heartbeat(
        &self,
        request: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        let state = self.state.lock().await;
        Ok(Response::new(HeartbeatResponse {
            alive: true,
            global_step: state.global_step,
        }))
    }

    async fn register(
        &self,
        request: Request<RegisterRequest>,
    ) -> Result<Response<RegisterResponse>, Status> {
        let req = request.into_inner();
        let mut state = self.state.lock().await;
        let rank = state.peers.len() as u32;
        state.peers.push(req.address);

        Ok(Response::new(RegisterResponse {
            rank,
            world_size: state.world_size,
            peer_addresses: state.peers.clone(),
        }))
    }

    async fn all_reduce_gradients(
        &self,
        request: Request<GradientChunk>,
    ) -> Result<Response<GradientChunk>, Status> {
        let req = request.into_inner();
        let param_idx = req.param_index;
        let world_size = req.world_size;

        let local_grads = bytes_to_f32(&req.data);

        // Accumulate
        {
            let mut state = self.state.lock().await;
            let accum = state.grad_accum.entry(param_idx).or_insert_with(|| vec![0.0; local_grads.len()]);
            for (a, g) in accum.iter_mut().zip(local_grads.iter()) {
                *a += g;
            }
            let count = state.grad_count.entry(param_idx).or_insert(0);
            *count += 1;

            if *count >= world_size {
                self.grad_ready.notify_waiters();
            }
        }

        // Wait until all workers have submitted
        loop {
            let state = self.state.lock().await;
            if *state.grad_count.get(&param_idx).unwrap_or(&0) >= world_size {
                // Average and return
                let accum = state.grad_accum.get(&param_idx).unwrap();
                let averaged: Vec<f32> = accum.iter().map(|v| v / world_size as f32).collect();
                return Ok(Response::new(GradientChunk {
                    param_index: param_idx,
                    rank: 0,
                    world_size,
                    data: f32_to_bytes(&averaged),
                    shape: req.shape,
                }));
            }
            drop(state);
            self.grad_ready.notified().await;
        }
    }

    async fn barrier(
        &self,
        request: Request<BarrierRequest>,
    ) -> Result<Response<BarrierResponse>, Status> {
        let req = request.into_inner();

        {
            let mut state = self.state.lock().await;
            state.barrier_count += 1;
            if state.barrier_count >= state.world_size {
                state.barrier_count = 0;
                state.global_step = req.step;
                // Clear gradient accumulators for next step
                state.grad_accum.clear();
                state.grad_count.clear();
                self.barrier_ready.notify_waiters();
            }
        }

        loop {
            let state = self.state.lock().await;
            if state.barrier_count == 0 {
                return Ok(Response::new(BarrierResponse { proceed: true }));
            }
            drop(state);
            self.barrier_ready.notified().await;
        }
    }
}

fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Start the coordinator gRPC server.
pub async fn start_coordinator(
    addr: &str,
    world_size: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = addr.parse()?;
    let service = CoordinatorService::new(world_size);
    eprintln!("[sptorch] coordinator listening on {} (world_size={})", addr, world_size);
    tonic::transport::Server::builder()
        .add_service(service.into_server())
        .serve(addr)
        .await?;
    Ok(())
}
