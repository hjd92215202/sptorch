use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tauri::Emitter;
use versioning::{
    BufferPointers, EvolutionMetrics, FencePhase, FenceState, HardwareState, LayerPolicy, TensorLayoutSnapshot,
    UpdatePolicy, VersionNode, VersionedStorage, EVENT_FENCE, EVENT_HARDWARE_STATE, EVENT_METRICS,
    EVENT_VERSION_COMMIT,
};

#[derive(Clone)]
pub struct EngineBridge {
    pub storage: Arc<RwLock<VersionedStorage>>,
    pub metrics_tx: broadcast::Sender<EvolutionMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStatusDto {
    pub global_version: u64,
    pub active_version: u64,
    pub chain_head: Option<VersionNode>,
    pub layer_policies: Vec<LayerPolicy>,
    pub tensors: Vec<TensorLayoutSnapshot>,
}

impl EngineBridge {
    pub fn new(storage: Arc<RwLock<VersionedStorage>>) -> Self {
        let (metrics_tx, _rx) = broadcast::channel(256);
        Self { storage, metrics_tx }
    }

    pub fn snapshot_status(&self) -> Result<EngineStatusDto, String> {
        let snap = self
            .storage
            .read()
            .map_err(|_| "storage lock poisoned".to_string())?
            .clone();

        Ok(EngineStatusDto {
            global_version: snap.global_version,
            active_version: snap.active_version,
            chain_head: snap.chain.last().cloned(),
            layer_policies: snap.layer_policies,
            tensors: snap.tensors,
        })
    }

    pub fn push_metric(&self, metric: EvolutionMetrics) {
        let _ = self.metrics_tx.send(metric);
    }

    pub fn commit_version(&self, reason: impl Into<String>) -> Result<VersionNode, String> {
        let mut guard = self
            .storage
            .write()
            .map_err(|_| "storage lock poisoned".to_string())?;
        let parent = Some(guard.active_version);
        guard.global_version += 1;
        guard.active_version = guard.global_version;
        let node = VersionNode {
            version_id: guard.active_version,
            parent_version: parent,
            committed_at_ms: now_ms(),
            reason: reason.into(),
        };
        guard.chain.push(node.clone());
        Ok(node)
    }

    pub fn trigger_atomic_swap_simulated(&self) -> Result<Vec<FenceState>, String> {
        let phases = [
            (FencePhase::Prepare, 0.15, 8, "prepare swap"),
            (FencePhase::WaitFence, 0.35, 6, "waiting fence signal"),
            (FencePhase::Swap, 0.65, 4, "atomic pointer swap"),
            (FencePhase::Commit, 0.90, 2, "commit version"),
            (FencePhase::Done, 1.0, 0, "swap done"),
        ];

        let mut out = Vec::with_capacity(phases.len());
        for (phase, progress, queue_depth, msg) in phases {
            out.push(FenceState {
                phase,
                progress,
                queue_depth,
                message: msg.to_string(),
            });
        }

        let _ = self.commit_version("atomic_swap_simulated")?;
        Ok(out)
    }
}

pub struct AppState {
    pub bridge: Arc<EngineBridge>,
}

#[tauri::command]
pub async fn get_engine_status(state: tauri::State<'_, AppState>) -> Result<EngineStatusDto, String> {
    state.bridge.snapshot_status()
}

#[tauri::command]
pub async fn start_evolution_stream(app: tauri::AppHandle, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let mut rx = state.bridge.metrics_tx.subscribe();
    tauri::async_runtime::spawn(async move {
        while let Ok(m) = rx.recv().await {
            let _ = app.emit(EVENT_METRICS, m);
        }
    });
    Ok(())
}

#[tauri::command]
pub async fn trigger_atomic_swap(app: tauri::AppHandle, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let fence_states = state.bridge.trigger_atomic_swap_simulated()?;

    for fence in &fence_states {
        let _ = app.emit(EVENT_FENCE, fence);
    }

    if let Some(last) = fence_states.last() {
        let hw = HardwareState {
            backend: "simulated-hal-ffi".to_string(),
            queue_depth: last.queue_depth,
            online: !matches!(last.phase, FencePhase::Error),
        };
        let _ = app.emit(EVENT_HARDWARE_STATE, hw);
    }

    let status = state.bridge.snapshot_status()?;
    if let Some(head) = status.chain_head {
        let _ = app.emit(EVENT_VERSION_COMMIT, head);
    }

    Ok(())
}

pub fn bootstrap_default_storage() -> VersionedStorage {
    VersionedStorage {
        global_version: 1,
        active_version: 1,
        chain: vec![VersionNode {
            version_id: 1,
            parent_version: None,
            committed_at_ms: now_ms(),
            reason: "bootstrap".to_string(),
        }],
        layer_policies: vec![
            LayerPolicy {
                layer_name: "embedding".to_string(),
                policy: UpdatePolicy::Single,
            },
            LayerPolicy {
                layer_name: "transformer.block0.attn".to_string(),
                policy: UpdatePolicy::Double,
            },
            LayerPolicy {
                layer_name: "lm_head".to_string(),
                policy: UpdatePolicy::Single,
            },
        ],
        tensors: vec![
            TensorLayoutSnapshot {
                tensor_id: "embedding.weight".to_string(),
                shape: vec![32000, 64],
                strides: vec![64, 1],
                offset: 0,
                numel: 32000 * 64,
                dtype: "F32".to_string(),
                device: "CPU".to_string(),
                pointers: BufferPointers {
                    active_ptr: "arc:embedding_active".to_string(),
                    shadow_ptr: None,
                    active_version: 1,
                    shadow_version: None,
                },
            },
            TensorLayoutSnapshot {
                tensor_id: "transformer.block0.attn.q_proj.weight".to_string(),
                shape: vec![64, 64],
                strides: vec![64, 1],
                offset: 0,
                numel: 4096,
                dtype: "F32".to_string(),
                device: "CPU".to_string(),
                pointers: BufferPointers {
                    active_ptr: "arc:attn_active".to_string(),
                    shadow_ptr: Some("arc:attn_shadow".to_string()),
                    active_version: 1,
                    shadow_version: Some(2),
                },
            },
        ],
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
