use live_evolution::events::{publish, subscribe, LiveEvolutionEvent};
use versioning::{EvolutionMetrics, FencePhase, FenceState, HardwareState, VersionNode};

#[tokio::test]
async fn test_events_roundtrip_variants() {
    let mut rx = subscribe();

    publish(LiveEvolutionEvent::Metrics(EvolutionMetrics {
        ts_ms: 1,
        loss: 0.9,
        grad_norm: 0.2,
        grad_scale_factor: 1.0,
        accum_current: 1,
        accum_target: 4,
        version_id: 1,
        fence: None,
    }));
    publish(LiveEvolutionEvent::Fence(FenceState {
        phase: FencePhase::Swap,
        progress: 0.7,
        queue_depth: 3,
        message: "swap".to_string(),
    }));
    publish(LiveEvolutionEvent::VersionCommit(VersionNode {
        version_id: 2,
        parent_version: Some(1),
        committed_at_ms: 2,
        reason: "test".to_string(),
    }));
    publish(LiveEvolutionEvent::HardwareState(HardwareState {
        backend: "sim".to_string(),
        queue_depth: 1,
        online: true,
    }));

    let mut seen = [false; 4];
    for _ in 0..4 {
        match rx.recv().await.expect("recv event") {
            LiveEvolutionEvent::Metrics(_) => seen[0] = true,
            LiveEvolutionEvent::Fence(_) => seen[1] = true,
            LiveEvolutionEvent::VersionCommit(_) => seen[2] = true,
            LiveEvolutionEvent::HardwareState(_) => seen[3] = true,
        }
    }

    assert!(seen.iter().all(|x| *x));
}
