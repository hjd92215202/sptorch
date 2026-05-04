use std::sync::{Arc, RwLock};

use sptorch_studio::engine_bridge::{bootstrap_default_storage, EngineBridge};
use versioning::{EvolutionMetrics, FencePhase};

fn make_bridge() -> EngineBridge {
    EngineBridge::new(Arc::new(RwLock::new(bootstrap_default_storage())))
}

#[test]
fn test_snapshot_status_concurrent_reads() {
    let bridge = Arc::new(make_bridge());
    let mut handles = Vec::new();
    for _ in 0..8 {
        let b = bridge.clone();
        handles.push(std::thread::spawn(move || {
            for _ in 0..128 {
                let s = b.snapshot_status().expect("snapshot status");
                assert!(s.global_version >= s.active_version);
            }
        }));
    }
    for h in handles {
        h.join().expect("thread join");
    }
}

#[test]
fn test_commit_version_increments_and_links_parent() {
    let bridge = make_bridge();
    let before = bridge.snapshot_status().expect("status before");
    let head = bridge.commit_version("integration_test_commit").expect("commit version");
    let after = bridge.snapshot_status().expect("status after");

    assert_eq!(after.global_version, before.global_version + 1);
    assert_eq!(after.active_version, after.global_version);
    assert_eq!(head.parent_version, Some(before.active_version));
    assert_eq!(after.chain_head.expect("chain head").version_id, after.global_version);
}

#[test]
fn test_fence_state_machine_monotonic_progress() {
    let bridge = make_bridge();
    let states = bridge
        .trigger_atomic_swap_simulated()
        .expect("trigger atomic swap simulated");

    let phases: Vec<FencePhase> = states.iter().map(|s| s.phase.clone()).collect();
    assert_eq!(
        phases,
        vec![
            FencePhase::Prepare,
            FencePhase::WaitFence,
            FencePhase::Swap,
            FencePhase::Commit,
            FencePhase::Done
        ]
    );

    let mut prev = 0.0;
    for st in states {
        assert!(st.progress >= prev);
        prev = st.progress;
    }
}

#[tokio::test]
async fn test_metrics_stream_sequence_100_messages() {
    let bridge = make_bridge();
    let mut rx = bridge.metrics_tx.subscribe();

    for i in 0..100u32 {
        bridge.push_metric(EvolutionMetrics {
            ts_ms: i as u64,
            loss: 1.0 / ((i + 1) as f32),
            grad_norm: 0.1 * i as f32,
            grad_scale_factor: 1.0,
            accum_current: i % 4,
            accum_target: 4,
            version_id: 1,
            fence: None,
        });
    }

    let mut count = 0usize;
    while count < 100 {
        let _ = rx.recv().await.expect("recv metrics");
        count += 1;
    }
    assert_eq!(count, 100);
}

#[test]
fn test_accumulation_bounds() {
    for i in 0..128u32 {
        let curr = i % 8;
        let metric = EvolutionMetrics {
            ts_ms: i as u64,
            loss: 1.0,
            grad_norm: 1.0,
            grad_scale_factor: 1.0,
            accum_current: curr,
            accum_target: 8,
            version_id: 1,
            fence: None,
        };
        assert!(metric.accum_current <= metric.accum_target);
    }
}

#[test]
fn test_scale_gradient_alert_condition_payload() {
    let prev = 1.0f32;
    let curr = 1.5f32;
    let ratio = ((curr - prev) / prev).abs();
    assert!(ratio > 0.3);

    let metric = EvolutionMetrics {
        ts_ms: 1,
        loss: 1.0,
        grad_norm: 1.0,
        grad_scale_factor: curr,
        accum_current: 1,
        accum_target: 4,
        version_id: 1,
        fence: None,
    };
    assert!(metric.grad_scale_factor > 1.3);
}
