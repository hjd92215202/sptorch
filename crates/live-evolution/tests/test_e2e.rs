use core_tensor::Tensor;
use live_evolution::double_buffer::DoubleBufferParams;
use live_evolution::incremental::IncrementalTrainer;
use live_evolution::monitor::{TrainingMonitor, MonitorAction};
use optim::SGD;

/// End-to-end live evolution test:
/// DoubleBuffer + IncrementalTrainer + Monitor working together.
/// Simulates: push data → trigger training → monitor loss → swap params.
#[test]
fn test_live_evolution_end_to_end() {
    // Create a simple "model" with 1 parameter
    let params = vec![Tensor::with_grad(vec![0.5, 0.5, 0.5, 0.5], vec![2, 2], true)];

    // Double buffer
    let db = DoubleBufferParams::new(&params);

    // Incremental trainer (micro_batch_size=2)
    let opt = SGD::new(params.clone(), 0.01, 0.0);
    let mut trainer = IncrementalTrainer::new(opt, params.clone(), 2);

    // Monitor (window=3, threshold=50%)
    let mut monitor = TrainingMonitor::new(3, 0.5);

    // Simulate data arriving
    let samples = vec![
        (vec![0, 1], vec![1, 2]),
        (vec![2, 3], vec![3, 4]),
        (vec![4, 5], vec![5, 6]),
        (vec![6, 7], vec![7, 0]),
    ];

    let mut triggered_count = 0;

    for (input, target) in &samples {
        let should_train = trainer.push_sample(input.clone(), target.clone());
        if should_train {
            triggered_count += 1;
            let _batch = trainer.drain_batch();
            trainer.step_completed();

            // Simulate a decreasing loss
            let fake_loss = 2.0 - 0.3 * triggered_count as f32;
            let action = monitor.record_loss(fake_loss);

            match action {
                MonitorAction::Continue => {
                    // Training went well, swap params
                    db.swap();
                }
                MonitorAction::Rollback { .. } => {
                    // Would rollback in real scenario
                    db.sync_shadow_from_active();
                }
            }
        }
    }

    // Verify: 4 samples with batch_size=2 → 2 training triggers
    assert_eq!(triggered_count, 2);
    assert_eq!(trainer.total_steps(), 2);
    assert_eq!(monitor.total_samples(), 2);

    // Monitor should not have triggered rollback (loss was decreasing)
    assert_eq!(monitor.rollback_count(), 0);

    // Double buffer should have swapped twice
    // (active/shadow state depends on even/odd swaps)
    assert_eq!(db.num_params(), 1);
}

/// Test rollback scenario: monitor detects degradation, triggers rollback.
#[test]
fn test_live_evolution_rollback_scenario() {
    let params = vec![Tensor::with_grad(vec![1.0, 2.0], vec![2], true)];
    let db = DoubleBufferParams::new(&params);
    let mut monitor = TrainingMonitor::new(2, 0.1); // tight threshold

    // Good phase: loss = 1.0
    monitor.record_loss(1.0);
    monitor.record_loss(1.0); // best_avg = 1.0
    db.swap();

    // Bad phase: loss spikes
    let action = monitor.record_loss(2.0);
    match action {
        MonitorAction::Rollback { .. } => {
            // Rollback: restore shadow from active
            db.sync_shadow_from_active();
            monitor.reset_after_rollback();
        }
        MonitorAction::Continue => {
            // Might not trigger yet with window=2
            db.swap();
        }
    }

    let action2 = monitor.record_loss(2.0);
    if let MonitorAction::Rollback { .. } = action2 {
        db.sync_shadow_from_active();
        monitor.reset_after_rollback();
    }

    // Verify rollback was triggered at least once
    assert!(monitor.rollback_count() >= 1);
}
