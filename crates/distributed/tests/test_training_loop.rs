use distributed::allreduce::average_gradients;
use distributed::coordinator;
use distributed::worker::Worker;

/// End-to-end distributed training simulation:
/// 2 workers each compute local gradients, allreduce via coordinator,
/// verify averaged gradients match expected values across multiple steps.
#[tokio::test]
async fn test_distributed_training_loop() {
    let coord_handle = tokio::spawn(async {
        coordinator::start_coordinator("127.0.0.1:50060", 2).await.unwrap();
    });

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let mut w0 = Worker::connect("127.0.0.1:50060", "train-0", 2).await.unwrap();
    let mut w1 = Worker::connect("127.0.0.1:50060", "train-1", 2).await.unwrap();

    // Simulate 3 training steps
    for step in 0..3u64 {
        // Each worker has different "gradients"
        let g0 = vec![1.0 * (step as f32 + 1.0), 2.0 * (step as f32 + 1.0)];
        let g1 = vec![3.0 * (step as f32 + 1.0), 4.0 * (step as f32 + 1.0)];

        let (r0, r1) = tokio::join!(w0.allreduce(0, &g0, &[2]), w1.allreduce(0, &g1, &[2]),);

        let avg0 = r0.unwrap();
        let avg1 = r1.unwrap();

        // Both workers should get the same averaged gradient
        assert_eq!(avg0, avg1, "step {}: workers got different results", step);

        // Verify correctness: average of g0 and g1
        let expected = average_gradients(&[g0, g1]);
        for i in 0..2 {
            assert!(
                (avg0[i] - expected[i]).abs() < 1e-6,
                "step {}: mismatch at [{}]: got {} expected {}",
                step,
                i,
                avg0[i],
                expected[i]
            );
        }

        // Barrier sync
        let (b0, b1) = tokio::join!(w0.barrier(step + 1), w1.barrier(step + 1),);
        b0.unwrap();
        b1.unwrap();
    }

    // Verify global step advanced
    let global = w0.heartbeat(3).await.unwrap();
    assert_eq!(global, 3);

    coord_handle.abort();
}
