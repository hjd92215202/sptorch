use sptorch_distributed::coordinator;
use sptorch_distributed::worker::Worker;

#[tokio::test]
async fn test_register_and_heartbeat() {
    // Start coordinator in background
    let coord_handle = tokio::spawn(async {
        coordinator::start_coordinator("127.0.0.1:50051", 2).await.unwrap();
    });

    // Give server time to start
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    // Connect two workers
    let mut w1 = Worker::connect("127.0.0.1:50051", "worker-0", 2).await.unwrap();
    let w2 = Worker::connect("127.0.0.1:50051", "worker-1", 2).await.unwrap();

    assert_eq!(w1.rank, 0);
    assert_eq!(w2.rank, 1);
    assert_eq!(w1.world_size, 2);

    // Heartbeat
    let step = w1.heartbeat(0).await.unwrap();
    assert_eq!(step, 0);

    coord_handle.abort();
}

#[tokio::test]
async fn test_allreduce_two_workers() {
    let coord_handle = tokio::spawn(async {
        coordinator::start_coordinator("127.0.0.1:50052", 2).await.unwrap();
    });

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let mut w1 = Worker::connect("127.0.0.1:50052", "w0", 2).await.unwrap();
    let mut w2 = Worker::connect("127.0.0.1:50052", "w1", 2).await.unwrap();

    // Worker 0 has grads [1.0, 2.0, 3.0], Worker 1 has [3.0, 4.0, 5.0]
    // Expected average: [2.0, 3.0, 4.0]
    let (r1, r2) = tokio::join!(
        w1.allreduce(0, &[1.0, 2.0, 3.0], &[3]),
        w2.allreduce(0, &[3.0, 4.0, 5.0], &[3]),
    );

    let avg1 = r1.unwrap();
    let avg2 = r2.unwrap();

    assert_eq!(avg1, vec![2.0, 3.0, 4.0]);
    assert_eq!(avg2, vec![2.0, 3.0, 4.0]);

    coord_handle.abort();
}

#[tokio::test]
async fn test_barrier_sync() {
    let coord_handle = tokio::spawn(async {
        coordinator::start_coordinator("127.0.0.1:50053", 2).await.unwrap();
    });

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let mut w1 = Worker::connect("127.0.0.1:50053", "w0", 2).await.unwrap();
    let mut w2 = Worker::connect("127.0.0.1:50053", "w1", 2).await.unwrap();

    // Both workers hit barrier at step 1
    let (r1, r2) = tokio::join!(w1.barrier(1), w2.barrier(1),);

    r1.unwrap();
    r2.unwrap();

    // After barrier, global_step should be 1
    let step = w1.heartbeat(1).await.unwrap();
    assert_eq!(step, 1);

    coord_handle.abort();
}
