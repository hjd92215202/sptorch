//! End-to-end test: register CUDA backend, create tensors on Cuda(0),
//! run autograd ops (add/mul/matmul + backward), verify GPU dispatch works.

use core_tensor::{Tensor, Device};
use core_ops::{add, mul, matmul, sum};

#[test]
fn test_gpu_autograd_forward_backward() {
    // Register CUDA backend
    runtime_cuda::register_cuda_backend();

    let dev = Device::Cuda(0);

    // Create tensors and mark as Cuda(0)
    let a = Tensor::with_grad(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
    a.0.write().unwrap().device = dev;

    let b = Tensor::with_grad(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], true);
    b.0.write().unwrap().device = dev;

    // Forward: c = a + b (should dispatch to GPU)
    let c = add(&a, &b);
    assert_eq!(c.data(), vec![6.0, 8.0, 10.0, 12.0]);

    // Forward: d = a * b (element-wise, GPU)
    let d = mul(&a, &b);
    assert_eq!(d.data(), vec![5.0, 12.0, 21.0, 32.0]);

    // Forward: e = a @ b (matmul, cuBLAS)
    let e = matmul(&a, &b);
    // [1,2;3,4] @ [5,6;7,8] = [19,22;43,50]
    let e_data = e.data();
    assert!((e_data[0] - 19.0).abs() < 0.1);
    assert!((e_data[1] - 22.0).abs() < 0.1);
    assert!((e_data[2] - 43.0).abs() < 0.1);
    assert!((e_data[3] - 50.0).abs() < 0.1);

    // Backward through sum(e)
    let loss = sum(&e);
    loss.backward();

    // a should have gradients
    let a_grad = a.grad().expect("a should have gradient after backward");
    assert_eq!(a_grad.len(), 4);
    // d(sum(A@B))/dA = ones @ B^T = [[5+6, 7+8],[5+6, 7+8]] = ... actually:
    // d(sum(C))/dA where C=A@B: grad_C = ones[2,2], dA = grad_C @ B^T
    // B^T = [[5,7],[6,8]], ones@B^T = [[11,15],[11,15]]
    assert!((a_grad[0] - 11.0).abs() < 0.2, "a_grad[0]={}", a_grad[0]);
    assert!((a_grad[1] - 15.0).abs() < 0.2, "a_grad[1]={}", a_grad[1]);

    let b_grad = b.grad().expect("b should have gradient after backward");
    assert_eq!(b_grad.len(), 4);
    // dB = A^T @ grad_C, A^T = [[1,3],[2,4]], A^T @ ones = [[4,4],[6,6]]
    assert!((b_grad[0] - 4.0).abs() < 0.2, "b_grad[0]={}", b_grad[0]);
    assert!((b_grad[1] - 4.0).abs() < 0.2, "b_grad[1]={}", b_grad[1]);
}

#[test]
fn test_gpu_autograd_training_step() {
    runtime_cuda::register_cuda_backend();

    let dev = Device::Cuda(0);

    // Simple "model": y = x @ w, loss = sum(y)
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    x.0.write().unwrap().device = dev;

    let w = Tensor::with_grad(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], vec![3, 2], true);
    w.0.write().unwrap().device = dev;

    // Forward
    let y = matmul(&x, &w);
    let loss = sum(&y);
    let loss_val = loss.data()[0];

    // Backward
    loss.backward();

    let w_grad = w.grad().expect("w should have gradient");
    assert_eq!(w_grad.len(), 6);

    // Manual SGD step
    let lr = 0.01f32;
    let w_data = w.data();
    let new_w: Vec<f32> = w_data.iter().zip(w_grad.iter()).map(|(p, g)| p - lr * g).collect();

    // Verify weights changed
    let diff: f32 = w_data.iter().zip(new_w.iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 0.0, "SGD step should change weights");

    // Verify loss is finite
    assert!(loss_val.is_finite(), "loss should be finite: {}", loss_val);
}
