//! Multi-GPU DataParallel: split batch across devices, forward on each, allreduce gradients.

use core_tensor::{Tensor, Device};

/// DataParallel wrapper: distributes batches across multiple GPU devices.
pub struct DataParallel {
    pub devices: Vec<Device>,
    pub world_size: usize,
}

impl DataParallel {
    pub fn new(num_gpus: usize) -> Self {
        let devices: Vec<Device> = (0..num_gpus).map(|i| Device::Cuda(i)).collect();
        DataParallel {
            world_size: devices.len(),
            devices,
        }
    }

    /// Split a batch of data evenly across devices.
    /// Input shape: [batch, ...], returns Vec of [batch/world_size, ...] per device.
    pub fn scatter(&self, data: &[f32], batch_size: usize, feature_dim: usize) -> Vec<(Vec<f32>, Device)> {
        let chunk_size = batch_size / self.world_size;
        let remainder = batch_size % self.world_size;

        let mut chunks = Vec::new();
        let mut offset = 0;

        for (i, &dev) in self.devices.iter().enumerate() {
            let this_chunk = chunk_size + if i < remainder { 1 } else { 0 };
            let start = offset * feature_dim;
            let end = (offset + this_chunk) * feature_dim;
            chunks.push((data[start..end].to_vec(), dev));
            offset += this_chunk;
        }

        chunks
    }

    /// Allreduce gradients across devices (simple average).
    /// In a real multi-GPU setup this would use NCCL; here we simulate locally.
    pub fn allreduce_grads(&self, param_grads: &[Vec<Vec<f32>>]) -> Vec<Vec<f32>> {
        if param_grads.is_empty() {
            return Vec::new();
        }
        let num_params = param_grads[0].len();
        let mut averaged = Vec::with_capacity(num_params);

        for p in 0..num_params {
            let param_len = param_grads[0][p].len();
            let mut sum = vec![0.0f32; param_len];
            for device_grads in param_grads {
                for (s, g) in sum.iter_mut().zip(device_grads[p].iter()) {
                    *s += g;
                }
            }
            let avg: Vec<f32> = sum.iter().map(|v| v / self.world_size as f32).collect();
            averaged.push(avg);
        }

        averaged
    }
}

/// Simulate a DataParallel training step:
/// 1. Scatter batch across devices
/// 2. Forward + backward on each device (simulated)
/// 3. Allreduce gradients
/// 4. Update parameters
pub fn data_parallel_step(
    dp: &DataParallel,
    params: &[Tensor],
    batch_grads: &[Vec<Vec<f32>>], // [device][param][values]
    lr: f32,
) {
    let averaged = dp.allreduce_grads(batch_grads);

    // Apply averaged gradients to params
    for (param, avg_grad) in params.iter().zip(averaged.iter()) {
        let inner = param.0.read().unwrap();
        let mut storage = inner.storage.write().unwrap();
        let w = storage.as_cpu_slice_mut();
        for (w_i, g_i) in w.iter_mut().zip(avg_grad.iter()) {
            *w_i -= lr * g_i;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_parallel_scatter() {
        let dp = DataParallel::new(2);
        // batch=4, feature_dim=3
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let chunks = dp.scatter(&data, 4, 3);

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].0, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]); // first 2 samples
        assert_eq!(chunks[1].0, vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0]); // last 2 samples
        assert_eq!(chunks[0].1, Device::Cuda(0));
        assert_eq!(chunks[1].1, Device::Cuda(1));
    }

    #[test]
    fn test_data_parallel_scatter_uneven() {
        let dp = DataParallel::new(3);
        // batch=5, feature_dim=2 -> chunks of 2,2,1
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let chunks = dp.scatter(&data, 5, 2);

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].0.len(), 4); // 2 samples * 2 features
        assert_eq!(chunks[1].0.len(), 4); // 2 samples
        assert_eq!(chunks[2].0.len(), 2); // 1 sample
    }

    #[test]
    fn test_allreduce_grads() {
        let dp = DataParallel::new(2);
        // 2 devices, 1 param with 3 values each
        let grads = vec![
            vec![vec![1.0, 2.0, 3.0]], // device 0
            vec![vec![3.0, 4.0, 5.0]], // device 1
        ];
        let avg = dp.allreduce_grads(&grads);
        assert_eq!(avg, vec![vec![2.0, 3.0, 4.0]]);
    }

    #[test]
    fn test_data_parallel_step() {
        let dp = DataParallel::new(2);
        let params = vec![Tensor::new(vec![1.0, 2.0, 3.0], vec![3])];

        let batch_grads = vec![
            vec![vec![0.1, 0.2, 0.3]], // device 0 grads
            vec![vec![0.3, 0.4, 0.5]], // device 1 grads
        ];

        data_parallel_step(&dp, &params, &batch_grads, 1.0);

        let w = params[0].data();
        // avg grad = [0.2, 0.3, 0.4], new w = [1.0-0.2, 2.0-0.3, 3.0-0.4] = [0.8, 1.7, 2.6]
        assert!((w[0] - 0.8).abs() < 1e-6);
        assert!((w[1] - 1.7).abs() < 1e-6);
        assert!((w[2] - 2.6).abs() < 1e-6);
    }

    #[test]
    fn test_data_parallel_multi_param() {
        let dp = DataParallel::new(3);
        let params = vec![
            Tensor::new(vec![1.0, 1.0], vec![2]),
            Tensor::new(vec![2.0, 2.0, 2.0], vec![3]),
        ];

        let batch_grads = vec![
            vec![vec![0.3, 0.3], vec![0.6, 0.6, 0.6]],
            vec![vec![0.6, 0.6], vec![0.9, 0.9, 0.9]],
            vec![vec![0.9, 0.9], vec![1.2, 1.2, 1.2]],
        ];

        data_parallel_step(&dp, &params, &batch_grads, 0.1);

        let w0 = params[0].data();
        // avg grad for param0 = [(0.3+0.6+0.9)/3, ...] = [0.6, 0.6]
        // new w0 = [1.0 - 0.1*0.6, 1.0 - 0.1*0.6] = [0.94, 0.94]
        assert!((w0[0] - 0.94).abs() < 1e-6);
    }
}
