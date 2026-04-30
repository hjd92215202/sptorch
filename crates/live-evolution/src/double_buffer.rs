use core_tensor::Tensor;
use std::sync::{Arc, RwLock, atomic::{AtomicBool, Ordering}};

/// Double-buffered parameter store for concurrent train/inference.
///
/// Two copies of model parameters: "active" (used by inference) and "shadow"
/// (being updated by training). After a training step completes, an atomic
/// swap makes the shadow become active.
pub struct DoubleBufferParams {
    buf_a: Vec<Tensor>,
    buf_b: Vec<Tensor>,
    /// true = buf_a is active (inference), buf_b is shadow (training)
    a_is_active: Arc<AtomicBool>,
    /// Guards the swap operation
    swap_lock: Arc<RwLock<()>>,
}

impl DoubleBufferParams {
    /// Create from an initial set of parameters. Clones them into both buffers.
    pub fn new(params: &[Tensor]) -> Self {
        let buf_a: Vec<Tensor> = params.iter().map(clone_tensor).collect();
        let buf_b: Vec<Tensor> = params.iter().map(clone_tensor).collect();
        DoubleBufferParams {
            buf_a,
            buf_b,
            a_is_active: Arc::new(AtomicBool::new(true)),
            swap_lock: Arc::new(RwLock::new(())),
        }
    }

    /// Get a reference to the active (inference) parameters.
    /// Note: caller must not hold this reference across a `swap()` call.
    pub fn active_params(&self) -> &[Tensor] {
        if self.a_is_active.load(Ordering::Acquire) {
            &self.buf_a
        } else {
            &self.buf_b
        }
    }

    /// Get a reference to the shadow (training) parameters.
    /// Note: caller must not hold this reference across a `swap()` call.
    pub fn shadow_params(&self) -> &[Tensor] {
        if self.a_is_active.load(Ordering::Acquire) {
            &self.buf_b
        } else {
            &self.buf_a
        }
    }

    /// Get a clone of the active parameters' data (safe across swap).
    pub fn active_params_snapshot(&self) -> Vec<Vec<f32>> {
        let _guard = self.swap_lock.read().unwrap();
        let params = if self.a_is_active.load(Ordering::Acquire) {
            &self.buf_a
        } else {
            &self.buf_b
        };
        params.iter().map(|p| p.contiguous_data()).collect()
    }

    /// Atomic swap: shadow becomes active, old active becomes new shadow.
    /// Call this after a training step completes.
    pub fn swap(&self) {
        let _guard = self.swap_lock.write().unwrap();
        let prev = self.a_is_active.load(Ordering::Acquire);
        self.a_is_active.store(!prev, Ordering::Release);
    }

    /// Copy active params into shadow (reset shadow to current active state).
    /// Useful before starting a new training epoch.
    pub fn sync_shadow_from_active(&self) {
        let _guard = self.swap_lock.write().unwrap();
        let (src, dst) = if self.a_is_active.load(Ordering::Acquire) {
            (&self.buf_a, &self.buf_b)
        } else {
            (&self.buf_b, &self.buf_a)
        };
        for (s, d) in src.iter().zip(dst.iter()) {
            let src_data = s.contiguous_data();
            let d_inner = d.0.read().unwrap();
            let mut d_storage = d_inner.storage.write().unwrap();
            d_storage.as_cpu_slice_mut().copy_from_slice(&src_data);
        }
    }

    pub fn num_params(&self) -> usize {
        self.buf_a.len()
    }
}

fn clone_tensor(t: &Tensor) -> Tensor {
    let data = t.contiguous_data();
    let shape = t.shape();
    Tensor::with_grad(data, shape, t.requires_grad())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_double_buffer_initial_state() {
        let p = vec![Tensor::new(vec![1.0, 2.0, 3.0], vec![3])];
        let db = DoubleBufferParams::new(&p);
        assert_eq!(db.num_params(), 1);
        assert_eq!(db.active_params()[0].data(), vec![1.0, 2.0, 3.0]);
        assert_eq!(db.shadow_params()[0].data(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_double_buffer_swap() {
        let p = vec![Tensor::new(vec![1.0, 2.0], vec![2])];
        let db = DoubleBufferParams::new(&p);

        // Modify shadow
        {
            let shadow = db.shadow_params();
            let inner = shadow[0].0.read().unwrap();
            let mut storage = inner.storage.write().unwrap();
            let s = storage.as_cpu_slice_mut();
            s[0] = 10.0;
            s[1] = 20.0;
        }

        // Before swap: active is still [1, 2]
        assert_eq!(db.active_params()[0].data(), vec![1.0, 2.0]);

        // Swap
        db.swap();

        // After swap: active is now [10, 20]
        assert_eq!(db.active_params()[0].data(), vec![10.0, 20.0]);
        // Old active is now shadow: [1, 2]
        assert_eq!(db.shadow_params()[0].data(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_sync_shadow_from_active() {
        let p = vec![Tensor::new(vec![5.0, 6.0], vec![2])];
        let db = DoubleBufferParams::new(&p);

        // Modify shadow to something different
        {
            let shadow = db.shadow_params();
            let inner = shadow[0].0.read().unwrap();
            let mut storage = inner.storage.write().unwrap();
            storage.as_cpu_slice_mut()[0] = 99.0;
        }
        assert_eq!(db.shadow_params()[0].data()[0], 99.0);

        // Sync shadow from active
        db.sync_shadow_from_active();
        assert_eq!(db.shadow_params()[0].data(), vec![5.0, 6.0]);
    }
}
