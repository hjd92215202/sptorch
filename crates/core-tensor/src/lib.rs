use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use std::fmt;
use thiserror::Error;

// ============ Global Backend Registry ============

use std::collections::HashMap;
use std::sync::OnceLock;

/// Trait object for dispatching compute to registered backends.
pub trait BackendDispatch: Send + Sync {
    fn add_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]);
    fn mul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]);
    fn neg_f32(&self, a: &[f32], out: &mut [f32]);
    fn exp_f32(&self, a: &[f32], out: &mut [f32]);
    fn log_f32(&self, a: &[f32], out: &mut [f32]);
    fn relu_f32(&self, a: &[f32], out: &mut [f32]);
    fn gelu_f32(&self, a: &[f32], out: &mut [f32]);
    fn scale_f32(&self, a: &[f32], scalar: f32, out: &mut [f32]);
    fn matmul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize);
}

static BACKEND_REGISTRY: OnceLock<RwLock<HashMap<Device, Arc<dyn BackendDispatch>>>> = OnceLock::new();

fn registry() -> &'static RwLock<HashMap<Device, Arc<dyn BackendDispatch>>> {
    BACKEND_REGISTRY.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Register a backend for a specific device.
pub fn register_backend(device: Device, backend: Arc<dyn BackendDispatch>) {
    registry().write().unwrap().insert(device, backend);
}

/// Get the registered backend for a device, if any.
pub fn get_backend(device: &Device) -> Option<Arc<dyn BackendDispatch>> {
    registry().read().unwrap().get(device).cloned()
}

#[derive(Error, Debug)]
pub enum TensorError {
    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    #[error("device mismatch: expected {expected:?}, got {got:?}")]
    DeviceMismatch { expected: Device, got: Device },

    #[error("dtype mismatch: expected {expected:?}, got {got:?}")]
    DTypeMismatch { expected: DType, got: DType },

    #[error("invalid shape: {0}")]
    InvalidShape(String),

    #[error("lock poisoned")]
    LockPoisoned,

    #[error("device storage error: {0}")]
    DeviceError(String),
}

pub type Result<T> = std::result::Result<T, TensorError>;

pub trait Op: std::fmt::Debug + Send + Sync {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>>;
}

#[derive(Debug)]
pub struct Node {
    pub op: Box<dyn Op>,
    pub inputs: Vec<Tensor>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    CPU,
    Cuda(usize),
    Custom(u16),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DType { F32, F16, BF16 }

impl Default for DType {
    fn default() -> Self { DType::F32 }
}

// ============ F16/BF16 conversion utilities ============

pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    if exp == 255 {
        // Inf/NaN
        return (sign | 0x7C00 | (mant >> 13).min(1) as u32) as u16;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return (sign | 0x7C00) as u16; // overflow -> Inf
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign as u16; // underflow -> 0
        }
        let m = (mant | 0x800000) >> (1 - new_exp);
        return (sign | (m >> 13)) as u16;
    }

    (sign | ((new_exp as u32) << 10) | (mant >> 13)) as u16
}

pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut m = mant;
        let mut e = 0i32;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f32_exp = (127 - 15 + 1 + e) as u32;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }
    if exp == 31 {
        return f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13));
    }

    let f32_exp = exp + 127 - 15;
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
}

pub fn f32_to_bf16(val: f32) -> u16 {
    (val.to_bits() >> 16) as u16
}

pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Trait for opaque device memory buffers (GPU, NPU, etc.).
/// Implementors hold device-side memory and provide host transfer.
pub trait DeviceBuffer: Send + Sync + fmt::Debug {
    fn device(&self) -> Device;
    fn len(&self) -> usize;
    fn to_host(&self) -> Vec<f32>;
    fn from_host(data: &[f32], device: Device) -> std::result::Result<Box<dyn DeviceBuffer>, String>
    where
        Self: Sized;
}

pub enum Storage {
    Cpu(Vec<f32>),
    Device(Box<dyn DeviceBuffer>),
}

impl Storage {
    pub fn cpu(data: Vec<f32>) -> Self {
        Storage::Cpu(data)
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Storage::Cpu(_))
    }

    pub fn as_cpu_slice(&self) -> &[f32] {
        match self {
            Storage::Cpu(v) => v,
            Storage::Device(_) => panic!("cannot borrow device storage as CPU slice; call to_cpu_vec() first"),
        }
    }

    pub fn as_cpu_slice_mut(&mut self) -> &mut [f32] {
        match self {
            Storage::Cpu(v) => v,
            Storage::Device(_) => panic!("cannot mutably borrow device storage as CPU slice"),
        }
    }

    pub fn to_cpu_vec(&self) -> Vec<f32> {
        match self {
            Storage::Cpu(v) => v.clone(),
            Storage::Device(buf) => buf.to_host(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Storage::Cpu(v) => v.len(),
            Storage::Device(buf) => buf.len(),
        }
    }

    pub fn device(&self) -> Device {
        match self {
            Storage::Cpu(_) => Device::CPU,
            Storage::Device(buf) => buf.device(),
        }
    }
}

impl fmt::Debug for Storage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Storage::Cpu(v) => f.debug_tuple("Cpu").field(v).finish(),
            Storage::Device(buf) => f.debug_tuple("Device").field(buf).finish(),
        }
    }
}

#[derive(Debug)]
pub struct TensorInner {
    pub storage: Arc<RwLock<Storage>>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
    pub dtype: DType,
    pub device: Device,
    pub requires_grad: bool,
    pub grad: Option<Tensor>,
    pub creator: Option<Arc<Node>>,
}

#[derive(Clone)]
pub struct Tensor(pub Arc<RwLock<TensorInner>>);

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let strides = compute_strides(&shape);
        let inner = TensorInner {
            storage: Arc::new(RwLock::new(Storage::cpu(data))),
            shape,
            strides,
            offset: 0,
            dtype: DType::F32,
            device: Device::CPU,
            requires_grad: false,
            grad: None,
            creator: None,
        };
        Tensor(Arc::new(RwLock::new(inner)))
    }

    pub fn with_grad(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Self {
        let t = Self::new(data, shape);
        t.0.write().unwrap().requires_grad = requires_grad;
        t
    }

    pub fn data(&self) -> Vec<f32> {
        self.0.read().unwrap().storage.read().unwrap().to_cpu_vec()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.0.read().unwrap().shape.clone()
    }

    pub fn strides(&self) -> Vec<usize> {
        self.0.read().unwrap().strides.clone()
    }

    pub fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    pub fn device(&self) -> Device {
        self.0.read().unwrap().device
    }

    pub fn dtype(&self) -> DType {
        self.0.read().unwrap().dtype
    }

    /// Convert tensor to a different dtype. Returns a new tensor with converted data.
    /// Computation always happens in f32; this controls storage precision.
    pub fn to_dtype(&self, target: DType) -> Tensor {
        let current = self.dtype();
        if current == target {
            return self.clone();
        }
        let f32_data = self.contiguous_data();
        let shape = self.shape();

        let stored: Vec<f32> = match target {
            DType::F32 => f32_data,
            DType::F16 => {
                // Round-trip through f16 to simulate precision loss
                f32_data.iter().map(|&v| f16_to_f32(f32_to_f16(v))).collect()
            }
            DType::BF16 => {
                f32_data.iter().map(|&v| bf16_to_f32(f32_to_bf16(v))).collect()
            }
        };

        let t = Tensor::new(stored, shape);
        t.0.write().unwrap().dtype = target;
        t
    }

    /// Shorthand for to_dtype(DType::F16)
    pub fn half(&self) -> Tensor {
        self.to_dtype(DType::F16)
    }

    /// Shorthand for to_dtype(DType::BF16)
    pub fn bfloat16(&self) -> Tensor {
        self.to_dtype(DType::BF16)
    }

    /// Shorthand for to_dtype(DType::F32)
    pub fn float(&self) -> Tensor {
        self.to_dtype(DType::F32)
    }

    pub fn is_contiguous(&self) -> bool {
        let inner = self.0.read().unwrap();
        inner.strides == compute_strides(&inner.shape)
    }

    pub fn contiguous_data(&self) -> Vec<f32> {
        let inner = self.0.read().unwrap();
        let storage = inner.storage.read().unwrap();
        let expected_strides = compute_strides(&inner.shape);
        if inner.offset == 0 && inner.strides == expected_strides {
            return storage.to_cpu_vec();
        }
        let cpu_data = storage.to_cpu_vec();
        let numel: usize = inner.shape.iter().product();
        let ndim = inner.shape.len();
        let mut result = Vec::with_capacity(numel);
        let mut indices = vec![0usize; ndim];
        for _ in 0..numel {
            let physical: usize = inner.offset
                + indices.iter().zip(inner.strides.iter()).map(|(i, s)| i * s).sum::<usize>();
            result.push(cpu_data[physical]);
            for d in (0..ndim).rev() {
                indices[d] += 1;
                if indices[d] < inner.shape[d] {
                    break;
                }
                indices[d] = 0;
            }
        }
        result
    }

    pub fn requires_grad(&self) -> bool {
        self.0.read().unwrap().requires_grad
    }

    pub fn accum_grad(&self, grad_tensor: &Tensor) {
        let incoming_data = grad_tensor.data();

        let mut inner = self.0.write().unwrap();
        if !inner.requires_grad { return; }

        if let Some(existing_grad) = &inner.grad {
            let g_inner = existing_grad.0.write().unwrap();
            let mut g_storage = g_inner.storage.write().unwrap();
            let g_slice = g_storage.as_cpu_slice_mut();
            for i in 0..g_slice.len() {
                g_slice[i] += incoming_data[i];
            }
        } else {
            inner.grad = Some(Tensor::new(incoming_data, inner.shape.clone()));
        }
    }

    pub fn grad(&self) -> Option<Vec<f32>> {
        let inner = self.0.read().unwrap();
        inner.grad.as_ref().map(|g| g.data())
    }

    pub fn backward(&self) {
        let shape = self.shape();
        let size: usize = shape.iter().product();
        let seed = Tensor::new(vec![1.0; size], shape);

        self.accum_grad(&seed);

        let mut queue: VecDeque<(Tensor, Tensor)> = VecDeque::new();
        queue.push_back((self.clone(), seed));

        while let Some((tensor, grad_output)) = queue.pop_front() {
            let creator = {
                let inner = tensor.0.read().unwrap();
                inner.creator.clone()
            };

            if let Some(node) = creator {
                let input_grads = node.op.backward(&grad_output);

                for (input, maybe_grad) in node.inputs.iter().zip(input_grads.into_iter()) {
                    if let Some(grad) = maybe_grad {
                        input.accum_grad(&grad);

                        let has_creator = input.0.read().unwrap().creator.is_some();
                        if has_creator {
                            queue.push_back((input.clone(), grad));
                        }
                    }
                }
            }
        }
    }
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.0.read().unwrap();
        f.debug_struct("Tensor")
            .field("data", &inner.storage)
            .field("shape", &inner.shape)
            .field("requires_grad", &inner.requires_grad)
            .field("device", &inner.device)
            .field("dtype", &inner.dtype)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_f16_roundtrip() {
        let vals = [0.0f32, 1.0, -1.0, 0.5, 65504.0, -65504.0, 1e-4];
        for &v in &vals {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            assert!((back - v).abs() / (v.abs() + 1e-10) < 0.01,
                "f16 roundtrip failed for {}: got {}", v, back);
        }
    }

    #[test]
    fn test_f32_bf16_roundtrip() {
        let vals = [0.0f32, 1.0, -1.0, 3.14, 1e10, -1e10];
        for &v in &vals {
            let b = f32_to_bf16(v);
            let back = bf16_to_f32(b);
            assert!((back - v).abs() / (v.abs() + 1e-10) < 0.01,
                "bf16 roundtrip failed for {}: got {}", v, back);
        }
    }

    #[test]
    fn test_tensor_dtype() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(t.dtype(), DType::F32);
    }

    #[test]
    fn test_tensor_half() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let h = t.half();
        assert_eq!(h.dtype(), DType::F16);
        assert_eq!(h.shape(), vec![3]);
        let d = h.data();
        assert!((d[0] - 1.0).abs() < 1e-3);
        assert!((d[1] - 2.0).abs() < 1e-3);
        assert!((d[2] - 3.0).abs() < 1e-3);
    }

    #[test]
    fn test_tensor_bfloat16() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = t.bfloat16();
        assert_eq!(b.dtype(), DType::BF16);
        let d = b.data();
        assert!((d[0] - 1.0).abs() < 0.02);
        assert!((d[1] - 2.0).abs() < 0.02);
        assert!((d[2] - 3.0).abs() < 0.02);
    }

    #[test]
    fn test_tensor_float_roundtrip() {
        let t = Tensor::new(vec![1.5, 2.5, 3.5], vec![3]);
        let h = t.half();
        let back = h.float();
        assert_eq!(back.dtype(), DType::F32);
        let d = back.data();
        assert!((d[0] - 1.5).abs() < 1e-3);
    }

    #[test]
    fn test_tensor_to_dtype_noop() {
        let t = Tensor::new(vec![1.0], vec![1]);
        let same = t.to_dtype(DType::F32);
        // Should be the same Arc (clone, not copy)
        assert_eq!(same.data(), t.data());
    }

    #[test]
    fn test_f16_precision_loss() {
        // f16 has ~3 decimal digits of precision
        let t = Tensor::new(vec![1.001], vec![1]);
        let h = t.half();
        let d = h.data();
        // Should lose some precision but be close
        assert!((d[0] - 1.001).abs() < 0.002);
    }
}
