use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use std::fmt;
use thiserror::Error;

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Device { CPU }

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DType { F32, F16, BF16 }

impl Default for DType {
    fn default() -> Self { DType::F32 }
}

#[derive(Debug)]
pub struct Storage {
    pub data: Vec<f32>,
}

#[derive(Debug)]
pub struct TensorInner {
    pub storage: Arc<RwLock<Storage>>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
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
            storage: Arc::new(RwLock::new(Storage { data })),
            shape,
            strides,
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
        self.0.read().unwrap().storage.read().unwrap().data.clone()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.0.read().unwrap().shape.clone()
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
            for i in 0..g_storage.data.len() {
                g_storage.data[i] += incoming_data[i];
            }
        } else {
            inner.grad = Some(Tensor::new(incoming_data, inner.shape.clone()));
        }
    }

    pub fn grad(&self) -> Option<Vec<f32>> {
        let inner = self.0.read().unwrap();
        inner.grad.as_ref().map(|g| g.data())
    }

    /// 拓扑排序迭代式反向传播
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
            .finish()
    }
}
