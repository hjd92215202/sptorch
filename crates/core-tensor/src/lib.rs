use std::sync::{Arc, RwLock};
use std::fmt;

pub trait Op: std::fmt::Debug + Send + Sync {
    fn backward(&self, grad_output: &Tensor);
}

pub struct Node {
    pub op: Box<dyn Op>,
    pub inputs: Vec<Tensor>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Device { CPU }

pub struct Storage {
    pub data: Vec<f32>,
}

pub struct TensorInner {
    pub storage: Arc<RwLock<Storage>>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
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

    // --- 新增：获取数据拷贝 (用于前向计算) ---
    pub fn data(&self) -> Vec<f32> {
        self.0.read().unwrap().storage.read().unwrap().data.clone()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.0.read().unwrap().shape.clone()
    }

    // --- 新增：梯度累加逻辑 ---
    pub fn accum_grad(&self, grad_tensor: &Tensor) {
        let mut inner = self.0.write().unwrap();
        if !inner.requires_grad { return; }

        if let Some(existing_grad) = &inner.grad {
            let g_inner = existing_grad.0.write().unwrap();
            let mut g_storage = g_inner.storage.write().unwrap();
            let incoming_data = grad_tensor.data();
            for i in 0..g_storage.data.len() {
                g_storage.data[i] += incoming_data[i];
            }
        } else {
            inner.grad = Some(grad_tensor.clone());
        }
    }

    // --- 新增：反向传播递归入口 ---
    pub fn backward(&self) {
        let shape = self.shape();
        let size = shape.iter().product();
        // 种子梯度：dL/dloss = 1.0
        let grad_output = Tensor::new(vec![1.0; size], shape);
        self.backward_step(&grad_output);
    }

    pub fn backward_step(&self, grad_output: &Tensor) {
        self.accum_grad(grad_output);
        let inner = self.0.read().unwrap();
        if let Some(creator) = &inner.creator {
            creator.op.backward(grad_output);
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
        // 只打印核心元数据，不递归打印 creator，避免死循环
        f.debug_struct("Tensor")
            .field("shape", &inner.shape)
            .field("requires_grad", &inner.requires_grad)
            .field("device", &inner.device)
            .finish()
    }
}