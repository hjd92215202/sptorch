use core_tensor::{Tensor, Op, Node};
use std::sync::Arc;

#[derive(Debug)]
pub struct AddOp {
    pub input_a: Tensor,
    pub input_b: Tensor,
}

impl Op for AddOp {
    fn backward(&self, grad_output: &Tensor) {
        // 对于 z = a + b
        // dL/da = dL/dz * 1
        // dL/db = dL/dz * 1
        self.input_a.backward_step(grad_output);
        self.input_b.backward_step(grad_output);
    }
}

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let a_data = a.data();
    let b_data = b.data();
    let shape = a.shape();
    
    // 1. 真实的前向数学计算
    let res_data: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(x, y)| x + y).collect();
    let res = Tensor::new(res_data, shape);

    // 2. 只有当输入需要梯度时，才构建计算图
    let a_req = a.0.read().unwrap().requires_grad;
    let b_req = b.0.read().unwrap().requires_grad;

    if a_req || b_req {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(Arc::new(Node {
            op: Box::new(AddOp {
                input_a: a.clone(),
                input_b: b.clone(),
            }),
            inputs: vec![a.clone(), b.clone()],
        }));
    }
    
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autograd_basic() {
        // x = [2.0], y = [3.0]
        let x = Tensor::with_grad(vec![2.0], vec![1], true);
        let y = Tensor::with_grad(vec![3.0], vec![1], true);
        
        // z = x + y = [5.0]
        let z = add(&x, &y);
        assert_eq!(z.data(), vec![5.0]);

        // 反向传播
        z.backward();

        // 验证梯度：dz/dx = 1.0, dz/dy = 1.0
        let x_grad = x.0.read().unwrap().grad.as_ref().unwrap().data();
        let y_grad = y.0.read().unwrap().grad.as_ref().unwrap().data();
        
        assert_eq!(x_grad, vec![1.0]);
        assert_eq!(y_grad, vec![1.0]);
    }
}