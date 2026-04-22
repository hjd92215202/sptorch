use core_tensor::Tensor;
use core_ops::*;
use rand::Rng;

// ============ Module Trait ============

pub trait Module: Send + Sync {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;
}

// ============ Initialization ============

pub fn xavier_uniform(rows: usize, cols: usize) -> Tensor {
    let mut rng = rand::thread_rng();
    let limit = (6.0 / (rows + cols) as f32).sqrt();
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| rng.gen_range(-limit..limit))
        .collect();
    Tensor::with_grad(data, vec![rows, cols], true)
}

pub fn kaiming_normal(rows: usize, cols: usize) -> Tensor {
    let mut rng = rand::thread_rng();
    let std = (2.0 / rows as f32).sqrt();
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| {
            // Box-Muller transform for normal distribution
            let u1: f32 = rng.gen_range(1e-7..1.0);
            let u2: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
            std * (-2.0 * u1.ln()).sqrt() * u2.cos()
        })
        .collect();
    Tensor::with_grad(data, vec![rows, cols], true)
}

fn zeros_grad(size: usize) -> Tensor {
    Tensor::with_grad(vec![0.0; size], vec![size], true)
}

// ============ Linear ============

pub struct Linear {
    pub weight: Tensor, // [out_features, in_features]
    pub bias: Option<Tensor>, // [out_features]
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Self {
        let weight = xavier_uniform(out_features, in_features);
        let bias = if use_bias {
            Some(zeros_grad(out_features))
        } else {
            None
        };
        Linear { weight, bias }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        // input: [batch, in_features], weight: [out, in]
        // output = input @ weight^T + bias = [batch, out]
        let wt = transpose(&self.weight);
        let out = matmul(input, &wt);
        if let Some(ref bias) = self.bias {
            broadcast_add(&out, bias)
        } else {
            out
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }
}

// ============ Embedding ============

pub struct Embedding {
    pub weight: Tensor, // [num_embeddings, embedding_dim]
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..num_embeddings * embedding_dim)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        Embedding {
            weight: Tensor::with_grad(data, vec![num_embeddings, embedding_dim], true),
        }
    }

    pub fn forward_indices(&self, indices: &[usize]) -> Tensor {
        embedding_lookup(&self.weight, indices)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}

// ============ LayerNorm ============

pub struct LayerNorm {
    pub gamma: Tensor, // [normalized_shape]
    pub beta: Tensor,  // [normalized_shape]
    pub eps: f32,
    pub normalized_shape: usize,
}

impl LayerNorm {
    pub fn new(normalized_shape: usize) -> Self {
        LayerNorm {
            gamma: Tensor::with_grad(vec![1.0; normalized_shape], vec![normalized_shape], true),
            beta: Tensor::with_grad(vec![0.0; normalized_shape], vec![normalized_shape], true),
            eps: 1e-5,
            normalized_shape,
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        let data = input.contiguous_data();
        let gamma = self.gamma.contiguous_data();
        let beta = self.beta.contiguous_data();
        let dim = self.normalized_shape;

        let leading: usize = data.len() / dim;
        let mut normalized = vec![0.0f32; data.len()];
        let mut out = vec![0.0f32; data.len()];
        for b in 0..leading {
            let off = b * dim;
            let mean: f32 = (0..dim).map(|i| data[off + i]).sum::<f32>() / dim as f32;
            let var: f32 = (0..dim).map(|i| (data[off + i] - mean).powi(2)).sum::<f32>() / dim as f32;
            let inv_std = 1.0 / (var + self.eps).sqrt();
            for i in 0..dim {
                normalized[off + i] = (data[off + i] - mean) * inv_std;
                out[off + i] = gamma[i] * normalized[off + i] + beta[i];
            }
        }

        let res = Tensor::new(out, shape.clone());

        if input.requires_grad() || self.gamma.requires_grad() {
            let mut inner = res.0.write().unwrap();
            inner.requires_grad = true;
            inner.creator = Some(std::sync::Arc::new(core_tensor::Node {
                op: Box::new(LayerNormOp {
                    normalized: normalized,
                    gamma: gamma,
                    input_data: data,
                    eps: self.eps,
                    dim,
                    shape,
                }),
                inputs: vec![input.clone(), self.gamma.clone(), self.beta.clone()],
            }));
        }

        res
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}

#[derive(Debug)]
struct LayerNormOp {
    normalized: Vec<f32>, // (x - mean) / std for each element
    gamma: Vec<f32>,
    input_data: Vec<f32>,
    eps: f32,
    dim: usize,
    shape: Vec<usize>,
}

impl core_tensor::Op for LayerNormOp {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let g = grad_output.contiguous_data();
        let dim = self.dim;
        let leading = g.len() / dim;

        // d_gamma: sum over batch of grad * normalized
        let mut d_gamma = vec![0.0f32; dim];
        // d_beta: sum over batch of grad
        let mut d_beta = vec![0.0f32; dim];
        // d_input
        let mut d_input = vec![0.0f32; g.len()];

        for b in 0..leading {
            let off = b * dim;
            let mean: f32 = (0..dim).map(|i| self.input_data[off + i]).sum::<f32>() / dim as f32;
            let var: f32 = (0..dim).map(|i| (self.input_data[off + i] - mean).powi(2)).sum::<f32>() / dim as f32;
            let inv_std = 1.0 / (var + self.eps).sqrt();

            for i in 0..dim {
                d_gamma[i] += g[off + i] * self.normalized[off + i];
                d_beta[i] += g[off + i];
            }

            // d_input for this row
            // g_hat = g * gamma
            let mut g_hat = vec![0.0f32; dim];
            for i in 0..dim {
                g_hat[i] = g[off + i] * self.gamma[i];
            }
            let g_hat_mean: f32 = g_hat.iter().sum::<f32>() / dim as f32;
            let g_hat_norm_mean: f32 = g_hat.iter().zip(self.normalized[off..off+dim].iter())
                .map(|(gh, n)| gh * n).sum::<f32>() / dim as f32;

            for i in 0..dim {
                d_input[off + i] = inv_std * (g_hat[i] - g_hat_mean - self.normalized[off + i] * g_hat_norm_mean);
            }
        }

        vec![
            Some(Tensor::new(d_input, self.shape.clone())),
            Some(Tensor::new(d_gamma, vec![dim])),
            Some(Tensor::new(d_beta, vec![dim])),
        ]
    }
}

// ============ MultiHeadAttention ============

pub struct MultiHeadAttention {
    pub n_head: usize,
    pub d_model: usize,
    pub head_dim: usize,
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, n_head: usize) -> Self {
        assert_eq!(d_model % n_head, 0);
        let head_dim = d_model / n_head;
        MultiHeadAttention {
            n_head,
            d_model,
            head_dim,
            wq: Linear::new(d_model, d_model, false),
            wk: Linear::new(d_model, d_model, false),
            wv: Linear::new(d_model, d_model, false),
            wo: Linear::new(d_model, d_model, false),
        }
    }

    /// input: [seq_len, d_model], returns [seq_len, d_model]
    pub fn forward_causal(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        let seq_len = shape[0];
        let h = self.n_head;
        let hd = self.head_dim;

        // Q, K, V projections: [seq_len, d_model]
        let q = self.wq.forward(input);
        let k = self.wk.forward(input);
        let v = self.wv.forward(input);

        // Reshape to [n_head, seq_len, head_dim]
        let q3 = reshape_to_heads(&q, seq_len, h, hd);
        let k3 = reshape_to_heads(&k, seq_len, h, hd);
        let v3 = reshape_to_heads(&v, seq_len, h, hd);

        // Attention scores: [n_head, seq_len, seq_len]
        let kt = batch_transpose(&k3, h, seq_len, hd);
        let scores = batch_matmul(&q3, &kt);
        let scores = scale(&scores, 1.0 / (hd as f32).sqrt());

        // Causal mask
        let mut mask = vec![false; h * seq_len * seq_len];
        for hi in 0..h {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    if j > i {
                        mask[hi * seq_len * seq_len + i * seq_len + j] = true;
                    }
                }
            }
        }
        let scores = masked_fill(&scores, &mask, f32::NEG_INFINITY);

        // Softmax per row: reshape to [n_head * seq_len, seq_len] for 2D softmax
        let scores_2d = reshape(&scores, vec![h * seq_len, seq_len]);
        let attn = softmax(&scores_2d);
        let attn_3d = reshape(&attn, vec![h, seq_len, seq_len]);

        // Weighted values: [n_head, seq_len, head_dim]
        let out = batch_matmul(&attn_3d, &v3);

        // Reshape back to [seq_len, d_model]
        let out_2d = reshape_from_heads(&out, seq_len, h, hd);

        // Output projection
        self.wo.forward(&out_2d)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.wq.parameters());
        p.extend(self.wk.parameters());
        p.extend(self.wv.parameters());
        p.extend(self.wo.parameters());
        p
    }
}

/// [seq_len, d_model] -> [n_head, seq_len, head_dim]
fn reshape_to_heads(x: &Tensor, seq_len: usize, n_head: usize, head_dim: usize) -> Tensor {
    // x is [seq_len, d_model] where d_model = n_head * head_dim
    // We need [n_head, seq_len, head_dim]
    let data = x.contiguous_data();
    let d_model = n_head * head_dim;
    let mut out = vec![0.0f32; n_head * seq_len * head_dim];
    for s in 0..seq_len {
        for h in 0..n_head {
            for d in 0..head_dim {
                out[h * seq_len * head_dim + s * head_dim + d] =
                    data[s * d_model + h * head_dim + d];
            }
        }
    }

    let res = Tensor::new(out, vec![n_head, seq_len, head_dim]);

    if x.requires_grad() {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(std::sync::Arc::new(core_tensor::Node {
            op: Box::new(ReshapeToHeadsOp { seq_len, n_head, head_dim }),
            inputs: vec![x.clone()],
        }));
    }

    res
}

#[derive(Debug)]
struct ReshapeToHeadsOp {
    seq_len: usize,
    n_head: usize,
    head_dim: usize,
}

impl core_tensor::Op for ReshapeToHeadsOp {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // Reverse: [n_head, seq_len, head_dim] -> [seq_len, d_model]
        let g = grad_output.contiguous_data();
        let d_model = self.n_head * self.head_dim;
        let mut out = vec![0.0f32; self.seq_len * d_model];
        for s in 0..self.seq_len {
            for h in 0..self.n_head {
                for d in 0..self.head_dim {
                    out[s * d_model + h * self.head_dim + d] =
                        g[h * self.seq_len * self.head_dim + s * self.head_dim + d];
                }
            }
        }
        vec![Some(Tensor::new(out, vec![self.seq_len, d_model]))]
    }
}

/// [n_head, seq_len, head_dim] -> [seq_len, d_model]
fn reshape_from_heads(x: &Tensor, seq_len: usize, n_head: usize, head_dim: usize) -> Tensor {
    let data = x.contiguous_data();
    let d_model = n_head * head_dim;
    let mut out = vec![0.0f32; seq_len * d_model];
    for s in 0..seq_len {
        for h in 0..n_head {
            for d in 0..head_dim {
                out[s * d_model + h * head_dim + d] =
                    data[h * seq_len * head_dim + s * head_dim + d];
            }
        }
    }

    let res = Tensor::new(out, vec![seq_len, d_model]);

    if x.requires_grad() {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(std::sync::Arc::new(core_tensor::Node {
            op: Box::new(ReshapeFromHeadsOp { seq_len, n_head, head_dim }),
            inputs: vec![x.clone()],
        }));
    }

    res
}

#[derive(Debug)]
struct ReshapeFromHeadsOp {
    seq_len: usize,
    n_head: usize,
    head_dim: usize,
}

impl core_tensor::Op for ReshapeFromHeadsOp {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // Reverse: [seq_len, d_model] -> [n_head, seq_len, head_dim]
        let g = grad_output.contiguous_data();
        let d_model = self.n_head * self.head_dim;
        let mut out = vec![0.0f32; self.n_head * self.seq_len * self.head_dim];
        for s in 0..self.seq_len {
            for h in 0..self.n_head {
                for d in 0..self.head_dim {
                    out[h * self.seq_len * self.head_dim + s * self.head_dim + d] =
                        g[s * d_model + h * self.head_dim + d];
                }
            }
        }
        vec![Some(Tensor::new(out, vec![self.n_head, self.seq_len, self.head_dim]))]
    }
}

/// Transpose last two dims of [B, M, N] -> [B, N, M]
fn batch_transpose(x: &Tensor, batch: usize, rows: usize, cols: usize) -> Tensor {
    let data = x.contiguous_data();
    let mut out = vec![0.0f32; batch * rows * cols];
    for b in 0..batch {
        let off = b * rows * cols;
        for i in 0..rows {
            for j in 0..cols {
                out[b * cols * rows + j * rows + i] = data[off + i * cols + j];
            }
        }
    }

    let res = Tensor::new(out, vec![batch, cols, rows]);

    if x.requires_grad() {
        let mut inner = res.0.write().unwrap();
        inner.requires_grad = true;
        inner.creator = Some(std::sync::Arc::new(core_tensor::Node {
            op: Box::new(BatchTransposeOp { batch, rows, cols }),
            inputs: vec![x.clone()],
        }));
    }

    res
}

#[derive(Debug)]
struct BatchTransposeOp {
    batch: usize,
    rows: usize,
    cols: usize,
}

impl core_tensor::Op for BatchTransposeOp {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // Transpose back: [B, cols, rows] -> [B, rows, cols]
        let g = grad_output.contiguous_data();
        let mut out = vec![0.0f32; self.batch * self.rows * self.cols];
        for b in 0..self.batch {
            for i in 0..self.cols {
                for j in 0..self.rows {
                    out[b * self.rows * self.cols + j * self.cols + i] =
                        g[b * self.cols * self.rows + i * self.rows + j];
                }
            }
        }
        vec![Some(Tensor::new(out, vec![self.batch, self.rows, self.cols]))]
    }
}

// ============ TransformerBlock ============

pub struct TransformerBlock {
    pub ln1: LayerNorm,
    pub attn: MultiHeadAttention,
    pub ln2: LayerNorm,
    pub ffn_up: Linear,
    pub ffn_down: Linear,
}

impl TransformerBlock {
    pub fn new(d_model: usize, n_head: usize, d_ff: usize) -> Self {
        TransformerBlock {
            ln1: LayerNorm::new(d_model),
            attn: MultiHeadAttention::new(d_model, n_head),
            ln2: LayerNorm::new(d_model),
            ffn_up: Linear::new(d_model, d_ff, true),
            ffn_down: Linear::new(d_ff, d_model, true),
        }
    }

    /// input: [seq_len, d_model] -> [seq_len, d_model]
    pub fn forward_seq(&self, input: &Tensor) -> Tensor {
        // Pre-norm: LN -> MHA -> residual
        let normed = self.ln1.forward(input);
        let attn_out = self.attn.forward_causal(&normed);
        let x = add(input, &attn_out);

        // Pre-norm: LN -> FFN -> residual
        let normed2 = self.ln2.forward(&x);
        let ffn_out = gelu(&self.ffn_up.forward(&normed2));
        let ffn_out = self.ffn_down.forward(&ffn_out);
        add(&x, &ffn_out)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.ln1.parameters());
        p.extend(self.attn.parameters());
        p.extend(self.ln2.parameters());
        p.extend(self.ffn_up.parameters());
        p.extend(self.ffn_down.parameters());
        p
    }
}

// ============ GPT Model ============

pub struct GPT {
    pub token_emb: Embedding,
    pub pos_emb: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub ln_f: LayerNorm,
    pub lm_head: Linear,
    pub seq_len: usize,
}

impl GPT {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        n_head: usize,
        n_layer: usize,
        d_ff: usize,
        seq_len: usize,
    ) -> Self {
        let blocks = (0..n_layer)
            .map(|_| TransformerBlock::new(d_model, n_head, d_ff))
            .collect();
        GPT {
            token_emb: Embedding::new(vocab_size, d_model),
            pos_emb: Embedding::new(seq_len, d_model),
            blocks,
            ln_f: LayerNorm::new(d_model),
            lm_head: Linear::new(d_model, vocab_size, false),
            seq_len,
        }
    }

    /// token_ids: &[usize] of length <= seq_len
    /// returns logits: [seq_len, vocab_size]
    pub fn forward_ids(&self, token_ids: &[usize]) -> Tensor {
        let slen = token_ids.len();
        assert!(slen <= self.seq_len);

        let positions: Vec<usize> = (0..slen).collect();
        let tok = self.token_emb.forward_indices(token_ids);
        let pos = self.pos_emb.forward_indices(&positions);
        let mut x = add(&tok, &pos);

        for block in &self.blocks {
            x = block.forward_seq(&x);
        }

        let x = self.ln_f.forward(&x);
        self.lm_head.forward(&x)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.token_emb.parameters());
        p.extend(self.pos_emb.parameters());
        for block in &self.blocks {
            p.extend(block.parameters());
        }
        p.extend(self.ln_f.parameters());
        p.extend(self.lm_head.parameters());
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward_shape() {
        let linear = Linear::new(4, 3, true);
        let input = Tensor::new(vec![1.0; 8], vec![2, 4]);
        let out = linear.forward(&input);
        assert_eq!(out.shape(), vec![2, 3]);
    }

    #[test]
    fn test_linear_no_bias() {
        let linear = Linear::new(3, 2, false);
        assert!(linear.bias.is_none());
        assert_eq!(linear.parameters().len(), 1);
    }

    #[test]
    fn test_linear_with_bias() {
        let linear = Linear::new(3, 2, true);
        assert!(linear.bias.is_some());
        assert_eq!(linear.parameters().len(), 2);
    }

    #[test]
    fn test_embedding_forward() {
        let emb = Embedding::new(10, 4);
        let out = emb.forward_indices(&[0, 3, 7]);
        assert_eq!(out.shape(), vec![3, 4]);
    }

    #[test]
    fn test_layer_norm_forward_1d() {
        let ln = LayerNorm::new(3);
        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let out = ln.forward(&input);
        let d = out.data();
        // After layer norm with gamma=1, beta=0: mean should be ~0
        let mean: f32 = d.iter().sum::<f32>() / 3.0;
        assert!(mean.abs() < 1e-5);
    }

    #[test]
    fn test_layer_norm_forward_2d() {
        let ln = LayerNorm::new(4);
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]);
        let out = ln.forward(&input);
        assert_eq!(out.shape(), vec![2, 4]);
        let d = out.data();
        // Each row should have mean ~0
        let mean0: f32 = d[0..4].iter().sum::<f32>() / 4.0;
        let mean1: f32 = d[4..8].iter().sum::<f32>() / 4.0;
        assert!(mean0.abs() < 1e-5);
        assert!(mean1.abs() < 1e-5);
    }

    #[test]
    fn test_xavier_uniform_range() {
        let t = xavier_uniform(100, 100);
        let d = t.data();
        let limit = (6.0 / 200.0f32).sqrt();
        for v in &d {
            assert!(*v >= -limit && *v <= limit);
        }
    }

    #[test]
    fn test_kaiming_normal_stats() {
        let t = kaiming_normal(1000, 100);
        let d = t.data();
        let mean: f32 = d.iter().sum::<f32>() / d.len() as f32;
        // Mean should be close to 0
        assert!(mean.abs() < 0.1);
    }

    // --- MultiHeadAttention tests ---

    #[test]
    fn test_mha_forward_shape() {
        let mha = MultiHeadAttention::new(8, 2); // d_model=8, n_head=2
        let input = Tensor::new(vec![0.1; 4 * 8], vec![4, 8]); // seq_len=4
        let out = mha.forward_causal(&input);
        assert_eq!(out.shape(), vec![4, 8]);
    }

    #[test]
    fn test_mha_parameters_count() {
        let mha = MultiHeadAttention::new(8, 2);
        // 4 weight matrices (no bias): Wq, Wk, Wv, Wo
        assert_eq!(mha.parameters().len(), 4);
    }

    // --- TransformerBlock tests ---

    #[test]
    fn test_transformer_block_forward_shape() {
        let block = TransformerBlock::new(8, 2, 32); // d_model=8, n_head=2, d_ff=32
        let input = Tensor::new(vec![0.1; 4 * 8], vec![4, 8]);
        let out = block.forward_seq(&input);
        assert_eq!(out.shape(), vec![4, 8]);
    }

    // --- GPT tests ---

    #[test]
    fn test_gpt_forward_shape() {
        let gpt = GPT::new(16, 8, 2, 2, 32, 8);
        // vocab=16, d_model=8, n_head=2, n_layer=2, d_ff=32, seq_len=8
        let logits = gpt.forward_ids(&[0, 1, 2, 3]);
        assert_eq!(logits.shape(), vec![4, 16]); // [seq_len, vocab_size]
    }

    #[test]
    fn test_gpt_parameters() {
        let gpt = GPT::new(16, 8, 2, 1, 32, 8);
        let params = gpt.parameters();
        // token_emb(1) + pos_emb(1) + 1 block(ln1:2 + attn:4 + ln2:2 + ffn_up:2 + ffn_down:2 = 12) + ln_f(2) + lm_head(1) = 17
        assert_eq!(params.len(), 17);
    }

    #[test]
    fn test_gpt_backward_runs() {
        let gpt = GPT::new(8, 4, 2, 1, 16, 4);
        let logits = gpt.forward_ids(&[0, 1, 2]);
        let loss = cross_entropy_loss(&logits, &[1, 2, 3]);
        loss.backward();
        // Check that gradients exist on parameters
        let param_names = [
            "token_emb", "pos_emb",
            "ln1_gamma", "ln1_beta",
            "wq", "wk", "wv", "wo",
            "ln2_gamma", "ln2_beta",
            "ffn_up_w", "ffn_up_b", "ffn_down_w", "ffn_down_b",
            "ln_f_gamma", "ln_f_beta",
            "lm_head",
        ];
        for (i, p) in gpt.parameters().iter().enumerate() {
            let name = if i < param_names.len() { param_names[i] } else { "unknown" };
            assert!(p.grad().is_some(), "param[{}] '{}' has no gradient", i, name);
        }
    }
}
