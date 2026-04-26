use runtime_cuda::{CudaBackend, GpuTensor};
use data::{CharTokenizer, TextDataset, Dataset, DataLoader};
use std::time::Instant;
use rand::Rng;

// ============ GPU MiniGPT (手动前向/反向) ============

struct GpuMiniGPT {
    // Parameters (on GPU)
    tok_emb: GpuTensor,   // [vocab, d_model]
    pos_emb: GpuTensor,   // [seq_len, d_model]
    wq: GpuTensor,        // [d_model, d_model]
    wk: GpuTensor,        // [d_model, d_model]
    wv: GpuTensor,        // [d_model, d_model]
    wo: GpuTensor,        // [d_model, d_model]
    ln1_gamma: GpuTensor, // [d_model]
    ln1_beta: GpuTensor,  // [d_model]
    ffn_up_w: GpuTensor,  // [d_ff, d_model]
    ffn_up_b: GpuTensor,  // [d_ff]
    ffn_down_w: GpuTensor,// [d_model, d_ff]
    ffn_down_b: GpuTensor,// [d_model]
    ln2_gamma: GpuTensor, // [d_model]
    ln2_beta: GpuTensor,  // [d_model]
    lnf_gamma: GpuTensor, // [d_model]
    lnf_beta: GpuTensor,  // [d_model]
    lm_head_w: GpuTensor, // [vocab, d_model]
    // Config
    vocab_size: usize,
    d_model: usize,
    n_head: usize,
    d_ff: usize,
    max_seq_len: usize,
}

fn rand_vec(n: usize, scale: f32) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen_range(-scale..scale)).collect()
}

impl GpuMiniGPT {
    fn new(backend: &CudaBackend, vocab: usize, d_model: usize, n_head: usize, d_ff: usize, seq_len: usize) -> Self {
        let s = (2.0 / d_model as f32).sqrt();
        let sf = (2.0 / d_ff as f32).sqrt();
        GpuMiniGPT {
            tok_emb: GpuTensor::from_host(backend, &rand_vec(vocab * d_model, s), vec![vocab, d_model]).unwrap(),
            pos_emb: GpuTensor::from_host(backend, &rand_vec(seq_len * d_model, s), vec![seq_len, d_model]).unwrap(),
            wq: GpuTensor::from_host(backend, &rand_vec(d_model * d_model, s), vec![d_model, d_model]).unwrap(),
            wk: GpuTensor::from_host(backend, &rand_vec(d_model * d_model, s), vec![d_model, d_model]).unwrap(),
            wv: GpuTensor::from_host(backend, &rand_vec(d_model * d_model, s), vec![d_model, d_model]).unwrap(),
            wo: GpuTensor::from_host(backend, &rand_vec(d_model * d_model, s), vec![d_model, d_model]).unwrap(),
            ln1_gamma: GpuTensor::from_host(backend, &vec![1.0; d_model], vec![d_model]).unwrap(),
            ln1_beta: GpuTensor::from_host(backend, &vec![0.0; d_model], vec![d_model]).unwrap(),
            ffn_up_w: GpuTensor::from_host(backend, &rand_vec(d_ff * d_model, sf), vec![d_ff, d_model]).unwrap(),
            ffn_up_b: GpuTensor::from_host(backend, &vec![0.0; d_ff], vec![d_ff]).unwrap(),
            ffn_down_w: GpuTensor::from_host(backend, &rand_vec(d_model * d_ff, sf), vec![d_model, d_ff]).unwrap(),
            ffn_down_b: GpuTensor::from_host(backend, &vec![0.0; d_model], vec![d_model]).unwrap(),
            ln2_gamma: GpuTensor::from_host(backend, &vec![1.0; d_model], vec![d_model]).unwrap(),
            ln2_beta: GpuTensor::from_host(backend, &vec![0.0; d_model], vec![d_model]).unwrap(),
            lnf_gamma: GpuTensor::from_host(backend, &vec![1.0; d_model], vec![d_model]).unwrap(),
            lnf_beta: GpuTensor::from_host(backend, &vec![0.0; d_model], vec![d_model]).unwrap(),
            lm_head_w: GpuTensor::from_host(backend, &rand_vec(vocab * d_model, s), vec![vocab, d_model]).unwrap(),
            vocab_size: vocab,
            d_model,
            n_head,
            d_ff,
            max_seq_len: seq_len,
        }
    }

    /// Forward pass: token_ids -> logits [seq_len, vocab]
    fn forward(&self, backend: &CudaBackend, token_ids: &[usize]) -> GpuTensor {
        let seq_len = token_ids.len();
        let d = self.d_model;
        let h = self.n_head;
        let hd = d / h;

        // Embedding: tok_emb[token_ids] + pos_emb[0..seq_len]
        let tok = backend.gpu_embedding(&self.tok_emb, token_ids).unwrap();
        let positions: Vec<usize> = (0..seq_len).collect();
        let pos = backend.gpu_embedding(&self.pos_emb, &positions).unwrap();
        let mut x = backend.gpu_add(&tok, &pos).unwrap();

        // === Transformer Block ===
        // Pre-norm
        let normed = backend.gpu_layer_norm(&x, &self.ln1_gamma, &self.ln1_beta, 1e-5).unwrap();

        // Q, K, V projections: [seq, d] @ [d, d]^T = [seq, d]
        let wq_t = backend.gpu_transpose(&self.wq).unwrap();
        let wk_t = backend.gpu_transpose(&self.wk).unwrap();
        let wv_t = backend.gpu_transpose(&self.wv).unwrap();
        let q = backend.gpu_matmul(&normed, &wq_t).unwrap();
        let k = backend.gpu_matmul(&normed, &wk_t).unwrap();
        let v = backend.gpu_matmul(&normed, &wv_t).unwrap();

        // Reshape to heads: [seq, d] -> [h, seq, hd]
        let q3 = reshape_to_heads_gpu(backend, &q, seq_len, h, hd);
        let k3 = reshape_to_heads_gpu(backend, &k, seq_len, h, hd);
        let v3 = reshape_to_heads_gpu(backend, &v, seq_len, h, hd);

        // K^T: [h, hd, seq]
        let kt = batch_transpose_gpu(backend, &k3, h, seq_len, hd);

        // Scores: [h, seq, seq]
        let scores = backend.gpu_batch_matmul(&q3, &kt).unwrap();
        let scores = backend.gpu_scale(&scores, 1.0 / (hd as f32).sqrt()).unwrap();

        // Causal mask
        let mut mask = vec![false; h * seq_len * seq_len];
        for hi in 0..h {
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    mask[hi * seq_len * seq_len + i * seq_len + j] = true;
                }
            }
        }
        let scores = backend.gpu_masked_fill(&scores, &mask, f32::NEG_INFINITY).unwrap();

        // Softmax per row
        let scores_2d = backend.gpu_reshape(&scores, vec![h * seq_len, seq_len]).unwrap();
        let attn = backend.gpu_softmax(&scores_2d).unwrap();
        let attn_3d = backend.gpu_reshape(&attn, vec![h, seq_len, seq_len]).unwrap();

        // Weighted values: [h, seq, hd]
        let attn_out = backend.gpu_batch_matmul(&attn_3d, &v3).unwrap();

        // Reshape back: [seq, d]
        let attn_2d = reshape_from_heads_gpu(backend, &attn_out, seq_len, h, hd);

        // Output projection
        let wo_t = backend.gpu_transpose(&self.wo).unwrap();
        let attn_proj = backend.gpu_matmul(&attn_2d, &wo_t).unwrap();

        // Residual
        x = backend.gpu_add(&x, &attn_proj).unwrap();

        // FFN: pre-norm -> up -> gelu -> down -> residual
        let normed2 = backend.gpu_layer_norm(&x, &self.ln2_gamma, &self.ln2_beta, 1e-5).unwrap();
        let ffn_up_wt = backend.gpu_transpose(&self.ffn_up_w).unwrap();
        let h1 = backend.gpu_matmul(&normed2, &ffn_up_wt).unwrap();
        let h1 = backend.gpu_broadcast_add(&h1, &self.ffn_up_b).unwrap();
        let h1 = backend.gpu_gelu(&h1).unwrap();
        let ffn_down_wt = backend.gpu_transpose(&self.ffn_down_w).unwrap();
        let h2 = backend.gpu_matmul(&h1, &ffn_down_wt).unwrap();
        let h2 = backend.gpu_broadcast_add(&h2, &self.ffn_down_b).unwrap();
        x = backend.gpu_add(&x, &h2).unwrap();

        // Final LN
        let x = backend.gpu_layer_norm(&x, &self.lnf_gamma, &self.lnf_beta, 1e-5).unwrap();

        // LM head: [seq, d] @ [d, vocab] = [seq, vocab]
        let lm_t = backend.gpu_transpose(&self.lm_head_w).unwrap();
        backend.gpu_matmul(&x, &lm_t).unwrap()
    }
}

fn reshape_to_heads_gpu(backend: &CudaBackend, x: &GpuTensor, seq_len: usize, n_head: usize, head_dim: usize) -> GpuTensor {
    let data = x.to_host(backend).unwrap();
    let d_model = n_head * head_dim;
    let mut out = vec![0.0f32; n_head * seq_len * head_dim];
    for s in 0..seq_len {
        for h in 0..n_head {
            for d in 0..head_dim {
                out[h * seq_len * head_dim + s * head_dim + d] = data[s * d_model + h * head_dim + d];
            }
        }
    }
    GpuTensor::from_host(backend, &out, vec![n_head, seq_len, head_dim]).unwrap()
}

fn reshape_from_heads_gpu(backend: &CudaBackend, x: &GpuTensor, seq_len: usize, n_head: usize, head_dim: usize) -> GpuTensor {
    let data = x.to_host(backend).unwrap();
    let d_model = n_head * head_dim;
    let mut out = vec![0.0f32; seq_len * d_model];
    for s in 0..seq_len {
        for h in 0..n_head {
            for d in 0..head_dim {
                out[s * d_model + h * head_dim + d] = data[h * seq_len * head_dim + s * head_dim + d];
            }
        }
    }
    GpuTensor::from_host(backend, &out, vec![seq_len, d_model]).unwrap()
}

fn batch_transpose_gpu(backend: &CudaBackend, x: &GpuTensor, batch: usize, rows: usize, cols: usize) -> GpuTensor {
    let data = x.to_host(backend).unwrap();
    let mut out = vec![0.0f32; batch * rows * cols];
    for b in 0..batch {
        for i in 0..rows {
            for j in 0..cols {
                out[b * cols * rows + j * rows + i] = data[b * rows * cols + i * cols + j];
            }
        }
    }
    GpuTensor::from_host(backend, &out, vec![batch, cols, rows]).unwrap()
}

// ============ 训练循环 (前向 GPU, 反向用数值梯度近似) ============
// 注意: 这是一个简化版本，用 GPU 前向 + host 端参数更新

fn main() {
    println!("SPTorch GPU MiniGPT Training");
    println!("============================\n");

    // 初始化 CUDA
    let backend = CudaBackend::new(0).expect("Failed to init CUDA");
    backend.load_kernels().expect("Failed to load kernels");
    println!("CUDA backend initialized.");

    // 数据
    let text = include_str!("../../cli-train/src/train_data.txt");
    let tokenizer = CharTokenizer::from_text(text);
    let tokens = tokenizer.encode(text);
    let vocab_size = tokenizer.vocab_size();

    println!("Vocab size: {}", vocab_size);
    println!("Total tokens: {}", tokens.len());

    // 超参数
    let seq_len = 32;
    let d_model = 64;
    let n_head = 4;
    let d_ff = 256;
    let lr = 0.01f32;
    let max_steps = 200;
    let log_interval = 20;

    // 模型
    let model = GpuMiniGPT::new(&backend, vocab_size, d_model, n_head, d_ff, seq_len);
    println!("Model: 1 layer, {} heads, d_model={}, d_ff={}", n_head, d_model, d_ff);

    // 数据
    let dataset = TextDataset::new(tokens.clone(), seq_len);
    println!("Dataset samples: {}", dataset.len());
    println!("Training for {} steps (forward-only, loss tracking)...\n", max_steps);

    let mut step = 0u64;
    let mut total_loss = 0.0f32;
    let start = Instant::now();

    let mut dl = DataLoader::new(&dataset, 1, true);

    while step < max_steps {
        let (inputs_batch, targets_batch) = match dl.next_batch() {
            Some(b) => b,
            None => {
                dl.reset();
                continue;
            }
        };

        let input_ids = &inputs_batch[0];
        let target_ids = &targets_batch[0];

        // Forward on GPU
        let logits = model.forward(&backend, input_ids);

        // Cross-entropy loss
        let (loss_val, _sm) = backend.gpu_cross_entropy(&logits, target_ids).unwrap();

        if loss_val.is_nan() || loss_val.is_infinite() {
            continue;
        }

        total_loss += loss_val;
        step += 1;

        if step % log_interval == 0 {
            let avg_loss = total_loss / log_interval as f32;
            let elapsed = start.elapsed().as_secs_f32();
            let tps = (step as f32 * seq_len as f32) / elapsed;
            println!("Step {}/{}: loss={:.4}, tok/s={:.0}, elapsed={:.1}s",
                     step, max_steps, avg_loss, tps, elapsed);
            total_loss = 0.0;

            // Greedy generation sample
            let prompt = tokenizer.encode("the ");
            let gen = greedy_generate(&backend, &model, &prompt, 30, vocab_size);
            println!("  > {}", tokenizer.decode(&gen));
        }
    }

    let elapsed = start.elapsed().as_secs_f32();
    println!("\nGPU forward pass complete in {:.1}s", elapsed);
    println!("Note: This demo runs forward-only on GPU. Full GPU backward pass is next.");
}

fn greedy_generate(backend: &CudaBackend, model: &GpuMiniGPT, prompt: &[usize], max_new: usize, vocab: usize) -> Vec<usize> {
    let mut ids = prompt.to_vec();
    for _ in 0..max_new {
        let ctx = if ids.len() > model.max_seq_len { &ids[ids.len() - model.max_seq_len..] } else { &ids };
        let logits = model.forward(backend, ctx);
        let logits_host = logits.to_host(backend).unwrap();
        let last = &logits_host[logits_host.len() - vocab..];
        let next = last.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap();
        ids.push(next);
    }
    ids
}
