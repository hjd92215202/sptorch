use data::{CharTokenizer, DataLoader, Dataset, TextDataset, Tokenizer};
use rand::Rng;
use runtime_cuda::{CudaBackend, GpuTensor};
use std::time::Instant;

// ============ GPU MiniGPT with Attention + Manual Backward ============
// Architecture: Embedding -> SingleHeadAttention -> FFN(GELU) -> Residual -> LM Head

struct GpuAttentionGPT {
    tok_emb: GpuTensor,   // [vocab, d_model]
    pos_emb: GpuTensor,   // [seq_len, d_model]
    wq: GpuTensor,        // [d_model, d_model]
    wk: GpuTensor,        // [d_model, d_model]
    wv: GpuTensor,        // [d_model, d_model]
    wo: GpuTensor,        // [d_model, d_model]
    fc1_w: GpuTensor,     // [d_ff, d_model]
    fc1_b: GpuTensor,     // [d_ff]
    fc2_w: GpuTensor,     // [d_model, d_ff]
    fc2_b: GpuTensor,     // [d_model]
    lm_head_w: GpuTensor, // [vocab, d_model]
    vocab_size: usize,
    d_model: usize,
    d_ff: usize,
    max_seq_len: usize,
}

fn rand_vec(n: usize, scale: f32) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen_range(-scale..scale)).collect()
}

impl GpuAttentionGPT {
    fn new(backend: &CudaBackend, vocab: usize, d_model: usize, d_ff: usize, seq_len: usize) -> Self {
        let s = (2.0 / d_model as f32).sqrt() * 0.5;
        let sf = (2.0 / d_ff as f32).sqrt() * 0.5;
        GpuAttentionGPT {
            tok_emb: GpuTensor::from_host(backend, &rand_vec(vocab * d_model, s), vec![vocab, d_model]).unwrap(),
            pos_emb: GpuTensor::from_host(backend, &rand_vec(seq_len * d_model, s), vec![seq_len, d_model]).unwrap(),
            wq: GpuTensor::from_host(backend, &rand_vec(d_model * d_model, s), vec![d_model, d_model]).unwrap(),
            wk: GpuTensor::from_host(backend, &rand_vec(d_model * d_model, s), vec![d_model, d_model]).unwrap(),
            wv: GpuTensor::from_host(backend, &rand_vec(d_model * d_model, s), vec![d_model, d_model]).unwrap(),
            wo: GpuTensor::from_host(backend, &rand_vec(d_model * d_model, s), vec![d_model, d_model]).unwrap(),
            fc1_w: GpuTensor::from_host(backend, &rand_vec(d_ff * d_model, sf), vec![d_ff, d_model]).unwrap(),
            fc1_b: GpuTensor::from_host(backend, &vec![0.0; d_ff], vec![d_ff]).unwrap(),
            fc2_w: GpuTensor::from_host(backend, &rand_vec(d_model * d_ff, sf), vec![d_model, d_ff]).unwrap(),
            fc2_b: GpuTensor::from_host(backend, &vec![0.0; d_model], vec![d_model]).unwrap(),
            lm_head_w: GpuTensor::from_host(backend, &rand_vec(vocab * d_model, s), vec![vocab, d_model]).unwrap(),
            vocab_size: vocab,
            d_model,
            d_ff,
            max_seq_len: seq_len,
        }
    }

    fn forward(&self, backend: &CudaBackend, token_ids: &[usize]) -> FwdCache {
        let seq = token_ids.len();
        let d = self.d_model;

        // Embedding: [seq, d]
        let tok = backend.gpu_embedding(&self.tok_emb, token_ids).unwrap();
        let positions: Vec<usize> = (0..seq).collect();
        let pos = backend.gpu_embedding(&self.pos_emb, &positions).unwrap();
        let emb = backend.gpu_add(&tok, &pos).unwrap();

        // Q, K, V: [seq, d] @ [d, d] = [seq, d]
        let wq_t = backend.gpu_transpose(&self.wq).unwrap();
        let wk_t = backend.gpu_transpose(&self.wk).unwrap();
        let wv_t = backend.gpu_transpose(&self.wv).unwrap();
        let q = backend.gpu_matmul(&emb, &wq_t).unwrap();
        let k = backend.gpu_matmul(&emb, &wk_t).unwrap();
        let v = backend.gpu_matmul(&emb, &wv_t).unwrap();

        // Attention scores: Q @ K^T / sqrt(d) -> [seq, seq]
        let k_t = backend.gpu_transpose(&k).unwrap();
        let scores = backend.gpu_matmul(&q, &k_t).unwrap();
        let scores = backend.gpu_scale(&scores, 1.0 / (d as f32).sqrt()).unwrap();

        // Causal mask
        let mut mask = vec![false; seq * seq];
        for i in 0..seq {
            for j in (i + 1)..seq {
                mask[i * seq + j] = true;
            }
        }
        let scores_masked = backend.gpu_masked_fill(&scores, &mask, f32::NEG_INFINITY).unwrap();

        // Softmax: [seq, seq]
        let attn_weights = backend.gpu_softmax(&scores_masked).unwrap();

        // Attention output: [seq, seq] @ [seq, d] = [seq, d]
        let attn_out = backend.gpu_matmul(&attn_weights, &v).unwrap();

        // Output projection: [seq, d] @ [d, d] = [seq, d]
        let wo_t = backend.gpu_transpose(&self.wo).unwrap();
        let attn_proj = backend.gpu_matmul(&attn_out, &wo_t).unwrap();

        // Residual 1
        let x = backend.gpu_add(&emb, &attn_proj).unwrap();

        // FFN: FC1 -> GELU -> FC2
        let fc1_wt = backend.gpu_transpose(&self.fc1_w).unwrap();
        let h1_pre = backend.gpu_matmul(&x, &fc1_wt).unwrap();
        let h1_pre = backend.gpu_broadcast_add(&h1_pre, &self.fc1_b).unwrap();
        let h1 = backend.gpu_gelu(&h1_pre).unwrap();
        let fc2_wt = backend.gpu_transpose(&self.fc2_w).unwrap();
        let ffn_out = backend.gpu_matmul(&h1, &fc2_wt).unwrap();
        let ffn_out = backend.gpu_broadcast_add(&ffn_out, &self.fc2_b).unwrap();

        // Residual 2
        let out = backend.gpu_add(&x, &ffn_out).unwrap();

        // LM Head
        let lm_t = backend.gpu_transpose(&self.lm_head_w).unwrap();
        let logits = backend.gpu_matmul(&out, &lm_t).unwrap();

        FwdCache {
            token_ids: token_ids.to_vec(),
            positions,
            seq,
            emb,
            q,
            k,
            v,
            attn_weights,
            attn_out,
            x,
            h1_pre,
            h1,
            out,
            logits,
        }
    }

    fn backward_and_update(&mut self, backend: &CudaBackend, c: &FwdCache, targets: &[usize], lr: f32) -> f32 {
        let seq = c.seq;
        let d = self.d_model;
        let ff = self.d_ff;
        let v = self.vocab_size;

        // Loss
        let (loss, sm) = backend.gpu_cross_entropy(&c.logits, targets).unwrap();
        let dl = backend.gpu_cross_entropy_backward(&sm, targets).unwrap(); // [seq, vocab]

        // --- LM Head backward ---
        // logits = out @ lm_head_w^T
        let d_out = backend.gpu_matmul(&dl, &self.lm_head_w).unwrap(); // [seq, d]
        let dl_t = backend.gpu_transpose(&dl).unwrap();
        let d_lm = backend.gpu_matmul(&dl_t, &c.out).unwrap(); // [vocab, d]

        // --- Residual 2: out = x + ffn_out ---
        let d_x_r2 = &d_out; // [seq, d]
        let d_ffn = &d_out; // [seq, d]

        // --- FC2 backward: ffn_out = h1 @ fc2_w^T + fc2_b ---
        let d_h1 = backend.gpu_matmul(d_ffn, &self.fc2_w).unwrap(); // [seq, ff]
        let d_ffn_t = backend.gpu_transpose(d_ffn).unwrap();
        let d_fc2_w = backend.gpu_matmul(&d_ffn_t, &c.h1).unwrap(); // [d, ff]
        let d_fc2_b = sum_rows(backend, d_ffn, seq, d);

        // --- GELU backward ---
        let d_h1_pre = gelu_backward(backend, &d_h1, &c.h1_pre);

        // --- FC1 backward: h1_pre = x @ fc1_w^T + fc1_b ---
        let d_x_fc1 = backend.gpu_matmul(&d_h1_pre, &self.fc1_w).unwrap(); // [seq, d]
        let d_h1_pre_t = backend.gpu_transpose(&d_h1_pre).unwrap();
        let d_fc1_w = backend.gpu_matmul(&d_h1_pre_t, &c.x).unwrap(); // [ff, d]
        let d_fc1_b = sum_rows(backend, &d_h1_pre, seq, ff);

        // --- Residual 1: x = emb + attn_proj ---
        // d_x = d_x_r2 + d_x_fc1
        let d_x = backend.gpu_add(d_x_r2, &d_x_fc1).unwrap();
        let d_attn_proj = &d_x; // [seq, d]

        // --- Wo backward: attn_proj = attn_out @ wo^T ---
        let d_attn_out = backend.gpu_matmul(d_attn_proj, &self.wo).unwrap(); // [seq, d]
        let d_ap_t = backend.gpu_transpose(d_attn_proj).unwrap();
        let d_wo = backend.gpu_matmul(&d_ap_t, &c.attn_out).unwrap(); // [d, d]

        // --- Attention backward ---
        // attn_out = attn_weights @ V
        // d_attn_weights = d_attn_out @ V^T  [seq, seq]
        let v_t = backend.gpu_transpose(&c.v).unwrap();
        let d_attn_w = backend.gpu_matmul(&d_attn_out, &v_t).unwrap();
        // d_V = attn_weights^T @ d_attn_out  [seq, d]
        let aw_t = backend.gpu_transpose(&c.attn_weights).unwrap();
        let d_v = backend.gpu_matmul(&aw_t, &d_attn_out).unwrap();

        // --- Softmax backward ---
        // d_scores = softmax_backward(d_attn_w, attn_weights)
        let d_scores = softmax_backward(backend, &d_attn_w, &c.attn_weights, seq);

        // --- Scores backward: scores = Q @ K^T / sqrt(d) ---
        let d_scores_scaled = backend.gpu_scale(&d_scores, 1.0 / (d as f32).sqrt()).unwrap();
        // d_Q = d_scores_scaled @ K  [seq, d]
        let d_q = backend.gpu_matmul(&d_scores_scaled, &c.k).unwrap();
        // d_K = d_scores_scaled^T @ Q  [seq, d]
        let d_sc_t = backend.gpu_transpose(&d_scores_scaled).unwrap();
        let d_k = backend.gpu_matmul(&d_sc_t, &c.q).unwrap();

        // --- Wq/Wk/Wv backward ---
        // Q = emb @ Wq^T => d_emb_q = d_Q @ Wq, d_Wq = d_Q^T @ emb
        let d_emb_q = backend.gpu_matmul(&d_q, &self.wq).unwrap();
        let d_q_t = backend.gpu_transpose(&d_q).unwrap();
        let d_wq = backend.gpu_matmul(&d_q_t, &c.emb).unwrap();

        let d_emb_k = backend.gpu_matmul(&d_k, &self.wk).unwrap();
        let d_k_t = backend.gpu_transpose(&d_k).unwrap();
        let d_wk = backend.gpu_matmul(&d_k_t, &c.emb).unwrap();

        let d_emb_v = backend.gpu_matmul(&d_v, &self.wv).unwrap();
        let d_v_t = backend.gpu_transpose(&d_v).unwrap();
        let d_wv = backend.gpu_matmul(&d_v_t, &c.emb).unwrap();

        // --- Embedding backward ---
        // d_emb = d_x (from residual 1) + d_emb_q + d_emb_k + d_emb_v
        let d_emb = backend.gpu_add(&d_x, &d_emb_q).unwrap();
        let d_emb = backend.gpu_add(&d_emb, &d_emb_k).unwrap();
        let d_emb = backend.gpu_add(&d_emb, &d_emb_v).unwrap();

        let d_tok = backend.gpu_embedding_backward(&d_emb, &c.token_ids, v, d).unwrap();
        let d_pos = backend
            .gpu_embedding_backward(&d_emb, &c.positions, self.max_seq_len, d)
            .unwrap();

        // --- SGD updates ---
        backend.gpu_sgd_update(&mut self.lm_head_w, &d_lm, lr).unwrap();
        backend.gpu_sgd_update(&mut self.fc2_w, &d_fc2_w, lr).unwrap();
        backend.gpu_sgd_update(&mut self.fc2_b, &d_fc2_b, lr).unwrap();
        backend.gpu_sgd_update(&mut self.fc1_w, &d_fc1_w, lr).unwrap();
        backend.gpu_sgd_update(&mut self.fc1_b, &d_fc1_b, lr).unwrap();
        backend.gpu_sgd_update(&mut self.wo, &d_wo, lr).unwrap();
        backend.gpu_sgd_update(&mut self.wq, &d_wq, lr).unwrap();
        backend.gpu_sgd_update(&mut self.wk, &d_wk, lr).unwrap();
        backend.gpu_sgd_update(&mut self.wv, &d_wv, lr).unwrap();
        backend.gpu_sgd_update(&mut self.tok_emb, &d_tok, lr).unwrap();
        backend.gpu_sgd_update(&mut self.pos_emb, &d_pos, lr).unwrap();

        loss
    }
}

struct FwdCache {
    token_ids: Vec<usize>,
    positions: Vec<usize>,
    seq: usize,
    emb: GpuTensor,
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    attn_weights: GpuTensor,
    attn_out: GpuTensor,
    x: GpuTensor,
    h1_pre: GpuTensor,
    h1: GpuTensor,
    out: GpuTensor,
    logits: GpuTensor,
}

fn sum_rows(backend: &CudaBackend, x: &GpuTensor, rows: usize, cols: usize) -> GpuTensor {
    let data = x.to_host(backend).unwrap();
    let mut out = vec![0.0f32; cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c] += data[r * cols + c];
        }
    }
    GpuTensor::from_host(backend, &out, vec![cols]).unwrap()
}

fn gelu_backward(backend: &CudaBackend, d_out: &GpuTensor, input: &GpuTensor) -> GpuTensor {
    let g = d_out.to_host(backend).unwrap();
    let x = input.to_host(backend).unwrap();
    let sqrt_2_pi: f32 = (2.0 / std::f32::consts::PI).sqrt();
    let da: Vec<f32> = g
        .iter()
        .zip(x.iter())
        .map(|(gi, &xi)| {
            let k = sqrt_2_pi * (xi + 0.044715 * xi.powi(3));
            let tanh_k = k.tanh();
            let dk = sqrt_2_pi * (1.0 + 0.134145 * xi * xi);
            gi * (0.5 * (1.0 + tanh_k) + 0.5 * xi * (1.0 - tanh_k * tanh_k) * dk)
        })
        .collect();
    GpuTensor::from_host(backend, &da, d_out.shape.clone()).unwrap()
}

/// Softmax backward: for each row, d_input = s * (d_out - sum(d_out * s))
fn softmax_backward(backend: &CudaBackend, d_out: &GpuTensor, s: &GpuTensor, seq: usize) -> GpuTensor {
    let g = d_out.to_host(backend).unwrap();
    let sv = s.to_host(backend).unwrap();
    let mut da = vec![0.0f32; seq * seq];
    for r in 0..seq {
        let off = r * seq;
        let dot: f32 = (0..seq).map(|c| sv[off + c] * g[off + c]).sum();
        for c in 0..seq {
            da[off + c] = sv[off + c] * (g[off + c] - dot);
        }
    }
    GpuTensor::from_host(backend, &da, vec![seq, seq]).unwrap()
}

fn greedy_generate(backend: &CudaBackend, model: &GpuAttentionGPT, prompt: &[usize], max_new: usize) -> Vec<usize> {
    let vocab = model.vocab_size;
    let mut ids = prompt.to_vec();
    for _ in 0..max_new {
        let ctx = if ids.len() > model.max_seq_len {
            &ids[ids.len() - model.max_seq_len..]
        } else {
            &ids
        };
        let cache = model.forward(backend, ctx);
        let logits_host = cache.logits.to_host(backend).unwrap();
        let last = &logits_host[logits_host.len() - vocab..];
        let next = last
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        ids.push(next);
    }
    ids
}

fn sampled_generate(
    backend: &CudaBackend,
    model: &GpuAttentionGPT,
    prompt: &[usize],
    max_new: usize,
    temp: f32,
    top_k: usize,
) -> Vec<usize> {
    let vocab = model.vocab_size;
    let mut rng = rand::thread_rng();
    let mut ids = prompt.to_vec();
    for _ in 0..max_new {
        let ctx = if ids.len() > model.max_seq_len {
            &ids[ids.len() - model.max_seq_len..]
        } else {
            &ids
        };
        let cache = model.forward(backend, ctx);
        let logits_host = cache.logits.to_host(backend).unwrap();
        let last = &logits_host[logits_host.len() - vocab..];
        let scaled: Vec<f32> = last.iter().map(|x| x / temp.max(1e-8)).collect();
        let mut indexed: Vec<(usize, f32)> = scaled.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let k = top_k.min(vocab);
        let top = &indexed[..k];
        let max_val = top[0].1;
        let exps: Vec<f32> = top.iter().map(|(_, v)| (v - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();
        let r: f32 = rng.gen();
        let mut cumsum = 0.0;
        let mut next = top[0].0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                next = top[i].0;
                break;
            }
        }
        ids.push(next);
    }
    ids
}

fn main() {
    println!("SPTorch GPU Training (Attention Model)");
    println!("=======================================\n");

    let backend = CudaBackend::new(0).expect("Failed to init CUDA");
    backend.load_kernels().expect("Failed to load kernels");
    println!("CUDA backend initialized.");

    let text = include_str!("../../cli-train/src/train_data.txt");
    let tokenizer = CharTokenizer::from_text(text);
    let tokens = tokenizer.encode(text);
    let vocab_size = tokenizer.vocab_size();

    println!("Vocab size: {}", vocab_size);
    println!("Total tokens: {}", tokens.len());

    let seq_len = 32;
    let d_model = 64;
    let d_ff = 256;
    let lr = 0.05f32;
    let max_steps: u64 = 5000;
    let log_interval: u64 = 500;

    let mut model = GpuAttentionGPT::new(&backend, vocab_size, d_model, d_ff, seq_len);
    let total_params = vocab_size * d_model * 2
        + seq_len * d_model
        + d_model * d_model * 4
        + d_ff * d_model
        + d_ff
        + d_model * d_ff
        + d_model;
    println!("Model: Emb -> Attention(Q/K/V/O) -> FFN(GELU) -> Residual -> LM Head");
    println!("d_model={}, d_ff={}, params~{}", d_model, d_ff, total_params);

    let dataset = TextDataset::new(tokens.clone(), seq_len);
    println!("Dataset samples: {}", dataset.len());
    println!("Training for {} steps...\n", max_steps);

    let mut step = 0u64;
    let mut total_loss = 0.0f32;
    let start = Instant::now();

    while step < max_steps {
        let mut dl = DataLoader::new(&dataset, 1, true);
        while let Some((inputs_batch, targets_batch)) = dl.next_batch() {
            if step >= max_steps {
                break;
            }
            let input_ids = &inputs_batch[0];
            let target_ids = &targets_batch[0];

            let cache = model.forward(&backend, input_ids);
            let loss = model.backward_and_update(&backend, &cache, target_ids, lr);

            if loss.is_nan() || loss.is_infinite() {
                continue;
            }
            total_loss += loss;
            step += 1;

            if step % log_interval == 0 {
                let avg_loss = total_loss / log_interval as f32;
                let elapsed = start.elapsed().as_secs_f32();
                let tps = (step as f32 * seq_len as f32) / elapsed;
                println!(
                    "Step {}/{}: loss={:.4}, tok/s={:.0}, elapsed={:.1}s",
                    step, max_steps, avg_loss, tps, elapsed
                );
                total_loss = 0.0;

                let prompt = tokenizer.encode("the ");
                let gen = greedy_generate(&backend, &model, &prompt, 30);
                println!("  greedy > {}", tokenizer.decode(&gen));
                let gen = sampled_generate(&backend, &model, &prompt, 30, 0.8, 10);
                println!("  sample > {}", tokenizer.decode(&gen));
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f32();
    println!("\nGPU training complete in {:.1}s", elapsed);

    println!("\n--- Final Generation ---");
    for prompt in &["the ", "is a", "model", "learning "] {
        let ids = tokenizer.encode(prompt);
        let gen = sampled_generate(&backend, &model, &ids, 50, 0.8, 10);
        println!("{}", tokenizer.decode(&gen));
    }
}
