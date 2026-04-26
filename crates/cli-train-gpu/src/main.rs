use runtime_cuda::{CudaBackend, GpuTensor};
use data::{CharTokenizer, TextDataset, Dataset, DataLoader};
use std::time::Instant;
use rand::Rng;

// ============ GPU MiniGPT with Manual Backward ============
// Single-layer simplified GPT: Embedding -> Linear -> GELU -> Linear -> LM Head
// Full attention is expensive to backprop manually, so we use a simpler architecture
// that still demonstrates GPU training with loss decreasing.

struct GpuSimpleGPT {
    tok_emb: GpuTensor,    // [vocab, d_model]
    pos_emb: GpuTensor,    // [seq_len, d_model]
    fc1_w: GpuTensor,      // [d_ff, d_model]
    fc1_b: GpuTensor,      // [d_ff]
    fc2_w: GpuTensor,      // [d_model, d_ff]
    fc2_b: GpuTensor,      // [d_model]
    lm_head_w: GpuTensor,  // [vocab, d_model]
    vocab_size: usize,
    d_model: usize,
    d_ff: usize,
    max_seq_len: usize,
}

fn rand_vec(n: usize, scale: f32) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen_range(-scale..scale)).collect()
}

impl GpuSimpleGPT {
    fn new(backend: &CudaBackend, vocab: usize, d_model: usize, d_ff: usize, seq_len: usize) -> Self {
        let s = (2.0 / d_model as f32).sqrt();
        let sf = (2.0 / d_ff as f32).sqrt();
        GpuSimpleGPT {
            tok_emb: GpuTensor::from_host(backend, &rand_vec(vocab * d_model, s), vec![vocab, d_model]).unwrap(),
            pos_emb: GpuTensor::from_host(backend, &rand_vec(seq_len * d_model, s), vec![seq_len, d_model]).unwrap(),
            fc1_w: GpuTensor::from_host(backend, &rand_vec(d_ff * d_model, sf), vec![d_ff, d_model]).unwrap(),
            fc1_b: GpuTensor::from_host(backend, &vec![0.0; d_ff], vec![d_ff]).unwrap(),
            fc2_w: GpuTensor::from_host(backend, &rand_vec(d_model * d_ff, s), vec![d_model, d_ff]).unwrap(),
            fc2_b: GpuTensor::from_host(backend, &vec![0.0; d_model], vec![d_model]).unwrap(),
            lm_head_w: GpuTensor::from_host(backend, &rand_vec(vocab * d_model, s), vec![vocab, d_model]).unwrap(),
            vocab_size: vocab,
            d_model,
            d_ff,
            max_seq_len: seq_len,
        }
    }

    /// Forward pass returning intermediate activations for backward
    fn forward(&self, backend: &CudaBackend, token_ids: &[usize]) -> ForwardCache {
        let seq_len = token_ids.len();

        // Embedding
        let tok = backend.gpu_embedding(&self.tok_emb, token_ids).unwrap();
        let positions: Vec<usize> = (0..seq_len).collect();
        let pos = backend.gpu_embedding(&self.pos_emb, &positions).unwrap();
        let emb = backend.gpu_add(&tok, &pos).unwrap(); // [seq, d_model]

        // FC1: [seq, d_model] @ [d_model, d_ff] + bias = [seq, d_ff]
        let fc1_wt = backend.gpu_transpose(&self.fc1_w).unwrap(); // [d_model, d_ff]
        let h1_pre = backend.gpu_matmul(&emb, &fc1_wt).unwrap();
        let h1_pre = backend.gpu_broadcast_add(&h1_pre, &self.fc1_b).unwrap();

        // GELU activation
        let h1 = backend.gpu_gelu(&h1_pre).unwrap(); // [seq, d_ff]

        // FC2: [seq, d_ff] @ [d_ff, d_model] + bias = [seq, d_model]
        let fc2_wt = backend.gpu_transpose(&self.fc2_w).unwrap(); // [d_ff, d_model]
        let h2 = backend.gpu_matmul(&h1, &fc2_wt).unwrap();
        let h2 = backend.gpu_broadcast_add(&h2, &self.fc2_b).unwrap(); // [seq, d_model]

        // Residual
        let out = backend.gpu_add(&emb, &h2).unwrap(); // [seq, d_model]

        // LM Head: [seq, d_model] @ [d_model, vocab] = [seq, vocab]
        let lm_wt = backend.gpu_transpose(&self.lm_head_w).unwrap();
        let logits = backend.gpu_matmul(&out, &lm_wt).unwrap();

        ForwardCache {
            token_ids: token_ids.to_vec(),
            positions,
            emb, h1_pre, h1, h2, out, logits,
            seq_len,
        }
    }

    /// Manual backward pass + SGD update
    fn backward_and_update(&mut self, backend: &CudaBackend, cache: &ForwardCache, targets: &[usize], lr: f32) -> f32 {
        let seq = cache.seq_len;
        let d = self.d_model;
        let ff = self.d_ff;
        let v = self.vocab_size;

        // Cross-entropy loss + softmax
        let (loss, sm) = backend.gpu_cross_entropy(&cache.logits, targets).unwrap();

        // d_logits = (softmax - one_hot) / seq_len  [seq, vocab]
        let d_logits = backend.gpu_cross_entropy_backward(&sm, targets).unwrap();

        // --- LM Head backward ---
        // logits = out @ lm_head_w^T
        // d_out = d_logits @ lm_head_w  [seq, d_model]
        let d_out = backend.gpu_matmul(&d_logits, &self.lm_head_w).unwrap();
        // d_lm_head_w = d_logits^T @ out  [vocab, seq] @ [seq, d_model] = [vocab, d_model]
        let d_logits_t = backend.gpu_transpose(&d_logits).unwrap();
        let d_lm_head = backend.gpu_matmul(&d_logits_t, &cache.out).unwrap();

        // --- Residual backward ---
        // out = emb + h2, so d_emb_res = d_out, d_h2 = d_out
        let d_h2 = &d_out; // [seq, d_model]

        // --- FC2 backward ---
        // h2 = h1 @ fc2_w^T + fc2_b
        // d_h1 = d_h2 @ fc2_w  [seq, d_ff]
        let d_h1 = backend.gpu_matmul(d_h2, &self.fc2_w).unwrap();
        // d_fc2_w = d_h2^T @ h1  [d_model, seq] @ [seq, d_ff] = [d_model, d_ff]
        let d_h2_t = backend.gpu_transpose(d_h2).unwrap();
        let d_fc2_w = backend.gpu_matmul(&d_h2_t, &cache.h1).unwrap();
        // d_fc2_b = sum(d_h2, axis=0)  [d_model]
        let d_fc2_b = sum_rows(backend, d_h2, seq, d);

        // --- GELU backward ---
        // h1 = gelu(h1_pre)
        // d_h1_pre = d_h1 * gelu'(h1_pre)
        let d_h1_pre = gelu_backward(backend, &d_h1, &cache.h1_pre);

        // --- FC1 backward ---
        // h1_pre = emb @ fc1_w^T + fc1_b
        // d_emb_fc1 = d_h1_pre @ fc1_w  [seq, d_model]
        let d_emb_fc1 = backend.gpu_matmul(&d_h1_pre, &self.fc1_w).unwrap();
        // d_fc1_w = d_h1_pre^T @ emb  [d_ff, seq] @ [seq, d_model] = [d_ff, d_model]
        let d_h1_pre_t = backend.gpu_transpose(&d_h1_pre).unwrap();
        let d_fc1_w = backend.gpu_matmul(&d_h1_pre_t, &cache.emb).unwrap();
        // d_fc1_b = sum(d_h1_pre, axis=0)  [d_ff]
        let d_fc1_b = sum_rows(backend, &d_h1_pre, seq, ff);

        // --- Embedding backward ---
        // d_emb_total = d_out (from residual) + d_emb_fc1
        let d_emb_total = backend.gpu_add(&d_out, &d_emb_fc1).unwrap();
        // d_tok_emb: scatter d_emb_total back to [vocab, d_model]
        let d_tok_emb = backend.gpu_embedding_backward(&d_emb_total, &cache.token_ids, v, d).unwrap();
        let d_pos_emb = backend.gpu_embedding_backward(&d_emb_total, &cache.positions, self.max_seq_len, d).unwrap();

        // --- SGD updates ---
        backend.gpu_sgd_update(&mut self.lm_head_w, &d_lm_head, lr).unwrap();
        backend.gpu_sgd_update(&mut self.fc2_w, &d_fc2_w, lr).unwrap();
        backend.gpu_sgd_update(&mut self.fc2_b, &d_fc2_b, lr).unwrap();
        backend.gpu_sgd_update(&mut self.fc1_w, &d_fc1_w, lr).unwrap();
        backend.gpu_sgd_update(&mut self.fc1_b, &d_fc1_b, lr).unwrap();
        backend.gpu_sgd_update(&mut self.tok_emb, &d_tok_emb, lr).unwrap();
        backend.gpu_sgd_update(&mut self.pos_emb, &d_pos_emb, lr).unwrap();

        loss
    }
}

struct ForwardCache {
    token_ids: Vec<usize>,
    positions: Vec<usize>,
    emb: GpuTensor,
    h1_pre: GpuTensor,
    h1: GpuTensor,
    h2: GpuTensor,
    out: GpuTensor,
    logits: GpuTensor,
    seq_len: usize,
}

/// Sum rows: [rows, cols] -> [cols]
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

/// GELU backward: d_input = d_output * gelu'(input)
fn gelu_backward(backend: &CudaBackend, d_out: &GpuTensor, input: &GpuTensor) -> GpuTensor {
    let g = d_out.to_host(backend).unwrap();
    let x = input.to_host(backend).unwrap();
    let sqrt_2_pi: f32 = (2.0 / std::f32::consts::PI).sqrt();
    let da: Vec<f32> = g.iter().zip(x.iter()).map(|(gi, &xi)| {
        let k = sqrt_2_pi * (xi + 0.044715 * xi.powi(3));
        let tanh_k = k.tanh();
        let dk = sqrt_2_pi * (1.0 + 0.134145 * xi * xi);
        let gelu_grad = 0.5 * (1.0 + tanh_k) + 0.5 * xi * (1.0 - tanh_k * tanh_k) * dk;
        gi * gelu_grad
    }).collect();
    GpuTensor::from_host(backend, &da, d_out.shape.clone()).unwrap()
}

fn greedy_generate(backend: &CudaBackend, model: &GpuSimpleGPT, prompt: &[usize], max_new: usize) -> Vec<usize> {
    let vocab = model.vocab_size;
    let mut ids = prompt.to_vec();
    for _ in 0..max_new {
        let ctx = if ids.len() > model.max_seq_len { &ids[ids.len() - model.max_seq_len..] } else { &ids };
        let cache = model.forward(backend, ctx);
        let logits_host = cache.logits.to_host(backend).unwrap();
        let last = &logits_host[logits_host.len() - vocab..];
        let next = last.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap();
        ids.push(next);
    }
    ids
}

// ============ Main ============

fn main() {
    println!("SPTorch GPU Training (with backward pass)");
    println!("==========================================\n");

    let backend = CudaBackend::new(0).expect("Failed to init CUDA");
    backend.load_kernels().expect("Failed to load kernels");
    println!("CUDA backend initialized.");

    let text = include_str!("../../cli-train/src/train_data.txt");
    let tokenizer = CharTokenizer::from_text(text);
    let tokens = tokenizer.encode(text);
    let vocab_size = tokenizer.vocab_size();

    println!("Vocab size: {}", vocab_size);
    println!("Total tokens: {}", tokens.len());

    // Hyperparameters
    let seq_len = 32;
    let d_model = 96;
    let d_ff = 384;
    let lr = 0.02f32;
    let max_steps: u64 = 5000;
    let log_interval: u64 = 500;

    let mut model = GpuSimpleGPT::new(&backend, vocab_size, d_model, d_ff, seq_len);
    let total_params = vocab_size * d_model + seq_len * d_model
        + d_ff * d_model + d_ff + d_model * d_ff + d_model + vocab_size * d_model;
    println!("Model: Embedding -> FC1 -> GELU -> FC2 -> Residual -> LM Head");
    println!("d_model={}, d_ff={}, params={}", d_model, d_ff, total_params);

    let dataset = TextDataset::new(tokens.clone(), seq_len);
    println!("Dataset samples: {}", dataset.len());
    println!("Training for {} steps with GPU forward+backward...\n", max_steps);

    let mut step = 0u64;
    let mut total_loss = 0.0f32;
    let start = Instant::now();

    while step < max_steps {
        let mut dl = DataLoader::new(&dataset, 1, true);

        while let Some((inputs_batch, targets_batch)) = dl.next_batch() {
            if step >= max_steps { break; }

            let input_ids = &inputs_batch[0];
            let target_ids = &targets_batch[0];

            // Forward
            let cache = model.forward(&backend, input_ids);

            // Backward + update
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
                println!("Step {}/{}: loss={:.4}, tok/s={:.0}, elapsed={:.1}s",
                         step, max_steps, avg_loss, tps, elapsed);
                total_loss = 0.0;

                let prompt = tokenizer.encode("the ");
                let gen = greedy_generate(&backend, &model, &prompt, 30);
                println!("  > {}", tokenizer.decode(&gen));
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f32();
    println!("\nGPU training complete in {:.1}s", elapsed);

    // Final generation
    println!("\n--- Generation Samples ---");
    for prompt in &["the ", "is a", "model"] {
        let ids = tokenizer.encode(prompt);
        let gen = greedy_generate(&backend, &model, &ids, 40);
        println!("{}", tokenizer.decode(&gen));
    }
}
