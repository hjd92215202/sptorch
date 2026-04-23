use core_ops::*;
use nn::{GPT, generate_greedy, generate_with_sampling};
use optim::{AdamW, Optimizer, CosineScheduler, LrScheduler, clip_grad_norm};
use data::{CharTokenizer, TextDataset, Dataset, DataLoader};
use serialize::{save_checkpoint, load_checkpoint};
use std::time::Instant;

fn main() {
    println!("SPTorch MiniGPT Training");
    println!("========================\n");

    // 准备数据
    let text = include_str!("train_data.txt");

    let tokenizer = CharTokenizer::from_text(text);
    let tokens = tokenizer.encode(text);
    let vocab_size = tokenizer.vocab_size();

    println!("Vocab size: {}", vocab_size);
    println!("Total tokens: {}", tokens.len());

    // 超参数
    let seq_len = 64;
    let d_model = 96;
    let n_head = 4;
    let n_layer = 3;
    let d_ff = 384;
    let lr = 3e-3;
    let max_steps: u64 = 1000;
    let warmup_steps: u64 = 50;
    let log_interval = 100;
    let checkpoint_path = "minigpt_checkpoint.sptc";

    // 模型
    let model = GPT::new(vocab_size, d_model, n_head, n_layer, d_ff, seq_len);
    let params = model.parameters();
    let total_params: usize = params.iter().map(|p| p.numel()).sum();
    println!("Model: {} layers, {} heads, d_model={}, d_ff={}", n_layer, n_head, d_model, d_ff);
    println!("Total parameters: {}", total_params);

    // 优化器和调度器
    let mut optimizer = AdamW::default(params.clone(), lr);
    let scheduler = CosineScheduler::new(lr, warmup_steps, max_steps);

    // 数据
    let dataset = TextDataset::new(tokens.clone(), seq_len);
    println!("Dataset samples: {}", dataset.len());
    println!("Training for {} steps...\n", max_steps);

    let mut step: u64 = 0;
    let mut total_loss = 0.0f32;
    let mut total_tokens = 0usize;
    let start = Instant::now();

    while step < max_steps {
        let mut dl = DataLoader::new(&dataset, 1, true);

        while let Some((inputs_batch, targets_batch)) = dl.next_batch() {
            if step >= max_steps { break; }

            let input_ids = &inputs_batch[0];
            let target_ids = &targets_batch[0];

            // Update learning rate
            let current_lr = scheduler.get_lr(step);
            optimizer.set_lr(current_lr);

            // Forward
            let logits = model.forward_ids(input_ids);
            let loss = cross_entropy_loss(&logits, target_ids);
            let loss_val = loss.data()[0];

            if loss_val.is_nan() || loss_val.is_infinite() {
                eprintln!("Step {}: Loss is NaN/Inf, skipping", step);
                optimizer.zero_grad();
                continue;
            }

            // Backward
            optimizer.zero_grad();
            loss.backward();

            // Clip & step
            let grad_norm = clip_grad_norm(&params, 1.0);
            optimizer.step();

            total_loss += loss_val;
            total_tokens += seq_len;
            step += 1;

            if step % log_interval as u64 == 0 {
                let avg_loss = total_loss / log_interval as f32;
                let elapsed = start.elapsed().as_secs_f32();
                let tps = total_tokens as f32 / elapsed;
                println!("Step {}/{}: loss={:.4}, lr={:.6}, grad_norm={:.4}, tok/s={:.0}",
                         step, max_steps, avg_loss, current_lr, grad_norm, tps);
                total_loss = 0.0;

                // Sample generation
                let prompt_ids = tokenizer.encode("the ");
                let gen = generate_greedy(&model, &prompt_ids, 30, vocab_size);
                println!("  > {}", tokenizer.decode(&gen));
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f32();
    println!("\nTraining complete in {:.1}s", elapsed);

    // Save checkpoint
    println!("Saving checkpoint to {}...", checkpoint_path);
    save_checkpoint(checkpoint_path, &params).unwrap();
    println!("Checkpoint saved.");

    // Generate samples
    println!("\n--- Greedy Generation ---");
    for prompt in &["the ", "hello", "is a"] {
        let ids = tokenizer.encode(prompt);
        let gen = generate_greedy(&model, &ids, 50, vocab_size);
        println!("{}", tokenizer.decode(&gen));
    }

    println!("\n--- Sampled Generation (temp=0.8, top_k=10) ---");
    for prompt in &["the ", "hello", "is a"] {
        let ids = tokenizer.encode(prompt);
        let gen = generate_with_sampling(&model, &ids, 50, vocab_size, 0.8, 10);
        println!("{}", tokenizer.decode(&gen));
    }

    // Verify checkpoint load
    println!("\n--- Checkpoint Verification ---");
    let model2 = GPT::new(vocab_size, d_model, n_head, n_layer, d_ff, seq_len);
    let params2 = model2.parameters();
    load_checkpoint(checkpoint_path, &params2).unwrap();
    let test_ids = tokenizer.encode("the ");
    let gen1 = generate_greedy(&model, &test_ids, 20, vocab_size);
    let gen2 = generate_greedy(&model2, &test_ids, 20, vocab_size);
    if gen1 == gen2 {
        println!("Checkpoint load verified: generation matches.");
    } else {
        println!("WARNING: generation mismatch after checkpoint load!");
    }
}
