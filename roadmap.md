# SPTorch 开发路线图

> 用 Rust 从零实现的类 PyTorch 深度学习框架，最终目标：训练 MiniGPT。
> 两大核心目标：**硬件普惠**（多后端适配）+ **实时进化**（分钟级全参数更新）。

---

## 项目架构

```
crates/
  core-tensor/      Tensor 元信息、Shape、Stride、DType、Storage、autograd backward
  core-autograd/     计算图、反向调度（拓扑排序）
  core-ops/          20+ 算子（前向+反向+数值梯度检查），tiled matmul
  hal/               Hardware Abstraction Layer：Backend + KernelProvider trait
  nn/                Module trait、Linear、Embedding、LayerNorm、MHA、TransformerBlock、GPT、生成采样
  optim/             SGD、AdamW、CosineScheduler、clip_grad_norm、NaN/Inf 守卫
  data/              CharTokenizer、TextDataset、DataLoader（shuffle+reset）
  serialize/         Checkpoint 二进制保存/加载
  runtime-cuda/      CUDA 后端：nvrtc 编译 kernel + cuBLAS matmul + SGD update
  cli-train/         CPU MiniGPT 训练入口（Transformer，autograd 反向传播）
  cli-train-gpu/     GPU 训练入口（SimpleGPT，手动反向传播）
```

### 测试覆盖

| crate | 测试数 | 说明 |
|-------|--------|------|
| core-ops | 55 | 所有算子前向+反向+grad check |
| nn | 14 | Linear/Embedding/LayerNorm/MHA/TransformerBlock/GPT forward+backward |
| optim | 10 | SGD/AdamW/clip/zero_grad/NaN guard/CosineScheduler |
| data | 6 | Tokenizer/Dataset/DataLoader |
| runtime-cuda | 12 | GPU add/mul/neg/scale/exp/log/gelu/relu/matmul + CPU 对比 |
| serialize | 2 | checkpoint roundtrip + shape mismatch |
| core-autograd | 3 | 基础 autograd |
| hal | 4 | Backend trait |
| **合计** | **106** | **全部通过** |

---

## 开发计划与当前进度

### P0：地基修复与架构奠基 ✅ 已完成

- 修复 backward_step 锁作用域，拓扑排序迭代替代递归
- 引入 DType 抽象（F32/F16/BF16）
- HAL 层：Backend + KernelProvider trait
- TensorError 错误体系

### P1：最小训练闭环（CPU）✅ 已完成

- 20+ 算子全部含前向+反向+数值梯度检查
- nn 模块：Module trait, Linear, Embedding, LayerNorm
- optim 模块：SGD(+momentum), AdamW(+weight decay), zero_grad, clip_grad_norm
- Tiny LM 端到端训练：1000 步 loss 2.0→1.69

### P2：MiniGPT 训练达成 ✅ 已完成

- Transformer 核心组件：MultiHeadAttention, TransformerBlock, GPT 模型
- 新增算子：batch_matmul, gelu, masked_fill, broadcast_add, concat, scale
- 数据管道：CharTokenizer, TextDataset, DataLoader
- 训练工程：CosineScheduler(warmup+cosine decay), checkpoint 保存/加载
- 推理生成：greedy, top-k sampling with temperature
- LayerNorm 反向传播修复（LayerNormOp），Linear bias 用 broadcast_add 保持计算图

### P3：硬件普惠——GPU 适配 🔧 进行中

- **P3.1 CUDA 后端基础** ✅：cudarc 绑定，nvrtc 编译 8 个逐元素 kernel，cuBLAS sgemm
- **P3.2 GPU 前向推理** ✅：MiniGPT 在 RTX 3050 上 ~2100 tok/s
- **P3.3 GPU 训练闭环** ✅：手动反向传播 + SGD kernel，loss 稳步下降，~6000 tok/s
- P3.4 显存优化：待实现（AMP、梯度检查点、梯度累积）
- P3.5 性能工程：待实现（Profiler、算子融合、CUDA stream）

### P4：分布式训练 ⏳ 未开始

- 单机多卡 DataParallel（NCCL AllReduce）
- Pipeline / Tensor Parallelism

### P5：实时进化引擎 ⏳ 未开始

- 双缓冲参数架构（训练-推理共存）
- 增量训练调度器
- 灾难性遗忘缓解（EWC + 经验回放 + 知识蒸馏）
- 在线数据管道、监控与自动回滚

### P6：工程成熟度 ⏳ 持续

- CI/CD、文档、新后端扩展（ROCm/Metal/WebGPU）

---

## 训练实验记录

### 实验 1：CPU Tiny LM（P1 验收）

| 配置 | 值 |
|------|-----|
| 模型 | Embedding → 2×(Linear+ReLU) → Linear |
| 参数量 | ~107K |
| 数据 | 332 tokens，字符级 |
| 超参 | seq_len=8, d=32, hidden=64, lr=0.01, AdamW |
| 结果 | 1000 步，loss 2.0→1.69 |
| 速度 | ~420 tok/s (release) |
| 生成 | "hellora the the the the t"（重复，数据太少） |

**经验**：数据量太小（332 tokens）导致模型只能记忆高频 pattern。字符级 tokenizer 的理论最优 loss = ln(vocab_size)。

### 实验 2：CPU MiniGPT Transformer（P2 验收）

| 配置 | 值 |
|------|-----|
| 模型 | GPT: token/pos embedding + 2×TransformerBlock + LN + LM head |
| 参数量 | ~60K (d=48, 2层4头) |
| 数据 | 8646 tokens，字符级，31 vocab |
| 超参 | seq_len=32, lr=5e-3, warmup=100, cosine decay, AdamW |
| 结果 | 3000 步，loss 2.72→1.85，97 秒 |
| 速度 | ~1000 tok/s (release, tiled matmul) |
| 生成 (greedy) | "the the model the the the"（重复） |
| 生成 (sampling) | "the models", "cons and model"（出现类英文结构） |

**经验**：
- Greedy decoding 在 loss 还不够低时容易陷入重复，sampling 能展示模型实际学到的分布
- 字符级 tokenizer + 60K 参数的小模型能力有限，但框架正确性已验证
- Tiled matmul (32×32 分块) 比朴素三重循环快 ~50%
- Checkpoint roundtrip 验证通过

### 实验 3：CPU 大模型尝试（性能瓶颈发现）

| 配置 | 值 |
|------|-----|
| 模型 | GPT: 3层4头, d=96, d_ff=384 |
| 参数量 | ~347K |
| 结果 | 600 步后 loss 2.80→2.33，但速度降到 ~6 tok/s |

**经验**：
- CPU 上 Transformer 训练速度随模型增大急剧下降（O(n²) attention + O(n³) matmul）
- 3 层 96 维模型在 CPU 上每步需要 ~10 秒，不实际
- 这直接推动了 P3 CUDA 后端的开发

### 实验 4：GPU SimpleGPT 训练（P3 验收）

| 配置 | 值 |
|------|-----|
| 模型 | Embedding → FC1 → GELU → FC2 → Residual → LM Head（无 attention） |
| 参数量 | ~83K (d=96, d_ff=384) |
| 数据 | 8646 tokens，字符级 |
| 超参 | seq_len=32, lr=0.02, SGD |
| 结果 | 5000 步，loss 3.15→2.41，31 秒 |
| 速度 | ~5200 tok/s (GPU, release) |
| 生成 | "the the the the the the"（重复） |

**经验**：
- GPU 训练 ~5200 tok/s vs CPU ~1000 tok/s，约 5× 加速
- 无 attention 的简化模型无法学习位置相关的模式，生成质量差
- 手动反向传播可行但代码量大，对于完整 Transformer 不实际
- 复合操作（softmax/LayerNorm/embedding）目前走 host 端计算再传回 GPU，是性能瓶颈
- cuBLAS matmul 是真正的加速点，逐元素 kernel 对小 tensor 加速不明显

### 关键发现与经验总结

1. **生成质量 vs loss**：loss < 2.0 时 greedy 开始出现词级模式，loss < 1.5 才能生成可读文本。Sampling (temp=0.8, top_k=10) 比 greedy 更能展示模型能力。

2. **字符级 tokenizer 的局限**：vocab=31 时理论最优 loss = ln(31) ≈ 3.43。实际 loss 2.4 说明模型学到了大量 pattern，但字符级粒度限制了生成连贯性。BPE tokenizer 是下一步提升的关键。

3. **CPU matmul 是瓶颈**：Transformer 训练 80%+ 时间在 matmul。Tiled matmul 有帮助但不够，BLAS 库或 GPU 是必须的。

4. **GPU host↔device 传输开销**：当前复合操作（softmax/LayerNorm）走 host 端计算，每次 dtoh+htod 有 ~0.1ms 延迟。对于小 tensor 这个开销占比很大。下一步应实现纯 GPU kernel。

5. **手动 backward vs autograd**：手动 backward 对简单架构可行，但对完整 Transformer（attention + LayerNorm + 残差）代码量爆炸。正确的方向是让 autograd 框架支持 GPU tensor。

---

## 下一步优先级

1. **autograd 支持 GPU tensor**：让 core-tensor 的 Storage 支持 GPU 内存，autograd 自动调度 GPU kernel。这样 nn::GPT 可以直接在 GPU 上训练。
2. **纯 GPU softmax/LayerNorm kernel**：消除 host↔device 传输瓶颈。
3. **BPE tokenizer**：提升生成质量的关键。
4. **混合精度训练（AMP）**：FP16 前向 + FP32 master weights，显存减半，速度翻倍。

---

## 硬件环境

- CPU: Windows 10
- GPU: NVIDIA GeForce RTX 3050 (6GB, Compute 8.6)
- CUDA: 12.4 (nvcc + cudarc 0.12)
- Rust: stable
