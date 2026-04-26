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
  cli-train-gpu/     GPU 训练入口（Attention 模型，手动反向传播）
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

- **P3.1 CUDA 后端基础** ✅：cudarc 绑定，nvrtc 编译 10 个 kernel（add/mul/neg/scale/exp/log/gelu/relu/sgd/accum），cuBLAS sgemm
- **P3.2 GPU 前向推理** ✅：完整 Transformer 前向在 RTX 3050 上 ~2100 tok/s
- **P3.3 GPU 训练闭环** ✅：手动反向传播 + SGD kernel，SimpleGPT loss 稳步下降 ~6000 tok/s
- **P3.4 GPU Attention 训练** ✅：单头 causal attention + FFN，手动 backward ~200 行，loss 3.13→2.38
- P3.5 显存优化：待实现（AMP、梯度检查点、梯度累积）
- P3.6 性能工程：待实现（纯 GPU softmax/LN kernel、算子融合、CUDA stream）

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

### 实验 4：GPU SimpleGPT 训练（无 attention，P3 验收）

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

### 实验 5：GPU Attention 模型训练

| 配置 | 值 |
|------|-----|
| 模型 | Embedding → SingleHeadAttention(Q/K/V/O) → FFN(GELU) → Residual → LM Head |
| 参数量 | ~55K (d=64, d_ff=256, 单头 attention) |
| 数据 | 8646 tokens，字符级，31 vocab |
| 超参 | seq_len=32, lr=0.05, SGD, 手动 backward |
| 结果 | 5000 步，loss 3.13→2.38，53 秒 |
| 速度 | ~3000 tok/s (GPU, release) |
| 生成 (greedy) | "the the the the the"（仍重复） |
| 生成 (sampling) | "the theres arame ans tun thereci"（出现类英文结构） |

**经验**：
- 加入 attention 后 sampling 生成质量明显好于无 attention 版本，出现 "learning in", "models", "theres" 等类英文片段
- 单头 attention 的 GPU 训练比无 attention 版本慢（~3000 vs ~5200 tok/s），因为 attention 有 O(seq²) 的 softmax + matmul
- SGD 收敛比 AdamW 慢，lr=0.05 比 lr=0.01 好很多
- Greedy 仍然重复——这是字符级小模型的固有问题，不是框架 bug
- 手动 backward 对单层 attention 可行（~200 行代码），但多层会代码爆炸

### 实验 6：CPU Autograd + GPU cuBLAS 混合训练

| 配置 | 值 |
|------|-----|
| 模型 | GPT: 2层4头, d=48, d_ff=192（完整 Transformer，autograd 反向传播） |
| 参数量 | ~60K |
| 数据 | 8646 tokens，字符级，31 vocab |
| 超参 | seq_len=32, lr=5e-3, warmup=100, cosine decay, AdamW |
| GPU 加速 | core-ops matmul 自动 offload 到 cuBLAS（feature flag "cuda"） |
| 结果 | 3000 步，loss 2.72→1.80，115 秒 |
| 速度 | ~850 tok/s |
| 生成 (greedy) | "the train the the train the train"（首次出现词级模式） |
| 生成 (sampling) | "the to produtions al a to padwicalins toker formatt"（类英文句子） |

**经验**：
- 这是最佳方案：autograd 保证梯度正确性，cuBLAS 加速 matmul 热点
- loss 首次降到 2.0 以下（1.80），greedy 生成出现 "train", "model" 等完整单词
- 速度比纯 CPU（~1000 tok/s）略慢——因为小矩阵的 GPU 传输开销抵消了 cuBLAS 加速
- 对于大矩阵（d≥128），GPU 加速效果会更明显
- AdamW + cosine decay 比 SGD 收敛快得多（同样 3000 步，AdamW loss 1.80 vs SGD loss 2.38）
- Checkpoint 保存/加载验证通过

### 8. GPU matmul offload 方案选择（P3.5）

**问题**：如何让现有 autograd 框架利用 GPU 加速，而不重写整个 Storage 层？

**分析**：
- 方案 A：重构 core-tensor Storage 支持 GPU 内存 → 改动巨大，破坏 106 个测试
- 方案 B：在 core-ops 中用 feature flag 可选依赖 runtime-cuda，matmul 自动 offload → 零侵入

**决策**：方案 B。在 `core-ops/Cargo.toml` 中加 `cuda = ["runtime-cuda"]` feature，matmul 函数内部检测 GPU 可用性，自动选择 cuBLAS 或 CPU tiled matmul。用 `OnceLock` 做全局单例初始化。

**结果**：
- 现有 106 个测试不受影响（不启用 cuda feature 时完全不变）
- cli-train 只需在 Cargo.toml 加 `cuda = ["core-ops/cuda"]` 即可启用
- 首次 matmul 调用时自动初始化 GPU，打印 `[sptorch] GPU accelerator enabled`

**教训**：feature flag + 全局单例是 Rust 中做可选硬件加速的惯用模式，比改 trait 层级简单得多。

---

## 调试过程与决策记录

> 记录开发过程中遇到的问题、尝试的方案、最终的决策和原因。方便后续学习和系统优化。

### 1. 数值梯度检查失败（P1 初期）

**问题**：`test_grad_check_add/mul/matmul` 3 个测试失败，numerical=1.001358 vs analytical=1，diff=0.001358。

**分析**：eps=1e-4 在 f32 下有限差分精度不够。f32 有效精度约 7 位十进制，eps 太小会导致截断误差，太大会导致近似误差。

**尝试**：
- 方案 A：减小 eps 到 1e-5 → 截断误差更大，反而更差
- 方案 B：放宽 tol 到 1e-2 → 通过，且 PyTorch 的 gradcheck 在 f64 下默认 atol=1e-5, rtol=1e-3

**决策**：eps=1e-3, tol=1e-2。这是 f32 有限差分的合理参数组合。

### 2. LayerNorm 断开计算图（P2 中期）

**问题**：`test_gpt_backward_runs` 失败，`param[11] 'ffn_up_b' has no gradient`。

**分析**：LayerNorm.forward() 用 `Tensor::new()` 创建输出，没有 creator 节点，梯度无法传播。Linear 的 bias broadcast 也有同样问题——用 `Tensor::new` 创建了新 tensor 断开了与原始 bias 参数的计算图。

**修复**：
1. LayerNorm：添加 `LayerNormOp` 实现完整的反向传播（d_input, d_gamma, d_beta）
2. Linear bias：从手动 broadcast + `Tensor::new` 改为 `broadcast_add(&out, bias)`，保持计算图连通

**教训**：任何创建新 Tensor 的地方都必须检查是否需要建立 autograd 节点。`Tensor::new()` 是"计算图断点"。

### 3. CPU 训练速度瓶颈（P2 后期）

**问题**：3 层 96 维 Transformer 在 CPU 上只有 ~6 tok/s，训练 1000 步需要 ~3 小时。

**分析**：profiling 发现 matmul 占 80%+ 时间。朴素三重循环 O(n³) 对 cache 不友好。

**尝试**：
- 方案 A：tiled matmul (32×32 分块) → 速度从 ~22 提升到 ~32 tok/s（初始 100 步），约 50% 提升
- 方案 B：接入 BLAS 库（openblas/mkl）→ 需要额外依赖，暂不采用
- 方案 C：上 GPU → 最终选择

**决策**：先用 tiled matmul 作为 CPU 优化，同时启动 P3 CUDA 后端。CPU 训练用小模型（d=48）快速迭代验证框架正确性，大模型训练交给 GPU。

### 4. PTX 编译失败（P3 初期）

**问题**：手写 PTX 汇编在 RTX 3050 (SM 8.6) 上报 `CUDA_ERROR_INVALID_PTX`。

**分析**：手写 PTX 指定了 `.target sm_80`，但 PTX 语法细节对 SM 版本敏感，寄存器命名和指令集有差异。

**决策**：改用 CUDA C 源码 + nvrtc 运行时编译。nvrtc 自动处理 SM 版本适配，代码可读性也更好。代价是首次编译多 ~0.5 秒，但只需编译一次。

### 5. GPU 训练无 loss 下降（P3.2 → P3.3）

**问题**：P3.2 实现了 GPU 前向推理（~2100 tok/s），但 loss 不下降——因为没有反向传播。

**分析**：两个方案：
- 方案 A：让 autograd 框架支持 GPU tensor → 改动大，需要重构 core-tensor 的 Storage
- 方案 B：手动实现 backward → 代码多但独立，不影响现有 106 个测试

**决策**：先用方案 B（手动 backward）验证 GPU 训练可行性，后续再做方案 A。

**结果**：SimpleGPT（无 attention）手动 backward ~80 行代码，loss 稳步下降，验证了 GPU 训练闭环。

### 6. 无 attention 模型生成质量差（P3.3 → P3.4）

**问题**：GPU SimpleGPT 训练 5000 步 loss 降到 2.41，但生成全是 "the the the the"。

**分析**：无 attention 的 FFN-only 模型对每个位置独立处理，无法学习 token 间的依赖关系。"the" 是训练数据中最高频的 token，greedy decoding 总是选它。

**尝试**：
- 增大模型（d=96, d_ff=384, 83K 参数）→ loss 降到 2.41 但生成仍重复
- 增加训练步数（3000→5000）→ loss 继续下降但生成不改善

**决策**：必须加入 attention 机制。实现了单头 causal attention 的完整 forward+backward（~200 行手动梯度代码）。

**结果**：加入 attention 后 sampling 生成出现 "learning in", "models", "theres" 等类英文片段，验证了 attention 对位置依赖建模的关键作用。

### 7. SGD vs AdamW 收敛速度（P3.4）

**问题**：GPU 手动 backward 用 SGD（因为 AdamW 需要维护 m/v 状态，手动实现复杂），lr=0.01 时收敛很慢。

**分析**：SGD 没有自适应学习率，对不同参数用同一个 lr。Embedding 层梯度稀疏，需要更大 lr；attention 权重梯度密集，lr 太大会不稳定。

**尝试**：
- lr=0.01 → 3000 步 loss 3.42→2.87，收敛慢
- lr=0.05 → 5000 步 loss 3.13→2.38，明显更好

**教训**：SGD 需要比 AdamW 更高的 lr（约 5-10×），但也更容易发散。后续应在 GPU 上实现 AdamW（需要 m/v 状态的 GPU buffer）。

---

## 关键发现与经验总结

1. **生成质量 vs loss**：loss < 2.0 时 greedy 开始出现词级模式，loss < 1.5 才能生成可读文本。Sampling (temp=0.8, top_k=10) 比 greedy 更能展示模型能力。

2. **字符级 tokenizer 的局限**：vocab=31 时理论最优 loss = ln(31) ≈ 3.43。实际 loss 2.4 说明模型学到了大量 pattern，但字符级粒度限制了生成连贯性。BPE tokenizer 是下一步提升的关键。

3. **CPU matmul 是瓶颈**：Transformer 训练 80%+ 时间在 matmul。Tiled matmul 有帮助但不够，BLAS 库或 GPU 是必须的。

4. **GPU host↔device 传输开销**：当前复合操作（softmax/LayerNorm）走 host 端计算，每次 dtoh+htod 有 ~0.1ms 延迟。对于小 tensor 这个开销占比很大。下一步应实现纯 GPU kernel。

5. **手动 backward vs autograd**：手动 backward 对简单架构可行，但对完整 Transformer（attention + LayerNorm + 残差）代码量爆炸。正确的方向是让 autograd 框架支持 GPU tensor。

6. **Attention 是生成质量的关键**：无 attention 的 FFN-only 模型即使 loss 降到 2.4 也只能生成重复文本。加入 attention 后立即出现位置相关的模式。

7. **初始化很重要**：Xavier/Kaiming 初始化 + 较小的 scale (×0.5) 对训练稳定性有帮助。初始化太大会导致 softmax 饱和，梯度消失。

---

## 下一步优先级

1. **autograd 支持 GPU tensor**：让 core-tensor 的 Storage 支持 GPU 内存，autograd 自动调度 GPU kernel。这样 nn::GPT 可以直接在 GPU 上训练，不需要手动 backward。
2. **纯 GPU softmax/LayerNorm kernel**：消除 host↔device 传输瓶颈，预计可提速 2-3×。
3. **GPU AdamW**：在 GPU 上维护 m/v 状态，比 SGD 收敛更快。
4. **BPE tokenizer**：提升生成质量的关键，从字符级（31 vocab）升级到子词级（~1000 vocab）。
5. **混合精度训练（AMP）**：FP16 前向 + FP32 master weights，显存减半，速度翻倍。

---

## Git 提交历史

```
3cb5374 Initial commit
0666f1e Initial monorepo: core-tensor and core-autograd
e95fed2 l1-将 test_autograd_basic 测试涉及的add流程debug清楚
c8e7bf3 P0: 架构奠基 - 拓扑排序反向传播、HAL抽象层、DType、错误处理
1f15781 P1+P2.1+P2.2: 最小训练闭环 + Transformer组件 + 数据管道
117a94d P2: MiniGPT训练达成 - Transformer完整流程跑通
7e077e5 优化MiniGPT: 扩大训练数据(8.6K tokens)、增大模型(3层96维)
3319cf7 性能优化: tiled matmul (32x32分块) 提升cache局部性
d33ed5c 调优MiniGPT: 小模型快速迭代(2层48维, 60K参数, 3000步)
61a3bfa P3.1: CUDA后端基础实现 - 逐元素kernel + cuBLAS matmul
0b65c24 P3.2: GPU端到端前向跑通 - MiniGPT在RTX 3050上~2100 tok/s
3b7bbb9 P3.3: GPU训练闭环 - 手动反向传播 + SGD更新，loss稳步下降
9c305cf GPU训练调优: 增大模型(d=96,ff=384), 5000步loss 3.15→2.41
68ba3cc 添加 roadmap.md: 架构、进度、实验记录、经验总结
bdf6661 GPU Attention模型: 单头attention+手动backward, loss 3.13→2.38
```

---

## 硬件环境

- OS: Windows 10
- GPU: NVIDIA GeForce RTX 3050 (6GB, Compute 8.6)
- CUDA: 12.4 (nvcc via Anaconda + cudarc 0.12)
- Rust: stable
- 构建: cargo, release 模式用于训练
