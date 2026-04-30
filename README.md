# SPTorch — Rust 工业级异构 AI 引擎

用 Rust 从零构建的类 PyTorch 深度学习框架，覆盖张量计算、自动微分、GPU 加速、分布式训练、在线学习与垂直产品化。

**四大核心愿景：**

- **算力平权**：廉价以太网集群分布式训练（gRPC AllReduce）
- **打破垄断**：HAL 硬件抽象层 + C FFI，任意芯片即插即用
- **实时进化**：双缓冲参数 + EWC，分钟级全参数在线学习
- **垂直闭环**：Text2SQL 数据分析服务，Rust 单二进制交付（3.9MB）

## Workspace 结构（17 crates）

```text
crates/
  core-tensor/      Tensor、Shape、DType(F32/F16/BF16)、Storage(Cpu/Device)、autograd backward
  core-autograd/     计算图、反向调度（拓扑排序）
  core-ops/          23 个可微分算子（前向+反向+数值梯度检查）
  hal/               Hardware Abstraction Layer：Backend + KernelProvider(20个算子接口)
  hal-ffi/           C FFI 外部硬件接入桥接（libloading 动态加载）
  mock-npu/          Mock NPU cdylib（验证 FFI 全链路）
  nn/                Module trait、Linear、LoRA、Embedding、LayerNorm、MHA、Transformer、GPT
  optim/             SGD、AdamW、CosineScheduler、clip_grad_norm
  data/              CharTokenizer、BPE Tokenizer、TextDataset、DataLoader
  serialize/         Checkpoint 保存/加载 + safetensors 格式解析
  runtime-cuda/      CUDA 后端：nvrtc kernel + cuBLAS matmul
  distributed/       分布式引擎：gRPC coordinator/worker + AllReduce + Barrier
  live-evolution/    实时进化：双缓冲参数 + 增量训练 + EWC + 在线监控
  text2sql/          Text2SQL 产品：Axum API + SQLx + RAG + SQL 约束生成
  cli-train/         CPU MiniGPT 训练入口
  cli-train-gpu/     GPU 训练入口（Attention 模型）
  cli-text2sql/      Text2SQL 服务入口
```

## 快速开始

```bash
# 运行全量测试（188 个）
cargo test --workspace

# CPU 训练 MiniGPT
cargo run --release -p cli-train

# GPU 训练（需要 CUDA 12.x）
cargo run --release -p cli-train-gpu

# Text2SQL 服务
cargo run --release -p cli-text2sql
```

## 测试覆盖

| crate | 测试数 | 说明 |
|-------|--------|------|
| core-tensor | 8 | F16/BF16 转换、DType 接口 |
| core-ops | 56 | 所有算子前向+反向+grad check |
| nn | 23 | Linear/LoRA/Embedding/LayerNorm/MHA/Transformer/GPT |
| optim | 10 | SGD/AdamW/clip/CosineScheduler |
| data | 10 | CharTokenizer/BPE/Dataset/DataLoader |
| runtime-cuda | 12 | GPU 算子 + CPU 对比 |
| serialize | 7 | checkpoint roundtrip + safetensors |
| core-autograd | 3 | 基础 autograd |
| hal | 15 | Backend + KernelProvider 20 个算子 |
| hal-ffi | 10 | FFI 全链路集成测试 |
| distributed | 7 | AllReduce + gRPC + 多步训练 |
| live-evolution | 16 | 双缓冲/EWC/监控/端到端 |
| text2sql | 11 | Schema/RAG/SQL 约束/SQLite |
| **合计** | **188** | **全部通过** |

## 开发路线

详见 [roadmap.md](roadmap.md)（P0-P8 全部交付，PX 持续工程改进）。

## 硬件环境

- OS: Windows 10
- GPU: NVIDIA GeForce RTX 3050 (6GB, Compute 8.6)
- CUDA: 12.4
- Rust: stable (2021 edition)
