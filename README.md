# SPTorch — Rust 工业级异构 AI 引擎

用 Rust 从零构建的全栈 AI 平台，覆盖张量计算、自动微分、GPU 加速、分布式训练、在线学习、硬件抽象与分层 IDE 生态。

## 项目架构定位（统一术语）

- **平台层（母体）**：`SPTorch` 是全栈 AI 平台/框架体系（引擎 + 协议 + 硬件抽象 + IDE 生态）。
- **控制中枢层**：`SPTorch Studio` 是平台控制中枢，负责观测、编排、调试与交付。
- **产品层**：工业版 `Text2SQL` 是平台生态中的首个生产级训练产品（样板产品），用于验证平台能力闭环。
- **边界原则**：产品仅通过 `sptorch::v1` 稳定门面 API 依赖框架，不直接依赖内部子 crate。

**四大核心愿景：**

- **算力平权**：廉价以太网集群分布式训练（gRPC AllReduce）
- **硬件主权**：自研 DDR4 PCB，HAL 硬件抽象层 + C FFI，任意芯片即插即用
- **生命主权**：双缓冲参数 + EWC，分钟级全参数在线学习，模型在业务中自我进化
- **开发者主权**：分层 IDE（SPTorch Studio），让业务人员拥有 AI，让硬件工程师指挥模型

**能力验证：**

- **Text2SQL 一体机**：生态中的首个生产级样板产品，Rust 单二进制交付（3.9MB），验证平台从训练→推理→交付的完整闭环能力

## Workspace 结构（平台 crates + 产品 packages）

```text
crates/
  sptorch/          框架统一门面 crate（供外部产品依赖）
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
  cli-train/         CPU MiniGPT 训练入口
  cli-train-gpu/     GPU 训练入口（Attention 模型）
  versioning/        版本化张量协议（VersionedStorage/UpdatePolicy/FenceState/Metrics）
products/
  text2sql/          Text2SQL 产品服务层：Axum API + SQLx + RAG + SQL 约束生成（框架无关）
  cli-text2sql/      Text2SQL 产品运行时（训练/推理引擎适配 + 服务入口）
studio/
  src-tauri/         SPTorch Studio 控制中枢（Tauri 2 后端）
```

## 快速开始

```bash
# 运行框架 workspace 测试
cargo test --workspace

# CPU 训练 MiniGPT
cargo run --release -p sptorch-cli-train

# GPU 训练（需要 CUDA 12.x）
cargo run --release -p sptorch-cli-train-gpu

# Text2SQL 服务（生态样板产品）
cargo run --release --manifest-path products/Cargo.toml -p sptorch-text2sql

# 产品 workspace 测试
cargo test --manifest-path products/Cargo.toml --workspace
```

发布与版本策略见 [docs.release-strategy.md](docs.release-strategy.md)。
发布顺序清单见 [docs.publish-order.md](docs.publish-order.md)。

## 发布前清单（当前状态）

- [x] `sptorch` 已补齐发布元数据（repository/homepage/documentation/readme/keywords/categories）。
- [x] 非发布目标已标注 `publish = false`：`sptorch-cli-train`、`sptorch-cli-train-gpu`、`sptorch-mock-npu`、`sptorch-studio`、`products/*`。
- [x] `cargo package -p sptorch-core-tensor --no-verify` 可通过（作为链路基线）。
- [ ] `cargo package -p sptorch --no-verify` 仍阻塞：依赖的内部 crate（如 `sptorch-core-autograd`）尚未发布到 registry。

## 测试覆盖

| crate | 测试数 | 说明 |
|-------|--------|------|
| sptorch-core-tensor | 8 | F16/BF16 转换、DType 接口 |
| sptorch-core-ops | 56 | 所有算子前向+反向+grad check |
| sptorch-nn | 23 | Linear/LoRA/Embedding/LayerNorm/MHA/Transformer/GPT |
| sptorch-optim | 10 | SGD/AdamW/clip/CosineScheduler |
| sptorch-data | 10 | CharTokenizer/BPE/Dataset/DataLoader |
| sptorch-runtime-cuda | 12 | GPU 算子 + CPU 对比 |
| sptorch-serialize | 7 | checkpoint roundtrip + safetensors |
| sptorch-core-autograd | 3 | 基础 autograd |
| sptorch-hal | 15 | Backend + KernelProvider 20 个算子 |
| sptorch-hal-ffi | 10 | FFI 全链路集成测试 |
| sptorch-distributed | 7 | AllReduce + gRPC + 多步训练 |
| sptorch-live-evolution | 16 | 双缓冲/EWC/监控/端到端 |
| sptorch-text2sql | 11 | Schema/RAG/SQL 约束/SQLite |
| **合计** | **211** | **全部通过** |

## 开发路线

详见 [roadmap.md](roadmap.md)（本仓库唯一事实源 / Single Source of Truth）。

- 阶段口径以 roadmap 勾选状态与实验记录为准
- “已完成”与“进行中”状态不再由 README 单独维护
- P8（Tang 9k 硬件点亮）为当前最高优先级

## Studio 开发进度（同步）

- 已新增 `crates/versioning`：版本化张量协议（`VersionedStorage` / `UpdatePolicy` / `FenceState` / `EvolutionMetrics`）
- 已新增 `studio/`（Tauri 2 + React）：
  - 后端 `engine_bridge`：`get_engine_status`、`start_evolution_stream`、`trigger_atomic_swap`
  - 指标流已改为直接订阅 `live-evolution` 训练事件总线（`Metrics/VersionCommit/Fence/HardwareState`），不再使用本地定时模拟推送
  - v1.1 预埋适配：`hal-ffi` 新增可选遥测 ABI `sptorch_query_runtime`，Studio 启流时优先读取硬件队列深度/在线状态快照
  - 事件流稳态增强：`start_evolution_stream` 幂等启动，避免重复订阅导致的 commit/fence 重复推送
  - 前端核心面板：Versioned Dashboard、Memory Snapshot、Autograd Version Graph、Hardware Fence Panel
  - 控制中枢可用性增强：Version Timeline 按 `version_id` 去重、Memory 面板自动选中首个张量、Autograd 图增加空态提示
- 已新增最小测试基线：
  - 前端：Vitest + RTL（Dashboard / Memory / Fence 组件）
  - 前端：App 事件流集成测试（含 Fence Error 恢复分支）与 `api.ts` 桥接测试
  - Rust：`engine_bridge` 集成测试分层到 `studio/src-tauri/tests`
- CI 已接入前端测试：GitHub Actions 增加 `frontend-test` job（`npm ci` + `npm run test`）
- 当前状态：v1 指标流来自真实 `live-evolution` 训练过程；Fence/队列深度仍为可观测模拟信号，真实 `hal-ffi` ABI 扩展列入 v1.1
- 产品解耦进展：`text2sql` 已收敛为产品服务协议层；神经训练/推理实现迁移到 `cli-text2sql` 产品运行时，框架保持干净

## 硬件环境

- OS: Windows 10
- GPU: NVIDIA GeForce RTX 3050 (6GB, Compute 8.6)
- CUDA: 12.4
- Rust: 1.95.0 (stable, 2021 edition)
