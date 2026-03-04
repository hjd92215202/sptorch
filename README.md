# pytorch (Rust mini)

一个用 Rust 逐步实现「类 PyTorch」并最终训练 MiniGPT 的学习型工程仓库。

当前仓库处于早期阶段，已包含：
- `core-tensor`：`Tensor` 基础数据结构与梯度字段。
- `core-autograd`：以 `add` 为例的最小自动求导链路。

## Workspace 结构

```text
crates/
  core-tensor/
  core-autograd/
docs/
  rust_pytorch_minigpt_plan_zh.md
  phase_a_two_week_sprint.md
```

## 快速开始

### 1) 运行测试

```bash
cargo test --workspace
```

### 2) 当前状态

- 已有 `add` 前向 + 反向传播最小闭环。
- 还未覆盖阶段 A 所需全部算子与 `nn/optim` 最小模块。

## 阶段 A（两周冲刺版）

已整理为可执行任务清单：
- 详见：`docs/phase_a_two_week_sprint.md`
- 目标：2 周内完成“最小训练闭环”可演示版本。

## 近期里程碑（建议）

1. Week 1：补齐核心算子 + autograd + 测试基线。
2. Week 2：打通 `nn::Linear` + `optim::SGD/AdamW` + tiny LM 训练脚本（loss 下降）。

## 开发建议

- 每个新算子至少具备：
  - forward 单测
  - backward 单测
  - 简单数值梯度检查（有限差分）
- 提交前执行：

```bash
cargo test --workspace
```
