# 阶段 A 两周冲刺任务拆分清单（可执行版）

> 目标：在 2 周内，从当前原型推进到“最小训练闭环可演示”。

## 0. 冲刺目标（Definition of Done）

在 `CPU + f32` 前提下完成：
1. 基础张量算子（含反向）支持 tiny language model 训练。
2. 最小 `nn` 与 `optim` 可用（`Linear` + `Embedding` + `SGD/AdamW`）。
3. 跑通一个 tiny 训练脚本，`loss` 显著下降。
4. 有最小测试体系（单测 + 梯度检查）。

---

## 1. 拆分原则

- 以“可合并的小 PR”为单位推进（每 PR 半天~1 天）。
- 先正确性，再性能；先 API 稳定，再扩展算子。
- 每新增算子同时补 forward/backward 测试。

---

## 2. Week 1：核心算子与 autograd 完整化

### Day 1：工程基线与约束

- [ ] 增加基础检查命令到 README（已完成）。
- [ ] 在 `core-tensor` 引入 shape 校验工具函数（统一错误信息）。
- [ ] 修复/覆盖 stride 计算边界（特别是 0-d 或 1-d 情况）。
- [ ] 补充 `Tensor` 调试输出字段（shape/strides/requires_grad）。

**验收**：
- `cargo test --workspace` 通过。
- 新增 shape/stride 边界测试。

### Day 2：`mul` 与 `sum/mean`

- [ ] 实现 `mul(a, b)` 前向与反向。
- [ ] 实现 `sum`（先全量归约），`mean`。
- [ ] 处理梯度回传时的形状恢复（最小版，无广播）。

**验收**：
- 覆盖 `z=(x*y).sum()` 的梯度单测。

### Day 3：`matmul`（2D first）

- [ ] 实现 2D `matmul` 前向（`[m,k]@[k,n]`）。
- [ ] 实现反向（`dA=dY*B^T`, `dB=A^T*dY`）。
- [ ] 增加维度不匹配错误测试。

**验收**：
- `matmul` forward/backward 单测通过。

### Day 4：视图与变换（最小版）

- [ ] 实现 `reshape`（总元素数不变校验）。
- [ ] 实现 `transpose`（2D 版本即可）。
- [ ] 明确文档：视图语义先简化（必要时先复制实现）。

**验收**：
- `reshape/transpose` 基础行为 + 反向链路测试。

### Day 5：梯度检查工具

- [ ] 新增简易数值梯度检查工具（有限差分）。
- [ ] 至少覆盖 `add/mul/matmul` 三类算子。
- [ ] 输出误差阈值与失败信息。

**验收**：
- 梯度检查在 CI 本地可稳定通过。

---

## 3. Week 2：nn/optim 与 tiny 训练闭环

### Day 6：`nn::Module` 与参数注册最小版

- [ ] 新建 `crates/nn`（或同名模块）。
- [ ] 定义 `Module` trait：`forward`、`parameters`。
- [ ] 实现 `Linear`（权重、偏置可训练）。

**验收**：
- `Linear` 前向 shape 与参数数量测试通过。

### Day 7：`Embedding` 与简单初始化

- [ ] 实现 `Embedding(vocab_size, d_model)`。
- [ ] 增加初始化函数（如 uniform/xavier minimal）。
- [ ] 为 Embedding 增加梯度回传测试（索引位置更新）。

**验收**：
- Embedding 前向与反向可用。

### Day 8：`optim::SGD` 与 `zero_grad`

- [ ] 新建 `crates/optim`。
- [ ] 实现 `SGD::step()` 与 `zero_grad()`。
- [ ] 提供统一 `Parameter` 访问方式。

**验收**：
- 单参数二次函数最优化例子可下降。

### Day 9：`AdamW` 最小实现 + 梯度裁剪

- [ ] 实现 AdamW（`m/v`、偏置修正、weight decay）。
- [ ] 实现 `clip_grad_norm`（global norm）。
- [ ] 加入 NaN/Inf 训练守卫。

**验收**：
- 小模型训练 100~300 step，loss 稳定下降。

### Day 10：端到端脚本 + 文档收尾

- [ ] 新建 tiny LM 训练示例（字符级）。
- [ ] 输出关键日志：step/loss/lr/tokens_s。
- [ ] README 补“如何运行训练脚本”和预期结果。
- [ ] 整理已知限制与下一阶段入口。

**验收**：
- 一条命令可复现 tiny 训练，看到 loss 下降曲线。

---

## 4. 推荐 PR 切分（示例）

1. PR-01：shape/stride 校验与边界测试。
2. PR-02：`mul/sum/mean` + 测试。
3. PR-03：`matmul` + backward + 测试。
4. PR-04：`reshape/transpose` + 测试。
5. PR-05：梯度检查工具与用例。
6. PR-06：`nn::Linear` + 参数注册。
7. PR-07：`Embedding` + 初始化。
8. PR-08：`optim::SGD` + `zero_grad`。
9. PR-09：`AdamW` + `clip_grad_norm` + 训练守卫。
10. PR-10：tiny LM 训练 demo + 文档。

---

## 5. 风险与缓解

- 风险：算子形状规则复杂导致反向错误。
  - 缓解：先禁广播，后续增量支持；每个算子先写梯度检查。
- 风险：并发锁设计影响性能。
  - 缓解：阶段 A 仅保证正确性，阶段 C 再做内核优化。
- 风险：端到端训练不收敛。
  - 缓解：先用最小数据/模型，固定随机种子与学习率范围。

---

## 6. 冲刺期间每日检查清单

- [ ] `cargo test --workspace`
- [ ] 新增代码是否含单测
- [ ] 新增算子是否含 backward
- [ ] 是否补充文档/注释
- [ ] 是否可拆成独立 PR
