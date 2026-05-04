# SPTorch Framework/Product 发布与版本策略（v1）

本文档定义“框架是框架、产品是产品”的发布边界与版本策略，供后续拆分为两个仓库时直接复用。

## 1. 工程边界

- 框架工程：仓库根 `Cargo.toml` workspace（`crates/*` + `studio/src-tauri`）。
- 产品工程：`products/Cargo.toml` workspace（`text2sql` + `cli-text2sql`）。
- 产品通过 `sptorch` 门面 crate 引入框架能力，不直接依赖内部子 crate。

## 2. 稳定 API 约束

对外稳定 API 仅通过 `crates/sptorch` 暴露，首选命名空间：

- `sptorch::v1::nn`
- `sptorch::v1::optim`
- `sptorch::v1::ops`
- `sptorch::v1::checkpoint`
- `sptorch::v1::prelude`

非 `v1` 命名空间默认视为内部实现细节，不承诺稳定性。

## 3. 版本策略（SemVer）

- `sptorch` 使用 SemVer。
- 在 `v1` 命名空间内：
  - 兼容增强使用 MINOR（`0.x` 阶段可按团队策略先行，但发布后建议严格化）。
  - 破坏性变更必须通过新命名空间（如 `v2`）并保留 `v1` 过渡窗口。

## 4. 发布策略

框架发布顺序（未来上 crates.io）：

1. 发布底层依赖 crate（如 `core-tensor/core-ops/nn/optim/...`，可按需设 `publish = false`）。
2. 发布门面 crate `sptorch`。
3. 产品仓库将依赖切换为：`sptorch = "<version>"`。

过渡期（未上 crates.io）建议：

```toml
sptorch = { git = "<framework-repo>", rev = "<commit>" }
```

## 5. CI 建议

- 框架 CI：`cargo test --workspace`（根目录）。
- 产品 CI：`cargo test --manifest-path products/Cargo.toml --workspace`。
- 二者保持独立 job，避免耦合发布节奏。

## 6. 仓库拆分计划（后续）

- 仓库 A：`sptorch-framework`（保留 `crates/*`, `studio/`）。
- 仓库 B：`sptorch-text2sql`（保留 `products/*`）。
- 仓库 B 仅依赖仓库 A 的 `sptorch` 门面 crate。
