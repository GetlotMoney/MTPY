# DVSR 后续待办清单（路线图）

**当前进度**：
- ✅ Baseline 72.24
- ✅ G1 (Topo+KL) 72.61
- ✅ G2 (G1+MSDN) 72.41 → 续训 72.79
- ✅ G3 (G2+Conditional Text 修复版) 续训 **72.95 ⭐ 当前最佳**
- ⏳ H2 (DMP 池化级动态门控) — 当前在跑

---

## 📋 优先级路线（按推荐顺序）

### 🥇 Phase 1 — 三层动态门控完整跑完（本周内）

#### **H2-fixed** ⏳ 当前
- 配置: `pool_method=dmp, pool_dynamic=fixed, pool_lambda_fixed=0.5`
- 验证: mean+LaSt 等权融合是否有效
- 跑完决策:
  - H ≥ 73.0 → 进 H2-mlp
  - 持平 72.7-73 → 跳 H2-mlp 让模型自学
  - 跌 < 72.5 → 检查 dmp 代码

#### **H2-mlp** 🚧 待跑
- 改 yaml: `pool_dynamic: fixed → mlp`
- 让 sigmoid(MLP_pool(cls)) 自学 λ
- 关键看点: epoch 末 sigmoid(bias) 收敛值（>0.5 表示 LaSt 主导，<0.5 表示 mean 主导）

#### **H3** 🚧 待跑
- 在 H2-mlp 基础加 `weight_s2v_mode: mlp`（branch 级动态）
- 训练日志已实现 `Weight w (mlp): bias-baseline=...`

#### **H4** 🚧 待跑
- 在 H2-mlp 基础加 `gating_dynamic: mlp`（score 级动态）
- 训练日志已实现 `Gating α (mlp): bias-baseline=...`

#### **H5 — 三层全开** 🚧 待跑
- pool + branch + score 三层 mlp 同时开
- 论文核心实验，期望 H 73~74

---

### 🥈 Phase 2 — 文本来源对照（30 分钟）

#### **GPT-5.5 文本替换 Claude** 🚧 待跑
- 单变量改文本 source（其他配置保持 G3 续训最佳）
- 需要先 encode GPT-5.5 文本到 .pt 缓存（如果还是 raw text）
- 期望: H 72.79 ± 0.5 看哪个 LLM 更适合

---

### 🥉 Phase 3 — 论文必备 ablation（20 分钟）

跑完 H 系列后，**论文 Table 1 必备的单变量消融**：

| 实验 | 配置 | 用途 |
|------|------|------|
| **A0** | 全关辅助 | baseline 锚点 |
| **A1** | 仅 Topo | Topology Pearson 单独贡献 |
| **A2** | 仅 KL（cons 动态）| Consistency 单独贡献 |
| **A3** | 仅 MSDN | MSDN 互蒸馏单独贡献 |
| **A4** | 仅 G3（Conditional Text）| CoCoOp-style 单独贡献 |
| **A5** | 仅 H2（DMP）| Pool dynamic 单独贡献 |

每组 epoch=20，约 5 分钟一组。

---

### 🌟 Phase 4 — CGC 模块（论文标题级，1-2 天）

#### **C1** 实施 base + cond + cf 三分支 forward
- frozen base: `l⁰_c = cos(f_x, t_c)·τ` 不参与训练
- cond branch: 现有 G3
- cf branch: π̃ 用 batch shuffle / zero / noise

#### **C2** 加 class-wise gate g_c(x)
- gate input: `[f_x; t_c; f_x ⊙ t_c]`
- 替代当前 G3 的 broadcast 200 类

#### **C3** 加 L_gain
- `Δ = l^cond - l^cf`
- `L_gain = CE(Δ, y)`

#### **C4** 加 L_bias seen-unseen margin
- `m = max_seen(l^cond) - max_unseen(l^cond)`
- `L_bias = max(0, m - δ)`

#### **C5** 反事实采样策略对比
- shuffle vs zero vs noise 哪种最好

---

### 📋 Phase 5 — 12 项待补消融（论文写作前必扫）

| 项 | 待扫 | 优先级 |
|----|------|------|
| LaSt-ViT k-grid | k=2/4/16/32 | 🟡 中 |
| LaSt-ViT σ-grid | σ=5/10/20/50 | 🟡 中 |
| **P1 单独** | class_attention pool 不联合 alpha/w | 🟢 高 |
| Calibration λ-grid | 0.0005/0.002/0.005 | 🟡 中 |
| Random Holiday | 多组超参 | 🟢 高 |
| Attention Pool | hidden=64/128/256 + 多头 | 🟢 高 |
| L2-SP 单独 | 15-20 epoch 单独验证 | 🟢 高 |
| Cosine-only v3 真版本 | forward 通 backward 断 | 🟡 低 |
| CIG 22 升级 | MBG/TCG/双层 | 🟡 中 |
| Multi LLM 文本权重 | learnable cluster | 🟡 中 |
| α-grid 加权融合 | 温度/排序方法 | 🟡 中 |
| gate_net MLP vs alpha_net | 直接对比 | 🟡 低 |

---

### 📋 Phase 6 — 跨数据集 + 多 seed（论文写作前最后一步）

#### AWA2（50 类粗粒度动物）
- 用 H5 全开最佳配置
- `cuda:1` 跑（避免 CUB 占用）

#### SUN（722 类场景）
- 同上

#### 多 seed 平均
- seed = 5, 42, 2024 各跑 H5 全开
- 算 mean ± std

---

## 📊 已确认的"真·结构性失败"（不再做）

| 项 | H | 原因 |
|----|------|------|
| Cosine-only v1/v2 | 43-45 | 流形错位 |
| F5 全开 (P1+α/w) | 68 | P1 per-class pool 拖累 |
| 50ep 三合一全家桶 | 34→18 | Loss 量级失控 |

---

## 🎯 论文叙事链（已成型）

```
Part 1: Foundation
  Adapter + FAE + Bidirectional Transformer

Part 2: Loss-level Innovation
  - G1 Topology Pearson (角度拓扑保持)
  - G1 Consistency KL (动态权重 ★ 创新点)
  - G2 MSDN Mutual Distillation
  - lambda 动态权重 (跟 G1 cons 配套)

Part 3: Forward-level Innovation
  - G3 Conditional Text Adapter (仅 seen + normalize 限幅)

Part 4: Hierarchical Dynamic Gating (H 系列, 标题级)
  - Pool-level (H2 DMP)
  - Branch-level (H3 w_net)
  - Score-level (H4 alpha_net)

Part 5: Counterfactual Calibration (CGC, 标题级)
  - 反事实增益 + class-wise gate + seen-unseen balance

Part 6: Negative Results
  - Cosine-only 系列 (v1-v8)
  - LaSt-ViT 单独替代 (5 次失败)
  - F5 P1 per-class 池化

Part 7: Cross-Dataset Validation
  - AWA2, SUN
  - Multi-seed
```

---

## 🚦 下一步直接行动

1. **当前 H2-fixed 跑完** → 看 H 数值
2. **改 yaml 一行启 H2-mlp** → 5 分钟跑完
3. **再改一行启 H3** → 加 branch dynamic
4. ...

## ⏱️ 预估时间

| Phase | 预估 |
|-------|------|
| Phase 1 (H2-H5) | 4 次 × 5 分钟 = 20 分钟 |
| Phase 2 (GPT-5.5) | 30 分钟（含编码） |
| Phase 3 (A1-A6 ablation) | 30 分钟 |
| Phase 4 (CGC C1-C5) | 1-2 天 |
| Phase 5 (12 项消融) | 1 天 |
| Phase 6 (跨数据集 + seed) | 1 天 |
| **总计** | **3-5 天可全部完成** |
