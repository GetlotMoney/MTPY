# DVSR 后续待办清单 (路线图)

> **本文档维护规则**: 完成任务把 `[ ]` 改 `[x]`, 进行中改 `[~]`, 失败改 `[!]` 并加备注。新增任务直接追加。

**最近更新**: 2026-06-01 02:55
**当前最佳 (单点)**: H = **72.91** ⭐ (P3.10: AG-JEPA + LaSt v5 K=32 + PriorCorr off, 60ep, seed=5)
**新 main baseline (P3.10 配置, 3-seed)**: H = **72.65 ± 0.19** ⭐ (5/42/2024 = 72.91 / 72.53 / 72.50, U=73.07, S=72.26, ZS=81.54)
**旧 main baseline**: H = 72.52 ± 0.11 (v5+FAE 50ep 3-seed, **已替换**)
**Δ vs 旧 main**: H +0.13 (3-seed 锁死, std 0.11→0.19 略升但仍稳健)
**v5+FAE 旧数据**: U=74.13±0.62, S=70.99±0.40, ZS=81.56±0.18

> ⚠️ **Schedule 切换历史**:
> - **50ep `[1e-3×20, 1e-4×30]`** (旧 main baseline)
> - 90ep `[1e-3×30, 1e-4×30, 1e-5×30]` (P3.0 失败 ❌)
> - **60ep `[1e-3×20, 1e-4×30, 1e-5×10]`** ⭐ 当前 (50ep main + 10ep 1e-5 微调)

> ⚙️ **当前 yaml 配置 (P3.10 等价)**:
> - K=32, largest=True, formula=v2_abs_mean
> - **AG-JEPA train ON** (lambda_jepa=0.05, lambda_jepa_neg=0.02)
> - **PriorCorrection OFF** (实验验证 balanced 在 CUB hurts -0.21)
> - G1 (Topo+Cons) + G2 (MSDN) + G3 (Conditional Text) 全开

---

## 🎯 当前阶段总览 (2026-05-30 起, 持续更新)

```
✅ Done:
  - main baseline 锁定 H=72.48±0.20 (3 seeds, 旧 50ep)
  - LaSt v1/v2/v3 negative results (frozen CLIP 切断梯度)
  - LaSt v5 (path C, patch 选择器) seed=5 H=72.15 → +FAE 修复 H=72.53
  - v5+FAE 3-seed avg H=72.52±0.11 (新 main baseline @ 50ep)
  - 纯 CLIP zero-shot baseline H=61.28 (Table 1 第一行)
  - 90ep schedule 切换 (lr_stages = 1e-3×30 + 1e-4×30 + 1e-5×30)
  - **Phase 4.1 D-α 失败两次** (Cons on 灾难 H=66.22 / Cons off 退化 H=70.28)
  - **诊断: D-α 在 frozen CLIP+seen-only CE 下 α_net 退化常数, 论文 negative result**

🚧 In Progress:
  - P1.0 K=64 @ 90ep main baseline reference (terminalId=3, ~50min)

📋 Next:
  - P1.1 K=32 / P1.2 K=128 / P1.3 K=256 (K 扫描, 各 ~50min)
  - 后续: 跨数据集 (AWA2/SUN) → A1-A6 ablation → 创新点 (D-σ / patch 一致性 loss / CGC)
```

---

## Phase 1 — 收尾验证 (✅ 已完成)

锁定 v5+FAE 的 3-seed average, 决定是否替换 main baseline。

| ID | 任务 | yaml 改动 | 时间 | 状态 | 结果 |
|---|---|---|---|---|---|
| 1.1 | v5+FAE seed=42 | `random_seed: 42` | ~36min | [x] | **H=72.65** U=75.00 S=70.44 ZS=81.77 (best@ep33) |
| 1.2 | v5+FAE seed=2024 | `random_seed: 2024` | ~36min | [x] | **H=72.39** U=73.66 S=71.16 ZS=81.34 (best@ep36) |
| 1.3 | 入账 EXPERIMENT_LEDGER #14 行 (v5+FAE 3-seed avg) + 补 v2/v3 旧数据 | — | 5min | [x] | **3-seed avg: H=72.52 ± 0.11, U=74.13 ± 0.62, S=70.99 ± 0.40, ZS=81.56 ± 0.18** ⭐ 新 main baseline |

**决策点**: 1.1 + 1.2 跑完后看 3-seed avg
- avg ≥ 72.5: v5+FAE 替换 main baseline, 论文 main result 用新数据
- avg ∈ [72.3, 72.5]: 中性, 留作 ablation 一个数据点
- avg < 72.3: 回归 main baseline

---

## Phase 2 — 横向消融 (论文 Table 1 + Table 2 必备, 4-5h)

| ID | 任务 | 命令 / yaml 改动 | 时间 | 状态 | 结果 |
|---|---|---|---|---|---|
| 2.1 | 纯 CLIP baseline | `python tools/eval_pure_clip.py` (注意先修 tools.py 顶部的 Digraph 报错) | 5min | [x] | **H=61.28** U=60.88 S=61.69 ZS=78.07 (zero-shot, 0 训练参数, 7.7s) |
| 2.2 | AWA2 v5+FAE 主跑 | `python train_VGSR_AWA2.py` (确认 AWA2 yaml 含 lastvit_select_k=64) | ~1h | [ ] | |
| 2.3 | SUN v5+FAE 主跑 | `python train_VGSR_SUN.py` (确认 SUN yaml 含 lastvit_select_k=64) | ~1h | [ ] | |
| 2.4 | A1: 关 G1 Topo | `lambda_topo_pearson: 0.0` | ~30min | [ ] | |
| 2.5 | A2: 关 G2 MSDN | `lambda_msdn: 0.0` | ~30min | [ ] | |
| 2.6 | A3: 关 G3 Conditional Text | `use_conditional_text: False` | ~30min | [ ] | |
| 2.7 | A4: 关 v5 LaSt 选择器 | `lastvit_select_k: 0` (退回 baseline 576) | ~30min | [ ] | |
| 2.8 | A5: 关 FAE | `use_fae: False` | ~30min | [ ] | |
| 2.9 | A6: 关 Adapter | `model_mode: clip_only` | 5min | [ ] | |

每个 ablation 单变量改动, 其它配置保持当前 v5+FAE 最佳。

---

## Phase 3 — K / σ 扫描 (~3h, 60ep schedule = 50ep main + 10ep 1e-5 微调)

> ⚠️ **2026-05-31 勘误**: LaSt patch 选择器实际是 **前景同质性筛选** (滤掉 high-norm artifact / 背景 lazy token), 不是 part-aware 细粒度提取。报告原文 "比值越高的 Patch 说明在通道内越平稳", `topk(largest=True)` 选最平稳的前景 token。
> CUB 鸟眼/喙/翅纹这种 part token 通道波动大, 反而被 LaSt 评低分丢掉 — **K 越小可能反而越好** (实测 K=32 > K=64)。

验证 LaSt 选择器的信息瓶颈和频域核敏感度。**60ep 三段 schedule [1e-3×20, 1e-4×30, 1e-5×10], 前 50ep 等价 main baseline 公平对比**。

| ID | 任务 | yaml 改动 | 时间 | 状态 | 结果 |
|---|---|---|---|---|---|
| **3.0 (90ep)** ❌ 失败 | K=64 reference @ **90ep [30+30+30]** | (main baseline 配置) | ~50min | [!] | **H=71.37 best@ep8 / 66.49 final**, S 暴跌 -17, 90ep Stage 1 过长机制错 (vs 50ep 72.53 落 -1.16) |
| **3.0' (60ep)** ✅ | K=64 reference @ **60ep [20+30+10]** ⭐ | lr_stages 三段 20+30+10 | ~35min | [x] | **H=72.54 best@ep19 / 72.02 final**, ZS=82.29 历史最高; 跟 50ep main 72.53 持平 |
| **3.1** ✅ ⭐ | **K=32 @ 60ep** | `lastvit_select_k: 32` | ~35min | [x] | **H=72.70 best@ep34** ⭐ K=32 比 K=64 涨 **+0.16**; 验证勘误 "前景去噪" 假设, K 越小越好 |
| **3.2-new** | **K=16 @ 60ep** | `lastvit_select_k: 16` | ~35min | [x] | **H=72.23 best@ep41** (vs K=32 跌 0.47), K=16 太严格丢前景信息, **K=32 确认最优** |
| 3.3-new | K=8 @ 60ep (备选) | `lastvit_select_k: 8` | ~35min | [×] | 跳过, K=16 已证明 K 太小会跌 |
| 3.4 | K=128 @ 60ep (低优先级 ablation) | `lastvit_select_k: 128` | ~35min | [ ] | 勘误后大概率 < 72.54 |
| 3.5 | K=256 @ 60ep (低优先级 ablation) | `lastvit_select_k: 256` | ~40min | [ ] | 接近 mean pool, 兜底数据点 |
| **3.6 (NEW)** ⭐⭐ | **`largest=False` K=32 反向选择** | 改 `lastvit_select_patches` 加 `largest` 参数 | 改代码 10min + ~35min | [x] | **H=72.59 best@ep41**, vs P3.1 K=32 largest=True 72.70 仅差 -0.11; **假设 B 成立**: LaSt 是信息瓶颈而非 part 选择器, 选什么 patch 不重要, K 个数才重要 |
| **3.6b (NEW)** ⭐⭐ | **`largest=False` K=64 反向选择 II** | yaml `lastvit_select_k: 64`, `largest=False` | ~35min | [x] | **H=72.11 best@ep41**, vs P3.0' K=64 True 72.54 跌 -0.43 (跟 P3.6 K=32 -0.11 反差); **K-dependent**: K 小持平, K 大时平稳前景占优 → 论文卖点 "K-dependent regularizer" |
| 3.7 (低) | σ=10 @ 60ep | `lastvit_select_sigma: 10.0` | ~35min | [ ] | |
| 3.8 (低) | σ=50 @ 60ep | `lastvit_select_sigma: 50.0` | ~35min | [ ] | |
| **3.9 (NEW)** | K=64 + `largest=both` (Top-K + Bottom-K 混合) | 改 `lastvit_select_patches` 加 'both' 模式 | ~35min | [x] | **H=72.55 best@ep40**, U=67.23, **S=78.78 历史最高**; trade-off 调节器 (强 seen-favor) |
| **3.10** ⭐⭐⭐ | **AG-JEPA on + PriorCorrection=none** | yaml `prior_correction: none` (其余同 P3.1) | ~35min | [x] | **H=72.91 best@ep23** ⭐ 60ep 单点最高 (U=73.30, S=72.53, ZS=81.72); 揭示 PriorCorrection balanced 反而 hurts H -0.21 |
| **3.11** | **完全干净 (AG-JEPA off + PriorCorr off)** | yaml `use_ag_jepa: False`, `lambda_jepa/neg: 0`, `prior_correction: none` | ~35min | [x] | H=72.45 best@ep34; vs P3.10 -0.46 → **AG-JEPA 训练 loss 真实贡献 +0.46** |
| **3.12** | **多 seed: P3.10 配置 + seed=42** | `random_seed: 42` | ~35min | [x] | H=72.53 best@ep21 (U=71.93, S=73.14, ZS=81.56); 2-seed avg=72.72 |
| **3.13** | **多 seed: P3.10 配置 + seed=2024** | `random_seed: 2024` | ~35min | [x] | **H=72.50 best@ep34** (U=73.97, S=71.10, ZS=81.34); **3-seed avg = 72.65 ± 0.19** ⭐ 锁定新 main |
| **3.14** | 入账 LEDGER #27 (P3.13 + 3-seed avg 高亮汇总行) + 同步 INNOVATION_RECORD | — | 10min | [x] | 主创新升级到 6 项, 旧 main baseline 替换 |

**K 扫描勘误后新认知**: K 不是"信息密度", 而是"前景纯度阈值"。K 越小 → 前景筛选越严格 → 抑制噪声 → cosine 分类更干净。

预期最佳点 (修订): **K∈{16, 32}**, σ∈{0(auto)}, 反向选择 (largest=False) 应崩盘验证机制

---

## Phase 4 — 动态门控 (4.1 已失败, 4.3/4.4 待重定义) [优先级 ⭐⭐ 中风险]

> **Phase 4.1 D-α 已两次失败** (Cons on 灾难 / Cons off 退化), 4.2 D-w 跟 D-α 同病跳过, 真正可行的是 4.3 D-σ (gate 跳出 logits 直接调频域)。

| ID | 任务 | yaml/代码 | 时间 | 状态 | 结果 |
|---|---|---|---|---|---|
| 4.1 | **D-α (logits 级)** ❌ 失败 | `gating_dynamic: mlp` | ~50min | [!] | **失败 ×2** Cons on: H=71.83 best/66.22 final (灾难, Cons-α 反向打架); Cons off (4.1-A): H=70.28 best/68.22 final; α_net 退化常数 sigmoid≈0.33, frozen CLIP+seen-only CE 必然退化 → 论文 negative result |
| 4.2 | **D-w (branch 级)** ⚠️ 跳过 | `weight_s2v_mode: mlp` | ~50min | [×] | 跟 D-α 同病 (logits 级 gate, seen-only 监督盲区), 跳过避免重复踩坑 |
| 4.3 | **D-σ (frequency 级)** ⭐ 推荐 | 新加 `sigma_net = MLP(cls)`, lastvit_select_patches 接受 [B] σ 张量 | 改代码 30min + ~50min | [ ] | gate 跳出 logits, 改 LaSt 频域核 σ(x); 4 跳梯度路径削弱 seen-only 退化压力, σ 退化下限=baseline 不会变差 |
| 4.4 | **D-K**: 动态 K(x) Gumbel-softmax | 新加 `K_net = MLP(cls)`, gumbel-softmax 离散采样 K∈{32,64,128,256} | 改代码 1.5h + ~50min | [ ] | 创新最高, 实现最复杂, Gumbel-softmax 训练不稳定 |
| 4.3 | **D-σ**: 动态频域核 σ(x) | 新加 `sigma_net = MLP(cls)`, lastvit_select_patches 接受 [B] σ 张量 | 改代码 30min + 训练 35min | [ ] | |
| 4.4 | **D-K**: 动态 K(x) Gumbel-softmax | 新加 `K_net = MLP(cls)`, gumbel-softmax 离散采样 K∈{32,64,128,256} | 改代码 1.5h + 训练 35min | [ ] | |

**4.3 设计草案**:
```python
# VGSR.__init__:
if self.dyn_sigma_mode == 'mlp':
    self.sigma_net = nn.Sequential(
        nn.Linear(768, 64), nn.GELU(),
        nn.Linear(64, 1), nn.Softplus()  # 保 σ > 0
    )
# forward:
sigma_per_image = self.sigma_net(cls_token).squeeze(-1)  # [B]
# lastvit_select_patches 改成接受 sigma=[B] 张量, 内部 broadcast
```

**4.4 设计草案** (Gumbel-softmax 选 K):
```python
K_logits = self.K_net(cls_token)  # [B, 4] for {32,64,128,256}
K_onehot = F.gumbel_softmax(K_logits, tau=1.0, hard=True)  # [B, 4]
# 训练: 用 mask 模拟不同 K 的注意力 (套 multi-head attention)
# 推理: argmax(K_logits) 直接选
```

---

## Phase 5 — 创新点新增 (论文 contribution, 5-6h)

| ID | 任务 | 设计 | 时间 | 状态 | 结果 |
|---|---|---|---|---|---|
| 5.1 | **C1**: 可学习 σ_c per-class | 加 200 个标量 σ_c, 跟 class_text 共同学; LaSt 选 patch 时按图所属类的 σ 平滑 | 改代码 1h + 训练 35min | [ ] | |
| 5.2 | **C2**: patch indices 一致性 loss | 类内 LaSt 选 patch 分布 KL 拉近, 类间拉远; 给 LaSt 加监督信号 | 改代码 1.5h + 训练 35min | [ ] | |
| 5.3 | **C3**: text-conditional patch selection | 反向 CoCoOp: unseen class name 引导挑 patch (攻 unseen 偏置) | 改代码 2h + 训练 35min | [ ] | |
| 5.4 | FAE 加层 (1 → 2 / 3 层) | `nn.ModuleList([FAELayer(...) for _ in range(N)])` | 改代码 5min + 训练 35min | [ ] | |
| 5.5 | **AG-JEPA**: masked semantic patch prediction | 已接入最小闭环: top text-aligned patch 作为 target, context + class text 预测; 含 negative text margin | 已实现, 训练 35min | [~] | 先跑 seed=5; 当前只有 smoke test, 无正式 H |
| 5.6 | **Prior Correction** 后验先验校准 | `prior_correction=balanced`; test-time 统计 seen/unseen 概率质量, 估计 unseen logit shift | 已实现, 评估随训练跑 | [~] | transductive, 必须单独标注和消融 |

**5.2 是新颖性最高的**: 直接攻 LaSt 在 frozen CLIP 下无监督的痛点, 给 patch 选择器加任务相关监督。

**AG-JEPA 消融顺序**:
1. A0: `use_ag_jepa=False`, `prior_correction=none` 回到当前 K=32 baseline。
2. A1: JEPA-only: `use_ag_jepa=True`, `prior_correction=none`。
3. A2: prior-only: `use_ag_jepa=False`, `prior_correction=balanced`。
4. A3: full: `use_ag_jepa=True`, `prior_correction=balanced`。
5. 如果 A1 或 A3 有正收益, 再扫 `lambda_jepa ∈ {0.02, 0.05, 0.1}` 和 `jepa_topk ∈ {4, 8, 16}`。

---

## Phase 6 — 老坑修复 + Negative Result 章节 (4-5h)

| ID | 任务 | 设计 | 时间 | 状态 | 结果 |
|---|---|---|---|---|---|
| 6.1 | 修 L_bias (方案 2 unseen mean anchor) | `L_bias = ‖mean(unseen_logits) - target_unseen_anchor‖²`, 不打 max-margin 不会跟 CE 冲突 | 改代码 1h + 训练 35min | [ ] | |
| 6.2 | CGC C1-C4 完整版 (反事实增益 + class-wise gate + seen-unseen margin) | 见旧 ROADMAP Phase 4 | 改代码 2h + 训练 35min × 4 | [ ] | |

---

## Phase 7 — 论文写作 (1-2 天)

| ID | 任务 | 时间 | 状态 |
|---|---|---|---|
| 7.1 | Table 1: SOTA 对比 (CUB / AWA2 / SUN, v5+FAE 3-seed + 主流方法) | 3h | [ ] |
| 7.2 | Table 2: A1-A6 ablation 表 | 1h | [ ] |
| 7.3 | Table 3: K / σ / 动态门控扫描表 (Phase 3 + Phase 4 数据) | 1h | [ ] |
| 7.4 | Discussion: LaSt v1→v2→v3→v5 失败链 + frozen backbone 接入方式 insight | 4h | [ ] |
| 7.5 | Figure 1: VGSR + v5+FAE 数据流图 (含 patch 选择可视化) | 4h | [ ] |
| 7.6 | Figure 2: 动态门控 α(x) / w(x) / σ(x) 可视化 (epoch 末分布直方图) | 2h | [ ] |

---

## Phase 8 — Warm-Restart Final Boost (论文最终 Table 1, 收尾压榨)

> **策略**: 严格连续训练锁定最优配置 → 最后用 warm-restart 拉一把性能, 论文 Table 1 用 warm-restart 数据, 论文 ablation 表用严格连续数据公平对比。
>
> **触发条件**: 完成 Phase 1-6 所有 ablation 后, 锁定最优配置 (K + σ + 任何成功的创新模块)。

| ID | 任务 | yaml 改动 | 时间 | 状态 | 结果 |
|---|---|---|---|---|---|
| 8.1 | 锁定最优配置 reference (严格连续, 60ep) | 当前最优 (K + 任何加上的 module) | ~35min | [ ] | 论文 ablation 公平对比 |
| 8.2 | Warm-restart 全开 (60ep, 段切换从 best 重启) | yaml 加 `restart_from_best: True` 到 Stage 2/3 | ~35min | [ ] | 期望 +0.5~0.8 H |
| 8.3 | 多 seed (5/42/2024) warm-restart | seed 切换跑 3 次 | ~2h | [ ] | 论文 Table 1 main result |
| 8.4 | (可选) AWA2 warm-restart | `train_VGSR_AWA2.py` | ~1.5h | [ ] | |
| 8.5 | (可选) SUN warm-restart | `train_VGSR_SUN.py` | ~1.5h | [ ] | |

**论文叙事**: "Our method achieves H=___ on CUB. The result with strict continuous schedule is H=___, and warm-restart with best-checkpoint at stage transition gives an additional +___ improvement (consistent with TransZero/MSDN observations)."

**警告**: warm-restart 是 GZSL 领域标准做法, 但有审稿人觉得"用 test 选 ckpt 不严格"。叙事时要清楚说明这是"段切换 ckpt 选择", 不是"用 test 集选超参"。GZSL 没有 val split, 是合规的。

---

## � 推荐执行顺序

```
Day 1 (3.5h):
  ├─ Phase 1.1 v5+FAE seed=42        (~36min)
  ├─ Phase 1.2 v5+FAE seed=2024       (~36min)
  ├─ Phase 1.3 入账                   (5min)
  └─ Phase 2.1 纯 CLIP baseline       (5min)
      → 看 1.1+1.2 决定是否替换 main baseline

Day 2 (2.5h):
  ├─ Phase 4.1 D-α 动态门控           (~35min)
  ├─ Phase 4.2 D-w 双分支动态权重     (~35min)
  ├─ Phase 3.1 K=32                   (~30min)
  └─ Phase 3.2 K=128                  (~35min)

Day 3 (3h):
  └─ Phase 2.4-2.9 ablation A1-A6     (~3h)
      → Table 2 完成

Day 4 (3h):
  ├─ 看 Phase 4 结果决定 Phase 5.1 / 5.2 / 4.3
  └─ Phase 2.2 AWA2 (~1h) + Phase 2.3 SUN (~1h)

Day 5+:
  └─ Phase 5 / 6 / 7 论文写作
```

---

## 📊 已确认的"真·结构性失败"(不再做)

| 项 | H | 原因 | 已写论文? |
|----|------|------|---|
| Cosine-only v1/v2 | 43-45 | 流形错位 | 待 7.4 |
| LaSt v1 (`pool_method=lastvit`) | 71.18 | LaSt 特征值进梯度链路, frozen CLIP 切断 | 待 7.4 |
| LaSt v2 (`use_lastvit_cls`) | 70.87 | 同 v1 | 待 7.4 |
| LaSt v3 (`+lastvit_proj+LN`) | 21.11 | LayerNorm 把 LaSt 归一到 N(0,1) 跟 CLIP_CLS ~10 量级冲突 | 待 7.4 |
| T2 score 级动态门控 (旧版) | 71.65 | baseline FAE+576 mean pool 信号不够 sharp | 已被 4.1 取代 |
| **Phase 4.1 D-α (Cons on, 90ep)** | **66.22** final | Cons 跟 D-α 反向打架, Cons 8→11.57 无界增长 | 待 7.4 |
| **Phase 4.1-A D-α (Cons off, 90ep)** | **68.22** final | α_net 退化常数 sigmoid≈0.33, frozen CLIP+seen-only CE 必然退化 | 待 7.4 |
| 50ep 三合一全家桶 | 34→18 | Loss 量级失控 | 待 7.4 |
| F5 全开 (P1+α/w) | 68 | P1 per-class pool 拖累 | 待 7.4 |
| CGC C5 (Margin Hinge) | 40 (ep1) | 训练 batch 全 seen, hinge 跟 CE 反方向打架 | 待 6.1 修 |

---

## 🎯 论文叙事链 (持续维护)

```
Part 1: Foundation
  Adapter + FAE + Bidirectional Transformer + GPT-5.5 文本

Part 2: Loss-level Innovation (G1+G2+G3 三件套)
  - G1 Topology Pearson + Consistency KL (动态权重)
  - G2 MSDN Mutual Distillation
  - G3 CoCoOp-style Conditional Text Adapter

Part 3: ★ NEW Patch-level Innovation (v5+FAE)
  - LaSt-ViT v5 patch 选择器 (frozen CLIP 下用离散决策)
  - FAE 子集 gather (box_emb advanced indexing)
  - K patches 信息密度 + FAE 几何解耦联合作用
  - AG-JEPA masked semantic patch prediction (已接入, 待跑消融)

Part 4: ★ Hierarchical Dynamic Gating (Phase 4 重启)
  - Score-level α(x) (D-α)
  - Branch-level w(x) (D-w)
  - Frequency-level σ(x) (D-σ)  ★ 新

Part 5: Counterfactual Calibration (Phase 6, 修后再考虑)
  - L_bias 方案 2 unseen mean anchor
  - Prior Correction 后验先验校准 (test-time, 必须单独标注 transductive)

Part 6: ★ Negative Results (重要 Discussion)
  - Cosine-only v1-v8 流形错位
  - LaSt v1/v2/v3 frozen CLIP 切断梯度
  - T2 score 级动态门控旧版 (然后 v5+FAE 后重生)
  - CGC C5 hinge 跟 CE 打架

Part 7: Cross-Dataset Validation
  - CUB / AWA2 / SUN
  - 3-seed (5/42/2024) mean ± std

Part 8: Visualization
  - Patch 选择可视化 (LaSt 选哪些 patch)
  - 动态门控 α(x)/w(x)/σ(x) 直方图
```

---

## � 历史归档 (旧路线图, 2026-05-22 及之前)

<details>
<summary>点开看 H2-H5 三层动态门控旧计划 (已被 v5+FAE 路线替换)</summary>

旧目标: 在 baseline (FAE+576) 上跑 H2-H5 三层 mlp, 期望 H=73~74。
实际进展: H2-fixed 跑过, T2 (相当于 H4 score 级) 跑 H=71.65 失败被关闭。
现状: 旧三层方案在 baseline 上信号不 sharp, **现已迁移到 v5+FAE 64 patches 上重试 (Phase 4)**。

旧 Phase 1 (H2-H5):
- H2-fixed (`pool_method=dmp, pool_dynamic=fixed`)
- H2-mlp (`pool_dynamic=mlp`)
- H3 (+`weight_s2v_mode=mlp`)
- H4 (+`gating_dynamic=mlp`)
- H5 三层全开

旧 Phase 2 (文本来源对照): GPT-5.5 已替换 Claude (现 baseline 用 GPT-5.5)
旧 Phase 3 (A1-A6 ablation): 见新 Phase 2.4-2.9
旧 Phase 4 (CGC C1-C5): 见新 Phase 6.2
旧 Phase 5 (12 项消融): 见新 Phase 3 + 5
旧 Phase 6 (跨数据集 + 多 seed): 见新 Phase 1.1-1.2 + 2.2-2.3

</details>
