# 📋 CUB GZSL 实验记录

> 每次实验都按统一模板记录：配置 → 框架 → 结果 → 分析

**数据集**: CUB-200-2011 | **划分**: xlsa17 Proposed Split  
**训练**: 7057 张 / 150 seen | **测试 seen**: 1764 张 | **测试 unseen**: 2967 张 / 50 unseen  
**评估**: GZSL-U / GZSL-S / GZSL-H (调和均值) / ZSL  
**环境**: Windows + dassl_clip + cuda:0 + CLIP ViT-L/14@336px (frozen)

---

## 🏆 总览看板（按 H 排序）

| # | 时间 | 关键改动 | U | S | **H** | ZSL | best ep | 模型文件 |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-05-09 | 基线 ResNet+Adapter+GPT 10ep | 69.02 | 71.54 | 70.26 | 79.60 | 7 | (无) |
| 2 | 2026-05-10 | 串行双向 local=0.3 10ep | 66.13 | 74.47 | 70.05 | 79.38 | 8 | (无) |
| 3 | 2026-05-10 | 并行双向 local=0.3 10ep | 66.14 | 72.88 | 69.34 | 79.14 | 8 | (无) |
| 4 | 2026-05-11 | + unseen GPT 文本 20ep | 71.35 | 70.65 | 71.00 | 80.21 | 16 | (无) |
| 5 | 2026-05-11 | + local=0.5 20ep | 71.07 | 71.04 | 71.06 | 80.51 | 20 | (无) |
| 6 | 2026-05-17 | + 真 patch 评估缓存 15ep | 68.04 | 76.57 | 72.05 | 80.98 | 14 | (丢失) |
| ⭐7 | 2026-05-17 | + Claude 文本描述 15ep (seed=5) | 71.18 | 73.24 | 72.20 | 81.67 | 15 | (丢失) |
| 7' | 2026-05-17 | + Claude 文本描述 15ep (seed=42) | 70.88 | 73.80 | 72.31 | 81.79 | 15 | (丢失) |
| 8 | 2026-05-17 | 多 LLM 融合 (Merge 14 句) | - | - | 71.68 | - | - | (丢失) |
| 9 | 2026-05-17 | α-grid 加权融合 (max α=0.7) | - | - | 71.87 | - | - | (丢失) |
| × | 2026-05-17 | Cosine-only v1 | 30.57 | 77.41 | 43.83 | 69.99 | - | (丢失) |
| × | 2026-05-17 | Cosine-only v2 (unseen 绕过) | 31.22 | 80.88 | 45.05 | 77.77 | - | (丢失) |
| 10 | 2026-05-18 | + Calibration loss λ=0.001 | 71.65 | 72.73 | 72.19 | 81.73 | 15 | (丢失) |
| × | 2026-05-18 | + Calibration loss λ=0.01 | 78.97 | 56.75 | 66.04 | 82.08 | - | (丢失) |
| 11 | 2026-05-18 | FAE 消融 (use_fae=False) | 68.04 | 76.57 | 72.05 | 80.98 | - | (丢失) |
| × | 2026-05-18 | Random Holiday 机制 | - | - | 69.35 | - | - | (丢失) |
| × | 2026-05-18 | Attention Pooling | 72.11 | 71.56 | 71.83 | 81.50 | - | (丢失) |
| × | 2026-05-18 | gate_net MLP (CIG v1) | 64.65 | 77.85 | 70.64 | 81.62 | - | (丢失) |
| × | 2026-05-18 | CIG (α/τ 标量) | 67.07 | 77.24 | 71.80 | 81.28 | 13 | (丢失) |
| × | 2026-05-22 | Cons + L2SP + K=2 + 50ep | 69.87 | 74.29 | 72.01 | 82.01 | 10 | (丢失) |
| 12 | 2026-05-23 | **基线复现 (mean, gating=fixed)** | 70.04 | 74.58 | 72.24 | 81.66 | 15 | baseline_H7224_seed5_FIXED.pth |
| 🥇13 | 2026-05-23 | **LaSt-ViT k=8, 15ep** | 67.85 | 77.79 | **72.48** | 81.62 | 15 | best_..._H7248.pth |
| ❌14 | 2026-05-24 | LaSt-ViT k=8, 30ep | 74.83 | 68.87 | 71.73 | 81.94 | 9 | best_..._H7173.pth |

> 🥇 当前最佳；× 失败；⭐ 历史里程碑；(丢失) = 磁盘满或路径变迁导致 .pth 没了

---

## 📋 实验记录模板（每次新实验照这个格式填）

```
## 实验 N：标题

**时间**: YYYY-MM-DD HH:MM
**目的**: (一句话, 这次实验想验证什么)
**模型文件**: best_..._H{int}.pth
**训练日志**: training_log_CUB_YYYY-MM-DD_HH-MM-SS.txt

### 配置

| 项 | 值 | 说明 |
|---|---|---|
| epochs | 15 | |
| batch_size | 64 | |
| seed | 5 | |
| text_source | claude | |
| pool_method | mean | |
| gating | fixed | |
| score_mode | add | |
| use_fae | True | |
| 各 lambda | 全 0 | 关闭 |
| use_aug_cache | False | 单视角 |

### 框架（数据流）

(用文字或缩进图描述这次和之前的差异，无需重复整体架构)

### 结果

| Epoch | U | S | H | ZSL |
|---|---|---|---|---|
| 1 | ... | ... | ... | ... |
| ... | ... | ... | ... | ... |
| Best @ ep N | x | x | **x** | x |

### vs 基线

| 指标 | 基线 | 本次 | Δ |
|---|---|---|---|
| H | 72.24 | x | x |
| U | 70.04 | x | x |
| S | 74.58 | x | x |
| ZSL | 81.66 | x | x |

### 分析

- 现象：...
- 原因：...
- 教训：...

### 后续动作

- [ ] ...
```

---

## 实验 12：基线复现（mean 池化 + gating=fixed）⭐

**时间**: 2026-05-23 18:03  
**目的**: 重新跑出 H=72.26% 基线，作为后续所有实验的对照锚点  
**模型文件**: `train_log/CUB/baseline_H7224_seed5_FIXED.pth`  
**训练日志**: `training_log_CUB_2026-05-23_18-03-31.txt`

### 配置

| 项 | 值 |
|---|---|
| epochs | 15 |
| batch_size | 64 |
| seed | 5 |
| text_source | claude |
| pool_method | mean |
| gating | fixed |
| local_weight | 0.3 |
| score_mode | add |
| use_fae | True |
| lambda_cal/consist/l2sp | 0/0/0 |
| use_aug_cache | False |

### 框架

```
patches → embed_cv → vis → FAE → memory
              ┌────────┬────────┐
            v2s      s2v
              ↓        ↓
          F_p_v2s   F_p_s2v
              │        │ mean(dim=1)  ★ 当前用 mean
              │        ↓
              │   s2v_pooled [B,512]
              │        │
        score_v2s   score_s2v
              └────┬───┘
              local_score
                    │
        logits = base + 0.3 × local_score
                    │
              CE Loss only
```

### 结果

```
Best @ Epoch 15:
  GZSL-U : 70.04%
  GZSL-S : 74.58%
  GZSL-H : 72.24%   ← 与 5/17 基线 H=72.26% 偏差 -0.02%（噪声内）
  ZSL    : 81.66%
```

### 分析

- ✅ 完美复现历史基线，代码改动（gating 开关、resume 系统）没破坏行为
- ✅ 这是后续所有对比实验的**官方对照锚点**

---

## 🥇 实验 13：LaSt-ViT 频域 Top-K 池化（当前最佳）

**时间**: 2026-05-23 23:41  
**目的**: 替换 mean 池化为 LAST-ViT 频域选择，让模型自动跳过背景 patch  
**模型文件**: `train_log/CUB/best_model_CUB_2026-05-23_23-41-52_H7248.pth`  
**训练日志**: `training_log_CUB_2026-05-23_23-41-52.txt`

### 配置（vs 实验 12，只改 1 项）

| 项 | 实验 12 | 实验 13 |
|---|---|---|
| **pool_method** | mean | **lastvit** ⭐ |
| lastvit_k | (不读) | 8 |
| lastvit_sigma | (不读) | 10.0 |
| 其他全部 | 相同 | 相同 |

### 框架

```
F_p_s2v [B, 576, 512]
    │
    │ ★ LaSt-ViT 频域池化 (无参数)
    │   1. FFT 沿特征维 → 频域
    │   2. 高斯低通 σ=10 → x_lp
    │   3. 打分: F_p / |F_p - x_lp|
    │   4. topk(k=8) → 选 8 个"高频前景" patch
    │   5. mean over 8 patch
    ↓
s2v_pooled [B, 512]  (只用 8/576 个 patch)
```

### 结果

| Epoch | U | S | H | 备注 |
|---|---|---|---|---|
| 1 | 66.40 | 60.62 | 63.38 | |
| 5 | 66.75 | 71.51 | 69.05 | |
| 9 | 68.00 | 75.98 | 71.77 | 涨势 |
| 11 | 68.72 | 75.88 | 72.13 | |
| 13 | 67.24 | 77.98 | 72.22 | |
| 14 | 67.75 | 77.67 | 72.37 | |
| **15** | **67.85** | **77.79** | **72.48** | ⭐ 仍在涨 |

### vs 基线

| 指标 | mean 基线 | LaSt-ViT | Δ |
|---|---|---|---|
| H | 72.24 | **72.48** | **+0.24** ✅ |
| U | 70.04 | 67.85 | **−2.19** |
| S | 74.58 | **77.79** | **+3.21** |
| ZSL | 81.66 | 81.62 | -0.04 |

### 分析

- ✅ H 涨 0.24%（接近噪声边缘但确实涨）
- ✅ S +3.21%：lastvit 帮模型抓到 seen 类判别 patch
- ❌ U -2.19%：unseen 类的"全局形态"信息丢失
- 🔍 训练未收敛：epoch 14→15 仍 H 72.37→72.48

**机制理解**：
- LaSt 选"高频纹理 patch"，偏局部细节
- seen 类有训练监督 → 学到关键 patch
- unseen 类无监督 → 全局形态丢失 → U 跌
- 但因 lastvit 无可学参数，比 attention pool 失败模式好

### 后续动作

- [x] 跑 30 epoch 看 U 能否回涨（实验 14，进行中）
- [ ] K-grid: lastvit_k = 4 / 16 / 32 找最佳
- [ ] σ-grid: lastvit_sigma = 5 / 10 / 20

---

## ❌ 实验 14：LaSt-ViT k=8 长训练（30 epoch）— 失败

**时间**: 2026-05-24 01:03  
**目的**: 实验 13 在 epoch 15 仍在涨，跑 30 epoch 看是否进一步提升、U 是否回涨  
**模型文件**: `train_log/CUB/best_model_CUB_2026-05-24_01-03-02_H7173.pth`  
**训练日志**: `training_log_CUB_2026-05-24_01-03-02.txt`

### 配置（vs 实验 13，只改 epochs）

| 项 | 实验 13 | 实验 14 |
|---|---|---|
| **epochs** | 15 | **30** |
| 其他 | 相同 | 相同 |

### 结果（关键 epoch）

| Epoch | U | S | H | LR | 备注 |
|---|---|---|---|---|---|
| 5 | 69.90 | 69.30 | 69.60 | 0.000933 | 平稳上升 |
| 7 | 68.22 | 74.46 | 71.20 | 0.000872 | |
| **9 (Best)** | **74.83** | **68.87** | **71.73** | 0.000794 | ⭐ 全局最佳 |
| 10 | 70.47 | 72.72 | 71.58 | 0.000750 | |
| 11 | 77.31 | 61.99 | 68.81 | 0.000703 | S 暴跌 |
| 13 | 79.15 | 53.89 | 64.12 | 0.000604 | S=53.89% |
| 17 | 79.98 | 48.11 | 60.08 | ~0.0004 | S=48.11% |
| 22 | 80.17 | 50.39 | 61.88 | ~0.0003 | 没救回来 |
| (用户中断 @ 23) | - | - | - | - | 提前停训 |

### vs 实验 13（15 epoch）

| 指标 | 实验 13 (15ep) | 实验 14 (30ep) | Δ |
|---|---|---|---|
| H | **72.48** | 71.73 | **−0.75** ❌ |
| U | 67.85 | 74.83 | +6.98 |
| S | **77.79** | 68.87 | **−8.92** |
| ZSL | 81.62 | **81.94** | +0.32 |
| Best epoch | 15 | 9 | -6 |

### 🚨 失败模式

```
正常阶段 (epoch 1-9):
  H 稳步上升: 63 → 72
  Best 出现在 epoch 9 (H=71.73)

崩盘阶段 (epoch 10+):
  S 持续暴跌: 74.46 → 60 → 48
  U 反向飙升: 70 → 80
  H 整体下滑: 71.7 → 60 → 62
  ★ LR 已经从 0.0008 退到 0.0003, 仍然救不回来
```

### 分析：为什么 30ep 反而比 15ep 差

#### 1. 直接原因：LR 退火曲线不同

```
同一个 epoch, 30ep 设置的 LR 比 15ep 大 ~57%

epoch 7 时:
  15ep:  LR = 0.000552  ← 较小, 还在精修
  30ep:  LR = 0.000872  ← 较大, 步子大

epoch 9 时 (本次 Best):
  15ep:  LR = 0.000345
  30ep:  LR = 0.000794  ← 仍然偏大

实验 13 (15ep) 在 epoch 14-15 时:
  LR ≈ 0.00001, 模型在最小步进里精修
  Best @ epoch 15

实验 14 (30ep) 在 epoch 14-15 时:
  LR ≈ 0.00050, 模型仍在大步走
  → 已过拟合 unseen, 不是精修而是震荡
```

#### 2. 根本原因：CLIP + Adapter 的优化曲面太平坦

```
loss landscape 类比:

15ep 模型在 epoch 14-15:
  LR 已小到 1e-5, 走得很慢
  → 在最优点 "原地小步抖动" → 微调
  → 既找到了局部最优, 又没乱走

30ep 模型在 epoch 9-15:
  LR 还在 5e-4 以上, 步子大
  → 跨过局部最优点, 进入"过拟合 unseen"区域
  → 模型从"识别鸟"学成"猜 unseen 类"
  → S 暴跌, U 飙升
```

#### 3. 表象：U 涨 S 跌的极端化

```
正常训练:  CE loss 推动模型 "把 seen 类分对"
                          ↓
                S 高, U 因 base_logits 兜底也不差

过拟合表现:  模型在 seen 类上 train_loss < 0.3 (epoch 11+)
                          ↓
            模型开始"假装学 unseen": 
              - logits 把 unseen 列预测得过强
              - 训练时 unseen 列没监督, 但因 cross-modal 的间接梯度
                让 unseen 文本特征被推得"远离 seen", 反而靠近 cls 表示
              - 结果: unseen logits > seen logits → U 飙升
              - 但这是"假学", S 直接崩
```

#### 4. CosineAnnealingLR 的双刃剑

```
T_max=15: 后期 LR 快速归零 → 强制停止, 保住基线
T_max=30: 后期 LR 缓慢归零 → 给了模型"作妖"的时间
```

#### 5. 与之前方案 11 的相似性

```
方案 11 (Cons+L2SP+Aug+50ep):
  Best @ epoch 10: H=72.01
  Epoch 50: H=34.38   ← 崩盘到 30%

方案 14 (LaSt+30ep):
  Best @ epoch 9: H=71.73
  Epoch 22: H=61.88   ← 没那么严重但同方向

★ 共同点: epoch 数过大 + CE 主导 + 没有合适的早停机制
```

### 教训

1. **CLIP+Adapter 任务收敛快**：epoch 7-10 已经是甜点
2. **CosineAnnealing 的 T_max 不是越大越好**：15 epoch 是 sweet spot
3. **"看 H 还在涨"是误判**：实验 13 epoch 14→15 涨的 0.11% 是噪声
4. **best 模型保护机制虽好，但浪费 GPU 时间**：本次跑了 30 个 epoch 只用了 9 个

### 下次该怎么做

| 改进 | 说明 |
|---|---|
| ✅ 回到 epochs=15 | 已验证最佳 |
| ⏸️ 不再尝试 epochs > 15 | 浪费时间，不会更好 |
| 🆕 加 early stop（可选）| H 连续 5 epoch 不涨就停 |
| 🆕 K-grid 在 15 epoch 下做 | 不要混轮次变量 |

### 后续动作

- [x] 把 yaml `epochs` 改回 15
- [x] 记录失败原因到本文档
- [ ] **下次实验：LaSt-ViT K-grid (k=4/16/32, 15ep)**

---

## 📊 历史实验简录（概要）

详细架构见 git 历史 `v1-baseline` / `v2-cig-loss-aug` 标签。

### 已确认有效

- ✅ **真 patch 评估缓存修复**：评估时用真实 patch 而非 CLS×576 伪造（5/17）
- ✅ **Claude Opus 4.7 文本描述**：替代 GPT-4，6 句细节 + 1 句 caption（5/17，+1.26% H）
- ✅ **LaSt-ViT 频域池化**：替代 mean（5/23，+0.24% H）
- ✅ **G1 Topo Pearson + Consistency KL**：H=72.61，+0.37 H（5/25）

### 真·结构性失败（机制层面错，可以放弃）

- ❌ **Cosine-only v1/v2** (H=43-45, -27)：unseen 列绕过 proj_text 流形错位，机制错
- ❌ **F5 全开** (H=68.04, -4.2)：P1+α/w MLP 联合，P1 per-class pool 丢类间共性
- ❌ **Cons+L2SP+Aug 50ep 崩盘** (H 34→18)：Loss 量级失控，训练崩盘

### 效果不明显（**还没正经消融**，留作后续 ablation）

> 这些不是"失败"，是"单组实验跑出来 H 没涨/微跌"。论文写作前**必须**做单变量 ablation 给定论。每组 epoch=20 单独验证。

| 项 | 实测 H | Δ | 待补消融 |
|----|--------|------|----------|
| LaSt-ViT s2v 池化 (k=8) | 71.73 | -0.51 | k-grid (k=2/4/16/32) + σ-grid |
| CIG α/τ 标量 | 71.80 | -0.44 | 与 alpha_net MLP 对比 |
| 22 CIG 升级 (MBG/TCG/双层) | — | — | 未实现, 待做 |
| **P1 单独 (class_attention pool)** | 68 (F5 联合) | -4 | 单独跑过没？需要 P1 + 其他全关单独验证 |
| Calibration loss λ=0.001 | 72.19 | -0.05 | λ 网格 (0.0005/0.002/0.005) |
| Calibration loss λ=0.01 | 66.04 | -6 | 仅 1 个值, λ-grid |
| Random Holiday | 69.35 | -2.85 | 只 1 组超参, 重试不同丢弃比例 |
| Attention Pooling (旧 Linear) | 71.83 | -0.41 | hidden 维度 / 多头版没试 |
| gate_net MLP (CIG v1) | 70.64 | -1.60 | 已被 F5 alpha_net 部分替代但仍可对比 |
| Multi LLM 融合 (Merge 14 句) | 71.68 | -0.56 | 文本权重学习 / cluster 选择 |
| α-grid 加权融合 LLM | 71.87 max | -0.39 | 温度调整 / 排序方法 |
| L2-SP 单独 | — | — | 之前 50ep 崩盘, 没在 15-20ep 单独验证 |
| Cosine-only v3 流形共享+detach | 50.40 (v4) | -22 | v3/v4 内部未实现纯净 detach+manifold sharing, 真正 v3 待写 |

### 已实施开关（可随时切换）

- `gating: fixed | cig`
- `score_mode: add | cosine_only`
- `pool_method: mean | attention | lastvit`
- `use_fae: True | False`
- `use_aug_cache: True | False`
- `model_mode: vgsr | clip_only | adapter_only`
- `resume_from / resume_lr_schedule / extra_epochs`

---

## 🎯 待做实验队列（按优先级）

| 优先级 | 编号 | 实验 | 预期 |
|---|---|---|---|
| 🥇 | 15 | **LaSt-ViT k=4/16/32 K-grid (15ep)** ⭐ 下一步 | 找最佳 K, U 能否回涨 |
| 🥈 | 16 | FAE 尺度修复（可学 geo_scale）| ZSL +0.5~1% |
| 🥈 | 17 | Class-aware Attention Pooling | H +0.5~1.5% |
| 🥉 | 18 | gzsl_bias 网格搜索 | H +0.3~0.8% |
| 🥉 | 19 | unseen 也过 adapter 对照 | -1~+0.5% |
| 🥉 | 20 | Multi-seed ensemble (5/42/2024) | +0.2~0.4% |
| 🟡 | 21 | **Cosine-only v3: 流形共享 + detach** | 待测（v1/v2 失败 -27%）|

> 已确定的规则: epochs **永远不超过 15**（实验 14 验证 30ep 反而崩到 71.73）

---

## ✅ Cosine-only 救援实验路线 (E0-E8, 每组 epoch=20)

> **背景**: cosine_only 把 base_logits 从主路径踢成旁路 anchor, 之前 v1-v4 (H=43~50) 已证明 "纯 cosine_only 无 base 参与必崩"。v5 H=72.04 是 residual=1.0 退化等价 base 的结果, 不是真 cosine_only 成绩。
> 
> **修订思路**: 不再跑"纯 cosine_only"重复 v1-v4 的失败, 让 base 至少弱参与 (lambda_distill=0.1) 才是合理 baseline。每组都基于上一组叠加, 单变量观察。
> 
> **共同基线 yaml**:
> - `score_mode: cosine_only`
> - `epochs: 20`
> - `random_seed: 5`
> - `pool_method: mean`
> - `adapter_ratio: 0.2`
> - `text_residual: 0.5` / `visual_residual: 0.5`  (让 v_enh/t_enh 真的是 learned)

### 阶段一: 真 baseline (base 弱参与)

- [x] **E0 — cosine_only + Distill 弱**
  - 配置: `lambda_distill=0.1`, `distill_temp=4.0`
  - 关: `lambda_aux_s2v=0`, `lambda_aux_v2s=0`, `lambda_topo_pearson=0`, `lambda_v_anchor=0`, `lambda_t_unseen_anchor=0`
  - 目的: **真正 cosine_only baseline** (base 通过 KL teacher 弱回归), 论文 ablation 零点
  - 预期 H: 65~71 (低于固定 1.0/1.0 退化等价)
  - **实测 (2026-05-25)**: U=23.63 S=89.71 **H=37.41** ZSL=75.30 best_ep=20
  - **结论**: ❌ 比 v3/v4 (H=45-50) 还差! distill 0.1 软约束完全救不回, 极端 seen-bias (S/U=3.8 倍), Distill loss 从 0.94 只降到 0.60 几乎没动. 训练日志: `training_log_CUB_2026-05-25_00-10-49.txt`

### 阶段二: 加辅助监督

- [ ] **E1 — E0 + 双分支辅助 CE**
  - E0 基础上开: `lambda_aux_s2v=0.2`, `lambda_aux_v2s=0.2`, `aux_temp=14.28`
  - 目的: 让 s2v / v2s 各自也是合格 seen 分类器, 防止 decoder 退化
  - 预期 H: 70~72
  - 完成填写: U=___ S=___ H=___ ZSL=___ best_ep=___

- [ ] **E2 — E1 + 拓扑保持 (Pearson)**
  - E1 基础上开: `lambda_topo_pearson=0.05`
  - 目的: 保持类与类之间的角度拓扑, 防 seen-only CE 拉散 unseen
  - 预期 H: 71~72.5
  - 完成填写: U=___ S=___ H=___ ZSL=___ best_ep=___

### 阶段三: 加锚定 (修漂移)

- [ ] **E3 — E2 + 视觉锚定 v_anchor**
  - E2 基础上开: `lambda_v_anchor=0.1`
  - 目的: v_enh 锚回 cls_token, 修视觉端漂移
  - 预期 H: 71~72.5
  - 完成填写: U=___ S=___ H=___ ZSL=___ best_ep=___

- [ ] **E4 — E2 + Unseen 文本锚定 t_unseen_anchor**
  - E2 基础上开: `lambda_t_unseen_anchor=0.2`
  - 目的: t_enh[unseen] 锚回 CLIP 原始, 修共享 proj_text 间接漂移 (理论最对症)
  - 预期 H: 71.5~73
  - 完成填写: U=___ S=___ H=___ ZSL=___ best_ep=___

- [ ] **E5 — E2 + 双锚 (E3 + E4)**
  - E2 基础上同时开: `lambda_v_anchor=0.1`, `lambda_t_unseen_anchor=0.2`
  - 目的: 视觉 + 文本双管齐下
  - 预期 H: 72~73
  - 完成填写: U=___ S=___ H=___ ZSL=___ best_ep=___

### 阶段四: base 强参与 (条件触发)

- [ ] **E6 — E5 + Distill 加重**
  - E5 基础上改: `lambda_distill=0.3`
  - 触发条件: **仅当 E5 H < 72 时跑** (distill 软约束不够强, 加重看)
  - 目的: 看蒸馏强度上限
  - 预期 H: 72~73
  - 完成填写: U=___ S=___ H=___ ZSL=___ best_ep=___

- [ ] **E7 — Base 硬混合 cosine_base_blend** (✅ 已实现 2026-05-25)
  - **配置 (E0 失败 H=37.41 后紧急启用)**:
    - `cosine_base_blend: 0.3` (新开关, MyModel.py forward 末尾)
    - `lambda_distill: 0.0` (关掉, 避免双重作用)
    - 其他 E0 配置不变 (residual=0.5/0.5, aux/topo/anchor 全 0)
  - 公式: `logits_200 = 0.7 × cos(v_enh, t_enh)*scale  +  0.3 × base_logits`
  - 目的: 把 CLIP base 硬接回主路径, 修复 E0 distill 软约束完全无效的问题
  - 预期 H: 68~72 (直接拉回 add 基线附近)
  - 完成填写: U=___ S=___ H=___ ZSL=___ best_ep=___

### 阶段五: 动态 residual (论文创新点)

- [ ] **E8 — E5 + Learnable Split Residual** (✅ 已实现 2026-05-25, 直接跳过 E5 接 E7)
  - **配置 (E0/E7 fixed 双双失败 H≈37 后启用)**:
    - `score_mode: cosine_only`, `epochs: 20`, `pool_method: mean`
    - `cosine_base_blend_mode: learnable`, `cosine_base_blend_init: 1.0` (sigmoid起步=0.73 强 base 兜底)
    - `residual_mode: learnable_split`
    - `visual_residual_init: 0.0` (sigmoid=0.5)
    - `text_residual_seen_init: 0.0` (sigmoid=0.5)
    - `text_residual_unseen_init: 3.0` (sigmoid=0.953 强保护 unseen)
    - 其他 lambda 全 0 (干净归因)
  - **新增 4 个可学标量**: vr_logit / tr_seen_logit / tr_unseen_logit / cb_logit
  - **训练日志加打印**: 每 epoch 末打印 sigmoid(vr/tr_s/tr_u/cb) 收敛曲线
  - 目的: 让模型自学最优混合比, 把 base_logits 强势接回主路径同时让 unseen 文本受保护
  - 预期 H: 65~72 (起点已接近 add 基线, 看 learned 部分能否再涨)
  - 完成填写: U=___ S=___ H=___ ZSL=___ best_ep=___
  - 完成填写学到的系数: vr=___ tr_s=___ tr_u=___ cb=___

### 决策分支 (跑完每组后看)

- 如果 **E0 < 60**: cosine_only 框架结构性缺陷严重, **直接放弃, 回 add 模式 + P1 (Class-aware Attention Pool)**
- 如果 **E1 比 E0 涨 ≥ 1**: 双分支辅助 CE 是主要贡献, 继续叠加
- 如果 **E2 比 E1 持平**: topo 边际贡献低, 论文里写"拓扑保持无显著贡献" negative result
- 如果 **E5 比 E2 涨 ≥ 0.5**: 锚定有效, **完成核心叙事可写论文**, E6/E7 可以不跑
- 如果 **E5 比 E2 持平/掉**: 锚定边际效益低, 跳到 E6 (distill 加重) 或 E7 (硬混合) 兜底
- 如果 **E5 H ≥ 72.24** (add 基线): cosine_only 救回成功, **写论文叙事**: "cosine_only + 锚定 + 辅助 CE 框架可匹敌 add 模式且具更好可解释性"

### 最少完成集 (4 组, ~20 分钟)

时间紧只跑这 4 组:
- **E0** (真 baseline)
- **E2** (E0 + 双 aux + topo, 看辅助监督叠加效果)
- **E5** (E2 + 双锚, 看锚定全效)
- **E8** (E5 + 动态 residual, 创新点)

---

## 📁 关键文件路径

```
DVSR/
├── EXPERIMENT_RECORD.md            ← 本文件
├── 创新指导清单/INNOVATION_RECORD.md ← 详细创新方案与失败教训
├── Guide/
│   ├── PROJECT_REPORT.md           ← 模块技术细节汇报
│   └── COSINE_ONLY_REPORT.md       ← Cosine-only 失败专项报告
├── .kiro/docs/conversation_log.md  ← 对话历史摘要
├── config/VGSR_cub_gzsl.yaml       ← 训练配置
├── model/MyModel.py                ← 主模型代码
├── train_VGSR_CUB.py               ← 训练入口（含 resume 系统）
├── train_log/CUB/
│   ├── training_log_CUB_*.txt      ← 每次训练日志
│   ├── best_model_CUB_*_H*.pth     ← 最佳权重（带 H 后缀）
│   ├── ckpt_full_CUB_*.pth         ← 完整 ckpt（含 optimizer）
│   └── baseline_H7224_seed5_FIXED.pth   ← 永久基线快照
└── data/cache/                     ← CLIP 特征缓存
```

---

*最后更新: 2026-05-24*
