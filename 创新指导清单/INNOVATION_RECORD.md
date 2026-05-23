# DVSR 创新效果记录

**当前最佳**：H = **72.26 ± 0.08%**（Claude 描述 + 真 patch 评估 + 15 epoch，n=2 seeds）

**起点 baseline**：H = 70.35%（GPT-4 描述 + CLS×576 假 patch + 20 epoch）

**累计收益**：**+1.91%**

---

## 总体进度看板

| ID | 模块 | 状态 | H | 备注 |
|----|------|------|---|------|
| 1 | FAE 模块消融 | ✅ 完成 | 72.05 | FAE 微弱提升 |
| 2 | Claude 文本描述 | ✅ **当前最佳** | **72.26±0.08** | 严格对照 +1.26% |
| 3 | 多 LLM 融合 (Merge 14句) | ❌ 失败（感觉可以优化） | 71.68 | 稀释优势 |
| 4 | 加权融合LLLM (α-search) | ❌ 失败（感觉可以优化） | 71.87 max | 单调曲线，纯 Claude 最优 |
| 5 | Cosine-only 计分 | ❌ 失败 | 45.05 | v_enh 太弱 + unseen 训歪 |
| 6 | Calibration loss | ❌ 无效 | 72.19 | 0.001 中性，0.01 过强 |
| × | 随机放假机制 | ❌ 弃用 | 69.35 | 退化 |
| 12 | Attention Pooling (s2v) | ❌ 失败 | 71.83 | -0.37%，过聚焦少数 patch 丢全局 |
| 13 | 动态门控 v1 (gate_net MLP) | ❌ 失败 | 70.64 | -1.62%，gate 塌缩成全局降权器 |
| 14 | CIG 门控 (Confidence-Inverse Gating) | ❌ 失败 | 71.80 | -0.40%，α/τ 仍被压向少加 local |
| **15** | **Consistency+L2SP+Aug 三合一(50ep)** | ❌ **崩盘** | 72.01@ep10 | Cons loss 量级失控，后期 S 跌到 18% |
| **7** | **s2v 池化升级** | 🟡 **推荐下一步** | 期望 +0.5~1% | 见下方详细方案 |
| **8** | **门控增强 + 归一化 cosine** | 🟡 **推荐** | 期望 +0.3~1% | 见下方详细方案 |
| **9** | **一致性 Loss** | 🟡 推荐 | 期望 +0.3~0.5% | base 和 local 排序一致 |
| 10 | Caption 模板 ensemble | 🟡 备选 | 期望 +0.3~0.8% | CLIP 79-prompt 思路 |
| 11 | AWA2/SUN 推广 | 🟡 备选 | 跨数据集验证 | 论文必要 |

---

# 一、推荐的下一步创新方案

## 🥇 方案 14：CIG 门控（Confidence-Inverse Gating）⭐ 当前进行中

### 背景

之前尝试过两版门控，均失败：
- **Attention Pooling (方案 12)**: H=71.83% (-0.37%)，s2v 池化过度聚焦少数 patch，丢失 seen 类全局上下文
- **动态门控 v1 (方案 13)**: H=70.64% (-1.62%)，gate_net MLP 塌缩成"全局降权器"，unseen 跌 6.35%

### 失败根因

```
旧 gate_net 的问题:
  cls_token [B,768] → MLP → sigmoid → gate [B,1]
  logits = base_logits + gate × local_score
  
  ★ 监督只来自 seen 类 CE loss
  ★ unseen 类无梯度反传
  → 模型学到 "压低 gate 让 base 主导" 是 seen 上的最优策略
  → gate 塌缩 → unseen 跟着被压
```

### CIG 解决方案

**核心思想**：gate 不学习一个 MLP，而是 CLIP 自身置信度的简单函数。

```python
# __init__: 仅 2 个可学标量参数
self.gate_alpha = nn.Parameter(torch.tensor(1.0))   # 缩放系数
self.gate_tau   = nn.Parameter(torch.tensor(1.0))   # 温度

# forward
with torch.no_grad():                                   # 阻断 gate 梯度污染 base
    prob = F.softmax(base_logits, dim=-1)               # [B, 200]
    conf = prob.max(dim=-1).values                      # [B]
uncertainty = (1.0 - conf).clamp(min=0.0, max=1.0)      # [B]
alpha = F.softplus(self.gate_alpha)                     # 标量 > 0
tau   = F.softplus(self.gate_tau)                       # 标量 > 0
gate  = (alpha * uncertainty.pow(tau)).unsqueeze(-1)    # [B, 1]
logits_200 = base_logits + gate * local_score
```

### 关键设计

| 设计点 | 作用 |
|---|---|
| `base_logits.detach()` (no_grad) | 阻断 gate 梯度传到 base 分支，保护 CLIP 零样本能力 |
| `softmax` 在全 200 类上算 | unseen 也参与 conf 计算，**绕开监督盲区** |
| 仅 2 个可学标量 (α, τ) | 参数从 14.8 万降到 2，物理意义清晰 |
| `softplus` 包装 | 保证 α, τ 恒正且梯度平滑 |
| 对 seen/unseen 对称 | 同一公式作用所有类，不会塌缩成"全局降权器" |

### 行为预期

```
情况 1: CLIP 高度自信 (seen 简单图，conf=0.85)
  uncertainty = 0.15
  gate ≈ α × 0.15^τ → 很小 (假设 α=1, τ=1.5: gate≈0.058)
  → base 主导，不被 local 干扰

情况 2: CLIP 不确定 (unseen 困难图，conf=0.07)
  uncertainty = 0.93
  gate ≈ α × 0.93^τ → 接近 α (gate≈0.90)
  → local_score 救援发力
```

### 论文卖点

> *"We propose a parameter-free, confidence-conditional gating mechanism that lets CLIP's own zero-shot uncertainty drive local feature injection, eliminating the need for a learned gate network that suffers from supervision-coverage bias in GZSL."*

### 修改位置

`model/MyModel.py`：
- 删除 `self.gate_net = nn.Sequential(Linear, ReLU, Linear)`
- 新增 `self.gate_alpha`, `self.gate_tau` 两个 nn.Parameter
- forward 中 add 模式 gate 计算逻辑替换

### 后续方向

- 涨了 → 叠加方案 B (Class-aware Reliability Gate, CARG) 把 gate 升级为 [B, num_class] 每类独立
- 涨了 → 叠加方案 C (Per-Patch Gate) 在 s2v 池化前加 patch 级门控
- 中性 → 调 α/τ 初值 / 加 BatchNorm
- 跌了 → "门控削弱 base"思路本身错，回头探索其他方向

---

## 🥇 方案 7：s2v 池化升级（替换 mean）

### 问题

当前 s2v 分支的池化：
```python
s2v_pooled = F_p_s2v.mean(dim=1)   # [B, 576, 512] → [B, 512]
```

576 个 patch 平等对待，背景 patch（天空、树枝）占 5/6，判别 patch（鸟身）只占 1/6，**信号被噪声稀释 5 倍**。

### 三种升级方案

#### 方案 2a：Attention Pooling（推荐入门）

```python
# 新增参数：self.attn_head = nn.Linear(dim_com, 1)
attn_logits = self.attn_head(F_p_s2v)         # [B, 576, 1]
attn_weights = F.softmax(attn_logits, dim=1)   # [B, 576, 1]
s2v_pooled = (F_p_s2v * attn_weights).sum(dim=1)  # [B, 512]
```

- **新增参数**：512+1 = 513 个（几乎为 0）
- **效果**：鸟身 patch 自动获得高权重，背景低权重
- **风险**：极低

#### 方案 2b：Top-K Pooling

```python
score_per_patch = F_p_s2v.norm(dim=-1)         # [B, 576]
top_k_idx = score_per_patch.topk(k=64, dim=1).indices  # [B, 64]
top_k_patches = F_p_s2v.gather(1, top_k_idx.unsqueeze(-1).expand(-1,-1,512))
s2v_pooled = top_k_patches.mean(dim=1)         # [B, 512]
```

- **新增参数**：0
- **效果**：直接丢弃 512 个低响应 patch
- **风险**：K 值需要调

#### 方案 2c：Class-aware Attention（最强，配合门控最自然）

```python
# F_p_s2v: [B, 576, 512], txt_com: [N_cls, 512]
attn = F_p_s2v @ txt_com.T                     # [B, 576, N_cls]
attn = F.softmax(attn, dim=1)                  # 每类对 patch 的注意力
# 输出 [B, N_cls, 512]：每张图每类有自己的视觉表示
s2v_pooled = torch.einsum('bpd,bpc->bcd', F_p_s2v, attn)
```

- **新增参数**：0（用已有的 txt_com 做 query）
- **效果**：不同类关注不同 patch（"红头啄木鸟"关注头部，"长尾鸟"关注尾部）
- **输出形状变化**：[B, 512] → [B, N_cls, 512]，需要调整下游 score 计算
- **和"两端增强 cosine"天然配合**

### 推荐实施顺序

1. **先做 2a**（5 行代码，验证池化升级是否有效）
2. 如果涨了 → 做 2c（更强但需要调整 score 计算）
3. 如果 2a 没涨 → 说明池化不是瓶颈，转向其他方向

---

## 🥇 方案 8：门控增强 + 归一化 Cosine

### 核心思想

两端都以 CLIP 原始特征为基础，用门控动态决定增强强度，最终归一化后做 cosine：

```python
# 视觉端
v_delta = proj_visual(s2v_pool)                 # [B, 768]
gate_v = torch.sigmoid(self.gate_visual(CLS))   # [B, 768]
v_better = CLS + gate_v * v_delta               # [B, 768]

# 文本端（seen 列增强，unseen 保持）
t_delta = proj_text(F_p_v2s)                    # [B, 200, 768]
gate_t = torch.sigmoid(self.gate_text(all_text))  # [200, 768]
t_better = all_text.clone()
t_better[seenclass] = all_text[seenclass] + gate_t[seenclass] * t_delta.mean(0)[seenclass]

# 归一化 cosine
logits = F.normalize(v_better) @ F.normalize(t_better).T × logit_scale
```

### 与 add 模式的本质区别

- **add**：分数空间加偏移（线性）
- **本方案**：特征空间改方向（归一化是非线性操作）
- 门控让模型自适应：简单图 gate→0 回退 CLIP，难图 gate→1 全力增强

### 新增参数

| 模块 | 参数量 |
|------|--------|
| gate_visual (768→768) | ~590K |
| gate_text (768→768) | ~590K |
| proj_visual (512→768) | 已有 |
| proj_text (512→768) | 已有 |
| **总新增** | **~1.2M** |

### 安全性

- gate 输出 sigmoid(0~1)，不会爆炸
- gate→0 时回退到纯 CLIP（~70% H 下限）
- unseen 文本不经过可训练增强

### 如果配合方案 2c

```python
# 2c 输出 [B, N_cls, 512] → 每类有自己的视觉表示
s2v_class = class_aware_attention(F_p_s2v, txt_com)  # [B, 200, 512]
v_delta = proj_visual(s2v_class)                      # [B, 200, 768]
gate_v = sigmoid(gate_visual(CLS)).unsqueeze(1)       # [B, 1, 768]
v_better = CLS.unsqueeze(1) + gate_v * v_delta        # [B, 200, 768]

# 每类有自己的增强视觉 + 增强文本 → 逐类 cosine
logits[b,k] = cos(v_better[b,k], t_better[k])
```

这是**最完整的方案**：每类有独立的视觉表示 + 独立的文本表示 + 门控自适应。

---

## 🥈 方案 9：基线-局部一致性 Loss

### 思路

base_logits 和 local_score 对同一张图应该"意见一致"：

```python
base_pred = F.softmax(base_logits[:, seenclass] / temp, dim=-1)
local_pred = F.softmax(local_score[:, seenclass] / temp, dim=-1)
loss_consist = F.kl_div(local_pred.log(), base_pred.detach(), reduction='batchmean')

total_loss = loss_CE + lambda_consist × loss_consist
```

### 参数

```yaml
lambda_consist: 0.1
consist_temp: 2.0   # 温度，让分布更平滑
```

### 预期

- 防止 local_score 学到和 CLIP 矛盾的信号
- H +0.3~0.5%
- 风险极低（只是正则化）

---

# 二、已完成实验详细记录

## Claude 文本描述（当前最佳）

**严格对照（同 epoch=15 + 真 patch + 仅 text_source 差异）**:

| 指标 | GPT-4 (seed=5) | Claude (seed=5) | Claude (seed=42) | Claude mean±std |
|------|----------------|-----------------|------------------|-----------------|
| H | 71.00 | 72.20 | 72.31 | **72.26 ± 0.08** |
| U | 68.55 | 71.18 | 70.88 | 71.03 ± 0.21 |
| S | 73.64 | 73.24 | 73.80 | 73.52 ± 0.40 |
| ZSL | 80.85 | 81.67 | 81.79 | 81.73 ± 0.08 |

## Calibration Loss 实验

| lambda_cal | U | S | H | ZSL | 结论 |
|-----------|---|---|---|-----|------|
| 0（基线） | 71.18 | 73.24 | **72.20** | 81.67 | 最佳 |
| 0.001 | 71.65 | 72.73 | 72.19 | 81.73 | 中性（±0.01%） |
| 0.01 | 78.97 | 56.75 | 66.04 | 82.08 | 过强，S 暴跌 |

**结论**：cal loss 在本架构下无效。模型 U/S 已均衡，不需要额外推 unseen。

## Cosine-only 计分模式

| 版本 | U | S | H | ZSL | 问题 |
|------|---|---|---|-----|------|
| v1（暴雷） | 30.57 | 77.41 | 43.83 | 69.99 | unseen 文本训歪 |
| v2（方案3修复） | 31.22 | 80.88 | 45.05 | 77.77 | v_enh 太弱（mean 池化） |

**结论**：纯 cosine 框架在当前架构下不可行。核心原因：(1) mean 池化稀释信号 (2) 无 CLS 兜底。

## FAE 消融

| 配置 | 参数量 | U | S | H | ZSL |
|------|--------|---|---|---|-----|
| 有 FAE | 10.4M | 71.18 | 73.24 | 72.20 | 81.67 |
| 无 FAE | 8.3M | 68.04 | 76.57 | 72.05 | 80.98 |

**结论**：FAE 对 H 几乎无贡献（-0.15%），但对 ZSL 有 -0.69% 损失。几何解耦机制实测失效（attention scale 失配）。

## Attention Pooling (方案 12)

替换 `s2v_pooled = F_p_s2v.mean(dim=1)` 为：
```python
attn_logits = self.attn_pool(F_p_s2v)         # [B, 576, 1]
attn_weights = F.softmax(attn_logits, dim=1)
s2v_pooled = (F_p_s2v * attn_weights).sum(dim=1)
```

| 配置 | U | S | H | ZSL |
|------|---|---|---|-----|
| Mean (基线) | 71.18 | 73.24 | 72.20 | 81.67 |
| Attention Pool | 72.11 | 71.56 | **71.83** | 81.50 |

**结论**：H -0.37%。Attention Pooling 过度聚焦少数 patch，U 涨 0.93 但 S 跌 1.68，整体不划算。已改回 mean。

## 动态门控 v1 — gate_net MLP (方案 13)

新增 `gate_net = Linear(768→192) → ReLU → Linear(192→1)` (~14.8 万参数)：
```python
gate = sigmoid(gate_net(cls_token))           # [B, 1]
logits_200 = base_logits + gate * local_score
```

| 配置 | U | S | H | ZSL |
|------|---|---|---|-----|
| 固定 local_weight=0.3 (基线) | 71.18 | 73.24 | 72.20 | 81.67 |
| Dynamic gate_net | 64.65 | 77.85 | **70.64** | 81.62 |

**结论**：H -1.62%（远超 std=0.08% 噪声）。U -6.35%，S +4.35%。

**根因**：监督只覆盖 seen，gate 塌缩成"全局降权器"——CLIP 在 seen 上准 → 模型学到关 gate 让 base 主导更稳 → unseen 跟着被压。

**改进方向**：→ 方案 14 (CIG)，用 CLIP 自身置信度替代可学习 gate。

---

## 方案 15: Consistency Loss + L2-SP + 多视角增强 + 50 epoch (全家桶崩盘)

**配置**:
- `lambda_consist: 0.1`, `consist_temp: 2.0`
- `lambda_l2sp: 0.0005`
- `use_aug_cache: True` (K=2 视角)
- `epochs: 50` (从 15 加到 50)
- 同时叠加: Claude 文本 + 真 patch + use_fae=True + CIG 门控

| Epoch | U | S | H | ZSL | Cons Loss |
|---|---|---|---|---|---|
| 5 | 69.26 | 67.55 | 68.39 | 81.28 | ~3.0 |
| **10 (Best)** | **69.87** | **74.29** | **72.01** | **82.01** | ~5.4 |
| 20 | 80.23 | 39.68 | 53.10 | 81.60 | ~9.7 |
| 30 | 81.56 | 18.25 | 29.83 | 81.70 | ~12.0 |
| 50 | 81.42 | 21.79 | 34.38 | 81.62 | ~13.0 |

**结论**：H 从 epoch 10 的 72.01 跌到 epoch 50 的 34.38。模型彻底崩盘。

**根因**：
1. **Cons loss 量级失控**: 从 1.0 → 13.0（13倍增长）
2. **Cons 项贡献完全盖过 CE**: epoch 50 时 0.1×13=1.3 vs CE=0.05
3. **模型转而优化"local 模仿 base"**: 不再优化分类，S 暴跌到 18-22%
4. **三改动同时上无法定位**: 不知道是 Cons / L2SP / Aug 哪个是元凶

**教训**：
- KL 一致性 loss 在长训练下会无界增长，必须 clip 或大幅降权
- 多变量同时改的反例：本次 4 个变化同时上，无法定位
- ZSL 涨 0.34% 但 H 跌 0.25%，整体不划算

---

# 三、推荐实施优先级

```
第 1 步：方案 14（CIG 门控）⭐ 当前进行中
         替换失败的 gate_net MLP，2 个标量参数
         用 CLIP 置信度反推 gate，绕开监督盲区
         预期 H 持平或 +0.3~0.8%

第 2 步（涨了再做）：方案 2c（Class-aware Attention 池化）
         配合 CIG 形成"每类独立视觉表示 + 自适应门控"完整框架
         预期 H +0.5~1.5%（最大潜力）

第 3 步：方案 9（一致性 Loss）
         和上面任何方案叠加，只是正则化
         风险极低，预期 H +0.3~0.5%

备选：方案 7 (Attention Pool / Top-K)
         之前 Attention Pool 失败 -0.37%
         可改 Top-K 或与 CIG 共同跑
```

---

# 四、核心教训

1. **CLIP CLS 是强基线**：任何方案必须保留 CLS 的分类能力作为下限
2. **文本侧 ROI 最高**：Claude 描述 +1.26% 是目前最大单点增益
3. **cal loss 不适用**：模型 U/S 已均衡，强推 unseen 只会破坏平衡
4. **mean 池化是瓶颈，但替换需谨慎**：576 patch 平均稀释信号，但 Attention Pool 过聚焦少数 patch 反而 -0.37%
5. **归一化 ≠ 线性加法**：cosine 模式和 add 模式本质不同（非线性）
6. **unseen 文本必须保护**：任何让 unseen 经过可训练层的设计都会崩
7. **FAE 几何解耦名存实亡**：attention scale 失配导致 geo_weights 影响 < 1e-4
8. **可学习门控的监督盲区陷阱**：gate_net 接 cls_token 时，因 unseen 类无梯度反传，gate 会塌缩成"全局降权器"压低 unseen 信号。解决：用 CLIP 置信度等"对称信号"驱动 gate（方案 14 CIG）
9. **KL 一致性 loss 无界增长**：lambda_consist=0.1 在 50 epoch 下，cons 值从 1.0 涨到 13.0，最终完全主导 loss。修复：lambda 设到 0.01 以下，或加 clamp(max=2.0)
10. **多变量同时改的代价**：方案 15 同时上了 Cons + L2SP + Aug + 50ep 四个变化，整体 H 从 72.26 → 72.01 → 34.38，但因为同时改了 4 个变量根本无法定位元凶。原则：**一次只改一个变量**
