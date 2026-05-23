# DVSR / VGSR 项目工作汇报

> CLIP + Adapter + 跨模态 Transformer 的零样本细粒度鸟类识别

---

## ⚡ 一句话总结

🎯 **当前最佳结果**：CUB GZSL `H = 72.26 ± 0.08%`（双种子稳定复现）  
🚀 **累计提升**：基线 70.35% → 72.26%，共 **+1.91%**  
🔬 **唯一确认有效的改进**：Claude 文本描述（+1.26% H）

---

## 📊 总览看板

| # | 实验模块 | 状态 | H 结果 | Δ H | 是否采用 |
|---|---|:---:|:---:|:---:|:---:|
| ★ | **Claude 文本描述** | 🟢 成功 | **72.26 ± 0.08** | **+1.26** | ✅ 已采用 |
| 1 | 真 patch 评估缓存修复 | 🟢 成功 | 71.00 → 72.20 | 隐含 +0.7 | ✅ 已采用 |
| 2 | FAE 模块消融 | 🟡 中性 | 72.05 | -0.15 | ✅ 保留（ZSL 友好）|
| 3 | 多 LLM 文本融合 (Merge) | 🔴 失败 | 71.68 | -0.52 | ❌ |
| 4 | α-grid 文本加权融合 | 🔴 失败 | 71.87 max | -0.33 | ❌ |
| 5 | Cosine-only 计分模式 | 🔴 失败 | 45.05 | -27.15 | ❌ |
| 6 | Calibration loss | 🟡 中性 | 72.19 | -0.01 | ❌ |
| 7 | Random Holiday 机制 | 🔴 失败 | 69.35 | -2.85 | ❌ |
| 8 | Attention Pooling (s2v) | 🔴 失败 | 71.83 | -0.37 | ❌ |
| 9 | 动态门控 v1 (gate_net MLP) | 🔴 失败 | 70.64 | -1.62 | ❌ |
| 10 | CIG 门控 (无 MLP) | 🟡 改善但未达基线 | 71.80 | -0.40 | ❌ |

> 🟢 成功 / 🟡 中性或部分提升 / 🔴 显著回退

---

## 🏗️ 整体架构

### 模块拼装积木图

```
┌─────────────────────────────────────────────────────────────┐
│  🖼️  输入图像 [B, 3, 336, 336]                                 │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
                  ❄️ CLIP ViT-L/14@336px
                  (frozen, requires_grad=False)
                           ↓
            ┌──────────────┴──────────────┐
            ↓                             ↓
    🔵 cls_token [B,768]          🟦 patches [B,576,768]
            │                             │
            │                    ┌────────┴─────────┐
            │                    ↓                  ↓
            │         🟪 v2s 分支          🟧 s2v 分支
            │         文本 query 视觉      视觉 query 文本
            │         得到文本→视觉认知    得到视觉→文本认知
            │                    ↓                  ↓
            │         ┌──────────┴──────────────────┘
            │         ↓
            │    🔶 local_score [B, 200]
            │         │
            │  📚 Adapter (只动 seen 150 类)
            │         │
            │         ↓
            │    🟩 all_text [200, 768]
            │         │
            └───── ⚖️  base_logits [B, 200] = cos(cls, all_text)
                       │
                       +
                  🚪 gate × local_score   (动态融合)
                       │
                       ↓
                  🎯 logits [B, 200]
                       │
                       ↓
                  💥 CE Loss (only on seen)
```

### 关键尺寸表

| 张量 | 形状 | 说明 |
|---|---|---|
| `clip_features` | `[B, 577, 768]` | 1 个 CLS + 576 个 patch token |
| `cls_token` | `[B, 768]` | 全图语义 |
| `patches` | `[B, 576, 768]` | 24×24 空间 patch |
| `seen_text_embeds` | `[150, 768]` | 经 adapter 微调 |
| `unseen_text_embeds` | `[50, 768]` | CLIP 原始（保护零样本能力）|
| `all_text` | `[200, 768]` | 上面两者拼接 |
| `logits` 训练 | `[B, 150]` | 仅 seen 列 |
| `logits` 推理 | `[B, 200]` | 全类 |

---

## 🧪 各方案详细汇报

下面每个方案都给一个**模块卡片**，结构统一：
- 📌 想法 / 输入 / 输出
- 🔑 关键代码 (节选)
- 📊 实验结果
- 🧠 分析与教训

---

### 🌟 方案 ★：Claude 文本描述 ✅ 最大收益

**模块卡片**

| 项 | 内容 |
|---|---|
| 📌 想法 | 用 Claude Opus 4.7 替代 GPT-4 生成"6 句细节 + 1 句整体 caption"的描述 |
| 📥 输入 | CUB 类名 list（200 类）|
| 📤 输出 | `cub_claude.pt`：dict，类名 → 7 句话 |
| 🔑 修改 | `tools/generate_claude_descriptions.py` + 替换 `gpt_text_path` |

**严格对照实验**（同 epoch=15 + 真 patch + 仅 text 差异）

| 种子 | 描述源 | U | S | H | ZSL |
|---|---|---|---|---|---|
| 5 | GPT-4 | 68.55 | 73.64 | 71.00 | 80.85 |
| 5 | Claude | 71.18 | 73.24 | **72.20** | 81.67 |
| 42 | Claude | 70.88 | 73.80 | **72.31** | 81.79 |

> 🎯 Claude `mean ± std = 72.26 ± 0.08`，相对 GPT-4 **+1.26% H**

**🧠 教训**：文本端 ROI 最高，Claude 的 caption 句更接近 CLIP 训练分布。

---

### 🌟 方案 1：FAE 模块（核心组件，重点详细讲解）

#### 模块卡片

| 项 | 内容 |
|---|---|
| 📌 想法 | TransZero++ 风格的几何感知自注意力，"反向扣除"位置纠缠 |
| 📥 输入 | `x: [B, 576, 512]` 投影后的 patch 特征 |
| 📥 几何 | `geo_emb: [B, 576, 576, 64]` 预计算 |
| 📤 输出 | `x': [B, 576, 512]` 解耦位置纠缠后的 patch 表示 |
| 🔑 实现 | `BoxRelationalEmbedding` + `GeometryMultiHeadAttention` + `FAELayer` |

#### 三大组件关系（积木式）

```
  📐 BoxRelationalEmbedding (静态查找表)
  ──────────────────────────────────────
  网格 24×24 → 计算两两 patch 的相对几何
                    │
                    ▼
                正弦编码
                    │
                    ▼
       预计算 geometry_embedding [576, 576, 64]
                    │
                    │ (训练时 expand 成 [B,576,576,64])
                    ▼
  🎯 GeometryMultiHeadAttention (几何感知注意力)
  ──────────────────────────────────────
       att = QK/√d  ─  geo_weight   ★ 减号
                              ▲
            每个 head 独立 W_g · ReLU
                    │
                    ▼
       softmax → @ V → 输出投影 → Add(x) + LN
                    │
                    ▼
  🔧 FAELayer (单层 Transformer 块)
  ──────────────────────────────────────
       几何感知注意力
            │
            ▼
       FFN: 512 → 1024 → 512
            │
            ▼
       Add(残差) + LN → x'
```

#### 关键设计：**减号是核心**

```
标准多头注意力:        att = Q · K / √d
                              ↑
                           越大越关注

FAE 的注意力:        att = Q · K / √d  -  geo_weight
                                              ↑
                                          ≥ 0, 越靠近的 patch 这个值越大
                                          相减后该 patch 对越不容易被关注
```

**为什么是减号**：CLIP ViT 内部的 position embedding 让相邻 patch 特征趋同，模型容易"看身边"走捷径。FAE **扣减**几何相关度，逼模型抛开空间近邻，根据语义内容决定关注谁。对鸟类细粒度任务尤其重要——喙形 / 羽毛纹理这种跨区域判别必须靠纯语义关联。

#### 关键代码（核心：减号操作）

```python
# ★ FAE 的核心：几何感知注意力中的减号
# 标准注意力: att = QK/√d
# FAE 注意力: att = QK/√d  -  geo_weight   ← 减号扣减空间相关度

# geo_weight 由预计算的相对位置编码经 Linear + ReLU 得到 (≥ 0)
geo_weights = F.relu(torch.cat(geo_per_head, dim=1))   # [B, heads, N, N]
att = att - geo_weights                                 # 空间越近 → 被扣越多

# FAELayer 整体数据流
# 📥 x: [B, 576, 512]  geo_emb: [B, 576, 576, 64]
x = self.attn(x, geo_emb)          # 几何感知自注意力 + Add + LN
x = self.ln(x + self.dropout(      # FFN + 残差 + LN
    self.ffn(x)))
# 📤 x': [B, 576, 512]
```

#### FAE 消融实验

| 配置 | 参数量 | U | S | H | ZSL |
|---|:---:|:---:|:---:|:---:|:---:|
| ✅ 有 FAE | 10.4M | 71.18 | 73.24 | **72.20** | **81.67** |
| ❌ 无 FAE | 8.3M | 68.04 | 76.57 | 72.05 | 80.98 |

#### 🧠 诊断与教训

通过 `tools/diagnose_fae.py` 检查实际行为：

| 现象 | 数值 | 含义 |
|---|---|---|
| W_g 范数（4 个 head） | `1 个为 0.0001` | 1 个 head 几何机制完全死掉 |
| attention scale | `-5 ± 6` | 注意力分数本身很大 |
| geo_weights 平均 | `~0.1` | 几何偏置量级很小 |
| softmax 后影响 | `< 1e-4` | **几何项实际贡献可忽略** |

**结论**：FAE 几何解耦机制实测**名存实亡**，但因为它的 FFN 部分还在工作，移除会让 ZSL 跌 0.69%。所以保留模块本身，但要重新设计真正起作用的几何机制是后续课题。

---

### 方案 3-4：多 LLM 文本融合 ❌ 失败

| 项 | 内容 |
|---|---|
| 📌 想法 | 用 Claude（7 句）+ GPT-5.5（7 句）= 14 句，或加权融合 |
| 📥 输入 | 两份描述各 200 类 × 7 句 |
| 📤 输出 | 融合后的 `cub_merge.pt` |
| 🔑 修改 | `tools/sweep_text_alpha.py` |

```
方案 a: Merge (concat 14 句)            H = 71.68  (-0.52)
方案 b: α-grid (α=0.7~0.9 加权混合)     H = 71.87 max (-0.33)
方案 c: 25 epoch 长训练 α=0.9           H = 71.19 (更差)
```

**🧠 教训**：纯 Claude 是当前全局最优，融合反而稀释优势。

---

### 方案 5：Cosine-only 计分模式 ❌ 失败

#### 模块卡片

| 项 | 内容 |
|---|---|
| 📌 想法 | 不用 `base + β × local`，改用 `cos(v_enh, t_enh) × scale` 一锤定音 |
| 🔑 输出 | logits = 增强视觉 vs 增强文本 的纯余弦相似度 |

```
   v_enh:  s2v_pooled → proj_visual → 残差融合 cls_token  →  [B, 768]
   t_enh:  v2s 输出 → proj_text → 残差融合 all_text       →  [B, 200, 768]
                                                                    ↓
   ★ 关键修复 (方案 3): unseen 列直接用 all_text 绕过可训练 proj_text

   logits[b,k] = cos(v_enh[b], t_enh[b,k]) × scale
```

#### 实验结果

| 版本 | U | S | H | ZSL | 问题 |
|---|---|---|---|---|---|
| v1（暴雷） | 30.57 | 77.41 | 43.83 | 69.99 | unseen 文本经未训练 proj_text 训歪 |
| v2（unseen 绕过） | 31.22 | 80.88 | 45.05 | 77.77 | v_enh mean 池化太弱 + 无 base 兜底 |

**🧠 教训**：纯 cosine 框架在当前架构下不可行，原因 (1) mean 池化稀释 (2) 无 CLS base_logits 兜底。

---

### 方案 6：Calibration Loss ❌ 中性

| 项 | 内容 |
|---|---|
| 📌 想法 | 强迫模型给 unseen 类保留一定概率：`-log(mean(P(unseen)))` |
| 🔑 修改 | `compute_loss()` 加 `lambda_cal × loss_cal` |

| `lambda_cal` | U | S | H | ZSL |
|---|---|---|---|---|
| 0（基线）| 71.18 | 73.24 | **72.20** | 81.67 |
| 0.001 | 71.65 | 72.73 | 72.19 | 81.73 (中性) |
| 0.01 | 78.97 | 56.75 | 66.04 | 82.08 (过强) |

**🧠 教训**：模型 U/S 已均衡，cal loss 不适用。

---

### 方案 7：Random Holiday 机制 ❌ 失败

| 项 | 内容 |
|---|---|
| 📌 想法 | 训练后期随机让模型"放假"（部分 batch 做 label_smoothing） |
| 配置 | holiday_prob=0.1, start_epoch=10, smoothing=0.3 |

```
启动 holiday 前: H ≈ 69.35
启动 holiday 后: H 跳水到 62.98
```

**🧠 教训**：已收敛模型上突然 label_smoothing 产生巨大梯度冲击，全部回滚。

---

### 方案 8：Attention Pooling (s2v) ❌ 失败

#### 模块卡片

```
  当前 (mean):  s2v_pooled = F_p_s2v.mean(dim=1)   平等对待 576 patch
                ──────────────────────────
                问题: 5/6 是背景 (天空/树枝)
                信号被稀释 5 倍

  替换 (Attention):
                attn_logits = self.attn_pool(F_p_s2v)        # [B, 576, 1]
                attn_weights = F.softmax(attn_logits, dim=1)
                s2v_pooled = (F_p_s2v * attn_weights).sum(dim=1)
```

| 配置 | U | S | H | ZSL |
|---|---|---|---|---|
| Mean 基线 | 71.18 | 73.24 | **72.20** | 81.67 |
| Attention Pool | 72.11 | 71.56 | 71.83 | 81.50 |

**🧠 教训**：Attention Pool 过度聚焦少数 patch，U +0.93 但 S -1.68，整体 -0.37%。

---

### 方案 9：动态门控 v1 (gate_net MLP) ❌ 失败

#### 模块卡片

```
  cls_token [B, 768]
        │
        ▼
  Linear(768 → 192) + ReLU
        │
        ▼
  Linear(192 → 1)
        │
        ▼
  sigmoid → gate [B, 1]
        │
        ▼
  logits = base + gate × local_score
```

| 配置 | U | S | H | ZSL |
|---|---|---|---|---|
| 固定 local_weight=0.3 (基线) | 71.18 | 73.24 | **72.20** | 81.67 |
| 动态 gate_net | 64.65 | 77.85 | 70.64 | 81.62 |

**🧠 失败根因**：

```
监督盲区陷阱:
  1. gate 是 [B, 1] 全局标量, 对 200 类一视同仁
  2. CE loss 只覆盖 seen 150 类, unseen 50 类无梯度
  3. 模型学到 "压低 gate 让 base 主导, seen 准" 的 shortcut
  4. gate 塌缩成全局降权, unseen 跟着被压
```

---

### 方案 10：CIG 门控 (Confidence-Inverse Gating) 🟡 改善但未达基线

#### 模块卡片

| 项 | 内容 |
|---|---|
| 📌 想法 | gate 不学 MLP，由 CLIP 自身置信度反推：`gate = α × (1-conf)^τ` |
| 📥 输入 | `base_logits` 在全 200 类上的 max-prob |
| 📤 输出 | `gate ∈ [0, α]` 实时随每张图变化 |
| 参数 | 仅 2 个标量：`α`, `τ` |

```
  base_logits (detach, 全 200 类) → softmax → max → conf
                                                       │
                                                       ▼
                                       uncertainty = 1 - conf
                                                       │
                                                       ▼
                              gate = softplus(α) × uncertainty^softplus(τ)
                                                       │
                                                       ▼
                              logits = base + gate × local_score
```

#### 关键代码

```python
# __init__: 仅 2 个可学标量
self.gate_alpha = nn.Parameter(torch.tensor(1.0))
self.gate_tau   = nn.Parameter(torch.tensor(1.0))

# forward
with torch.no_grad():                              # 阻断 gate 反传到 base
    prob = F.softmax(base_logits, dim=-1)          # [B, 200]
    conf = prob.max(dim=-1).values                 # [B]
uncertainty = (1.0 - conf).clamp(0.0, 1.0)
alpha = F.softplus(self.gate_alpha)
tau   = F.softplus(self.gate_tau)
gate  = (alpha * uncertainty.pow(tau)).unsqueeze(-1)
logits_200 = base_logits + gate * local_score
```

#### 实验结果

| 配置 | U | S | H | ZSL |
|---|---|---|---|---|
| 基线（无门控） | 71.18 | 73.24 | **72.20** | 81.67 |
| gate_net MLP | 64.65 | 77.85 | 70.64 | 81.62 |
| **CIG** | 67.07 | 77.24 | 71.80 | 81.28 |

#### 🧠 分析

- ✅ 比 gate_net MLP 救回 **+1.16%**（说明监督盲区是部分原因）
- ❌ 仍未达基线 **-0.40%**（5 倍 std 噪声范围）
- 🔍 推断：α/τ 仍被 seen CE loss 推向"少加 local"方向，Conf-driven 思路只解决了部分问题

---

## 🎯 总体教训

```
┌─────────────────────────────────────────────────────────┐
│  1️⃣  CLIP CLS 是强基线                                    │
│      任何方案必须保留 CLS 分类能力作为下限                │
│                                                          │
│  2️⃣  文本侧 ROI 最高                                       │
│      Claude 描述 +1.26% 是目前最大单点增益                │
│                                                          │
│  3️⃣  unseen 文本必须保护                                   │
│      任何让 unseen 经过可训练层的设计都会崩               │
│                                                          │
│  4️⃣  全局可学门控陷阱                                      │
│      监督只覆盖 seen, gate 会塌缩成全局降权器             │
│      解决: 用 CLIP 置信度等"对称信号"驱动 gate            │
│                                                          │
│  5️⃣  归一化 ≠ 线性加法                                     │
│      cosine 模式和 add 模式本质不同 (非线性)              │
│                                                          │
│  6️⃣  cal loss 不适用                                        │
│      模型 U/S 已均衡, 强推 unseen 只破坏平衡              │
│                                                          │
│  7️⃣  FAE 几何解耦名存实亡                                 │
│      attention scale 失配, geo_weights 影响 < 1e-4        │
│      但 FFN 部分仍贡献 ZSL                                │
│                                                          │
│  8️⃣  收敛阈值                                              │
│      15 epoch + CosineAnnealingLR 已饱和                  │
│      加轮次解决不了结构性问题                             │
└─────────────────────────────────────────────────────────┘
```

---

## 🔮 下一步候选方向

| 优先级 | 方案 | 预期 H 增益 | 理由 |
|:---:|---|:---:|---|
| 🥇 | Class-aware Attention Pooling (方案 2c) | +0.5 ~ 1.5% | 每类独立 patch 权重，避开 mean 稀释 |
| 🥇 | unseen 也过 adapter 对照 | -1% ~ +0.5% | 验证"分叉保护"是否过度小心 |
| 🥈 | 一致性 Loss (KL between base / local) | +0.3 ~ 0.5% | 正则化，风险低 |
| 🥈 | Caption 模板 ensemble | +0.3 ~ 0.8% | CLIP 79-prompt 思路 |
| 🥉 | AWA2 / SUN 推广 | 跨数据集验证 | 论文必要性 |

---

## 📁 关键文件索引

```
DVSR/
├── model/MyModel.py                          # 主模型 (Adapter + CrossModal + VGSR)
├── tools/
│   ├── extract_features.py                   # 提取真 patch 特征缓存
│   ├── generate_claude_descriptions.py       # 生成 Claude 文本
│   ├── helper_func.py                        # GZSL 评估
│   ├── sweep_text_alpha.py                   # α-grid 文本融合
│   └── diagnose_fae.py                       # FAE 行为诊断
├── data/
│   ├── cache/                                # 真 patch 缓存
│   ├── gpt4_data/cub_claude.pt               # 当前最佳文本
│   └── gpt4_data/cub_merge.pt                # 多 LLM 合并文本（未采用）
├── train_log/CUB/                            # 历次训练日志
├── 创新指导清单/INNOVATION_RECORD.md          # 详细实验记录
├── Guide/
│   ├── PROJECT_GUIDE.md                      # 架构说明
│   ├── COSINE_ONLY_REPORT.md                 # cosine_only 失败报告
│   └── PROJECT_REPORT.md                     # 本汇报
└── train_VGSR_CUB.py                         # CUB 训练入口
```

---

## 📌 项目当前状态快照

```
✅ 当前最佳:  H = 72.26 ± 0.08% (双种子复测)
🔧 配置:      Claude 描述 + 真 patch 评估 + 15 epoch
              + add 模式 + mean 池化 + use_fae=True
              + lambda_cal=0 + gate=固定 local_weight=0.3
🌱 种子:      seed=5 (主) / seed=42 (复测)
🖥️ 环境:      Windows + dassl_clip + RTX 5070 Ti + cuda:0
📚 数据:      CUB-200-2011 (150 seen / 50 unseen)
🤖 文本:      Claude Opus 4.7 生成 (200 类 × 7 句)
```

---

*报告日期：2026-05-18*  
*总实验次数：13+（含失败）*  
*关键 commit：[GitHub MTPY](https://github.com/GetlotMoney/MTPY)*
