# DVSR / VGSR 项目工作汇报

> CLIP + Adapter + 跨模态 Transformer 的零样本细粒度鸟类识别

---

## ⚡ 一句话总结

🎯 **当前最佳结果**：CUB GZSL `H = 72.48%`（LaSt-ViT 频域池化, 单 run）
🥈 **稳定基线**：`H = 72.24%`（mean 池化, 已复现）  
🚀 **累计提升**：基线 70.35% → 72.48%，共 **+2.13%**  
🔬 **确认有效的改进**：① Claude 文本描述（+1.26%）② 真 patch 评估缓存修复（+0.7%）③ LaSt-ViT 池化（+0.24%）

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
| 11 | Cons + L2SP + Aug + 50ep | 🔴 崩盘 | 72.01 → 34.38 | -37.86 | ❌ |
| 12 | 基线复现（mean 池化）| 🟢 复现成功 | **72.24** | -0.02 | ✅ 锚点 |
| 🥇 13 | **LaSt-ViT 频域池化 (k=8)** | 🟢 **当前最佳** | **72.48** | **+0.24** | ✅ 已采用 |
| 14 | LaSt-ViT k=8 长训练 30ep | 🟡 进行中 | ? | ? | ? |
| 15 | LaSt-ViT K-grid (k=4/16/32) | 🟡 待做 | 期望 +0.3~0.8 | - | - |
| 16 | FAE 几何尺度修复 | 🟡 待做 | 期望 ZSL +0.5~1 | - | - |
| 17 | Class-aware Attention Pool | 🟡 待做 | 期望 +0.5~1.5 | - | - |
| 21 | Cosine-only v3 (流形共享+detach) | 🟡 待做 | 待测 | - | - |

> 🟢 成功 / 🟡 中性 or 进行中 / 🔴 显著回退

---

## 🏗️ 整体架构（当前最佳配置）

### 模块拼装积木图

```
┌─────────────────────────────────────────────────────────────┐
│  🖼️  输入图像 [B, 3, 336, 336]                                │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
                  ❄️ CLIP ViT-L/14@336px
                  (frozen, requires_grad=False)
                           ↓
            ┌──────────────┴──────────────┐
            ↓                             ↓
    🔵 cls_token [B,768]          🟦 patches [B,576,768]
            │                             │
            │                  embed_cv (768→512)
            │                             │
            │                       FAE 几何感知
            │                             │
            │                      memory [B,576,512]
            │                             │
            │              ┌──────────────┴──────────────┐
            │              ↓                              ↓
            │       🟪 v2s 分支                    🟧 s2v 分支
            │       文本 query 视觉                 视觉 query 文本
            │       F_p_v2s [B,200,512]             F_p_s2v [B,576,512]
            │              │                              │
            │              ↓                       ★ LaSt-ViT 频域 Top-K
            │       score_v2s [B,200]              选 8 个高频前景 patch
            │              │                              ↓
            │              │                     s2v_pooled [B,512]
            │              │                              │
            │              │                     score_s2v [B,200]
            │              └────────────┬─────────────────┘
            │                           ↓
            │                  local_score [B, 200]
            │                       (0.5×s2v + 0.5×v2s)
            │                           │
            │  📚 Adapter (只动 seen 150 类)
            │           │
            │           ↓
            │      all_text [200, 768]
            │           │
            └───── ⚖️  base_logits = cos(cls, all_text)
                       │
                       +
                  🚪 0.3 × local_score   (gating='fixed')
                       │
                       ↓
                  🎯 logits [B, 200]
                       │
                       ↓
                  💥 CE Loss only (lambda_cal=consist=l2sp=0)
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
| `F_p_s2v` | `[B, 576, 512]` | s2v 分支输出 |
| `s2v_pooled (LaSt)` | `[B, 512]` | LaSt-ViT 选 top-8 patch 平均 |
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

### 🌟 方案 ★：Claude 文本描述 ✅ 最大单点收益

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

### 🥇 方案 13：LaSt-ViT 频域 Top-K 池化 ✅ 当前最佳

#### 模块卡片

| 项 | 内容 |
|---|---|
| 📌 想法 | 用频域分析无参数地选"高频前景" patch，跳过背景稀释 |
| 📥 输入 | `F_p_s2v: [B, 576, 512]` s2v 分支输出 |
| 📤 输出 | `s2v_pooled: [B, 512]` 池化后的全局视觉 |
| 🔑 实现 | `lastvit_pool(F_p, k=8, sigma=10.0)` 工具函数 |
| 🔧 论文 | LAST-ViT (ChengShiest/LAST-ViT) |

#### 算法流程

```
F_p_s2v [B, 576, 512]
   │
   ├── FFT 沿特征维 (dim=-1)
   │
   ├── 高斯低通滤波 (σ=10.0) → x_lp
   │
   ├── 频域差分打分: score = F_p / |F_p - x_lp|
   │   patch_scores = score.mean(-1)   [B, 576]
   │
   ├── topk(scores, k=8) → 选 8 个最重要的 patch
   │
   └── 这 8 个 patch 平均 → [B, 512]
```

**核心思想**：高频残差稳定的 patch = 含丰富纹理 = 前景；规则与具体类无关，**对 seen/unseen 同等友好**（无可学参数）。

#### 关键代码

```python
def lastvit_pool(F_p, k=8, sigma=10.0):
    B, N, D = F_p.shape
    # 1. FFT 沿特征维
    x_freq = torch.fft.fft(F_p, dim=-1)
    # 2. 高斯低通
    gs_k = _gaussian_kernel_1d(D, sigma).to(F_p.device)
    x_freq = torch.fft.fftshift(x_freq, dim=-1) * gs_k
    x_freq = torch.fft.ifftshift(x_freq, dim=-1)
    x_lp = torch.fft.ifft(x_freq, dim=-1).real
    # 3. 打分
    diff = F_p / (torch.abs(x_lp - F_p) + 1e-6)
    patch_scores = diff.mean(dim=-1)                  # [B, N]
    # 4. 选 top-K
    _, idx = torch.topk(patch_scores, k=k, dim=1)
    sel = torch.gather(F_p, 1, idx.unsqueeze(-1).expand(-1, -1, D))
    # 5. K 个 patch 平均
    return sel.mean(dim=1)                            # [B, D]
```

#### 实验结果（vs mean 基线，单变量对照）

| 配置 | U | S | H | ZSL |
|---|---|---|---|---|
| mean (基线) | 70.04 | 74.58 | **72.24** | 81.66 |
| **LaSt-ViT k=8 (新)** | 67.85 | **77.79** | **72.48** | 81.62 |
| Δ | **−2.19** | **+3.21** | **+0.24** | -0.04 |

**关键观察**：
- ✅ H +0.24%（真实涨，但接近噪声边缘）
- ✅ S +3.21%（lastvit 帮模型抓 seen 类的判别 patch）
- ❌ U −2.19%（unseen 类的全局形态丢失）
- 🔍 训练 epoch 14→15 仍 H 在涨（72.37 → 72.48），未收敛

**和 Attention Pool 对比**（都是失败/边缘）：

| 池化 | U | S | H | 失败模式 |
|---|---|---|---|---|
| mean (基线) | 70.04 | 74.58 | 72.24 | 中庸 |
| Attention Pool (旧) | 72.11 | 71.56 | 71.83 | **U↑ S↓** (相反) |
| **LaSt-ViT k=8 (新)** | 67.85 | 77.79 | **72.48** | **U↓ S↑** |

**🧠 分析**：
- LaSt 选 8 个"高频纹理 patch"，偏向局部细节
- seen 类有训练监督 → 学到关键 patch（S 涨）
- unseen 类无监督 → 全局形态丢失（U 跌）
- 但**因为是无参数规则**，比 attention pool 失败模式好

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

**为什么是减号**：CLIP ViT 内部的 position embedding 让相邻 patch 特征趋同，模型容易"看身边"走捷径。FAE **扣减**几何相关度，逼模型抛开空间近邻，根据语义内容决定关注谁。

#### 关键代码

```python
geo_weights = F.relu(torch.cat(geo_per_head, dim=1))   # [B, heads, N, N]
att = att - geo_weights                                # 空间越近 → 被扣越多
```

#### FAE 消融实验

| 配置 | 参数量 | U | S | H | ZSL |
|---|:---:|:---:|:---:|:---:|:---:|
| ✅ 有 FAE | 10.4M | 71.18 | 73.24 | **72.20** | **81.67** |
| ❌ 无 FAE | 8.3M | 68.04 | 76.57 | 72.05 | 80.98 |

#### 🧠 已诊断未修复的 bug

通过 `tools/diagnose_fae.py` 检查实际行为：

| 现象 | 数值 | 含义 |
|---|---|---|
| W_g 范数（4 个 head） | `1 个为 0.0001` | 1 个 head 几何机制完全死掉 |
| attention scale | `-5 ± 6` | 注意力分数本身很大 |
| geo_weights 平均 | `~0.1` | 几何偏置量级很小 |
| softmax 后影响 | `< 1e-4` | **几何项实际贡献可忽略** |

**结论**：FAE 几何解耦机制实测**名存实亡**，但 FFN 部分仍贡献 ZSL（移除会让 ZSL 跌 0.69%）。

**待做（方案 16）**：加可学 `geo_scale` 让 geo_weights 真正起作用。

---

### 方案 3-4：多 LLM 文本融合 ❌ 失败

| 项 | 内容 |
|---|---|
| 📌 想法 | 用 Claude（7 句）+ GPT-5.5（7 句）= 14 句，或加权融合 |
| 📥 输入 | 两份描述各 200 类 × 7 句 |
| 📤 输出 | 融合后的 `cub_merge.pt` |

```
方案 a: Merge (concat 14 句)            H = 71.68  (-0.52)
方案 b: α-grid (α=0.7~0.9 加权混合)     H = 71.87 max (-0.33)
方案 c: 25 epoch 长训练 α=0.9           H = 71.19 (更差)
```

**🧠 教训**：纯 Claude 是当前全局最优，融合反而稀释优势。

---

### 方案 5：Cosine-only 计分模式 ❌ 失败（v1/v2）

#### 模块卡片#### 实验结果

| 版本 | U | S | H | ZSL | 问题 |
|---|---|---|---|---|---|
| v1（暴雷） | 30.57 | 77.41 | 43.83 | 69.99 | unseen 文本经未训练 proj_text 训歪 |
| v2（unseen 完全绕过） | 31.22 | 80.88 | 45.05 | 77.77 | seen/unseen 流形不匹配，余弦失真 |
| **v3（待做：流形共享+detach）** | ? | ? | ? | ? | 让 unseen 也过 proj_text 但 detach 截断输入梯度 |

**🧠 教训**：纯 cosine 框架在当前架构下不可行，原因 (1) v_enh mean 池化稀释 (2) 无 base_logits 兜底 (3) seen/unseen 流形不匹配。

**改进方向（方案 21）**：

```python
# v2 (当前): unseen 完全绕过 proj_text
t_enh[:, unseenclass, :] = all_text[unseenclass, :]   # CLIP 原始, A 区
t_enh[:, seenclass, :]   = proj_text(F_seen) 残差融合  # B 区, 流形错位

# v3 (待做): unseen 也过 proj_text, 但 detach 截断输入梯度
F_unseen_d = F_p_v2s[:, unseenclass, :].detach()
t_enh_unseen = proj_text(F_unseen_d) 残差融合          # 也在 B 区, 流形对齐
                                                      # 但不引入 unseen 监督信号
```

合规性：detach 阻断 unseen 输入梯度，proj_text 参数仍只被 seen CE 推训练，符合 GZSL "unseen 不参与训练监督"的规则。

---

### 方案 6：Calibration Loss ❌ 中性

| 📌 想法 | 强迫模型给 unseen 类保留概率 `-log(mean(P(unseen)))` |

| `lambda_cal` | U | S | H | ZSL |
|---|---|---|---|---|
| 0（基线）| 71.18 | 73.24 | **72.20** | 81.67 |
| 0.001 | 71.65 | 72.73 | 72.19 | 81.73 (中性) |
| 0.01 | 78.97 | 56.75 | 66.04 | 82.08 (过强) |

**🧠 教训**：模型 U/S 已均衡，cal loss 不适用。

---

### 方案 7：Random Holiday 机制 ❌ 失败

| 📌 想法 | 训练后期随机让模型"放假"（部分 batch 做 label_smoothing） |

```
启动 holiday 前: H ≈ 69.35
启动 holiday 后: H 跳水到 62.98
```

**🧠 教训**：已收敛模型上突然 label_smoothing 产生巨大梯度冲击，全部回滚。

---

### 方案 8：Attention Pooling (s2v) ❌ 失败

```python
attn_logits  = self.attn_pool(F_p_s2v)        # Linear 512→1
attn_weights = F.softmax(attn_logits, dim=1)
s2v_pooled   = (F_p_s2v * attn_weights).sum(dim=1)
```

| 配置 | U | S | H | ZSL |
|---|---|---|---|---|
| Mean 基线 | 71.18 | 73.24 | **72.20** | 81.67 |
| Attention Pool | 72.11 | 71.56 | 71.83 | 81.50 |

**🧠 教训**：Attention Pool 过度聚焦少数 patch，U +0.93 但 S -1.68，整体 -0.37%。

**和 LaSt-ViT 的对比**：
- Attention Pool 是**可学参数**，被 seen 监督推偏 → S 反而跌
- LaSt-ViT 是**无参数频域规则** → S 涨（因为高频本就利于细粒度）

---

### 方案 9：动态门控 v1 (gate_net MLP) ❌ 失败

```python
gate = sigmoid(MLP(cls_token))                # [B, 1]
logits = base + gate × local_score
```

| 配置 | U | S | H | ZSL |
|---|---|---|---|---|
| 固定 local_weight=0.3 (基线) | 71.18 | 73.24 | **72.20** | 81.67 |
| 动态 gate_net | 64.65 | 77.85 | 70.64 | 81.62 |

**🧠 失败根因 — 监督盲区陷阱**：
1. gate 是 [B, 1] 全局标量，对 200 类一视同仁
2. CE loss 只覆盖 seen 150 类，unseen 50 类无梯度
3. 模型学到 "压低 gate 让 base 主导, seen 准" 的 shortcut
4. gate 塌缩成全局降权, unseen 跟着被压

---

### 方案 10：CIG 门控 (Confidence-Inverse Gating) 🟡 改善但未达基线

| 📌 想法 | gate 不学 MLP，由 CLIP 自身置信度反推：`gate = α × (1-conf)^τ` |
| 参数 | 仅 2 个标量：`α`, `τ` |

```python
with torch.no_grad():
    prob = F.softmax(base_logits, dim=-1)
    conf = prob.max(dim=-1).values
uncertainty = (1.0 - conf).clamp(0.0, 1.0)
gate = (F.softplus(α) * uncertainty.pow(F.softplus(τ))).unsqueeze(-1)
logits = base + gate × local_score
```

| 配置 | U | S | H | ZSL |
|---|---|---|---|---|
| 基线（无门控） | 71.18 | 73.24 | **72.20** | 81.67 |
| gate_net MLP | 64.65 | 77.85 | 70.64 | 81.62 |
| **CIG** | 67.07 | 77.24 | 71.80 | 81.28 |

**🧠 分析**：
- ✅ 比 gate_net MLP 救回 +1.16%（说明监督盲区是部分原因）
- ❌ 仍未达基线 -0.40%
- 🔍 推断：α/τ 仍被 seen CE loss 推向"少加 local"方向

---

### 方案 11：Cons + L2SP + Aug 三合一 (50ep) 🔴 崩盘

**配置**: lambda_consist=0.1 + lambda_l2sp=0.0005 + use_aug_cache=True (K=2) + epochs=50

| Epoch | U | S | H | Cons Loss |
|---|---|---|---|---|
| 5 | 69.26 | 67.55 | 68.39 | ~3.0 |
| **10 (Best)** | **69.87** | **74.29** | **72.01** | ~5.4 |
| 20 | 80.23 | 39.68 | 53.10 | ~9.7 |
| 30 | 81.56 | 18.25 | 29.83 | ~12.0 |
| 50 | 81.42 | 21.79 | 34.38 | ~13.0 |

**🧠 失败根因**：
1. KL 一致性 loss 量级失控（1.0 → 13.0，13倍）
2. Cons 项贡献完全盖过 CE（epoch 50: 0.1×13=1.3 vs CE=0.05）
3. 模型转而优化"local 模仿 base"，不再优化分类
4. 三改动同时上无法定位元凶

**教训**：长训练下 KL loss 必须 clip 或大幅降权（≤0.01）；多变量同时改的代价巨大。

---

### 方案 12：基线复现（mean 池化）⭐ 锚点

**目的**：用整理后的代码重新跑 H=72.26% 基线，作为后续所有实验的对照锚点。

| 配置 | 值 |
|---|---|
| epochs | 15 |
| seed | 5 |
| pool_method | mean |
| gating | fixed |
| local_weight | 0.3 |
| 所有 lambda | 0 |
| use_aug_cache | False |

**结果**：
```
Best @ Epoch 15: U=70.04, S=74.58, H=72.24, ZSL=81.66
```

vs 历史基线 H=72.26 偏差 -0.02%（噪声内）。**完美复现，代码改动未破坏行为**。

模型快照：`baseline_H7224_seed5_FIXED.pth`（永久备份）

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
│      让 unseen 经过未训练的可学层会崩 (cosine_only v1)    │
│      让 unseen 完全绕过又造成流形错位 (v2)                │
│      解决: 流形共享 + detach (v3, 待实施)                 │
│                                                          │
│  4️⃣  全局可学门控陷阱                                      │
│      监督只覆盖 seen, gate 会塌缩成全局降权器             │
│      解决: 用 CLIP 置信度等"对称信号"驱动 gate            │
│                                                          │
│  5️⃣  归一化 ≠ 线性加法                                     │
│      cosine 模式和 add 模式本质不同 (非线性)              │
│                                                          │
│  6️⃣  cal loss 不适用                                       │
│      模型 U/S 已均衡, 强推 unseen 只破坏平衡              │
│                                                          │
│  7️⃣  FAE 几何解耦名存实亡                                 │
│      attention scale 失配, geo_weights 影响 < 1e-4        │
│      但 FFN 部分仍贡献 ZSL, 待修复 (方案 16)              │
│                                                          │
│  8️⃣  KL 一致性 loss 长训练会爆炸                           │
│      量级失控时, λ × Cons 可超过 CE → 模型优化方向跑偏     │
│                                                          │
│  9️⃣  无参规则 > 可学权重                                  │
│      LaSt-ViT 无参频域选择 比 Attention Pool 可学权重好   │
│      (无监督偏差, GZSL 友好)                              │
│                                                          │
│  🔟  多变量同时改无法定位                                 │
│      原则: 一次只改一个变量                               │
└─────────────────────────────────────────────────────────┘
```

---

## 🔮 下一步候选方向（按优先级）

| 优先级 | 方案 | 预期 H 增益 | 理由 |
|:---:|---|:---:|---|
| 🥇 | LaSt-ViT 30ep（实验 14, 进行中） | +0.3~0.5% | epoch 14→15 仍涨, 训练未收敛 |
| 🥇 | LaSt-ViT K-grid (k=4/16/32) | +0.3~0.8% | 找最佳 K, 看 U 能否回涨 |
| 🥈 | FAE 几何尺度修复 (方案 16) | ZSL +0.5~1% | 已诊断的 bug, 可学 geo_scale |
| 🥈 | Class-aware Attention Pooling | +0.5~1.5% | 每类独立 patch 权重, 无可学参数 |
| 🥉 | gzsl_bias 网格搜索 | +0.3~0.8% | 0 代码改动, 5 次实验 |
| � | unseen 也过 adapter 对照 | -1~+0.5% | 验证"分叉保护"是否过度小心 |
| � | Multi-seed ensemble | +0.2~0.4% | 5/42/2024 三种子加权 |
| 🟡 | Cosine-only v3（流形共享+detach）| 待测 | 修复 v1/v2 失败, 论文卖点强 |
| 🟡 | AWA2 / SUN 推广 | 跨数据集验证 | 论文必要性 |

---

## 📁 关键文件索引

```
DVSR/
├── EXPERIMENT_RECORD.md                      # 实验数据流水账（每次必填）
├── 创新指导清单/INNOVATION_RECORD.md         # 创新方案与失败教训
├── Guide/
│   ├── PROJECT_REPORT.md                     # 本汇报（模块技术细节）
│   ├── PROJECT_GUIDE.md                      # 架构说明
│   └── COSINE_ONLY_REPORT.md                 # cosine_only 失败专项报告
├── .kiro/docs/conversation_log.md            # 对话历史摘要
├── model/MyModel.py                          # 主模型
├── train_VGSR_CUB.py                         # CUB 训练入口（含 resume 系统）
├── tools/
│   ├── extract_features.py                   # 提取真 patch 缓存
│   ├── generate_claude_descriptions.py       # 生成 Claude 文本
│   ├── helper_func.py                        # GZSL 评估
│   ├── sweep_text_alpha.py                   # α-grid 文本融合
│   └── diagnose_fae.py                       # FAE 行为诊断
├── data/
│   ├── cache/                                # 真 patch 缓存
│   └── gpt4_data/cub_claude.pt               # 当前最佳文本
├── train_log/CUB/
│   ├── training_log_CUB_*.txt                # 每次训练日志
│   ├── best_model_CUB_*_H*.pth               # 带 H 后缀的最佳权重
│   ├── ckpt_full_CUB_*.pth                   # 完整 ckpt（含 optimizer 用于 resume）
│   └── baseline_H7224_seed5_FIXED.pth        # 永久基线快照 ⭐
└── config/VGSR_cub_gzsl.yaml                 # 训练配置
```

---

## 📌 项目当前状态快照

```
🥇 当前最佳:   H = 72.48% (LaSt-ViT k=8, 单 run, epoch 15)
🥈 稳定基线:   H = 72.24% (mean 池化, 复现自历史 72.26%)

🔧 当前配置 (yaml):
   text_source       = claude            ← 当前最佳文本
   pool_method       = lastvit            ← LaSt-ViT 频域池化
   lastvit_k         = 8
   lastvit_sigma     = 10.0
   gating            = fixed
   local_weight      = 0.3
   score_mode        = add
   use_fae           = True
   use_aug_cache     = False              ← 单视角
   model_mode        = vgsr               ← 全功能
   epochs            = 30                 ← 当前正在跑
   random_seed       = 5
   lambda_cal        = 0
   lambda_consist    = 0
   lambda_l2sp       = 0
   resume_from       = ''                 ← 从头训练

🌱 种子:        seed=5 (主) / seed=42 (复测)
🖥️ 环境:        Windows + dassl_clip + RTX 5070 Ti + cuda:0
📚 数据:        CUB-200-2011 (150 seen / 50 unseen)
🤖 文本:        Claude Opus 4.7 生成 (200 类 × 7 句)
💾 缓存:        真 patch 单视角 (~10GB) + 多视角已删
🎛️ 已实施开关:   gating / score_mode / pool_method / use_fae /
                use_aug_cache / model_mode / resume_from
```

---

## 🆕 已实施的资源管理改进

| 改进 | 说明 |
|---|---|
| **模块快照打印** | 每次训练日志开头打印完整模块配置（pool/gating/loss/resume）|
| **Resume 系统** | yaml 加 `resume_from`/`resume_lr_schedule`/`extra_epochs`，支持续训 |
| **完整 checkpoint** | 保存 `ckpt_full_*.pth` 含 optimizer/scheduler/epoch 状态 |
| **权重 H 后缀** | `best_model_*_H7248.pth` 文件名带 H 值，跨 run 排序 |
| **GitHub 双标签** | `v1-baseline` + `v2-cig-loss-aug` 两个版本可独立 clone |

---

*报告日期：2026-05-24*  
*总实验次数：13 (含失败 + 待做)*  
*关键 commit：[GitHub MTPY](https://github.com/GetlotMoney/MTPY)*
