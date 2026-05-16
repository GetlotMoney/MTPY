# CUB GZSL 实验记录

**数据集**: CUB-200-2011 | **划分**: xlsa17 Proposed Split
**训练**: 7057张 / 150 seen类 | **测试seen**: 1764张 | **测试unseen**: 2967张 / 50 unseen类
**评估指标**: GZSL-U / GZSL-S / GZSL-H (调和均值) / ZSL

---

## 实验对比总览

| 实验 | 核心改动 | U | S | H | ZSL |
|------|---------|---|---|---|-----|
| 实验1 基线 | CLIP + Adapter + GPT，10轮 | 69.02% | 71.54% | 70.26% | 79.60% |
| 实验2 加分方式 | 串行双向，local=0.3，10轮 | 66.13% | 74.47% | 70.05% | 79.38% |
| 实验3 真正双分支 | 并行双向，local=0.3，10轮 | 66.14% | 72.88% | 69.34% | 79.14% |
| 实验4 最优配置 | 并行双向+unseen GPT，local=0.3，20轮 | 71.35% | 70.65% | 71.00% | 80.21% |
| **实验5 当前最佳** | **并行双向+unseen GPT，local=0.5，20轮** | **71.07%** | **71.04%** | **71.06%** | **80.51%** |

---

## 实验 1：基线（CLIP + Adapter + GPT）

### 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│  输入                                                            │
│  图片 [B,3,336,336]    200类文本（GPT 7句均值）                   │
└──────────┬──────────────────────────┬──────────────────────────┘
           ↓                          ↓
┌──────────────────┐      ┌───────────────────────────────────────┐
│  CLIP ViT-L/14   │      │  seen 150类 → Adapter → seen_text     │
│  (冻结，无梯度)   │      │  unseen 50类 → 原始CLIP文本            │
│  输出:            │      │  拼成 all_text [200, 768]             │
│  CLS [B,768]     │      └───────────────────────────────────────┘
└──────────┬───────┘                  │
           │                          │
           ↓                          ↓
┌─────────────────────────────────────────────────────────────────┐
│  base_logits = CLS @ all_text.T × logit_scale   [B, 200]        │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
                    训练: logits[:, seen_150]  [B, 150]
                    评估: logits              [B, 200]
                           ↓
                    CE Loss (seen 150类)

参数更新 (反向传播):
  loss → logits → logit_scale ✓
                → all_text → Adapter参数 ✓ (只有seen类文本经过Adapter)
  CLIP参数 ✗ (冻结)
  unseen文本 ✗ (不经过Adapter，无梯度)
```

### 结果

| U | S | H | ZSL |
|---|---|---|-----|
| 69.02% | 71.54% | **70.26%** | 79.60% |

最佳 @ Epoch 7

---

## 实验 2：加分方式（串行双向）

### 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│  输入                                                            │
│  图片 → CLIP(冻结) → CLS [B,768] + patches [B,576,768]          │
│  200类文本 → all_text [200,768]                                  │
└──────────┬──────────────────────────┬──────────────────────────┘
           │                          │
           ↓                          ↓
    base_logits                CrossModalTransformer
    CLS @ all_text.T           ┌─────────────────────────────┐
    [B, 200]                   │  patches → embed_cv → vis   │
    (CLIP原始，无梯度)           │  vis → FAE(几何感知) → memory│
                               │                             │
                               │  ── 串行 ──                 │
                               │  v2s: txt Q, mem KV         │
                               │  → F_p_v2s [B,200,512]      │
                               │       ↓ (串行依赖)           │
                               │  s2v: F_p_v2s Q, mem KV     │
                               │  → F_p_s2v [B,200,512]      │
                               │       ↓                     │
                               │  local_score [B,200]        │
                               └─────────────┬───────────────┘
                                             │
           ↓                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│  logits = base_logits + 0.3 × local_score   [B, 200]            │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
                    CE Loss (seen 150类)

参数更新 (反向传播):
  loss → logits → local_score → F_p_s2v → decoder_s2v ✓
                              → F_p_v2s → decoder_v2s ✓  (串行，s2v梯度流回v2s)
                              → memory  → FAE ✓
                              → vis     → embed_cv ✓
                → logit_scale ✓
                → Adapter ✓ (通过all_text里的seen_text)
  base_logits这路: CLS无梯度(CLIP冻结)，all_text通过Adapter有梯度

双向体现: s2v的Q是F_p_v2s(v2s的输出)，梯度从s2v流回v2s，串行耦合
```

### 结果

| U | S | H | ZSL |
|---|---|---|-----|
| 66.13% | 74.47% | **70.05%** | 79.38% |

最佳 @ Epoch 8

---

## 实验 3：真正双分支（并行双向）

### 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│  输入                                                            │
│  图片 → CLIP(冻结) → CLS [B,768] + patches [B,576,768]          │
│  200类文本 → all_text [200,768]                                  │
└──────────┬──────────────────────────┬──────────────────────────┘
           │                          │
           ↓                          ↓
    base_logits                CrossModalTransformer
    CLS @ all_text.T           ┌─────────────────────────────┐
    [B, 200]                   │  patches → embed_cv → vis   │
                               │  vis → FAE(几何感知) → memory│
                               │            │                 │
                               │     ┌──────┴──────┐         │
                               │     ↓             ↓         │
                               │  v2s分支       s2v分支       │
                               │  txt Q         mem Q        │
                               │  mem KV        txt KV       │
                               │  ↓             ↓            │
                               │  F_p_v2s       F_p_s2v      │
                               │  [B,200,512]   [B,576,512]  │
                               │  ↓             ↓            │
                               │  score_v2s     score_s2v    │
                               │  [B,200]       [B,200]      │
                               │     └──────┬──────┘         │
                               │            ↓                │
                               │  local_score = 0.5*s2v      │
                               │             + 0.5*v2s       │
                               └─────────────┬───────────────┘
                                             │
           ↓                                 ↓
┌─────────────────────────────────────────────────────────────────┐
│  logits = base_logits + 0.3 × local_score   [B, 200]            │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
                    CE Loss (seen 150类)

参数更新 (反向传播):
  loss → logits → local_score
                    ├→ score_v2s → F_p_v2s → decoder_v2s ✓
                    │                      → memory → FAE ✓ ← 两路梯度汇聚
                    └→ score_s2v → F_p_s2v → decoder_s2v ✓
                                           → memory → FAE ✓ ← 两路梯度汇聚
                → logit_scale ✓
                → Adapter ✓

双向体现:
  v2s: 文本 Query 视觉 → 学"文本对应哪些视觉区域"
  s2v: 视觉 Query 文本 → 学"视觉区域对应哪些语义"
  FAE同时接收两路梯度，被双向约束，学到更通用的视觉表示
  两路并行独立，互不依赖
```

### 结果

| U | S | H | ZSL |
|---|---|---|-----|
| 66.14% | 72.88% | **69.34%** | 79.14% |

最佳 @ Epoch 8

---

## 实验 4：最优配置（unseen GPT + local=0.3 + 20轮）

### 架构图（与实验3相同，改动在数据侧）

```
┌─────────────────────────────────────────────────────────────────┐
│  关键改动（相比实验9）                                            │
│                                                                 │
│  实验4:  unseen_text = class_text_embeds[unseenclass]           │
│          ← 类名模板，1句话，信息量少                              │
│                                                                 │
│  实验5: unseen_text = gpt_text_embeds[unseenclass]             │
│          ← GPT 7句描述均值，语义更丰富                           │
│                                                                 │
│  训练轮数: 10轮 → 20轮（峰值在Epoch 16）                         │
└─────────────────────────────────────────────────────────────────┘

数据流（与实验9完全相同，只是 unseen_text 质量更高）:

图片 → CLIP(冻结) → CLS + patches
                         ↓
all_text:
  seen[150]  = Adapter(GPT文本)    ← 训练优化
  unseen[50] = GPT文本(无Adapter)  ← CLIP原始，更丰富的语义
                         ↓
base_logits = CLS @ all_text.T
local_score = CrossModalTransformer(patches, all_text)
logits = base_logits + 0.3 × local_score
                         ↓
              CE Loss (seen 150类)
```

### 结果

| U | S | H | ZSL |
|---|---|---|-----|
| 71.35% | 70.65% | **71.00%** | 80.21% |

最佳 @ Epoch 16，**首次超过基线**

---

## 实验 5：当前最佳（local=0.5 + 20轮）

### 架构图（与实验4相同，只改 local_weight）

```
┌─────────────────────────────────────────────────────────────────┐
│  关键改动（相比实验4）                                           │
│                                                                 │
│  local_weight: 0.3 → 0.5                                        │
│  logits = base_logits + 0.5 × local_score                       │
│                                                                 │
│  双分支贡献更大，U/S 更均衡                                       │
└─────────────────────────────────────────────────────────────────┘

完整数据流:

图片 [B,3,336,336]
    ↓ CLIP ViT-L/14@336px (冻结)
CLS [B,768] + patches [B,576,768]
    │                    │
    │              ┌─────┴──────────────────────────────────────┐
    │              │         CrossModalTransformer               │
    │              │                                             │
    │              │  embed_cv(768→512)                          │
    │              │  patches → vis [B,576,512]                  │
    │              │  FAE(几何感知自注意力)                        │
    │              │  vis → memory [B,576,512]                   │
    │              │                                             │
    │              │  embed_text(768→512)                        │
    │              │  all_text → txt_com [200,512]               │
    │              │  txt_batch = expand → [B,200,512]           │
    │              │                                             │
    │              │  ┌─── v2s ────┐  ┌─── s2v ────┐           │
    │              │  │ Q=txt_batch│  │ Q=memory   │           │
    │              │  │ KV=memory  │  │ KV=txt_batch│           │
    │              │  │ ↓          │  │ ↓           │           │
    │              │  │F_p_v2s     │  │F_p_s2v      │           │
    │              │  │[B,200,512] │  │[B,576,512]  │           │
    │              │  │ ↓          │  │ ↓ mean(1)   │           │
    │              │  │score_v2s   │  │score_s2v    │           │
    │              │  │[B,200]     │  │[B,200]      │           │
    │              │  └─────┬──────┘  └──────┬──────┘           │
    │              │        └────────┬────────┘                  │
    │              │    local_score = 0.5*s2v + 0.5*v2s          │
    │              │    [B, 200]                                  │
    │              └─────────────────┬──────────────────────────┘
    │                                │
    ↓                                ↓
base_logits = CLS @ all_text.T × scale    local_score [B,200]
[B, 200]  (CLIP原始，无梯度)
    │                                │
    └────────────────┬───────────────┘
                     ↓
         logits = base_logits + 0.5 × local_score
         [B, 200]
                     ↓
         训练: logits[:, seen_150] [B,150]
                     ↓
              CE Loss → 反向传播

反向传播路径:
  loss
   ↓
  logits
   ├─→ base_logits: CLS无梯度(CLIP冻结)
   │                all_text → Adapter参数 ✓ (seen类)
   │                logit_scale ✓
   └─→ local_score
        ├─→ score_v2s → F_p_v2s → decoder_v2s ✓
        │                        → memory → FAE ✓ ←┐ 两路梯度
        └─→ score_s2v → F_p_s2v → decoder_s2v ✓    │ 同时更新
                                 → memory → FAE ✓ ←┘ FAE参数
                                 → embed_cv ✓
                       txt_com  → embed_text ✓

双向约束:
  v2s梯度: loss→score_v2s→F_p_v2s→decoder_v2s→memory→FAE
  s2v梯度: loss→score_s2v→F_p_s2v→decoder_s2v→memory→FAE
  FAE同时被两路约束 → 学到对两种交互都有效的视觉表示
```

### 结果

| U | S | H | ZSL |
|---|---|---|-----|
| 71.07% | 71.04% | **71.06%** | 80.51% |

最佳 @ Epoch 20，**当前最佳结果**

---

## 关键结论

1. **CLIP 的 CLS token 不能修改**：任何改变 CLS 的操作都会破坏 CLIP 语义空间，ZSL 大幅下降
2. **加分方式是正确的**：`logits = base_logits + β × local_score`，CLIP 能力完整保留
3. **unseen 用 GPT 描述有效**：ZSL 从 79.60% 提升到 80.51%，unseen 类语义更丰富
4. **20 轮比 10 轮好**：模型需要更多轮次收敛，峰值在 Epoch 16~20
5. **local_weight=0.5 最优**：U/S 极度均衡（71.07% vs 71.04%），H=71.06%
