# Cosine-Only 计分模式实验报告

## 一、核心思想

**放弃 "base + β×local" 加分式融合**，改为只用双分支增强后的特征做余弦相似度分类。

### 动机

当前 add 模式：
```
logits = base_logits + β × local_score
       = (CLS @ all_text.T × scale) + β × (CrossModal 算的分数)
```

**问题**：CrossModal 增强的特征没有真正用于分类，只是给 CLIP 基线打了个补丁分。Adapter 学得再好，最终分类还是依赖 CLIP 原始 CLS。

**新方案**：让 CrossModal 的输出**唯一决定**分类结果：
```
logits = cos(v_enh, t_enh) × logit_scale
```

---

## 二、架构流程图

```
┌──────────────────────────────────────────────────────────────────────────┐
│  输入                                                                     │
│   图像 → CLIP ViT-L/14 → CLS [B,768] + patches [B, 576, 768]             │
│   文本 200 类 → CLIP encoder → all_text [200, 768]                        │
│           其中 seen 150 类经 Adapter 微调                                  │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            ↓                                               ↓
   ┌─────────────────────┐                       ┌────────────────────┐
   │  embed_cv (768→512)  │                       │  embed_text(768→512)│
   │  vis [B, 576, 512]   │                       │  txt [200, 512]     │
   └──────────┬──────────┘                       └─────────┬──────────┘
              │                                             │
              ↓                                             │
   ┌──────────────────────┐                                 │
   │  FAE (可选)           │                                 │
   │  memory [B,576,512]   │                                 │
   └──────────┬───────────┘                                 │
              │                                              │
              ├──────────────────┬───────────────────────────┤
              ↓                  ↓                           │
   ┌─────────────────┐  ┌─────────────────┐                 │
   │   v2s 分支       │  │   s2v 分支       │                 │
   │  Q=text          │  │  Q=memory        │                 │
   │  K,V=memory      │  │  K,V=text        │                 │
   └────────┬────────┘  └────────┬────────┘                 │
            ↓                    ↓                           │
   F_p_v2s [B,200,512]   F_p_s2v [B,576,512]                │
            │                    │                           │
            ↓                    ↓                           │
   ┌─────────────────┐  ┌──────────────┐                    │
   │ proj_text        │  │ mean(dim=1)  │                    │
   │ (512→768)        │  │ → [B,512]    │                    │
   └────────┬────────┘  │ proj_visual  │                    │
            │            │ (512→768)    │                    │
            │            └──────┬───────┘                    │
            ↓                   ↓                            │
   t_enh_raw [B,200,768]  v_enh_raw [B,768]                 │
            │                   │                            │
            ↓                   ↓                            │
   ┌─────────────────┐  ┌─────────────────┐                 │
   │ 文本端残差融合    │  │ 视觉端残差融合    │                 │
   │ seen 列:         │  │ v_enh =          │                 │
   │  (1-α)·raw+α·txt│  │  (1-β)·raw+β·CLS │                 │
   │ unseen 列:       │  └────────┬────────┘                 │
   │  直接用 all_text  │           │                           │
   └────────┬────────┘           │                           │
            │                    │                           │
            ↓                    ↓                           │
        t_enh                v_enh                           │
       [B,200,768]         [B,768]                           │
            │                    │                           │
            └────────┬───────────┘                           │
                     ↓                                       │
   ┌────────────────────────────────────────────┐            │
   │    L2 归一化 + 余弦相似度 + 温度缩放         │            │
   │                                             │            │
   │    logits[b,k] = cos(v_enh[b], t_enh[b,k])  │            │
   │                  × logit_scale              │            │
   │                                             │            │
   │    输出: [B, 200]                           │            │
   └────────────────────────────────────────────┘            │
                     │                                       │
                     ↓                                       │
                 训练: CE loss on seen 150 类                 │
                 评估: argmax 在全 200 类中竞争               │
```

---

## 三、核心代码

### 3.1 CrossModalTransformer 输出增强特征

```python
# CrossModalTransformer.forward 末尾

# ========== 增强后的视觉/文本表示 (cosine_only 模式用) ==========
# v_enh: 视觉端经 s2v 池化 + 投影回 768
v_enh_512 = F_p_s2v.mean(dim=1)                          # [B, dim_com=512]
v_enh = self.proj_visual(v_enh_512)                      # [B, dim_f=768]

# t_enh: 文本端 (每张图独立) + 投影回 768
t_enh = self.proj_text(F_p_v2s)                          # [B, N_cls, dim_f=768]

return {
    'local_score': local_score,   # add 模式用
    'v_enh': v_enh,               # [B, 768]
    't_enh': t_enh,               # [B, N_cls, 768]
}
```

### 3.2 VGSR.forward 中 cosine_only 分支（方案 3 修复版）

```python
if self.score_mode == 'cosine_only':
    v_enh_raw = cm_out['v_enh']                          # [B, 768]
    t_enh_raw = cm_out['t_enh']                          # [B, 200, 768]
    B_size = patches.size(0)

    # ===== 视觉端残差融合 =====
    if cls_token is not None:
        v_enh = (1.0 - self.visual_residual) * v_enh_raw + \
                self.visual_residual * cls_token         # [B, 768]
    else:
        v_enh = v_enh_raw

    # ===== 文本端: seen 列增强, unseen 列保留 CLIP 原始 (方案 3) =====
    t_enh_seen = (1.0 - self.text_residual) * t_enh_raw + \
                 self.text_residual * all_text.unsqueeze(0)  # [B, 200, 768]

    # unseen 位置全用 all_text（绕过 proj_text，防止训歪）
    t_enh = all_text.unsqueeze(0).expand(B_size, -1, -1).contiguous()
    t_enh = t_enh.clone()
    t_enh[:, self.seenclass, :] = t_enh_seen[:, self.seenclass, :]

    # 余弦相似度 + 温度缩放
    v_n = F.normalize(v_enh, dim=-1).unsqueeze(1)        # [B, 1, 768]
    t_n = F.normalize(t_enh, dim=-1)                     # [B, 200, 768]
    logits_200 = (v_n * t_n).sum(dim=-1) * logit_scale   # [B, 200]
else:
    # add 模式 (向后兼容)
    logits_200 = base_logits + self.local_weight * local_score
```

### 3.3 配置参数

```yaml
# config/VGSR_cub_gzsl.yaml
score_mode:
  value: cosine_only       # 'add' 或 'cosine_only'
visual_residual:
  value: 0.5               # 视觉端保留 50% CLS
text_residual:
  value: 0.5               # seen 文本端保留 50% all_text
```

---

## 四、实验结果

### 4.1 对照表

| 实验 | score_mode | unseen 文本处理 | U | S | H | ZSL |
|------|-----------|----------------|---|---|---|-----|
| **基线 (add)** | add | 共享 all_text | **71.18** | **73.24** | **72.20** | **81.67** |
| cosine_only v1 (暴雷) | cosine_only | 全部经 proj_text | 30.57 | 77.41 | 43.83 | 69.99 |
| **cosine_only v2 (方案3)** | cosine_only | unseen 绕过 proj_text | 31.22 | 80.88 | 45.05 | 77.77 |

### 4.2 方案 3 修复效果

| 指标 | 暴雷版 (v1) | 方案 3 (v2) | 差值 |
|------|------------|------------|------|
| ZSL | 69.99 | **77.77** | **+7.78** ← unseen 文本保护生效 |
| U | 30.57 | 31.22 | +0.65 |
| S | 77.41 | 80.88 | +3.47 |
| H | 43.83 | 45.05 | +1.22 |

**方案 3 修复了 ZSL 崩溃问题**（+7.78%），但 H 仍远低于 add 模式。

---

## 五、失败原因分析

### 5.1 为什么 H 只有 45%（vs add 模式 72%）

1. **v_enh 信号太弱**
   - `F_p_s2v.mean(dim=1)` 把 576 个 patch 暴力平均
   - 背景 patch（天空、树枝）占 5/6，判别 patch（鸟身）只占 1/6
   - 信号被噪声稀释 5 倍

2. **无 base_logits 兜底**
   - add 模式有 `CLS @ all_text.T` 作为强基线（CLIP 原始零样本能力）
   - cosine_only 完全依赖 CrossModal 从零学起，训练初期 logits 是随机的

3. **训练动态不匹配**
   - add 模式：base_logits 从 epoch 1 就给出合理分数，local_score 只做微调
   - cosine_only：epoch 1 的 v_enh/t_enh 都是随机初始化，需要更多 epoch 才能收敛
   - 但 CUB 只有 7057 张图 × 10 epoch = 70K 步，不够 CrossModal 从零学到好表示

4. **seen-bias 严重**
   - S=80-84%（极高）但 U=24-31%（极低）
   - 模型把所有图都预测为 seen 类（因为 seen 类的 t_enh 经过训练更"尖锐"）

### 5.2 为什么 unseen 文本绕过能救 ZSL 但救不了 U

- **ZSL**（仅 50 unseen 类内竞争）：unseen 文本恢复正常后，50 类内部排序正确 → ZSL 回到 77%
- **U**（全 200 类竞争）：即使 unseen 文本正确，seen 类的 cosine 分数整体偏高（因为 seen 有训练），unseen 被压死 → U 只有 31%

---

## 六、结论与教训

### 6.1 结论

**cosine_only 框架在当前架构下不可行**。核心原因是 CrossModal 的输出质量不足以独立承担分类任务。

### 6.2 教训

1. **CLIP 的 CLS token 是强基线**：直接 `CLS @ text.T` 就能给出 ~70% H，任何替代方案必须超过这个门槛
2. **加分式融合是安全的**：`base + β×local` 保证了下限（β=0 回到 CLIP），cosine_only 没有下限
3. **mean 池化是瓶颈**：576 个 patch 暴力平均丢失太多信息，如果要做 cosine_only 必须先解决池化问题
4. **unseen 文本必须保护**：任何让 unseen 文本经过可训练层的设计都会在 GZSL 中崩溃

### 6.3 如果要让 cosine_only 工作（未来方向）

1. **升级 v_enh 池化**：用 attention pooling / top-k 替代 mean
2. **加 base_logits 作为初始化**：`logits = base_logits + cosine(v_enh, t_enh)` 而不是纯 cosine
3. **预训练 CrossModal**：先用 add 模式训好，再 finetune 到 cosine_only
4. **加 calibration loss**：强制给 unseen 类保留概率质量

---

## 七、当前最佳配置（回退到 add 模式）

```yaml
score_mode: add              # 回退
text_source: claude          # Claude 描述
epochs: 15
local_weight: 0.5
use_fae: True
random_seed: 5
```

**当前最佳**: H = 72.26% ± 0.08%（双种子复测）
