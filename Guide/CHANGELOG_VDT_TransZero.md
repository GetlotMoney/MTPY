# DVSR 项目变更记录：CLIP + Adapter + GPT (VDT-TransZero)

> 日期: 2026-05-10
> 目的: 实现纯 CLIP + Adapter + GPT-4 描述的零样本分类方法 (来自 VDT-TransZero 论文)
> 数据集: CUB-200-2011
> 运行环境: `F:\Anaconda\envs\dassl_clip`

---

## 一、当前模型架构（极简）

```
Image → CLIP ViT-L/14@336px (冻结) → 全局特征 [B, 768]
                                          ↓
                                    normalize
                                          ↓
                              cosine similarity × scale
                                          ↑
GPT-4 descriptions → CLIP text encoder (冻结) → [200, 768]
                                          ↓
                          Adapter 残差增强 (唯一可训练模块)
                                          ↓
                                    normalize
                                          ↓
                              logits [B, 200] → CrossEntropy
```

**可训练参数: 294,913** (Adapter 294,912 + logit_scale 1)

---

## 二、文件清单

| 文件 | 说明 |
|------|------|
| `model/VGSR.py` | 模型定义: Adapter + VGSR 主模型 |
| `train_VGSR_CUB.py` | CUB 训练入口 |
| `config/VGSR_cub_gzsl.yaml` | 超参数配置 |
| `tools/dataset.py` | 数据加载 (未修改) |
| `tools/helper_func.py` | CLIP 特征提取 + GZSL 评估 (未修改) |
| `data/gpt4_data/cub.pt` | GPT-4 类别描述数据 |
| `data/clip_att/CUB_attribute.pkl` | CLIP 属性特征 (本次未使用) |
| `data/CUB/images/` → 符号链接 | CUB 图片 |
| `data/xlsa17/data/` → 符号链接 | xlsa17 数据划分 |

---

## 三、运行方法

```bash
conda activate dassl_clip
cd C:\Users\Administrator\Desktop\DVSR
python train_VGSR_CUB.py
```

---

## 四、配置参数 (config/VGSR_cub_gzsl.yaml)

| 参数 | 值 | 说明 |
|------|-----|------|
| num_class | 200 | CUB 总类别数 |
| batch_size | 64 | 每批图片数 |
| epochs | 10 | 训练轮数 (可调) |
| dim_f_clip | 768 | CLIP 特征维度 |
| adapter_ratio | 0.2 | Adapter 残差比例 |
| is_bias | False | GZSL bias 开关 |
| random_seed | 5 | 随机种子 |

---

## 五、评估协议 (GZSL)

- **数据划分**: xlsa17 Proposed Split
  - 训练: 7,057 张 (150 seen classes)
  - 测试 Seen: 1,764 张 (150 seen classes)
  - 测试 Unseen: 2,967 张 (50 unseen classes)
- **指标**:
  - S: seen 类 per-class accuracy
  - U: unseen 类 per-class accuracy (GZSL, 搜索空间=全部200类)
  - H: 2×S×U/(S+U) 调和平均 (核心指标)
  - ZS: unseen 类 accuracy (ZSL, 搜索空间=仅50个unseen类)

---

## 六、与原模板的区别

原模板有 DSME 属性推理分支 + 多种 Loss，本次全部删除，只保留：
- CLIP 视觉编码 (冻结)
- CLIP 文本编码 (冻结)
- GPT-4 描述加载
- Adapter 残差增强 (可训练)
- CrossEntropy Loss
- GZSL 标准评估


---

# 双向 Transformer 集成 (TransZero++ 风格)

日期: 2026-05-10
目标: 在原有 CLIP + Adapter + GPT 框架基础上，加入视觉-语义双向 Transformer 交互，提升 GZSL 精度。

## 改动原则

**最小改动原则**：只改 1 个代码文件 + 1 个配置文件，不新建任何代码文件，不触碰模板其他部分。

## 基线 (改动前)

| 模型 | H (GZSL) | U | S | ZS |
|------|---------|---|---|-----|
| CLIP + Adapter + GPT | 70.26% | 69.02% | 71.54% | 79.60% |

## 改动清单

### 1. `model/VGSR.py` — 主要改动

**原架构**: `image_feat (CLS) @ adapted_text.T → logits`
**新架构**: 双向 Transformer 交互

文件内新增两个模块（与原有 `Adapter`、`VGSR` 同文件，保持模块化）:

```
model/VGSR.py
├── Adapter                      (原有, 未改)
├── BoxRelationalEmbedding       (新增) 24×24 grid 几何位置编码
├── GeometryMultiHeadAttention   (新增) 带几何偏置的多头注意力
├── FAELayer                     (新增) Feature Augmentation Encoder 层
├── CrossModalTransformer        (新增) 双向 Transformer 封装
│     ├── FAE 视觉自编码
│     ├── s2v 分支: text Query, visual KV
│     ├── v2s 分支: visual Query, text KV
│     └── 加权融合: w*s2v + (1-w)*v2s
└── VGSR                         (修改) 实例化并调用 CrossModalTransformer
```

**关键变化**:
- `VGSR.forward` 不再只用 CLS token，改为使用完整的 576 个 patch
- 新增 `_prepare_patches` 辅助方法，兼容 `[B, 577, 768]` / `[B, 576, 768]` / `[B, 1, 768]` / `[B, 768]` 多种输入
- `is_train=True` → `logits [B, 150]`，仅用 seen 类文本
- `is_train=False` → `logits [B, 200]`，seen 用 Adapter 增强，unseen 用 CLIP 原始，按全局索引拼接

**可训练参数**:
- 原模型: 294,913 (Adapter + logit_scale)
- 新模型: 9,494,277 (Adapter + CrossModalTransformer + logit_scale)

### 2. `config/VGSR_cub_gzsl.yaml` — 新增超参

在文件末尾追加 TransZero++ 双向 Transformer 配置:

```yaml
# === [TransZero++ 双向 Transformer 参数] ===
tf_common_dim:  { value: 512 }   # 公共投影维度 (768 → 512)
tf_heads:       { value: 4 }     # 多头注意力头数
tf_dropout:     { value: 0.1 }   # Dropout
weight_s2v:     { value: 0.5 }   # s2v/v2s 融合权重
```

所有参数通过 `getattr(config, 'key', default)` 获取，未设置时使用默认值，向后兼容。

### 3. 删除旧缓存

```
del data/cache/CUB_train_features.pt
del data/cache/CUB_train_labels.pt
```

原因: 旧缓存只存了 CLS token `[N, 768]`，双向 Transformer 需要完整 patch `[N, 576, 768]`。
删除后，训练脚本的 `USE_CACHE` 判断自动失败，走实时 CLIP 提取路径。

### 4. 未改动的文件

**严格保持不变**:
- `tools/dataset.py`
- `tools/helper_func.py`
- `tools/extract_features.py`
- `train_VGSR_CUB.py`
- `train_VGSR_AWA2.py`
- `train_VGSR_SUN.py`

接口契约保持不变:
- `model(clip_features, is_train)` 返回的字典仍含 `'clip_S_pp'` 键
- 评估函数 `eval_zs_gzsl` 无需修改

## 数据流对比

| 阶段 | 改动前 | 改动后 |
|------|--------|--------|
| CLIP 输出 | `[B, 577, 768]` | `[B, 577, 768]` (相同) |
| 取出特征 | `[:, 0, :]` CLS token `[B, 768]` | `[:, 1:, :]` 576 patch `[B, 576, 768]` |
| 文本路径 | Adapter `[150, 768]` | Adapter `[150, 768]` (相同) |
| 交互方式 | 余弦相似度 | 双向 Transformer (FAE + s2v + v2s) |
| 输出 | `image @ text.T` `[B, N_cls]` | `w*s2v + (1-w)*v2s` `[B, N_cls]` |

## 模块职责

| 模块 | 输入 | 输出 | 作用 |
|------|------|------|------|
| `BoxRelationalEmbedding` | batch_size | `[B, 576, 576, 64]` | 预计算 24×24 相对位置编码 |
| `GeometryMultiHeadAttention` | `[B, 576, 512]`, geo | `[B, 576, 512]` | 几何感知的自注意力 |
| `FAELayer` | `[B, 576, 512]`, geo | `[B, 576, 512]` | 消除 patch 几何纠缠 |
| `CrossModalTransformer` | patches+text | logits | 双向交互 + 融合 |
| `VGSR` | CLIP特征 | logits字典 | 主模型, 组合全部模块 |

## 如何回退

如需回到单分支基线:
1. `git checkout` 恢复 `model/VGSR.py` 到上一版本
2. 或在 `config/VGSR_cub_gzsl.yaml` 中设置 `weight_s2v: 0.0`（但这只是降级，不能完全回退架构）
3. 恢复缓存: `python tools/extract_features.py` 重新生成

## 预期效果

- 基线 H = 70.26%
- 加入 FAE: 视觉特征消除空间纠缠，预期 U +1~2%
- 加入 s2v 双向: 语义定位视觉区域，预期 S +1~3%
- 预期综合 H = 72~75%

实际效果以训练结果为准。


## 补充：Patch 缓存 (2026-05-10)

### 问题
之前"最小改动"只删除了旧 CLS 缓存，未生成 patch 缓存，导致训练走实时 CLIP 提取路径（慢）。

### 补充改动

**`tools/extract_features.py`** — 扩展提取逻辑:
- `extract_features()` 新增 `return_patches=True` 参数
- 同时保存 CLS (float32) 和 patch (float16) 特征
- 测试集仍然不缓存（评估只跑一次）

**`train_VGSR_CUB.py`** — 扩展缓存加载:
- 新增 `CACHE_TRAIN_PATCH = 'data/cache/CUB_train_patch_features.pt'`
- 缓存优先级: **patch > CLS > 实时提取**
- `USE_CACHE` 从 bool 改成字符串 `'patch' / 'cls' / None`
- patch 缓存放 **CPU 内存** (5.9GB), 按 batch 搬到 GPU, 避免 GPU 显存压力

### 新的缓存文件

| 文件 | 形状 | dtype | 大小 |
|------|------|-------|------|
| `CUB_train_patch_features.pt` | `[7057, 576, 768]` | float16 | 5.9 GB |
| `CUB_train_features.pt` (legacy) | `[7057, 768]` | float32 | 20 MB |
| `CUB_train_labels.pt` | `[7057]` | int | <1 MB |

### 修正 CHANGELOG 中的错误表格

之前第一版写的"改动后数据流"表格有误，实际情况:

| 阶段 | 改动前 (基线) | 改动后 (本版) |
|------|--------------|--------------|
| 训练输入 | `[B, 1, 768]` (CLS缓存) | `[B, 576, 768]` (patch缓存, float16→float32) |
| 训练缓存 | `CUB_train_features.pt` (20MB) | `CUB_train_patch_features.pt` (5.9GB) |
| 视觉特征 | CLS token `[B, 768]` | FAE编码后的 patch `[B, 576, 512]` |
| 语义交互 | `image @ text.T` | s2v + v2s 双分支融合 |
| 评估阶段 | 实时 CLIP 提取 | 实时 CLIP 提取 (相同) |

### 内存占用

- CPU 内存: +5.9 GB (patch 缓存常驻)
- GPU 显存: +每 batch `64 × 576 × 768 × 4B = 113 MB` (动态搬运)
- 32GB 系统内存足够

### 生成缓存命令
```
python tools/extract_features.py
```


---

# 对称双分支设计 (2026-05-10 第三次迭代)

## 背景

前两版双分支都失败了 (H=27% / H=15%), 失败的核心原因:
- 版本1: CrossModalTransformer 直接输出 logits, 在 512 维投影空间做余弦, unseen 类在该空间未对齐
- 版本2: 只增强视觉特征一端, 文本端没有变化, 没有真正利用"双向"

经过与 TransZero++ 论文的对照和多轮讨论, 确定正确的设计原则:
1. **两端都增强**: s2v 增强视觉端, v2s 增强文本端
2. **在 CLIP 768 维原始空间做分类**: 不破坏 CLIP 的零样本能力
3. **强残差保护 unseen 文本**: 避免 Transformer 扰动未见过的 unseen 类文本

## 设计思路 (对称双分支)

```
视觉分支 (s2v):
  patches [B, 576, 768]
      ↓ embed_cv_s2v → FAE_s2v → decoder_s2v (text Q, visual KV)
      ↓ mean over N_cls → proj_visual
  enhanced_visual [B, 768]  ← 增强的视觉特征

语义分支 (v2s):
  text [N_cls, 768]
      ↓ embed_text_v2s / embed_cv_v2s → FAE_v2s → decoder_v2s (text Q, visual KV)
      ↓ mean over batch → proj_text → text_delta [N_cls, 768]
      ↓ 强残差: enhanced_text = text + α * text_delta  (α=0.2)
  enhanced_text [N_cls, 768]  ← 增强的文本特征 (unseen 类主要保留原始)

最终分类 (基线余弦):
  enhanced_visual @ enhanced_text.T × logit_scale → logits [B, 200]
```

## 改动

### `model/VGSR.py` - CrossModalTransformer

**新增参数**: `text_residual=0.2` (v2s 路文本残差系数)

**新增两路独立的投影和 FAE** (避免梯度干扰):
- `embed_cv_s2v` / `embed_cv_v2s`: 视觉 768→dim_com
- `embed_text_s2v` / `embed_text_v2s`: 文本 768→dim_com
- `fae_s2v` / `fae_v2s`: 视觉自编码 (独立参数)
- `proj_visual`: dim_com→768 (视觉分支输出)
- `proj_text`: dim_com→768 (语义分支输出)

**forward 返回改为元组**:
```python
return enhanced_visual, enhanced_text
# enhanced_visual: [B, 768]
# enhanced_text:   [N_cls, 768]
```

### `model/VGSR.py` - VGSR.forward

改用两端增强做余弦相似度:
```python
enhanced_visual, enhanced_text = self.cross_tf(patches, all_text)
visual_n = F.normalize(enhanced_visual, dim=1)
text_n   = F.normalize(enhanced_text,   dim=1)
logits_200 = visual_n @ text_n.T * logit_scale
```

### `config/VGSR_cub_gzsl.yaml`

新增:
```yaml
text_residual:
  value: 0.2
```

## 参数量对比

| 版本 | 可训练参数 |
|------|-----------|
| 基线 (Adapter only) | 294,913 |
| 版本1 (FAE+s2v+v2s in 512dim) | 9,647,877 |
| 版本2 (单路 enhanced_visual) | 9,888,261 |
| **版本3 (对称双分支)** | **13,172,745** |

参数量增加是因为两路 FAE 独立, 每路约 5M 参数.

## 预期效果

| 方案 | U | S | H |
|------|---|---|---|
| 基线 | 69.02% | 71.54% | 70.26% |
| 版本3 目标 | ≥70% | ≥70% | ≥72% |

关键是 unseen 类的 text_residual=0.2 必须足够小, 不然 Transformer 的扰动会破坏 CLIP 零样本能力.

## 何时用什么 residual

| text_residual | 含义 | 适用场景 |
|--------------|------|---------|
| 0.0 | 完全保留 CLIP 文本 | 等价于基线 + 视觉增强 |
| 0.1~0.2 | 弱增强, 保护 unseen | **推荐**, 平衡提升和保护 |
| 0.3~0.5 | 中等增强 | seen 类提升更多, 但 unseen 可能下降 |
| 1.0 | 完全信任 Transformer | 等价于无残差, unseen 崩溃 |

## 如何回退

- 设置 `text_residual: 0.0` → v2s 分支失效, 等价于只有视觉增强
- 删除 `cross_tf` 实例 → 回到基线



---

# 串行双向耦合 (2026-05-10 第四次迭代)

## 背景

第三版对称双分支虽然两端都增强了, 但两路参数完全独立, 梯度只通过 CE loss 耦合,
严格意义上不是"双向交互", 只是"两路特征各自增强后相乘分类".

真正的双向交互需要:
1. 两路看到一致的视觉理解 (共享 FAE)
2. 一路的输出作为另一路的输入 (cross-connect)
3. 梯度能在两路之间直接流动

## 设计: 共享 FAE + 串行 cross-connect

```
patches [B, 576, 768] ──┬→ embed_cv → FAE → memory [B, 576, dim_com]  ← 共享
                        │                        │
text [N_cls, 768] ─────→ embed_text → txt_batch │
                                         │       │
                      ┌──────────────────┴───────┤
                      ↓                          ↓
               decoder_v2s (tgt=text, mem=memory)
                      ↓ mean(B) + proj + 残差
               enhanced_text [N_cls, 768]  ← 第一个输出
                      │
                      ↓ embed_text → enhanced_text_com
                      ↓
               decoder_s2v (tgt=enhanced_text_com, mem=memory)
                                              ↑
                                     关键: s2v 用 v2s 增强过的文本
                      ↓ mean(N_cls) + proj
               enhanced_visual [B, 768]  ← 第二个输出

最终: enhanced_visual @ enhanced_text.T → logits  (基线余弦不变)
```

## 梯度流向 (真正的双向)

```
loss
  ↓
enhanced_visual ──┐
enhanced_text   ──┼→ 两路耦合的梯度源头
                  │
                  ↓
s2v decoder 的梯度: 通过 enhanced_text (Query) 流回 v2s decoder
                  ↓
v2s decoder 的梯度: 接收来自两路的合成梯度
                  ↓
共享 FAE 的梯度: 接收两路的合成梯度
                  ↓
共享 embed_cv/embed_text: 接收所有梯度
```

不是两路独立, 是真正的双向流动.

## 改动 (`model/VGSR.py`)

### CrossModalTransformer 重构

**参数变化**:
| 模块 | 之前 (对称双分支) | 现在 (串行耦合) |
|------|-----------------|----------------|
| 视觉投影 | 两路独立 (2个 Linear) | **共享 1 个** |
| 文本投影 | 两路独立 (2个 Linear) | **共享 1 个** |
| FAE | 两路独立 (2个 FAE) | **共享 1 个** |
| decoder_v2s | 独立 | 独立 (但先跑) |
| decoder_s2v | 独立 | 独立 (用 v2s 输出作 Q) |
| proj_visual | 独立 | 独立 |
| proj_text | 独立 | 独立 |

**forward 流程改变**:
- 之前: v2s 和 s2v 完全并行, 互不相关
- 现在: v2s 先跑 → enhanced_text → s2v 用 enhanced_text 作 Q 再跑
- 串行依赖形成梯度链: `loss → s2v → v2s → 底层`

**CLIP 框架保持不变**:
- 最终还是 `enhanced_visual @ enhanced_text.T * logit_scale`
- unseen 文本通过 `text_residual=0.2` 强残差保护

## 参数量

| 版本 | 可训练参数 |
|------|-----------|
| 基线 | 294,913 |
| 版本3 (对称双分支) | 13,172,745 |
| **版本4 (串行耦合)** | **10,282,245** |

参数减少因为共享了 embed_cv / embed_text / FAE.

## 优势

1. **真正的双向交互**: s2v 依赖 v2s 的输出, 梯度真正流动
2. **参数更少**: 共享视觉理解, 减少 22% 参数
3. **CLIP 框架不变**: 最终仍是两特征余弦相似度
4. **unseen 保护**: text_residual=0.2 限制 v2s 的扰动

## 如何回退

- `text_residual: 0.0` → v2s 的 delta 完全抑制, enhanced_text = text, 退化为纯视觉增强
- 移除 `enhanced_text_com` 这一步的 `embed_text` 再投影, 改为直接用 `txt_batch` → 退化为对称双分支



---

# Bug 修复 (2026-05-11 第五次迭代)

## 上一版问题 (5 轮训练 H=0%)

训练完全崩溃, 核心诊断两个 bug:

### Bug 1: `embed_text` 被二次用于不同分布

```python
# v2s 阶段: 对原始 text 投影
txt_com = self.embed_text(text)           # 原始 CLIP 文本分布

# s2v 阶段: 又对 enhanced_text 投影
enhanced_text_com = self.embed_text(enhanced_text)  # 已加了 delta 的分布
```

同一个 Linear 要学会映射两种不同分布, 训练不收敛.

### Bug 2: `enhanced_visual` 完全替换 CLIP 视觉

```python
enhanced_visual = self.proj_visual(feat_visual_com)  # 完全替换
```

CLIP 原始视觉特征被丢弃, ZSL 从 80% 暴跌到 3%.

## 修复方案

### 修复 1: s2v Query 直接用 F_p_v2s

```python
# 改前 (错)
enhanced_text_com = self.embed_text(enhanced_text)  # 二次投影
F_p_s2v = self.decoder_s2v(tgt=enhanced_text_com, memory=memory)

# 改后 (对)
# 直接用 v2s 的 512 维输出 F_p_v2s 作为 s2v 的 Query
F_p_s2v = self.decoder_s2v(tgt=F_p_v2s, memory=memory)
```

好处:
- 避免 `embed_text` 处理两种分布
- 串行依赖更直接: `F_p_s2v ← F_p_v2s`, 梯度流动更清晰
- 减少一次不必要的投影

### 修复 2: 视觉端加残差

```python
# 改前 (错)
enhanced_visual = self.proj_visual(feat_visual_com)  # 完全替换

# 改后 (对)
cls_base = patches.mean(dim=1)                       # [B, 768] CLIP 视觉基础
visual_delta = self.proj_visual(feat_visual_com)     # [B, 768] 残差修正量
enhanced_visual = cls_base + α_v * visual_delta       # 残差增强
```

好处:
- 保留 CLIP 原始视觉能力 (基础)
- Transformer 只学"修正量", 训练更稳定
- 视觉和文本都是"原始 + 残差"对称设计

## 配置新增

`config/VGSR_cub_gzsl.yaml`:

```yaml
visual_residual:   # 新增
  value: 0.2
```

## 对称残差设计

修复后, 视觉和文本端完全对称:

```
文本端: enhanced_text   = text      + α_t * text_delta
视觉端: enhanced_visual = cls_base  + α_v * visual_delta
最终:   enhanced_visual @ enhanced_text.T * scale
```

两端都:
- 保留原始 CLIP 特征作基础
- 只允许小幅度残差修正 (α=0.2)
- unseen 类和 seen 类都不会被过度扰动

## 梯度流 (真正的双向耦合)

```
loss
  ↓
enhanced_visual = cls_base + α_v * visual_delta
                              ↓
                         proj_visual
                              ↓
                      feat_visual_com.mean(1)
                              ↓
                         F_p_s2v
                              ↓
                      decoder_s2v (tgt=F_p_v2s)
                              ↓
                         F_p_v2s  ← 这里把梯度传到 v2s
                              ↓
                      decoder_v2s
                              ↓
                         memory (共享)
                              ↓
                           FAE
                              ↓
                         patches / text
```

enhanced_text 也有独立的梯度路径: enhanced_text → proj_text → F_p_v2s → v2s

梯度真正在两路之间流动, 不是独立.

## 参数量不变

10,282,245 (和上一版相同, 只是改了连接方式和残差系数)



---

# CLS Token 修复 (2026-05-11 第六次迭代)

## 上一版问题

第五版虽然加了视觉残差, 但 `cls_base = patches.mean(dim=1)` 导致:
- patch 平均和真正的 CLS token 完全不同
- CLS 是 CLIP Transformer 专门聚合的全局表示, patch 平均只是粗暴平均
- ZSL 从 80% 暴跌到 30%, U 几乎为 0%

## 修复

### `tools/extract_features.py`
(不改, 本来就同时提取了 CLS + patch 两份缓存)

### `train_VGSR_CUB.py`

**加载时同时加载 CLS 缓存**:
```python
train_patches = torch.load(CACHE_TRAIN_PATCH)  # [N, 576, 768] float16
train_cls     = torch.load(CACHE_TRAIN_FEAT)   # [N, 768]     float32
```

**训练 batch 时拼接成完整 577**:
```python
cls_batch   = train_cls[idx].unsqueeze(1)     # [B, 1, 768]
patch_batch = train_patches[idx]              # [B, 576, 768]
clip_features = torch.cat([cls_batch, patch_batch], dim=1)  # [B, 577, 768]
```

### `model/VGSR.py`

**`VGSR.forward` 分离 CLS 和 patches**:
```python
if clip_features.size(1) == 577:
    cls_token = clip_features[:, 0, :]    # [B, 768]
    patches   = clip_features[:, 1:, :]   # [B, 576, 768]
enhanced_visual, enhanced_text = self.cross_tf(patches, all_text, cls_token)
```

**`CrossModalTransformer.forward` 接收 cls_token**:
```python
def forward(self, patches, text, cls_token=None):
    # ...
    if cls_token is not None:
        cls_base = cls_token              # 真正的 CLIP CLS
    else:
        cls_base = patches.mean(dim=1)    # fallback (不推荐)
    enhanced_visual = cls_base + self.visual_residual * visual_delta
```

## 关键意义

CLS token 是 CLIP 和 text 共享语义空间的关键:
- CLIP 训练时就是用 `image_feat @ text_feat.T` 做对比学习
- 这里的 `image_feat` 就是 CLS token
- `cls_base + α * delta` 保证 enhanced_visual 仍在这个共享空间里

用 patch.mean 作基础, enhanced_visual 脱离共享空间, 余弦相似度失去意义.

## 数据流 (完整)

```
训练:
  缓存: CUB_train_features.pt (CLS) + CUB_train_patch_features.pt (patch)
  batch: torch.cat([cls, patch], dim=1) → [B, 577, 768]
       → VGSR.forward 分离 cls/patches → CrossModalTransformer

评估:
  实时 CLIP 提取 → [B, 577, 768] (已经是完整)
       → VGSR.forward 分离 cls/patches → CrossModalTransformer
```

训练和评估的输入格式完全一致, 不存在分布差异.



---

# 优化迭代记录 (2026-05-11)

## 实验结果汇总

| 版本 | 配置 | U | S | H | ZSL |
|------|------|---|---|---|-----|
| 基线 | Adapter only | 69.02% | 71.54% | 70.26% | 79.60% |
| 串行双向 | local=0.3, unseen=类名 | 66.13% | 74.47% | 70.05% | 79.38% |
| 真正双分支 | local=0.3, unseen=类名 | 66.14% | 72.88% | 69.34% | 79.14% |
| 双分支+GPT | local=0.1, unseen=GPT | 65.06% | 75.41% | 69.85% | 80.30% |
| **双分支+GPT** | **local=0.3, unseen=GPT, 20轮** | **71.35%** | **70.65%** | **71.00%** | **80.21%** |

## 本次改动清单

### 1. `model/VGSR.py` — 真正双分支

`CrossModalTransformer.forward` 改为真正双分支（方向相反，并行独立）:

```
改前（串行）:
  F_p_v2s = decoder_v2s(Q=txt_batch, KV=memory)
  F_p_s2v = decoder_s2v(Q=F_p_v2s,   KV=memory)  ← s2v 依赖 v2s 输出

改后（真正双分支）:
  F_p_v2s = decoder_v2s(Q=txt_batch, KV=memory)   ← 文本 query 视觉
  F_p_s2v = decoder_s2v(Q=memory,    KV=txt_batch) ← 视觉 query 文本（方向相反）
  score_v2s = normalize(F_p_v2s) · normalize(txt_batch)  [B, N_cls]
  score_s2v = normalize(F_p_s2v.mean(1)) @ normalize(txt_com).T  [B, N_cls]
  local_score = w * score_s2v + (1-w) * score_v2s
```

### 2. `model/VGSR.py` — Self-calibration loss

`compute_loss` 新增 self-calibration loss:

```python
loss = loss_CE + lambda_cal * loss_cal
loss_cal = -log(mean(P(unseen)))  # 强制给 unseen 类保留概率
```

注意: `lambda_cal=0.1` 太大会导致 S 崩溃，当前设为 0.0（关闭）

### 3. `train_VGSR_CUB.py` — unseen 改用 GPT 描述

```python
# 改前
unseen_clip_embeds = class_text_embeds[dataloader.unseenclasses]  # 类名模板

# 改后
unseen_clip_embeds = gpt_text_embeds[dataloader.unseenclasses]    # GPT 7句平均
```

### 4. `train_VGSR_CUB.py` — 评估时传入 gzsl_bias

```python
gzsl_bias = getattr(config, 'gzsl_bias', 0.0)
acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(
    dataloader, clip_model, model, config.device,
    bias_unseen=gzsl_bias)
```

### 5. `tools/helper_func.py` — 评估函数支持 bias_unseen

`_eval_from_cache` 和 `_eval_unseen_from_cache` 新增 `bias_unseen` 参数:

```python
if bias_unseen != 0:
    logits[:, unseen_classes] += bias_unseen
```

### 6. `config/VGSR_cub_gzsl.yaml` — 新增超参

```yaml
local_weight:  0.5      # 双分支加分系数（从 0.3 调到 0.5）
gzsl_bias:     0.5      # 评估时给 unseen 类加的固定偏置
lambda_cal:    0.0      # self-calibration loss 权重（当前关闭）
epochs:        20       # 训练轮数（从 10 增加到 20）
```

## 关键结论

1. **unseen 用 GPT 描述有效**：ZSL 从 79.38% 提升到 80.21%
2. **20 轮比 10 轮好**：峰值在 Epoch 16，说明模型需要更多轮次收敛
3. **真正双分支 vs 串行**：真正双分支（69.34%）略差于串行（70.05%），但加上 GPT+20轮后超过基线
4. **self-calibration loss 需要极小的 lambda**：0.1 会导致 S 崩溃，需要 ≤0.01

