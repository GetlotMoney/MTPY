# MTPY: 基于 CLIP + Adapter + GPT 的零样本图像分类

广义零样本学习 (GZSL) 研究项目，基于 VDT-TransZero 思路，使用 CLIP ViT-L/14@336px 作为骨干。

## 核心思想

```
图像 → CLIP(冻结) → CLS + patches
                       │
       ┌───────────────┼─────────────────┐
       ↓                                 ↓
  base_logits = CLS @ all_text.T    CrossModalTransformer
                                    (并行双向 v2s + s2v)
                                         ↓
                                    local_score
       └───────────────┬─────────────────┘
                       ↓
          logits = base + 0.5 × local_score
                       ↓
                CE Loss (seen 150 类)
```

- **seen 类文本**：经 Adapter 优化（参数可训练）
- **unseen 类文本**：使用 GPT-4 生成的 7 句描述均值，无 Adapter
- **CLIP 主干**：始终冻结，保留原始零样本能力

## 实验结果（CUB GZSL）

| 实验 | 配置 | U | S | H | ZSL |
|------|------|---|---|---|-----|
| 1 基线 | CLIP+Adapter+GPT, 10 轮 | 69.02 | 71.54 | 70.26 | 79.60 |
| 2 串行双向 | CrossModal, local=0.3, 10 轮 | 66.13 | 74.47 | 70.05 | 79.38 |
| 3 并行双向 | 独立双分支, local=0.3, 10 轮 | 66.14 | 72.88 | 69.34 | 79.14 |
| 4 unseen GPT | 并行+unseen GPT, local=0.3, 20 轮 | 71.35 | 70.65 | 71.00 | 80.21 |
| **5 当前最佳** | **local=0.5, 20 轮** | **71.07** | **71.04** | **71.06** | **80.51** |

## 项目结构

```
.
├── model/VGSR.py              # 主模型 (Adapter + CrossModalTransformer)
├── tools/
│   ├── dataset.py             # CUB / AWA2 / SUN DataLoader
│   ├── helper_func.py         # CLIP 特征提取 + GZSL 评估
│   └── extract_features.py    # 预提取 CLIP 特征到缓存
├── config/
│   ├── VGSR_cub_gzsl.yaml
│   ├── VGSR_awa2_gzsl.yaml
│   └── VGSR_sun_gzsl.yaml
├── train_VGSR_CUB.py          # CUB 训练入口
├── train_VGSR_AWA2.py
├── train_VGSR_SUN.py
├── Guide/                     # 详细代码讲解文档
└── 创新指导清单/              # 模块创新效果记录
```

## 环境

- OS: Windows
- Python 3.x，conda 环境（推荐 `dassl_clip`）
- CUDA: cuda:0 (CUB/SUN), cuda:1 (AWA2)
- 主要依赖: `torch`, `clip` (OpenAI), `numpy`, `scipy`, `yaml`, `Pillow`

## 使用

```bash
# 1. 数据准备（自行下载 xlsa17 + CUB images，放到 data/）
# 2. 预提取 CLIP 特征（CUB 加速）
python tools/extract_features.py

# 3. 训练
python train_VGSR_CUB.py
python train_VGSR_AWA2.py
python train_VGSR_SUN.py
```

## 数据要求（不在仓库中）

需自行准备以下数据，放到对应目录：

- `data/xlsa17/data/{CUB,AWA2,SUN}/` — xlsa17 标准划分
- `data/CUB/images/` — CUB 原始图片
- `data/gpt4_data/cub.pt` — GPT-4 类描述
- `data/cache/CUB_*.pt` — 自动生成的 CLIP 特征缓存

## 详细文档

- `Guide/PROJECT_GUIDE.md` — 项目架构总览
- `Guide/VGSR_CODE_GUIDE.md` — 模型代码逐行讲解
- `Guide/TRAIN_CUB_CODE_GUIDE.md` — 训练流程讲解
- `EXPERIMENT_RECORD.md` — 完整实验记录
- `创新指导清单/INNOVATION_RECORD.md` — 模块创新效果对比
