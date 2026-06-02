# DVSR / VGSR: 基于 CLIP + Adapter + GPT 的零样本图像分类

广义零样本学习 (GZSL) 研究项目，基于 VDT-TransZero 思路，使用 CLIP ViT-L/14@336px 作为骨干。

> 当前唯一项目汇报: `DVSR_标准项目汇报.md`
>
> 旧的 report / 汇报 / 交接文档已清理。日常接手、开实验、写论文优先读 `DVSR_标准项目汇报.md`。

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
- **unseen 类文本**：使用 GPT 描述均值，无 Adapter
- **CLIP 主干**：始终冻结，保留原始零样本能力

## 实验结果（CUB GZSL）

| 口径 | 配置 | U | S | H | ZSL |
|------|------|---:|---:|---:|---:|
| 新 main baseline | P3.10 配置, 3 seeds, strict 60ep | 73.07 +/- 0.85 | 72.26 +/- 0.85 | 72.65 +/- 0.19 | 81.54 +/- 0.16 |
| 单点最高 strict | P3.10 seed=5 | 73.30 | 72.53 | 72.91 | 81.72 |
| 旧 main baseline | v5+FAE, 3 seeds, strict 50ep | 74.13 +/- 0.62 | 70.99 +/- 0.40 | 72.52 +/- 0.11 | 81.56 +/- 0.18 |
| 纯 CLIP zero-shot | no train | 60.88 | 61.69 | 61.28 | 78.07 |

注意: `H=73.05` / `H=73.20` 属于 warm-restart 或离群刷分口径，不作为当前 strict main baseline。

## 项目结构

```
.
├── DVSR_标准项目汇报.md        # 当前唯一项目汇报
├── model/MyModel.py           # 主模型 (Adapter + LaSt v5 + CrossModalTransformer)
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
├── experiments/               # 后续实验管理区：替换模块 / 消融 / 调参 / 跨数据集 / 最终结果
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

- `DVSR_标准项目汇报.md` — 当前唯一项目汇报
- `experiments/README.md` — 后续实验文件夹规范
- `train_log/CUB/training_log_*.txt` — 原始训练日志证据
- `Guide/*_CODE_GUIDE.md` / `Guide/*TUTORIAL.md` — 代码学习材料
