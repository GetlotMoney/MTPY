# DVSR / VGSR: 基于 CLIP + Adapter + GPT 的零样本图像分类

广义零样本学习 (GZSL) 研究项目，基于 VDT-TransZero 思路，使用 CLIP ViT-L/14@336px 作为骨干。

> 当前唯一项目汇报: `DVSR_标准项目汇报.md`
>
> 旧的报告 / 汇报 / 交接文档已清理。日常接手、开实验、写论文优先读 `DVSR_标准项目汇报.md`。

## 核心思想

```
图像 → CLIP(冻结) → CLS + 补丁特征
                       │
       ┌───────────────┼─────────────────┐
       ↓                                 ↓
  全局分类分数                      双向视觉-文本 Transformer
                                    (并行双向 v2s + s2v)
                                         ↓
                                    局部分数
       └───────────────┬─────────────────┘
                       ↓
          最终分数 = 全局分类分数 + 局部分数
                       ↓
                交叉熵损失 (seen 150 类)
```

- **seen 类文本**：经 Adapter 优化（参数可训练）
- **unseen 类文本**：使用 GPT 描述均值，无 Adapter
- **CLIP 主干**：始终冻结，保留原始零样本能力

## 实验结果（CUB GZSL）

| 口径 | 配置 | U | S | H | ZSL |
|------|------|---:|---:|---:|---:|
| 当前 baseline / 当前主基线 | 局部补丁选择 + 几何感知编码 + 双向视觉-文本交互 + 条件文本扰动 + AG-JEPA 辅助训练，多 seed 候选池取最高 H，严格连续 60 轮，seed=5 | 73.30 | 72.53 | 72.91 | 81.72 |
| 多 seed 参考 | 同一框架，seed=5 / 42 / 2024 全部记录，但不取平均作为主结果 | 见项目汇报 | 见项目汇报 | 最高 H=72.91 | 见项目汇报 |
| 旧主基线历史记录 | 局部补丁选择 + 几何感知编码，旧均值口径，仅作参考，待按最高 H 口径复核 | 74.13 +/- 0.62 | 70.99 +/- 0.40 | 72.52 +/- 0.11 | 81.56 +/- 0.18 |
| 纯 CLIP 零样本 | 不训练 | 60.88 | 61.69 | 61.28 | 78.07 |

注意：项目正式比较口径不是多 seed 均值，而是在预先记录的 seed 候选池中取主指标 H 的最大值。`H=73.05` / `H=73.20` 属于热重启或离群刷分口径，不作为当前严格连续主基线。

## 项目结构

```
.
├── DVSR_标准项目汇报.md        # 当前唯一项目汇报
├── model/MyModel.py           # 主模型：Adapter + LaSt v5 + 双向视觉-文本 Transformer
├── tools/
│   ├── dataset.py             # CUB / AWA2 / SUN 数据加载器
│   ├── helper_func.py         # CLIP 特征提取 + GZSL 评估
│   └── extract_features.py    # 预提取 CLIP 特征到缓存
├── config/
│   ├── VGSR_cub_gzsl.yaml
│   ├── VGSR_awa2_gzsl.yaml
│   └── VGSR_sun_gzsl.yaml
├── train_VGSR_CUB.py          # CUB 训练入口
├── train_VGSR_AWA2.py
├── train_VGSR_SUN.py
├── experiments/               # 实验结果、审查、日志副本和框架图
├── Guide/                     # 详细代码讲解文档
└── backlog.md                 # 当前实验执行窗口
```

创新指导清单已独立到项目上级目录：

```text
C:\Users\Administrator\Desktop\项目\创新指导清单
```

其中维护论文、创意树和长期分类队列；本仓库内只保留 `backlog.md` 当前执行窗口和 `experiments/` 实验结果。

实验 ID 分类规范见 `experiments/EXPERIMENT_REGISTRY.md`，当前固定类型包括：

```text
MOD-xxx       单模块创新实验
COMBO-xxx     组合模块实验
REV-MOD-xxx   单模块复核
TUNE-xxx      调参实验
ABL-xxx       消融实验
XDS-xxx       跨数据集实验
FINAL-xxx     最终复核
```

对应 GitHub 总控分支：

| 实验类型 | 分支 |
|---|---|
| `MOD-xxx` | `experiment/single-module-innovation` |
| `COMBO-xxx` | `experiment/module-combination` |
| `REV-MOD-xxx` | `experiment/single-module-review` |
| `TUNE-xxx` | `experiment/hyperparameter-tuning` |
| `ABL-xxx` | `experiment/ablation` |
| `XDS-xxx` | `experiment/cross-dataset` |
| `FINAL-xxx` | `experiment/final-review` |

旧分支 `experiment/innovation`、`experiment/tuning`、`experiment/final-runs` 只作为历史兼容入口；`experiment/ablation` 和 `experiment/cross-dataset` 名字已经完整，继续作为正式分支使用。

## 实验自动化框架

当前项目采用“创新树驱动实验”的闭环框架。创新想法、论文证据和长期队列放在外部创新指导清单；本仓库只保存当前执行窗口、实验记录、审查结果、日志副本和代码框架图。

```text
创新指导清单 / 创意树
  ↓
分类队列 queues/*.md
  ↓
backlog.md 当前执行窗口
  ↓
Git 本地 checkpoint
  ↓
Codex 实现最小代码或配置改动
  ↓
Codex 自审
  ↓
Claude 固定三轮审查
  ↓
审查通过后运行实验
  ↓
记录 README / logs / EXPERIMENT_REGISTRY / framework flow
  ↓
反馈 idea_tree.json / 创意树.md
  ↓
创新树给下一步意见
  ↓
更新分类队列和 backlog
```

关键约束：

- 每个实验必须先绑定创意树节点；如果是临时用户指定实验，跑完后必须回填创意树。
- 代码实现前必须检查 Git 状态并创建本地 checkpoint，方便回滚。
- 新模块默认关闭，只在实验 `config.yaml` 中打开；关闭后必须退回 baseline。
- Claude 审查固定三轮：代码/配置正确性、实验设计与可复现性、最终运行计划。
- 实验结果不取多 seed 平均值；项目选型看预注册候选 seed 中的最大 `H`，但所有 seed 结果必须完整记录。
- 每个实验结束后必须生成 `experiments/08_framework_flow_records/<EXP-ID>_<slug>.md`，沉淀代码框架图、流程说明和数据。
- 实验结束后必须反馈创新树，更新节点状态、权重、证据和下一步建议。

默认选择顺序：

```text
REV-MOD 单模块复核
  ↓
COMBO 有明确协同假设的组合实验
  ↓
MOD 新单模块初筛
  ↓
TUNE 调参
  ↓
XDS / FINAL
```

使用 Codex skill 时可以直接说：

```text
使用 $cv实验 跑 5 个创新模块实验
```

该 skill 会读取外部创新指导清单、同步 `backlog.md`、执行审查和实验记录流程。不会自动 push；push 需要用户明确要求。

完整项目级规范位于：

```text
docs/experiment_workflow/
```

其中 `README.md` 是总入口，`github_branching.md` 规定 GitHub 分支和 push 规则，`experiment_taxonomy.md` 规定 `MOD / COMBO / REV-MOD / TUNE / ABL / XDS / FINAL`，`idea_tree_feedback.md` 规定实验如何反哺创新树。

## 环境

- 操作系统：Windows
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
