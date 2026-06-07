# MOD-005 语义补丁 JEPA v2

日期: 2026-06-07

类型: 创新模块 / 新增代码模块

状态: 已完成

## 1. 实验目的

验证在现有 AG-JEPA 辅助目标中引入近邻类别语义原型，是否能更准确地遮蔽和预测判别性 patch，从而提升 CUB GZSL 主指标 H。

当前 baseline:

| 数据集 | seed | U | S | H | ZS |
|---|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 73.30 | 72.53 | 72.91 | 81.72 |

## 2. 来源复核与迁移规格

| 项 | 内容 |
|---|---|
| 创意来源 | `semantic-patch-jepa-v2` 创意树节点 |
| 本地证据 | ABL-002：关闭 AG-JEPA 后 H=71.08，相对 baseline 72.91 下降 1.83 |
| 外部论文 | 无直接论文迁移要求 |
| 本次是否参考论文原文 | 不适用，本模块来自本项目本地实验和用户 idea |
| 本次是否参考官方代码 | 不适用 |

原 AG-JEPA:

```text
patches + class_text
  -> class_text 选择 top-K 相关 patch
  -> 遮蔽 top-K patch
  -> context + class_text 预测被遮蔽 patch 表示
  -> cyclic next seen class 作为负文本
```

MOD-005 迁移设计:

```text
class_text + nearest seen class prototypes
  -> patch_score = sim(patch, class_text) - w * sim(patch, neighbor_text)
  -> 遮蔽更判别的 top-K patch
  -> context + (class_text - w * neighbor_text) 预测 target patch
  -> nearest neighbor text 作为 hard negative
```

关闭等价性:

- `use_ag_jepa_v2=False`
- 关闭时完全沿用原 `_ag_jepa_loss`：原 patch 选择、原 class_text 条件、原 cyclic negative。
- 主配置默认关闭，只有本实验配置打开。

## 3. 改动内容

| 配置项 | 主配置默认 | MOD-005 实验值 |
|---|---:|---:|
| `use_ag_jepa_v2` | `False` | `True` |
| `jepa_v2_neighbor_topk` | `5` | `5` |
| `jepa_v2_neighbor_weight` | `0.2` | `0.2` |
| `lambda_jepa` | `0.05` | `0.05` |
| `lambda_jepa_neg` | `0.02` | `0.02` |
| `jepa_topk` | `8` | `8` |

最小 diff 文件:

- `model/MyModel.py`
- `train_VGSR_CUB.py`
- `config/VGSR_cub_gzsl.yaml`
- `experiments/01_single_module_innovation/MOD-005_semantic_patch_jepa_v2/config.yaml`

## 4. 训练配置

| 项 | 值 |
|---|---|
| 数据集 | CUB |
| 随机种子 | 5 |
| 命令 | `F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/01_single_module_innovation/MOD-005_semantic_patch_jepa_v2/config.yaml` |
| 正式比较口径 | 固定 seed=5，与同 seed baseline 对照 |

## 5. 审查记录

Codex 自审: ACCEPTED，见 `codex-self-review.md`

Claude 三轮审查: ACCEPTED，见 `claude-review.md`

## 6. 结果

| 数据集 | seed | U | S | H | ZS | 最佳 epoch | baseline H | ΔH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 73.10 | 72.03 | 72.56 | 81.55 | 23 | 72.91 | -0.35 |

日志与模型产物:

- 原始日志: `train_log/CUB/training_log_CUB_2026-06-07_20-30-43.txt`
- 实验日志副本: `experiments/01_single_module_innovation/MOD-005_semantic_patch_jepa_v2/logs/MOD-005_CUB_seed5_20260607-203043.txt`
- 最佳模型: `train_log/CUB/best_model_CUB_2026-06-07_20-30-43_H7256.pth`
- 完整 checkpoint: `train_log/CUB/ckpt_full_CUB_2026-06-07_20-30-43.pth`

## 7. 结论

MOD-005 没有提升主指标。H 从 72.91 降到 72.56，下降 0.35；U 从 73.30 降到 73.10，S 从 72.53 降到 72.03，ZS 从 81.72 降到 81.55。

结论：当前“类别文本 - 近邻类别文本”的判别式 AG-JEPA v2 不纳入主框架。它可能把被遮蔽 patch 目标变得过窄或过硬，使 AG-JEPA 从原来的稳定辅助目标变成更强的 hard-negative 约束，削弱了泛化。后续如果继续做 AG-JEPA 方向，应优先尝试更温和的目标混合或只替换 hard negative，不建议保留当前 v2 公式。

## 8. 后续动作

- [x] Codex 自审。
- [x] Claude 三轮审查。
- [x] 审查通过后运行训练。
- [x] 复制并命名实验日志。
- [x] 更新 `experiments/EXPERIMENT_REGISTRY.md`。
- [x] 更新 `backlog.md` 和 `创新指导清单/queues/01_module_replacement.md`。
- [x] 生成 `experiments/08_framework_flow_records/MOD-005_semantic_patch_jepa_v2.md`。
- [x] 根据结果更新创意树权重。
