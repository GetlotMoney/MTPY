# MOD-004 属性引导的局部补丁 OT 对齐

日期: 2026-06-07

类型: 创新模块 / 新增代码模块

状态: 已完成

## 1. 实验目的

验证在 FAE 后的局部 patch memory 与当前类别 top-K 属性文本原型之间加入 Sinkhorn 软 OT 辅助 loss，是否能提升 CUB GZSL 主指标 H。

当前 baseline:

| 数据集 | seed | U | S | H | ZS |
|---|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 73.30 | 72.53 | 72.91 | 81.72 |

## 2. 来源复核与迁移规格

| 项 | 内容 |
|---|---|
| 论文 | Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model |
| 方法简称 | LaZSL |
| 本地 PDF | `创新指导清单/papers/Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model.pdf` |
| 官方代码 | `https://github.com/shiming-chen/LaZSL` |
| 官方代码状态 | 之前已确认仓库存在，但 clone/API/raw 访问失败或超时 |
| 本次是否参考论文原文 | 是，沿用 MOD-001 已读论文复核 |
| 本次是否参考官方代码 | 否，标记为 `未参考官方代码` |

迁移设计:

```text
CLIP patches
  -> LaSt patch selection
  -> FAE geometry-aware memory
  -> patch_attr_sim = cos(memory_patch, attr_text)
  -> 当前类别 class_attr top-K 属性
  -> Sinkhorn soft OT
  -> loss_attr_patch_ot = 1 - E_transport[sim]
```

关闭等价性:

- `use_attr_patch_ot=False`
- `lambda_attr_patch_ot=0.0`
- 关闭时不传属性文本矩阵，不返回 `attr_patch_sim`，不计算 OT loss。
- 主配置默认关闭，只有本实验配置打开。

## 3. 改动内容

| 配置项 | 主配置默认 | MOD-004 实验值 |
|---|---:|---:|
| `use_attr_patch_ot` | `False` | `True` |
| `lambda_attr_patch_ot` | `0.0` | `0.02` |
| `attr_patch_ot_topk` | `16` | `16` |
| `attr_patch_ot_temp` | `10.0` | `10.0` |
| `attr_patch_ot_iter` | `3` | `3` |

最小 diff 文件:

- `model/MyModel.py`
- `train_VGSR_CUB.py`
- `config/VGSR_cub_gzsl.yaml`
- `experiments/01_single_module_innovation/MOD-004_attribute_guided_patch_ot/config.yaml`

## 4. 训练配置

| 项 | 值 |
|---|---|
| 数据集 | CUB |
| 随机种子 | 5 |
| 命令 | `F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/01_single_module_innovation/MOD-004_attribute_guided_patch_ot/config.yaml` |
| 正式比较口径 | 固定 seed=5，与同 seed baseline 对照 |

## 5. 审查记录

Codex 自审: ACCEPTED，见 `codex-self-review.md`

Claude 三轮审查: ACCEPTED，见 `claude-review.md`

## 6. 结果

| 数据集 | seed | U | S | H | ZS | 最佳 epoch | baseline H | ΔH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 72.86 | 72.95 | 72.90 | 81.61 | 30 | 72.91 | -0.01 |

日志与模型产物:

- 原始日志: `train_log/CUB/training_log_CUB_2026-06-07_19-56-46.txt`
- 实验日志副本: `experiments/01_single_module_innovation/MOD-004_attribute_guided_patch_ot/logs/MOD-004_CUB_seed5_20260607-195646.txt`
- 最佳模型: `train_log/CUB/best_model_CUB_2026-06-07_19-56-46_H7290.pth`
- 完整 checkpoint: `train_log/CUB/ckpt_full_CUB_2026-06-07_19-56-46.pth`

## 7. 结论

MOD-004 基本追平 baseline，但没有超过当前主框架。H 从 72.91 变为 72.90，下降 0.01；S 从 72.53 提升到 72.95，ZS 从 81.72 降到 81.61，U 从 73.30 降到 72.86。

结论：当前 `lambda_attr_patch_ot=0.02` 的训练期 OT 辅助 loss 不纳入主框架默认路径。这个方向比 MOD-001 到 MOD-003 更接近 baseline，可保留为中性候选，后续只建议做更小权重、推理期局部分数或更稳定的属性选择机制，不建议直接沿用当前配置。

## 8. 后续动作

- [x] Codex 自审。
- [x] Claude 三轮审查。
- [x] 审查通过后运行训练。
- [x] 复制并命名实验日志。
- [x] 更新 `experiments/EXPERIMENT_REGISTRY.md`。
- [x] 更新 `backlog.md` 和 `创新指导清单/queues/01_module_replacement.md`。
- [x] 生成 `experiments/08_framework_flow_records/MOD-004_attribute_guided_patch_ot.md`。
- [x] 根据结果更新创意树权重。
