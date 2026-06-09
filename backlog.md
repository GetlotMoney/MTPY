# 当前实验执行窗口

更新时间：2026-06-09

这个文件只放“近期马上执行”的短队列，不承载全部想法。

长期想法和分类待做清单放在：

- `C:\Users\Administrator\Desktop\项目\创新指导清单\paper-idea-tree\创意树.md`
- `C:\Users\Administrator\Desktop\项目\创新指导清单\queues\01_module_replacement.md`
- `C:\Users\Administrator\Desktop\项目\创新指导清单\queues\02_ablation.md`
- `C:\Users\Administrator\Desktop\项目\创新指导清单\queues\03_hyperparam_tuning.md`
- `C:\Users\Administrator\Desktop\项目\创新指导清单\queues\04_cross_dataset.md`
- `C:\Users\Administrator\Desktop\项目\创新指导清单\queues\05_final_runs.md`

## 当前目标

当前目标：先通过创新模块把 CUB GZSL 主指标推到 `H >= 74`，达到 74 后再系统调参。

当前严格连续 baseline：

| 数据集 | seed | U | S | H | ZS |
|---|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 73.10 | 73.61 | 73.35 | 81.44 |

当前执行窗口比较口径：

- 单个创新模块实验默认固定 `seed=5`，与同 seed baseline 对照。
- 当前 baseline 来源：`TUNE-004_topo_pearson_01`，`lambda_topo_pearson=0.1`，最佳 epoch 51。
- 如果某个实验要补跑多个 seed，必须在实验 README 中预先写明 seed 列表。
- 多 seed 结果必须全部保留；是否按最高 `H` 作为项目选型参考，需要在该实验结论里单独标注，不能隐藏其它 seed。
- 不使用“事后临时挑 seed”作为单个实验的正式结论。

## 实验 ID 类型

| 前缀 | 含义 | 当前窗口默认队列 |
|---|---|---|
| `MOD-xxx` | 单模块创新实验 | `queues/01_module_replacement.md` |
| `COMBO-xxx` | 组合模块实验，需要写组成模块、协同假设、冲突分析和组合收益 | `queues/01_module_replacement.md` |
| `REV-MOD-xxx` | 单模块复核，用于 near_tie、win 或用户指定模块 | `queues/01_module_replacement.md` |
| `TUNE-xxx` | 调参实验 | `queues/03_hyperparam_tuning.md` |
| `ABL-xxx` | 消融实验 | `queues/02_ablation.md` |
| `XDS-xxx` | 跨数据集实验 | `queues/04_cross_dataset.md` |
| `FINAL-xxx` | 最终复核 | `queues/05_final_runs.md` |

执行优先级：`REV-MOD` 候选复核优先于有明确协同假设的 `COMBO`，`COMBO` 优先于新的 `MOD` 初筛，`TUNE` 默认在达到 `H >= 74` 后再系统执行，除非用户明确指定。

## 分支归属

| 实验分类 | 分支 | 分类队列 | 实验目录 |
|---|---|---|---|
| `MOD-xxx` 单模块创新实验 | `experiment/single-module-innovation` | `C:\Users\Administrator\Desktop\项目\创新指导清单\queues\01_module_replacement.md` | `experiments/01_single_module_innovation/` |
| `COMBO-xxx` 组合模块实验 | `experiment/module-combination` | `C:\Users\Administrator\Desktop\项目\创新指导清单\queues\01_module_replacement.md` | `experiments/02_module_combination/` |
| `REV-MOD-xxx` 单模块复核 | `experiment/single-module-review` | `C:\Users\Administrator\Desktop\项目\创新指导清单\queues\01_module_replacement.md` | `experiments/03_single_module_review/` |
| `TUNE-xxx` 调参实验 | `experiment/hyperparameter-tuning` | `C:\Users\Administrator\Desktop\项目\创新指导清单\queues\03_hyperparam_tuning.md` | `experiments/04_hyperparameter_tuning/` |
| `ABL-xxx` 消融实验 | `experiment/ablation` | `C:\Users\Administrator\Desktop\项目\创新指导清单\queues\02_ablation.md` | `experiments/05_ablation/` |
| `XDS-xxx` 跨数据集 | `experiment/cross-dataset` | `C:\Users\Administrator\Desktop\项目\创新指导清单\queues\04_cross_dataset.md` | `experiments/06_cross_dataset/` |
| `FINAL-xxx` 最终复核 | `experiment/final-review` | `C:\Users\Administrator\Desktop\项目\创新指导清单\queues\05_final_runs.md` | `experiments/07_final_review/` |

历史兼容分支：`experiment/innovation`、`experiment/tuning`、`experiment/final-runs` 保留为旧记录入口；`experiment/ablation` 和 `experiment/cross-dataset` 名字已经完整，继续作为正式分支使用。

## P0

| ID | 状态 | 分类 | 实验 | 来源队列 | 备注 |
|---|---|---|---|---|---|
| MOD-001 | done | 创新模块 | 几何感知属性路由 | `queues/01_module_replacement.md` | seed=5 H=72.45，低于 baseline 72.91，当前版本不保留 |

## P1

| ID | 状态 | 分类 | 实验 | 来源队列 | 备注 |
|---|---|---|---|---|---|
| MOD-002 | done | 创新模块 | 拓扑感知的自适应文本属性库 | `queues/01_module_replacement.md` | seed=5 H=69.24，低于 baseline 72.91，当前版本不保留 |
| MOD-003 | done | 创新模块 | 带不确定性门控的分支一致性蒸馏 | `queues/01_module_replacement.md` | seed=5 H=72.61，低于 baseline 72.91，当前版本不保留 |
| MOD-004 | done | 创新模块 | 属性引导的局部补丁 OT 对齐 | `queues/01_module_replacement.md` | seed=5 H=72.90，基本持平但低于 baseline 72.91，当前版本不保留；方向保留为中性候选 |
| MOD-005 | done | 创新模块 | 语义补丁 JEPA v2 | `queues/01_module_replacement.md` | seed=5 H=72.56，低于 baseline 72.91；当前 v2 不保留 |
| MOD-006 | done | 创新模块 | 反事实负文本挖掘 | `queues/01_module_replacement.md` | seed=5 H=72.39，低于 baseline 72.91；当前训练期 margin 不保留 |

## 更新规则

- `backlog.md` 只作为当前执行窗口，建议保留 3 到 10 个近期实验。
- 新想法先进入创意树或对应分类队列，不直接塞进 backlog。
- 启动某一分类实验时，先读对应分类队列，再把最高优先级项目同步到 backlog。
- 实验开始前，把对应项从 `open` 改成 `running`。
- 实验结束后，改成 `done`、`blocked`、`rejected`，或保留 `open` 并写清原因。
- 每个实验完成后，更新实验 README、`experiments/EXPERIMENT_REGISTRY.md`、框架图和对应分类队列。
