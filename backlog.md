# 当前实验执行窗口

更新时间：2026-06-07

这个文件只放“近期马上执行”的短队列，不承载全部想法。

长期想法和分类待做清单放在：

- `创新指导清单/paper-idea-tree/创意树.md`
- `创新指导清单/queues/01_module_replacement.md`
- `创新指导清单/queues/02_ablation.md`
- `创新指导清单/queues/03_hyperparam_tuning.md`
- `创新指导清单/queues/04_cross_dataset.md`
- `创新指导清单/queues/05_final_runs.md`

## 当前目标

当前目标：先通过创新模块把 CUB GZSL 主指标推到 `H >= 74`，达到 74 后再系统调参。

当前严格连续 baseline：

| 数据集 | seed | U | S | H | ZS |
|---|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 73.30 | 72.53 | 72.91 | 81.72 |

正式比较口径：

- 同一设置可以测试多个 seed。
- 主结果不取平均值。
- 主结果按候选 seed 中主指标 `H` 的最大值记录。
- 其它 seed 作为复查记录保留。

## 分支归属

| 实验分类 | 分支 | 分类队列 | 实验目录 |
|---|---|---|---|
| 创新模块 | `experiment/innovation` | `创新指导清单/queues/01_module_replacement.md` | `experiments/01_module_replacement/` |
| 消融实验 | `experiment/ablation` | `创新指导清单/queues/02_ablation.md` | `experiments/02_ablation/` |
| 调参实验 | `experiment/tuning` | `创新指导清单/queues/03_hyperparam_tuning.md` | `experiments/03_hyperparam_tuning/` |
| 跨数据集 | `experiment/cross-dataset` | `创新指导清单/queues/04_cross_dataset.md` | `experiments/04_cross_dataset/` |
| 最终复核 | `experiment/final-runs` | `创新指导清单/queues/05_final_runs.md` | `experiments/05_final_runs/` |

## P0

| ID | 状态 | 分类 | 实验 | 来源队列 | 备注 |
|---|---|---|---|---|---|
| MOD-001 | open | 创新模块 | 几何感知属性路由 | `queues/01_module_replacement.md` | 当前最高选择分；开始创新模块实验时优先执行 |

## P1

| ID | 状态 | 分类 | 实验 | 来源队列 | 备注 |
|---|---|---|---|---|---|
| MOD-002 | open | 创新模块 | 拓扑感知的自适应文本属性库 | `queues/01_module_replacement.md` | 连接 ABL-003 与 TPR，潜力高但改动略大 |
| MOD-003 | open | 创新模块 | 带不确定性门控的分支一致性蒸馏 | `queues/01_module_replacement.md` | 低侵入，基于 ABL-004 |

## 更新规则

- `backlog.md` 只作为当前执行窗口，建议保留 3 到 10 个近期实验。
- 新想法先进入创意树或对应分类队列，不直接塞进 backlog。
- 启动某一分类实验时，先读对应分类队列，再把最高优先级项目同步到 backlog。
- 实验开始前，把对应项从 `open` 改成 `running`。
- 实验结束后，改成 `done`、`blocked`、`rejected`，或保留 `open` 并写清原因。
- 每个实验完成后，更新实验 README、`experiments/EXPERIMENT_REGISTRY.md`、框架图和对应分类队列。
