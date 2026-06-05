# 实验 backlog

更新时间：2026-06-05

这个文件是短期实验队列，只记录下一批要做什么。长期总表见 `experiments/EXPERIMENT_REGISTRY.md`。

项目正式比较口径：同一设置可以测试多个 seed，但主结果不取平均值；按主指标 H 取候选 seed 中的最大值。其它 seed 只作为复查记录保留。

## P0

| ID | 状态 | 实验 | 假设 | 配置 | 运行命令 | 备注 |
|---|---|---|---|---|---|---|
| `ABL-001` | open | 去掉局部补丁选择 | 32 个局部补丁选择是否是当前 CUB GZSL 结果的核心贡献来源 | `experiments/02_ablation/ABL-001_disable_patch_selection/config.yaml` | `F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/02_ablation/ABL-001_disable_patch_selection/config.yaml` | 第一个从头测试 skill 的实验；先审查，通过后再训练 |

## P1

| ID | 状态 | 实验 | 假设 | 配置 | 运行命令 | 备注 |
|---|---|---|---|---|---|---|
| `ABL-002` | open | 去掉 AG-JEPA 辅助训练 | AG-JEPA 是否带来真实增益 | 待创建 | 待确认 | ABL-001 完成后优先考虑 |
| `ABL-003` | open | 去掉文本拓扑保持 | 文本语义结构约束是否有效 | 待创建 | 待确认 | 论文消融候选 |
| `ABL-004` | open | 去掉双分支互蒸馏 | 两条视觉-文本分支是否需要互相约束 | 待创建 | 待确认 | 论文消融候选 |

## 更新规则

- 每次实验开始前，把对应项从 `open` 改成 `running`。
- 每次实验结束后，把对应项改成 `done`、`blocked`、`rejected` 或保留 `open` 并写清原因。
- 新想法先放入 P2/P3；只有假设清楚、指标明确、成本可控时才升到 P0/P1。
