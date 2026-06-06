# 实验 backlog

更新时间：2026-06-06

这个文件是短期实验队列，只记录下一批要做什么。长期总表见 `experiments/EXPERIMENT_REGISTRY.md`。

项目正式比较口径：同一设置可以测试多个 seed，但主结果不取平均值；按主指标 H 取候选 seed 中的最大值。其它 seed 只作为复查记录保留。

当前 baseline：局部补丁选择 + 几何感知编码 + 双向视觉-文本交互 + 条件文本扰动 + AG-JEPA 辅助训练；CUB GZSL seed=5，U=73.30，S=72.53，H=72.91，ZS=81.72。后续新增模块、消融和调参实验默认与该 baseline 比较。

## 分支归属

后续实验按大模块分支维护，`main` 只保留当前 baseline 代码、主配置和总说明。

| 分支 | 用途 |
|---|---|
| `experiment/ablation` | 消融实验，包含 `ABL-001` 到 `ABL-006` 的完整记录 |
| `experiment/innovation` | 新增创新模块和模块替换实验 |
| `experiment/tuning` | 超参数、loss 权重、top-K、训练日程扫描 |
| `experiment/cross-dataset` | AWA2 / SUN 等跨数据集迁移 |
| `experiment/final-runs` | 最终多 seed 复核、热重启上限、正式结果表 |

## P0

| ID | 状态 | 实验 | 假设 | 配置 | 运行命令 | 备注 |
|---|---|---|---|---|---|---|
| - | - | - | - | - | - | main 不排队具体实验；到对应大模块分支维护 |

## P1

| ID | 状态 | 实验 | 假设 | 配置 | 运行命令 | 备注 |
|---|---|---|---|---|---|---|

## 更新规则

- 每次实验开始前，把对应项从 `open` 改成 `running`。
- 每次实验结束后，把对应项改成 `done`、`blocked`、`rejected` 或保留 `open` 并写清原因。
- 新想法先放入 P2/P3；只有假设清楚、指标明确、成本可控时才升到 P0/P1。
