# ABL-004：去掉双分支互蒸馏

日期：2026-06-06

类型：消融实验

状态：已完成

## 1. 实验目的

验证 s2v 与 v2s 两条视觉-文本分支之间的互蒸馏约束是否是当前 CUB GZSL 结果的有效贡献来源。

## 2. 改动内容

| 项目 | 基线设置 | 本实验设置 |
|---|---|---|
| `lambda_msdn` | `0.05` | `0.0` |
| 双分支互蒸馏 loss | 开启 | 关闭 |
| `lr_stages.value` | 根配置里的多段热重启 | `null`，使用严格连续训练流程 |

本实验只允许改变上述实验变量，不允许修改模型核心代码或根目录基线配置。

## 3. 训练配置

| 项目 | 值 |
|---|---|
| 数据集 | CUB |
| 随机种子 | 5 |
| 训练流程 | 严格连续 |
| Python | `F:\Anaconda\envs\dvsr_gpu\python.exe` |
| 命令 | `F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/02_ablation/ABL-004_disable_branch_distillation/config.yaml` |
| 配置文件 | `experiments/02_ablation/ABL-004_disable_branch_distillation/config.yaml` |
| 设备 | `cuda:0` |
| epoch | `30` |

## 4. 实验假设

如果双分支互蒸馏有效，那么关闭它以后，CUB GZSL 的 H 应该低于当前主基线。

项目正式比较口径：如果同一设置测试多个 seed，正式结果取主指标 H 的最大值，不取多 seed 平均值。

## 5. 对比基线

| 对比对象 | 口径 | H |
|---|---|---:|
| 当前主基线 | 严格连续训练，seed 候选池取最高 H，来源 seed=5 | 72.91 |
| 本实验 | seed=5，严格连续训练，关闭双分支互蒸馏 | 71.00 |

## 6. 审查记录

| 审查项 | 结果 | 备注 |
|---|---|---|
| Git checkpoint | 已完成 | `f10a663`，提交信息：`Prepare ABL-004 workflow checkpoint` |
| Codex 自查 | ACCEPTED | 配置只关闭 `lambda_msdn`，保留其它主框架变量；训练命令、seed、日志命名和结果记录计划完整 |
| Claude Code 三轮审查 | ACCEPTED | MCP job `ABL-004.round-1.ba53f9ea`，三轮均为 `ACCEPTED` |

审查通过前不允许运行训练。

## 7. 结果

| seed | U | S | H | ZS | 最佳轮次 | 原始日志 | 实验日志副本 |
|---:|---:|---:|---:|---:|---:|---|---|
| 5 | 73.30 | 68.83 | 71.00 | 81.52 | 9 | `train_log/CUB/training_log_CUB_2026-06-06_00-11-33.txt` | `experiments/02_ablation/ABL-004_disable_branch_distillation/logs/ABL-004_CUB_seed5_20260606-001133.txt` |

## 8. 结论

状态：完成。

观察事实：关闭双分支互蒸馏后，seed=5 的最佳 H 为 71.00，低于当前主基线 H=72.91，下降 1.91。

结论：在当前 CUB 设置下，s2v 与 v2s 两条视觉-文本分支之间的互蒸馏约束有正贡献。该实验支持继续保留 `lambda_msdn=0.05`；后续如果要进一步优化，应优先做互蒸馏权重扫描，而不是移除该节点。

## 9. 后续动作

- [x] 创建 ABL-004 实验前 Git checkpoint。
- [x] Codex 自审。
- [x] Claude Code 固定三轮审查。
- [x] 审查全部通过后运行训练。
- [x] 复制训练日志到本实验 `logs/` 目录，并使用 `ABL-004_CUB_seed5_<YYYYMMDD-HHMMSS>.txt` 命名。
- [x] 生成 `experiments/06_framework_flows/ABL-004_disable_branch_distillation.md`，记录代码框架图、流程说明和本实验数据。
- [x] 更新 `experiments/EXPERIMENT_REGISTRY.md`。
- [x] 更新 `backlog.md`。
