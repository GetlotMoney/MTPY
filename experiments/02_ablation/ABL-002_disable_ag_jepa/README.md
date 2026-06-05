# ABL-002：去掉 AG-JEPA 辅助训练

日期：2026-06-05

类型：消融实验

状态：已完成

## 1. 实验目的

验证 AG-JEPA 辅助训练是否是当前 CUB GZSL 结果的真实增益来源。

## 2. 改动内容

| 项目 | 基线设置 | 本实验设置 |
|---|---|---|
| `use_ag_jepa` | `True` | `False` |
| `lambda_jepa` | `0.05` | `0.0` |
| `lambda_jepa_neg` | `0.02` | `0.0` |
| `lr_stages.value` | 根配置里的多段热重启 | `null`，使用严格连续训练流程 |

本实验只允许改变上述实验变量，不允许修改模型核心代码或根目录基线配置。

## 3. 训练配置

| 项目 | 值 |
|---|---|
| 数据集 | CUB |
| 随机种子 | 5 |
| 训练流程 | 严格连续 |
| Python | `F:\Anaconda\envs\dvsr_gpu\python.exe` |
| 命令 | `F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/02_ablation/ABL-002_disable_ag_jepa/config.yaml` |
| 配置文件 | `experiments/02_ablation/ABL-002_disable_ag_jepa/config.yaml` |
| 设备 | `cuda:0` |
| epoch | `30` |

## 4. 实验假设

如果 AG-JEPA 辅助训练有效，那么关闭它以后，CUB GZSL 的 H 应该低于当前主基线。

项目正式比较口径：如果同一设置测试多个 seed，正式结果取主指标 H 的最大值，不取多 seed 平均值。

## 5. 对比基线

| 对比对象 | 口径 | H |
|---|---|---:|
| 当前主基线 | 严格连续训练，seed 候选池取最高 H，来源 seed=5 | 72.91 |
| 本实验 | seed=5，严格连续训练，关闭 AG-JEPA | 71.08 |

## 6. 审查记录

| 审查项 | 结果 | 备注 |
|---|---|---|
| Codex 自查 | ACCEPTED | 仅使用实验目录配置；未修改模型核心代码；checkpoint `3d828ad` 已建立 |
| Claude Code 三轮审查 | ACCEPTED | MCP job `ABL-002.round-1.0d4b14c5`；三轮均为 ACCEPTED |

审查通过前不允许运行训练。

## 7. 结果

| seed | U | S | H | ZS | 最佳轮次 | 原始日志 | 实验日志副本 |
|---:|---:|---:|---:|---:|---:|---|---|
| 5 | 76.00 | 66.76 | 71.08 | 81.66 | 11 | `train_log/CUB/training_log_CUB_2026-06-05_23-39-36.txt` | `experiments/02_ablation/ABL-002_disable_ag_jepa/logs/ABL-002_CUB_seed5_20260605-233936.txt` |

## 8. 结论

状态：完成。

观察事实：关闭 AG-JEPA 后，seed=5 的最佳 H 为 71.08，低于当前主基线 H=72.91，下降 1.83。

结论：在当前 CUB 设置下，AG-JEPA 辅助训练对主框架有效；关闭后 seen/unseen 平衡和主指标 H 都变差。该实验支持继续保留 AG-JEPA，并优先做 AG-JEPA 权重扫描或负文本 margin 扫描。

## 9. 后续动作

- [x] 创建 ABL-002 实验前 Git checkpoint。
- [x] Codex 自审。
- [x] Claude Code 固定三轮审查。
- [x] 审查全部通过后运行训练。
- [x] 复制训练日志到本实验 `logs/` 目录，并使用 `ABL-002_CUB_seed5_<YYYYMMDD-HHMMSS>.txt` 命名。
- [x] 生成 `experiments/06_framework_flows/ABL-002_disable_ag_jepa.md`，记录代码框架图、流程说明和本实验数据。
- [x] 更新 `experiments/EXPERIMENT_REGISTRY.md`。
- [x] 更新 `backlog.md`。
