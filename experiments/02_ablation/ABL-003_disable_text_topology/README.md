# ABL-003：去掉文本拓扑保持

日期：2026-06-05

类型：消融实验

状态：进行中，等待审查放行

## 1. 实验目的

验证文本类别原型拓扑保持约束是否是当前 CUB GZSL 结果的有效贡献来源。

## 2. 改动内容

| 项目 | 基线设置 | 本实验设置 |
|---|---|---|
| `lambda_topo_pearson` | `0.05` | `0.0` |
| 文本拓扑保持 loss | 开启 | 关闭 |
| `lr_stages.value` | 根配置里的多段热重启 | `null`，使用严格连续训练流程 |

本实验只允许改变上述实验变量，不允许修改模型核心代码或根目录基线配置。

## 3. 训练配置

| 项目 | 值 |
|---|---|
| 数据集 | CUB |
| 随机种子 | 5 |
| 训练流程 | 严格连续 |
| Python | `F:\Anaconda\envs\dvsr_gpu\python.exe` |
| 命令 | `F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/02_ablation/ABL-003_disable_text_topology/config.yaml` |
| 配置文件 | `experiments/02_ablation/ABL-003_disable_text_topology/config.yaml` |
| 设备 | `cuda:0` |
| epoch | `30` |

## 4. 实验假设

如果文本拓扑保持有效，那么关闭它以后，CUB GZSL 的 H 应该低于当前主基线。

项目正式比较口径：如果同一设置测试多个 seed，正式结果取主指标 H 的最大值，不取多 seed 平均值。

## 5. 对比基线

| 对比对象 | 口径 | H |
|---|---|---:|
| 当前主基线 | 严格连续训练，seed 候选池取最高 H，来源 seed=5 | 72.91 |
| 本实验 | 待运行 | - |

## 6. 审查记录

| 审查项 | 结果 | 备注 |
|---|---|---|
| Codex 自查 | ACCEPTED | 仅使用实验目录配置；未修改模型核心代码；checkpoint `9a5f6ab` 已建立 |
| Claude Code 三轮审查 | 待执行 | 通过 `claude_code_worker` MCP 的默认 `cli` 模式执行 |

审查通过前不允许运行训练。

## 7. 结果

| seed | U | S | H | ZS | 最佳轮次 | 原始日志 | 实验日志副本 |
|---:|---:|---:|---:|---:|---:|---|---|
| 5 |  |  |  |  |  |  |  |

## 8. 结论

状态：待定

决策：待训练和结果分析后填写。

## 9. 后续动作

- [x] 创建 ABL-003 实验前 Git checkpoint。
- [x] Codex 自审。
- [ ] Claude Code 固定三轮审查。
- [ ] 审查全部通过后运行训练。
- [ ] 复制训练日志到本实验 `logs/` 目录，并使用 `ABL-003_CUB_seed5_<YYYYMMDD-HHMMSS>.txt` 命名。
- [ ] 生成 `experiments/06_framework_flows/ABL-003_disable_text_topology.md`，记录代码框架图、流程说明和本实验数据。
- [ ] 更新 `experiments/EXPERIMENT_REGISTRY.md`。
- [ ] 更新 `backlog.md`。
