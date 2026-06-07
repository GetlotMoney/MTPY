# ABL-001：去掉局部补丁选择

日期：2026-06-05

类型：消融实验

状态：已完成

## 1. 实验目的

验证局部补丁选择器是否是当前 CUB GZSL 结果的核心贡献来源。

## 2. 改动内容

| 项目 | 基线设置 | 本实验设置 |
|---|---|---|
| `lastvit_select_k` | `32` | `0` |
| 局部补丁选择器 | 开启，选择 32 个补丁 | 关闭，几何感知编码看到全部 576 个补丁 |
| `lr_stages.value` | 分段训练设置 | `null`，使用严格连续训练流程 |

本实验只允许改变上述实验变量，不允许修改模型核心代码或根目录基线配置。

## 3. 训练配置

| 项目 | 值 |
|---|---|
| 数据集 | CUB |
| 随机种子 | 5 |
| 训练流程 | 严格连续 |
| Python | `F:\Anaconda\envs\dvsr_gpu\python.exe` |
| 命令 | `F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/05_ablation/ABL-001_disable_patch_selection/config.yaml` |
| 配置文件 | `experiments/05_ablation/ABL-001_disable_patch_selection/config.yaml` |
| 设备 | `cuda:0` |
| epoch | `30` |

## 4. 实验假设

如果 32 个局部补丁选择器确实重要，那么关闭它以后，CUB GZSL 的 H 应该低于当前主基线。

项目正式比较口径：如果同一设置测试多个 seed，正式结果取主指标 H 的最大值，不取多 seed 平均值。

## 5. 对比基线

| 对比对象 | 口径 | H |
|---|---|---:|
| 当前主基线 | 严格连续训练，seed 候选池取最高 H，来源 seed=5 | 72.91 |
| 本实验 | seed=5，严格连续训练，关闭局部补丁选择 | 71.55 |

## 6. 审查记录

| 审查项 | 结果 | 备注 |
|---|---|---|
| Codex 自查 | ACCEPTED | 仅使用实验目录配置；未修改模型核心代码；checkpoint `0874c5d` 已建立 |
| Claude Code 三轮审查 | ACCEPTED | MCP job `ABL-001.round-1.bdc20be6`；三轮均为 ACCEPTED |

审查通过前不允许运行训练。

## 7. 结果

| seed | U | S | H | ZS | 最佳轮次 | 原始日志 | 实验日志副本 |
|---:|---:|---:|---:|---:|---:|---|---|
| 5 | 74.22 | 69.07 | 71.55 | 81.84 | 9 | `train_log/CUB/training_log_CUB_2026-06-05_22-43-55.txt` | `experiments/05_ablation/ABL-001_disable_patch_selection/logs/ABL-001_CUB_seed5_20260605-224355_attempt2.txt` |

运行备注：

- `train_log/CUB/training_log_CUB_2026-06-05_22-40-02.txt` 是第一次启动失败日志，原因是 Windows GBK 控制台无法编码日志里的 `⚠` 字符，训练未进入 epoch。
- 第二次运行只设置进程环境变量 `PYTHONIOENCODING=utf-8`，未修改模型代码或实验配置，训练完整跑完 30 epoch。
- 失败日志副本：`experiments/05_ablation/ABL-001_disable_patch_selection/logs/ABL-001_CUB_seed5_20260605-224002_attempt1_failed_encoding.txt`。

## 8. 结论

状态：完成。

观察事实：关闭局部补丁选择后，seed=5 的最佳 H 为 71.55，低于当前主基线 H=72.91，下降 1.36。

结论：在当前 CUB 设置下，局部补丁选择对主框架有效；去掉它会削弱 GZSL-H。该实验支持继续保留局部补丁选择，并优先做更细的补丁数量或选择策略消融。

## 9. 后续动作

- [x] 启动 skill 后创建实验前 Git checkpoint。
- [x] Codex 自审。
- [x] Claude Code 固定三轮审查。
- [x] 审查全部通过后运行训练。
- [x] 复制训练日志到本实验 `logs/` 目录，并使用 `ABL-001_CUB_seed5_<YYYYMMDD-HHMMSS>.txt` 命名。
- [x] 生成 `experiments/08_framework_flow_records/ABL-001_disable_patch_selection.md`，记录代码框架图、流程说明和本实验数据。
- [x] 更新 `experiments/EXPERIMENT_REGISTRY.md`。
- [x] 更新 `backlog.md`。
