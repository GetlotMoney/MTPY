# 审查包：ABL-004

## 上下文

- 实验编号：`ABL-004`
- 实验名称：去掉双分支互蒸馏
- 当前分支：`experiment/batch-ablation-cub-20260605`
- Git checkpoint：`f10a663`，提交信息 `Prepare ABL-004 workflow checkpoint`
- 实验记录：`experiments/02_ablation/ABL-004_disable_branch_distillation/README.md`
- 实验配置：`experiments/02_ablation/ABL-004_disable_branch_distillation/config.yaml`
- 基线配置：`config/VGSR_cub_gzsl.yaml`
- 审查输出：`experiments/02_ablation/ABL-004_disable_branch_distillation/claude-review.md`

## 审查请求

请审查当前实验准备是否允许进入训练阶段。不要修改代码，不要运行训练。

Codex 自审结果：`ACCEPTED`。理由：当前准备只新增 ABL-004 实验目录并更新队列状态，实验配置只把 `lambda_msdn.value` 从基线的 `0.05` 改为 `0.0`，同时保持 `lastvit_select_k=32`、`use_ag_jepa=True`、`lambda_topo_pearson=0.05`、`random_seed=5` 和严格连续训练流程；未修改模型核心代码或根目录基线配置。

必须执行固定三轮审查：

1. Pass 1：代码/配置正确性、改动最小性、无关改动风险。
2. Pass 2：实验设计、指标有效性、基线可比性、seed/config 可复现性、数据泄漏风险。
3. Pass 3：最终运行计划、环境、命令、日志命名、回滚安全性、是否允许开跑。

## 预期实验

ABL-004 用于验证 s2v 与 v2s 两条视觉-文本分支之间的互蒸馏约束是否是 CUB GZSL 当前结果的真实增益来源。

该实验只应改变一个主要变量：

- 基线：`lambda_msdn.value = 0.05`
- 本实验：`lambda_msdn.value = 0.0`

该实验配置还通过以下设置关闭分段训练：

```yaml
lr_stages:
  value: null
```

这样做是为了遵守项目规范：消融实验使用严格连续训练流程。

## 当前配置证据

实验配置中关键字段应为：

```yaml
dataset:
  value: CUB
device:
  value: cuda:0
epochs:
  value: 30
random_seed:
  value: 5
lastvit_select_k:
  value: 32
use_ag_jepa:
  value: True
lambda_topo_pearson:
  value: 0.05
lambda_msdn:
  value: 0.0
lr_stages:
  value: null
```

## 项目评价口径

项目正式比较口径不是多 seed 平均值，而是：

- 同一设置可以测试多个 seed。
- 每个 seed 的 U / S / H / ZS / 最佳轮次 / 日志路径都必须记录。
- 正式比较值取主指标 H 的最大值。
- 当前主基线 H = 72.91，来源为严格连续训练 seed=5。

本实验第一轮测试只跑 seed=5。

## 计划运行命令

只有 Codex 自审和 Claude 三轮审查全部通过后，才允许运行：

```powershell
$env:PYTHONIOENCODING='utf-8'; F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/02_ablation/ABL-004_disable_branch_distillation/config.yaml
```

`PYTHONIOENCODING=utf-8` 只用于避免 Windows 控制台 Unicode 输出失败，不改变训练配置。

主指标：CUB GZSL H。

辅助指标：U、S、ZS、最佳轮次、原始日志路径、实验日志副本路径。

实验日志副本命名规则：

```text
experiments/02_ablation/ABL-004_disable_branch_distillation/logs/ABL-004_CUB_seed5_<YYYYMMDD-HHMMSS>.txt
```

## 审查重点

- 是否使用实验文件夹内的配置，而不是修改根目录基线配置？
- `lambda_msdn: 0.0` 是否能在 `model/MyModel.py` 中正确关闭双分支互蒸馏 loss？
- 是否保留当前主框架其它关键模块：`lastvit_select_k=32`、`use_ag_jepa=True`、`lambda_topo_pearson=0.05`？
- `lr_stages.value: null` 在当前代码库中是否代表严格连续训练流程？
- 是否避免了无关模型或训练改动？
- 命令是否使用 `F:\Anaconda\envs\dvsr_gpu\python.exe`？
- 日志命名和结果记录是否符合实验 workflow？
- 实验完成后是否会生成 `experiments/06_framework_flows/ABL-004_disable_branch_distillation.md`，并包含流程图、流程说明、指标数据和日志路径？

## 期望输出格式

Markdown 审查结果靠前位置必须包含：

```text
Pass 1 Decision: ACCEPTED 或 REJECTED
Pass 2 Decision: ACCEPTED 或 REJECTED
Pass 3 Decision: ACCEPTED 或 REJECTED
Overall Decision: ACCEPTED 或 REJECTED
```

只有三轮全部 `ACCEPTED` 时，`Overall Decision` 才能是 `ACCEPTED`。
