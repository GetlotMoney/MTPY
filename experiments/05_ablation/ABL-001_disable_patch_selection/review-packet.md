# 审查包：ABL-001

## 上下文

- 实验编号：`ABL-001`
- 实验名称：去掉局部补丁选择
- 当前分支：`experiment/ABL-001-disable-patch-selection`
- 实验记录：`experiments/05_ablation/ABL-001_disable_patch_selection/README.md`
- 实验配置：`experiments/05_ablation/ABL-001_disable_patch_selection/config.yaml`
- 基线配置：`config/VGSR_cub_gzsl.yaml`
- 实验前 checkpoint：`0874c5d Prepare ABL-001 workflow checkpoint`
- 审查输出：`experiments/05_ablation/ABL-001_disable_patch_selection/claude-review.md`

## 审查请求

请审查当前实验准备是否允许进入训练阶段。不要修改代码，不要运行训练。

必须执行固定三轮审查：

1. Pass 1：代码/配置正确性、改动最小性、无关改动风险。
2. Pass 2：实验设计、指标有效性、基线可比性、seed/config 可复现性、数据泄漏风险。
3. Pass 3：最终运行计划、环境、命令、日志命名、回滚安全性、是否允许开跑。

## 预期实验

ABL-001 用于验证局部补丁选择器是否是 CUB GZSL 当前结果的核心贡献来源。

该实验只应改变一个主要变量：

- 基线：`lastvit_select_k.value = 32`
- 本实验：`lastvit_select_k.value = 0`

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
  value: 0
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
F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/05_ablation/ABL-001_disable_patch_selection/config.yaml
```

主指标：CUB GZSL H。

辅助指标：U、S、ZS、最佳轮次、原始日志路径、实验日志副本路径。

实验日志副本命名规则：

```text
experiments/05_ablation/ABL-001_disable_patch_selection/logs/ABL-001_CUB_seed5_<YYYYMMDD-HHMMSS>.txt
```

## 审查重点

- 是否使用实验文件夹内的配置，而不是修改根目录基线配置？
- `lastvit_select_k: 0` 是否能在 `model/MyModel.py` 中正确关闭局部补丁选择？
- `lr_stages.value: null` 在当前代码库中是否代表严格连续训练流程？
- 是否避免了无关模型或训练改动？
- 命令是否使用 `F:\Anaconda\envs\dvsr_gpu\python.exe`？
- 日志命名和结果记录是否符合实验 workflow？
- 实验完成后是否会生成 `experiments/08_framework_flow_records/ABL-001_disable_patch_selection.md`，并包含流程图、流程说明、指标数据和日志路径？
- 是否应该在训练前先创建新的 Git checkpoint？

## 期望输出格式

Markdown 审查结果靠前位置必须包含：

```text
Pass 1 Decision: ACCEPTED 或 REJECTED
Pass 2 Decision: ACCEPTED 或 REJECTED
Pass 3 Decision: ACCEPTED 或 REJECTED
Overall Decision: ACCEPTED 或 REJECTED
```

只有三轮全部 `ACCEPTED` 时，`Overall Decision` 才能是 `ACCEPTED`。
