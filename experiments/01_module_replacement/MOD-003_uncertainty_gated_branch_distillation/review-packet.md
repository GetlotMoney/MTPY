# MOD-003 Claude 审查包

请对本实验做固定三轮审查。每一轮必须返回 `ACCEPTED` 或 `REJECTED`，最终给出 Overall 决策。只审查，不修改代码。

## 实验

- ID: MOD-003
- 名称: 带不确定性门控的分支一致性蒸馏
- 目录: `experiments/01_module_replacement/MOD-003_uncertainty_gated_branch_distillation/`
- 目标: 验证在现有 MSDN 双分支互蒸馏 loss 外加入不确定性门控，是否能减少错误共识并提升 CUB GZSL H。
- 正式比较口径: 固定 `seed=5` 单次对照，只与同 seed当前主框架 baseline 比较。

## 关键文件

- `model/MyModel.py`
- `train_VGSR_CUB.py`
- `config/VGSR_cub_gzsl.yaml`
- `experiments/01_module_replacement/MOD-003_uncertainty_gated_branch_distillation/config.yaml`
- `experiments/01_module_replacement/MOD-003_uncertainty_gated_branch_distillation/README.md`
- `experiments/01_module_replacement/MOD-003_uncertainty_gated_branch_distillation/codex-self-review.md`

## 来源复核

- MSDN/MSDN++ 论文 PDF 下载不完整，不能完整解析。
- 公开代码未能稳定访问。
- 标记: `未完整参考论文原文`；`未参考官方代码`。
- 当前项目已有 MSDN 风格互蒸馏实现，本实验只给现有 loss 加低侵入门控，不声称复现 MSDN++。

## 迁移设计

```text
score_s2v[:, seen], score_v2s[:, seen]
  -> p_s2v, p_v2s
  -> entropy confidence
  -> JS agreement
  -> gate = gate_min + (1 - gate_min) * confidence * agreement
  -> loss += lambda_msdn * mean(gate) * loss_msdn
```

关闭等价性:

- 根配置默认 `use_uncertainty_msdn_gate=False`
- 关闭时仍是 `loss += lambda_msdn * loss_msdn`
- 实验配置才打开 `use_uncertainty_msdn_gate=True`

## 已执行验证

```text
F:\Anaconda\envs\dvsr_gpu\python.exe -m py_compile model\MyModel.py train_VGSR_CUB.py
```

Dummy forward/loss:

- 打开模块: `clip_S_pp` 形状 `[2,150]`，`loss_msdn_gate≈0.200`，loss 可计算。
- 关闭模块: `clip_S_pp` 形状 `[2,150]`，`loss_msdn_gate=1.0`，路径等价固定 MSDN。

## 三轮审查要求

Pass 1: 代码/配置正确性。

Pass 2: 实验设计和可复现性。

Pass 3: 运行计划和放行。

## 输出要求

请把审查结果写入:

```text
experiments/01_module_replacement/MOD-003_uncertainty_gated_branch_distillation/claude-review.md
```

格式必须包含 Overall / Pass 1 / Pass 2 / Pass 3 的 `ACCEPTED` 或 `REJECTED`。
