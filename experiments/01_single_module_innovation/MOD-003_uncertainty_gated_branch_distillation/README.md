# MOD-003 带不确定性门控的分支一致性蒸馏

日期: 2026-06-07

类型: 创新模块 / 新增代码模块

状态: 已完成 / 不保留

## 1. 实验目的

验证在现有 MSDN 双分支互蒸馏 loss 外加入不确定性门控，是否能减少错误共识并提升 CUB GZSL 主指标 H。

当前 baseline:

| 数据集 | seed | U | S | H | ZS |
|---|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 73.30 | 72.53 | 72.91 | 81.72 |

本次正式比较口径: 固定 `seed=5` 单次对照，只与同 seed 当前主框架 baseline 比较。

## 2. 来源复核与迁移规格

| 项 | 内容 |
|---|---|
| 论文 | MSDN / MSDN++: Mutually Semantic Distillation / Mutually Causal Semantic Distillation Network for Zero-Shot Learning |
| 本地 PDF | `创新指导清单/papers/MSDN Mutually Semantic Distillation Network for Zero-Shot Learning.pdf` |
| PDF 状态 | 已尝试下载到本地，但当前文件解析报 `STREAM_TRUNCATED_PREMATURELY`，不能作为完整论文原文读取 |
| 官方/相关代码 | 已尝试查找 MSDN/MSDN++ 公开代码；本轮未能稳定访问可审查源码 |
| 本次是否参考论文原文 | 否，标记为 `未完整参考论文原文` |
| 本次是否参考官方代码 | 否，标记为 `未参考官方代码` |

当前项目已有 MSDN 风格互蒸馏:

```text
score_s2v[:, seen] <-> score_v2s[:, seen]
loss_msdn = KL(p_s2v.detach || p_v2s) + KL(p_v2s.detach || p_s2v)
```

MOD-003 不改这个结构，只在外层给 `lambda_msdn * loss_msdn` 加动态 gate:

```text
confidence = 1 - mean(entropy(p_s2v), entropy(p_v2s))
agreement  = exp(-alpha * JS(p_s2v, p_v2s))
gate       = gate_min + (1 - gate_min) * confidence * agreement
loss      += lambda_msdn * mean(gate) * loss_msdn
```

关闭等价性:

- `use_uncertainty_msdn_gate=False`
- 关闭时 `loss += lambda_msdn * loss_msdn`，完全等价当前 baseline。
- 主配置默认关闭，只有本实验配置打开。

## 3. 改动内容

| 项 | 原设置 | 新设置 |
|---|---|---|
| 主配置默认 | 固定 `lambda_msdn` | 新增默认关闭门控 |
| 实验配置 | `use_uncertainty_msdn_gate=False` | `True` |
| 最小门控 | 无 | `msdn_gate_min=0.2` |
| JS 衰减系数 | 无 | `msdn_gate_js_alpha=4.0` |

最小 diff 文件:

- `model/MyModel.py`
- `train_VGSR_CUB.py`
- `config/VGSR_cub_gzsl.yaml`
- `experiments/01_single_module_innovation/MOD-003_uncertainty_gated_branch_distillation/config.yaml`

## 4. 训练配置

| 项 | 值 |
|---|---|
| 数据集 | CUB |
| 随机种子 | 5 |
| 命令 | `F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/01_single_module_innovation/MOD-003_uncertainty_gated_branch_distillation/config.yaml` |
| 配置文件 | `experiments/01_single_module_innovation/MOD-003_uncertainty_gated_branch_distillation/config.yaml` |
| 正式比较口径 | 固定 seed=5，与同 seed baseline 对照 |

## 5. 审查记录

Codex 自审: ACCEPTED，见 `codex-self-review.md`

Claude 三轮审查: ACCEPTED，见 `claude-review.md`

## 6. 结果

| seed | U | S | H | ZS | 最佳轮次 | 原始日志 | 实验日志副本 |
|---:|---:|---:|---:|---:|---:|---|---|
| 5 | 72.83 | 72.39 | 72.61 | 81.15 | 51 | `train_log/CUB/training_log_CUB_2026-06-07_19-37-58.txt` | `experiments/01_single_module_innovation/MOD-003_uncertainty_gated_branch_distillation/logs/MOD-003_CUB_seed5_20260607-193758.txt` |

MGate 观察: 训练中大多数 step 位于 `0.20~0.21`，长期贴近 `msdn_gate_min=0.2`。

## 7. 结论

保留 / 放弃 / 待补跑: 放弃当前版本

理由: 固定 seed=5 下 H=72.61，低于同 seed baseline 72.91，下降 0.30。S=72.39 接近 baseline，但 U=72.83 低于 baseline U=73.30。门控基本长期贴近最小值 0.20，说明当前 entropy + JS gate 把 MSDN 互蒸馏削弱太多，没有减少错误共识带来的收益。后续若继续这个方向，应先把 `msdn_gate_min` 提高到 0.5 或改成逐样本 loss 加权，而不是继续当前版本。

## 8. 后续动作

- [x] Codex 自审。
- [x] Claude 三轮审查。
- [x] 审查通过后运行训练。
- [x] 复制并命名实验日志。
- [x] 更新 `experiments/EXPERIMENT_REGISTRY.md`。
- [x] 更新 `backlog.md` 和 `创新指导清单/queues/01_module_replacement.md`。
- [x] 生成 `experiments/08_framework_flow_records/MOD-003_uncertainty_gated_branch_distillation.md`。
- [x] 根据结果更新创意树权重。
