# MOD-003 Codex 自审

日期: 2026-06-07

结论: ACCEPTED

## 1. 代码与配置正确性

决策: ACCEPTED

- 新模块由 `use_uncertainty_msdn_gate` 控制，主配置默认关闭。
- 关闭时仍执行 `loss += lambda_msdn * loss_msdn`。
- 打开时 gate 在 `torch.no_grad()` 中由分支熵和 JS 分歧计算，只缩放 MSDN loss，不改变 logits。

## 2. 实验设计与可复现性

决策: ACCEPTED

- 实验配置独立存放。
- 本次只改变一个主要变量：MSDN loss 外层动态门控。
- 来源复核标记 `未完整参考论文原文` / `未参考官方代码`，本实验只验证本项目已有 MSDN loss 的保守改造。

## 3. 轻量验证

已执行:

```text
F:\Anaconda\envs\dvsr_gpu\python.exe -m py_compile model\MyModel.py train_VGSR_CUB.py
```

已执行 dummy forward/loss:

- 打开 `use_uncertainty_msdn_gate=True`: logits `[2,150]`，`loss_msdn_gate≈0.200`，loss 可计算。
- 关闭 `use_uncertainty_msdn_gate=False`: logits `[2,150]`，`loss_msdn_gate=1.0`。
