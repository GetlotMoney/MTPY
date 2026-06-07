# MOD-004 Codex 自审

日期: 2026-06-07

结论: ACCEPTED

## 1. 代码与配置正确性

决策: ACCEPTED

- 新模块由 `use_attr_patch_ot` 和 `lambda_attr_patch_ot` 控制，主配置默认关闭。
- 关闭时不传属性文本矩阵、不返回 `attr_patch_sim`，不改变 baseline。
- 打开时使用 FAE 后 memory 与 CUB 属性文本原型计算 patch-attribute 相似度。

## 2. 实验设计与可复现性

决策: ACCEPTED

- 本次只改变一个主要变量：新增属性引导 patch OT 辅助 loss。
- 论文原文已在 MOD-001 读过；官方代码未能访问，标记 `未参考官方代码`。

## 3. 轻量验证

已执行:

```text
F:\Anaconda\envs\dvsr_gpu\python.exe -m py_compile model\MyModel.py train_VGSR_CUB.py
```

已执行 dummy forward/loss:

- 关闭 `use_attr_patch_ot=False`: logits `[2,150]`，`loss_attr_patch_ot=0.0`，不返回 `attr_patch_sim`。
- 打开 `use_attr_patch_ot=True`: logits `[2,150]`，`loss_attr_patch_ot=0.8877`，返回 `attr_patch_sim`。
