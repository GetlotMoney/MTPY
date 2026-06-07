# COMBO 组合模块实验

本目录用于 `COMBO-xxx` 实验，判断两个或多个机制是否存在协同贡献。

组合实验必须写清：

- 组成模块：原始实验 ID、模块名、单模块结果和结论。
- 协同假设：为什么这些模块可能互补。
- 冲突分析：是否修改同一位置、同一 loss、同一梯度目标或同一语义空间。
- 对照表：baseline、每个组成模块单独结果、组合结果。
- 组合收益：

```text
combo_gain = COMBO_H - max(baseline_H, module_A_H, module_B_H, ...)
```

只有 `combo_gain > 0`，才说明组合超过 baseline 和各单模块最好结果。

