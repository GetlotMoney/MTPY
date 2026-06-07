# 实验 ID 与多模块规范

## 实验类型

| 前缀 | 类型 | 默认目录 | 目的 |
|---|---|---|---|
| `MOD-xxx` | 单模块创新实验 | `experiments/01_module_replacement/` | 判断单个机制是否有独立贡献。 |
| `COMBO-xxx` | 组合模块实验 | `experiments/01_module_replacement/`，必要时可建 `experiments/07_combo/` | 判断两个或多个机制是否有协同贡献。 |
| `REV-MOD-xxx` | 单模块复核 | `experiments/01_module_replacement/` | 对 near_tie、win 或用户指定模块做多 seed / 同配置复查。 |
| `TUNE-xxx` | 调参实验 | `experiments/03_hyperparam_tuning/` | 扫超参、loss 权重、top-K、margin、训练日程。 |
| `ABL-xxx` | 消融实验 | `experiments/02_ablation/` | 验证已有模块、loss 或路径是否必须保留。 |
| `XDS-xxx` | 跨数据集实验 | `experiments/04_cross_dataset/` | 验证 AWA2 / SUN 等泛化能力。 |
| `FINAL-xxx` | 最终复核 | `experiments/05_final_runs/` | 服务正式汇报、论文表格或最终主结果。 |

## 默认选择顺序

```text
REV-MOD 候选复核
  ↓
COMBO 有明确协同假设的组合模块
  ↓
MOD 新单模块初筛
  ↓
TUNE 调参
  ↓
XDS / FINAL
```

达到 `H >= 74` 之前，系统调参不是主路径，除非用户明确指定。

## 单模块实验

`MOD` 必须满足：

- 有明确创意树节点或临时用户指定原因。
- 一次只改一个主要机制。
- 新代码默认关闭，只在实验 `config.yaml` 打开。
- 关闭开关后必须退回当前 baseline。
- README 写清插入点、配置键、默认值、实验值和最小 diff。

## 替换而不是简单堆叠

创新模块不能默认堆叠。实现前必须判断它和 baseline 模块的关系：

| 关系 | 含义 | 实验设计 |
|---|---|---|
| `complementary` | 输入、目标或梯度方向互补 | 可以叠加，但必须默认关闭并控制权重。 |
| `overlap` | 处理同一信号或同一 loss 目标 | 优先降权、共享中间量，或只替换局部目标。 |
| `replacement` | 新机制意图替代已有路径 | 关闭被替代模块，用新机制替换。 |
| `conflict` | 可能拉相反梯度、重复校准或破坏同一语义空间 | 不允许直接堆叠，先做隔离或替换实验。 |
| `unknown` | 关系不清楚 | 写清不确定点，优先低风险隔离设计。 |

## 组合实验

`COMBO` 只能在以下情况创建：

- 两个模块理论互补。
- 单模块是 `near_tie` 或小幅 `soft_negative`，且有明确互补解释。
- 一个模块主要提升 U，另一个主要提升 S。
- 创意树节点存在连接关系、来自同一论文、同一机制链或同一设计原则。
- 用户明确指定组合尝试。

禁止组合：

- 两个模块都是 `hard_negative`。
- 两个模块改同一个位置、同一分支或同一 loss 目标。
- 只是因为“都试过了所以堆一下”。
- 没有写组成模块、协同假设、冲突分析和对照基线。
- 同时改模块、seed、学习率、epoch、评估 bias 或训练日程。

`COMBO` README 必须包含：

- 组成模块：原始实验 ID、模块名、单模块结果和结论。
- 协同假设：为什么互补。
- 冲突分析：是否存在同一位置、同一 loss、同一梯度目标或同一语义空间冲突。
- 对照表：baseline、每个组成模块单独结果、组合结果。

组合收益：

```text
combo_gain = COMBO_H - max(baseline_H, module_A_H, module_B_H, ...)
```

只有 `combo_gain > 0`，才说明组合超过 baseline 和各单模块最好结果。

## 单模块复核

`REV-MOD` 用于复核：

- `win`
- `near_tie`
- 用户指定的重点模块

复核必须预注册：

- 原始 `MOD` ID。
- 继承的代码/config。
- seed 列表。
- 需要补跑的 seed。
- 判断口径：仍看候选 seed 最大 `H`，但保留所有 seed。

