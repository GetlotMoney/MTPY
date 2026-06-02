# Ablation Experiments

放“消融实验”。

例子：

- 关闭 AG-JEPA
- 关闭文本拓扑保持
- 关闭双分支互蒸馏
- 关闭条件文本
- 关闭局部 patch 选择
- 关闭几何感知编码

规则：

- 消融实验必须保持同一个 strict schedule。
- 一次只关一个模块。
- 不要用 warm-restart 做消融对比。
