# Hyperparameter Tuning Experiments

放“调参实验”。

例子：

- patch 数量 K
- sigma
- loss 权重 lambda
- AG-JEPA top-k
- 学习率 schedule
- local score 权重

规则：

- 每次只扫一类参数。
- 记录完整参数列表，不要只记录最优结果。
- 如果只跑了单 seed，结论只能写“候选”，不能写最终结论。
