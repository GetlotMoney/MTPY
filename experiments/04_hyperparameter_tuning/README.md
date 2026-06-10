# 调参实验

放“调参实验”。

例子：

- 补丁数量 K
- sigma
- 损失权重 lambda
- AG-JEPA top-k
- 学习率流程
- 局部分数权重

规则：

- 每次只扫一类参数。
- 记录完整参数列表，不要只记录最优结果。
- 如果只跑了单个随机种子，结论只能写“候选”，不能写最终结论。

## 2026-06-10 组合调参批次

当前 baseline: TUNE-004 (lambda_topo_pearson=0.1, H=73.35)
批次 start: main 9533b0d -> exp/TUNE-combo-baseline-20260610

| ID | 调参变量 | 变化 |
|---|---:|---:|
| TUNE-021 | lambda_topo_pearson=0.1, conditional_text_ratio=0.01 | 73.72 |
| TUNE-022 | lambda_topo_pearson=0.1, lambda_jepa_neg=0.01 | 73.70 |
| TUNE-023 | lambda_topo_pearson=0.1, lambda_jepa=0.0 | 73.70 |
| TUNE-024 | lambda_topo_pearson=0.1, conditional_text_ratio=0.01, lambda_jepa_neg=0.01 | 74.09 |
| TUNE-025 | lambda_topo_pearson=0.1, conditional_text_ratio=0.01, lambda_jepa=0.0 | 73.91 |
| TUNE-026 | lambda_topo_pearson=0.1, lambda_jepa=0.0, lambda_jepa_neg=0.01 | 73.79 |
| TUNE-027 | lambda_topo_pearson=0.15 | 73.98 |
| TUNE-028 | lambda_topo_pearson=0.08 | 73.28 |
| TUNE-029 | lambda_topo_pearson=0.1, lastvit_select_k=16 | 73.37 |
| TUNE-030 | lambda_topo_pearson=0.1, conditional_text_ratio=0.01, lastvit_select_k=16 | 73.40 |

批次总结: TUNE-BATCH-20260610.md
全量参数矩阵: PARAMETER_TUNING_MATRIX.md
