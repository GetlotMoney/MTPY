# TUNE-025 组合调参: topo=0.1 + cond_text=0.01 + jepa=0.0

类型: TUNE
范式: TUNE-LITE
状态: completed

## 调参变量

| key | TUNE-004 | 本实验 |
|---|---:|---:|
| lambda_topo_pearson | 0.05 -> 0.10 | 0.10 |
| conditional_text_ratio | 0.005 | 0.010 |
| lambda_jepa | 0.05 | 0.0 |

## 结果

| 数据集 | seed | U | S | H | ZS | 最佳 epoch |
|---|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 71.93 | 76.01 | 73.91 | 82.13 | 26 |

结论: 73.91, 低于 TUNE-024(74.09), 不提升 baseline。
