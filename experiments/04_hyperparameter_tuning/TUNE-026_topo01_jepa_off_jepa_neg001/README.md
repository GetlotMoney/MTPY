# TUNE-026 组合调参: topo=0.1 + jepa=0.0 + jepa_neg=0.01

类型: TUNE
范式: TUNE-LITE
状态: completed

## 调参变量

| key | TUNE-004 | 本实验 |
|---|---:|---:|
| lambda_topo_pearson | 0.05 -> 0.10 | 0.10 |
| lambda_jepa | 0.05 | 0.0 |
| lambda_jepa_neg | 0.02 | 0.01 |

## 结果

| 数据集 | seed | U | S | H | ZS | 最佳 epoch |
|---|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 73.44 | 74.13 | 73.79 | 82.63 | 23 |

结论: 73.79, 低于 TUNE-024(74.09), 不提升 baseline。但 ZS 82.63 为本批次最高。
