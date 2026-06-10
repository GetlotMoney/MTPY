# TUNE-030 组合调参: topo=0.1 + cond_text=0.01 + lastvit_k=16

类型: TUNE
范式: TUNE-LITE
状态: completed

## 调参变量: lambda_topo_pearson=0.1, conditional_text_ratio=0.01, lastvit_select_k: 32 -> 16

## 结果

| 数据集 | seed | U | S | H | ZS | 最佳 epoch |
|---|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 71.32 | 75.60 | 73.40 | 81.28 | 32 |

结论: H=73.40, 低于 TUNE-024(74.09), 不提升 baseline。
