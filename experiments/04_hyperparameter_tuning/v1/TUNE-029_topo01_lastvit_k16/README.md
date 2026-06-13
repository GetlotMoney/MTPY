# TUNE-029 组合调参: topo=0.1 + lastvit_k=16

类型: TUNE
范式: TUNE-LITE
状态: completed

## 调参变量: lambda_topo_pearson=0.1, lastvit_select_k: 32 -> 16

## 结果

| 数据集 | seed | U | S | H | ZS | 最佳 epoch |
|---|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 73.03 | 73.72 | 73.37 | 81.52 | 43 |

结论: H=73.37, 与 TUNE-004(73.35)接近, topo=0.1下 k=16 可接受但不提升 baseline。
