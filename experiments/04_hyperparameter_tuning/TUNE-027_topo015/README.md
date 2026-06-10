# TUNE-027 单一调参: topo=0.15

类型: TUNE
范式: TUNE-LITE
状态: completed

## 调参变量: lambda_topo_pearson: 0.10 -> 0.15

## 结果

| 数据集 | seed | U | S | H | ZS | 最佳 epoch |
|---|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 71.89 | 76.20 | 73.98 | 81.78 | 32 |

结论: topo=0.15 高于 TUNE-004(73.35) 但低于 TUNE-024(74.09)。S 较高但 U 略低。
