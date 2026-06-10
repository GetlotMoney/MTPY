# TUNE-028 单一调参: topo=0.08

类型: TUNE
范式: TUNE-LITE
状态: completed

## 调参变量: lambda_topo_pearson: 0.10 -> 0.08

## 结果

| 数据集 | seed | U | S | H | ZS | 最佳 epoch |
|---|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 72.86 | 73.71 | 73.28 | 82.05 | 51 |

结论: topo=0.08 H=73.28, 低于 topo=0.1 的 TUNE-004(73.35)。确认 0.1 优于 0.08。
