# TUNE-038 topo=0.08

类型: TUNE
状态: done

## 参数变化
- 基于 TUNE-024 配置
- lambda_topo_pearson: 0.10 → 0.08

## 结果

| 数据集 | seed | U | S | H | ZS | Best |
|---|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 72.12 | 74.90 | 73.48 | 81.58 | ep43 |

vs TUNE-024 baseline (H=73.85): **-0.37** (soft_negative)

## 结论

- topo 从 0.10 降到 0.08，H 下降 0.37
- 确认 topo 最优值在 0.10 附近
