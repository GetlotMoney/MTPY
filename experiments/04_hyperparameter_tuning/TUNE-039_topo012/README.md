# TUNE-039 topo=0.12

类型: TUNE
状态: done

## 参数变化
- 基于 TUNE-024 配置
- lambda_topo_pearson: 0.10 → 0.12

## 结果

| 数据集 | seed | U | S | H | ZS | Best |
|---|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 70.23 | 77.54 | 73.71 | 81.17 | ep32 |

vs TUNE-024 baseline (H=73.85): **-0.14** (soft_negative)

## 结论

- topo 从 0.10 升到 0.12，H 下降 0.14
- 确认 topo 最优值在 0.10 附近