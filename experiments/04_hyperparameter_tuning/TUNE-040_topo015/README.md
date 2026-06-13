# TUNE-040 topo=0.15

类型: TUNE
状态: done

## 参数变化
- 基于 TUNE-024 配置
- lambda_topo_pearson: 0.10 → 0.15

## 结果

| 数据集 | seed | U | S | H | ZS | Best |
|---|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 71.73 | 76.04 | 73.82 | 81.66 | ep20 |

vs TUNE-024 baseline (H=73.85): **-0.03** (near_tie)

## 结论

- topo 0.15 与 baseline 几乎持平（-0.03）
- 确认 topo 最优区间：0.10 ~ 0.15