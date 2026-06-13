# TUNE-041 topo=0.20

类型: TUNE
状态: done

## 参数变化
- 基于 TUNE-024 配置
- lambda_topo_pearson: 0.10 → 0.20

## 结果

| 数据集 | seed | U | S | H | ZS | Best |
|---|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 70.84 | 76.95 | 73.77 | 81.64 | ep24 |

vs TUNE-024 baseline (H=73.85): **-0.08** (soft_negative)

## 结论

- topo 0.20 下降 0.08
- topo 最优区间：0.10 ~ 0.15
