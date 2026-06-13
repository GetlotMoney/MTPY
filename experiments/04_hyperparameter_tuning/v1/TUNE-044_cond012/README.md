# TUNE-044 cond_text=0.012

类型: TUNE
状态: done

## 参数变化
- 基于 TUNE-024 配置
- conditional_text_ratio: 0.010 → 0.012

## 结果

| 数据集 | seed | U | S | H | ZS | Best |
|---|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 71.31 | 76.06 | 73.61 | 81.47 | ep26 |

vs TUNE-024 baseline (H=73.85): **-0.24** (soft_negative)

## 结论

- cond_text 0.012 比 baseline 0.010 低 0.24
- 继续下探 0.015, 0.020