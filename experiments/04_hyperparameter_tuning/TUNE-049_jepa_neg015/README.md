# TUNE-049 jepa_neg=0.015

类型: TUNE
状态: done

## 参数变化
- 基于 TUNE-024 配置
- lambda_jepa_neg: 0.010 → 0.015

## 结果

| 数据集 | seed | U | S | H | ZS | Best |
|---|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 71.23 | 76.12 | 73.59 | 81.42 | ep26 |

vs TUNE-024 baseline (H=73.85): **-0.26** (soft_negative)

## 结论

- jepa_neg 0.015 比 baseline 0.010 低 0.26
- jepa_neg 最优值确认：**0.005**（TUNE-047 H=73.91）
