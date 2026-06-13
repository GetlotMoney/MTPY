# TUNE-048 jepa_neg=0.008

类型: TUNE
状态: done

## 参数变化
- 基于 TUNE-024 配置
- lambda_jepa_neg: 0.010 → 0.008

## 结果

| 数据集 | seed | U | S | H | ZS | Best |
|---|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 71.10 | 76.19 | 73.55 | 81.02 | ep26 |

vs TUNE-024 baseline (H=73.85): **-0.30** (soft_negative)

## 结论

- jepa_neg 0.008 比 baseline 0.010 低 0.30
- jepa_neg 最优值确认：**0.005**（TUNE-047 H=73.91）
