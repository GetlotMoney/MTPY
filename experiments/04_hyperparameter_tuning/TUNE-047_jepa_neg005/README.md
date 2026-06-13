# TUNE-047 jepa_neg=0.005

类型: TUNE
状态: done

## 参数变化
- 基于 TUNE-024 配置
- lambda_jepa_neg: 0.010 → 0.005

## 结果

| 数据集 | seed | U | S | H | ZS | Best |
|---|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 71.59 | 76.37 | 73.91 | 81.73 | ep26 |

vs TUNE-024 baseline (H=73.85): **+0.06** (near_tie)

## 结论

- jepa_neg 0.005 略优于 baseline 0.010（+0.06）
- 继续验证 0.008, 0.015, 0.020