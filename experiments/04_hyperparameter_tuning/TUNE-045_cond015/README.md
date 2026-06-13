# TUNE-045 cond_text=0.015

类型: TUNE
状态: done

## 参数变化
- 基于 TUNE-024 配置
- conditional_text_ratio: 0.010 → 0.015

## 结果

| 数据集 | seed | U | S | H | ZS | Best |
|---|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 71.58 | 75.92 | 73.69 | 81.72 | ep19 |

vs TUNE-024 baseline (H=73.85): **-0.16** (soft_negative)

## 结论

- cond_text 0.015 比 baseline 0.010 低 0.16
- 继续验证 0.020