# TUNE-043 cond_text=0.008

类型: TUNE
状态: done

## 参数变化
- 基于 TUNE-024 配置
- conditional_text_ratio: 0.010 → 0.008

## 结果

| 数据集 | seed | U | S | H | ZS | Best |
|---|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 72.36 | 75.57 | 73.93 | 81.62 | ep26 |

vs TUNE-024 baseline (H=73.85): **+0.08** (near_tie)

## 结论

- cond_text 0.008 略优于 baseline 0.010（+0.08）
- 仍在 near_tie 范围内，继续验证 0.012