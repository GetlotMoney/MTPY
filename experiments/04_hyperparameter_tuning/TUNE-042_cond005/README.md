# TUNE-042 cond_text=0.005

类型: TUNE
状态: done

## 参数变化
- 基于 TUNE-024 配置
- conditional_text_ratio: 0.010 → 0.005

## 结果

| 数据集 | seed | U | S | H | ZS | Best |
|---|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 73.17 | 73.83 | 73.50 | 81.76 | ep26 |

vs TUNE-024 baseline (H=73.85): **-0.35** (soft_negative)

## 结论

- cond_text 从 0.010 降到 0.005，H 下降 0.35
- cond_text 最优值在 0.010 附近