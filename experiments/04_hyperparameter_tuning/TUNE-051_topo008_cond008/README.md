# TUNE-051 topo=0.08, cond=0.008

类型: TUNE
状态: done

## 参数变化
- 基于 TUNE-024 配置
- lambda_topo: 0.10 → 0.08
- conditional_text_ratio: 0.010 → 0.008
- lambda_jepa_neg: 0.010 → 0.005

## 结果

| 数据集 | seed | U | S | H | ZS | Best |
|---|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 72.23 | 75.23 | 73.70 | 81.55 | ep26 |

vs TUNE-024 baseline (H=73.85): **-0.15** (soft_negative)

## Phase 3 2D 网格

| 实验 | topo | cond | H | 排名 |
|------|------|------|---|------|
| TUNE-043 | 0.10 | 0.008 | 73.93 | 🥇 最佳 |
| TUNE-051 | 0.08 | 0.008 | 73.70 | 第2 |

## 结论
- topo 0.08 + cond 0.008 组合不如 baseline 0.10 + 0.008
- 继续验证其他组合
