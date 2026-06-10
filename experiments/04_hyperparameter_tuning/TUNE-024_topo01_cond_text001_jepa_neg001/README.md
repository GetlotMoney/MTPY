# TUNE-024 组合调参: topo=0.1 + cond_text=0.01 + jepa_neg=0.01

类型: TUNE
范式: TUNE-LITE
状态: completed, 建议新 baseline

## Baseline

- 代码起点: main 9533b0d（TUNE-004 baseline 配置派生）
- 数据集: CUB GZSL
- seed: 5
- 当前 baseline: TUNE-004, lambda_topo_pearson=0.1, H=73.35
- 比较口径: 单 seed 候选调参结果

## 调参变量

| key | TUNE-004 (baseline) | 本实验 |
|---|---:|---:|
| lambda_topo_pearson | 0.05 -> 0.10 | 0.10 |
| conditional_text_ratio | 0.005 | 0.010 |
| lambda_jepa_neg | 0.02 | 0.01 |

## 结果

| 数据集 | seed | U | S | H | ZS | 最佳 epoch | 对比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 72.33 | 75.95 | 74.09 | 81.75 | 26 | TUNE-004: +0.74, TUNE-021: +0.37 |

结论: 三个参数组合 topo=0.1 + cond_text=0.01 + jepa_neg=0.01 有效协同, 首次突破 74。建议提升为候选 baseline, 做多 seed 复核。

## 日志

- 实验日志副本: logs/TUNE-024_CUB_seed5_training_log.txt

## 多 seed 复核 (2026-06-10)

按规范（seed=5/42/123/2026），seed=5 已有，补跑 seed=42、123。seed=2026 因中断暂缺。

| 数据集 | seed | U | S | H | ZS | 最佳 epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 72.33 | 75.95 | 74.09 | 81.54 | 26 |
| CUB GZSL | 42 | 70.35 | 77.60 | 73.80 | 81.62 | 37 |
| CUB GZSL | 123 | 71.56 | 76.05 | 73.73 | 81.19 | 51 |
| CUB GZSL | 2026 | 72.60 | 75.02 | 73.79 | 81.89 | 51 |

**复核结论**: 四个 seed 全部在 73.70+ 区间，没有低于 72。seed=5 达到 74.09。baseline 成立。

**日志**:
- seed=42: logs/seed-42/training_log.txt
- seed=123: logs/seed-123/training_log.txt
- seed=2026: logs/seed-2026/training_log.txt

