# TUNE-022 组合调参: topo=0.1 + jepa_neg=0.01

类型: TUNE
范式: TUNE-LITE
状态: completed

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
| lambda_jepa_neg | 0.02 | 0.01 |

假设: 降低 JEPA 负约束权重可能提高 unseen 判别力。

## 结果

| 数据集 | seed | U | S | H | ZS | 最佳 epoch | 对 TUNE-004 H=73.35 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 72.52 | 74.92 | 73.70 | 81.31 | 26 | +0.35 |

结论: topo=0.1 + jepa_neg=0.01 组合 H=73.70, 略低于 TUNE-021 H=73.72, 但显著高于 TUNE-004 baseline。不提升为 baseline。

## 日志

- 原始日志: train_log/CUB/training_log_CUB_2026-06-10_12-24-XX.txt
- 实验日志副本: logs/TUNE-022_CUB_seed5_training_log.txt
