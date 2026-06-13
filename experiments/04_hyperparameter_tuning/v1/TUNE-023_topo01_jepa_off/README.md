# TUNE-023 组合调参: topo=0.1 + jepa=0.0

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
| lambda_jepa | 0.05 | 0.0 |

假设: 关闭 JEPA loss 在以前的旧口径 topo=0.05 下未崩, 测试 topo=0.1 下的效果。

## 结果

| 数据集 | seed | U | S | H | ZS | 最佳 epoch | 对 TUNE-004 H=73.35 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 73.04 | 74.38 | 73.70 | 82.04 | 23 | +0.35 |

结论: topo=0.1 + jepa=0.0 组合 H=73.70, 与 TUNE-022(73.70) 持平, 低于 TUNE-021(73.72)。不提升为 baseline, 但 ZS 最高 82.04 值得关注。

## 日志

- 实验日志副本: logs/TUNE-023_CUB_seed5_training_log.txt
