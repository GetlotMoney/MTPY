# TUNE-021 组合调参: topo=0.1 + cond_text=0.01

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
| conditional_text_ratio | 0.005 | 0.010 |

假设: 提高 cond_text 能增加文本扰动多样性, 与更强的拓扑约束协同。

## 结果

| 数据集 | seed | U | S | H | ZS | 最佳 epoch | 对 TUNE-004 H=73.35 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 72.22 | 75.28 | 73.72 | 81.54 | 54 | +0.37 |

结论: topo=0.1 + cond_text=0.01 组合 H=73.72, 超过当前 TUNE-004 H=73.35。建议提升为候选 baseline, 后续做多 seed 复核。

## 日志

- 原始日志: train_log/CUB/training_log_CUB_2026-06-10_12-13-06.txt
- 实验日志副本: logs/TUNE-021_CUB_seed5_2026-06-10_12-13-06.txt
