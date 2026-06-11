# TUNE-032 组合调参: topo=0.1 + cond_text=0.015 + jepa_neg=0.01

类型: TUNE
范式: TUNE-LITE
状态: running

## Baseline

- 代码起点: main (TUNE-024 配置派生)
- 数据集: CUB GZSL
- seed: 5
- 当前 baseline: TUNE-024, H=74.09

## 调参变量

| key | TUNE-024 (baseline) | 本实验 |
|---|---:|---:|
| lambda_topo_pearson | 0.10 | 0.10 |
| conditional_text_ratio | 0.010 | 0.015 |
| lambda_jepa_neg | 0.01 | 0.01 |

**假设**: cond_text 从 0.005→0.01 已从 TUNE-021(73.72) 提升到 TUNE-024(74.09)。
继续加到 0.015 可能进一步增强条件化文本效果。

## 结果

| 数据集 | seed | U | S | H | ZS | 最佳 epoch | 对比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | - | - | - | - | - | - |

## 日志

- 实验日志副本: logs/
