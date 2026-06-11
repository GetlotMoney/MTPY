# TUNE-033 组合调参: topo=0.1 + cond_text=0.01 + jepa_neg=0.005

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
| conditional_text_ratio | 0.010 | 0.010 |
| lambda_jepa_neg | 0.01 | 0.005 |

**假设**: jepa_neg 从 0.02→0.01 已从 TUNE-022(73.70) 提升到 TUNE-024(74.09)。
继续减到 0.005 可能进一步减轻负样本约束，让模型更自由地学习。

## 结果

| 数据集 | seed | U | S | H | ZS | 最佳 epoch | 对比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | - | - | - | - | - | - |

## 日志

- 实验日志副本: logs/
