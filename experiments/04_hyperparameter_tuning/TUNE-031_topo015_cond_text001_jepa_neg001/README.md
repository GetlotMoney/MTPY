# TUNE-031 组合调参: topo=0.15 + cond_text=0.01 + jepa_neg=0.01

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
| lambda_topo_pearson | 0.10 | 0.15 |
| conditional_text_ratio | 0.010 | 0.010 |
| lambda_jepa_neg | 0.01 | 0.01 |

**假设**: TUNE-027 (topo=0.15, 旧 cond_text=0.005, 旧 jepa_neg=0.02) 得到 H=73.98。
现在配合 cond_text=0.01 + jepa_neg=0.01 (TUNE-024 优化组合)，topo=0.15 可能进一步提升。

## 结果

| 数据集 | seed | U | S | H | ZS | 最佳 epoch | 对比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | - | - | - | - | - | - |

## 日志

- 实验日志副本: logs/
