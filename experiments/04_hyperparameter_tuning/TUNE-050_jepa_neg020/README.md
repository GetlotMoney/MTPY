# TUNE-050 jepa_neg=0.020

类型: TUNE
状态: done

## 参数变化
- 基于 TUNE-024 配置
- lambda_jepa_neg: 0.010 → 0.020

## 结果

| 数据集 | seed | U | S | H | ZS | Best |
|---|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 71.44 | 76.39 | 73.83 | 81.64 | ep26 |

vs TUNE-024 baseline (H=73.85): **-0.02** (near_tie)

## Phase 2 jepa_neg 轴向总结

| 实验 | jepa_neg | H | 排名 |
|------|----------|---|------|
| TUNE-047 | 0.005 | 73.91 | 🥇 最佳 |
| TUNE-050 | 0.020 | 73.83 | 🥈 次佳 |
| TUNE-049 | 0.015 | 73.59 | 第3 |
| TUNE-048 | 0.008 | 73.55 | 第4 |

## Phase 2 全部完成

- topo 最优: 0.10~0.15
- cond_text 最优: 0.008
- jepa_neg 最优: 0.005

## 进入 Phase 3: 2D 网格
基于最优参数: topo=0.10, cond=0.008, jepa_neg=0.005
