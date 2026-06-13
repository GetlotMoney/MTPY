# TUNE-033 实验框架

## Baseline

- 基于: TUNE-024 (H=74.09, seed=5)
- 框架: topo=0.1 框架 (局部补丁选择 + 几何感知编码 + 双向视觉-文本交互 + AG-JEPA 辅助训练)
- 不改代码，只改 config.yaml

## 与 baseline 的关系

| 参数 | TUNE-024 (baseline) | TUNE-033 (本实验) |
|---|---|---|
| lambda_topo_pearson | 0.10 | 0.10 |
| conditional_text_ratio | 0.010 | 0.010 |
| lambda_jepa_neg | 0.01 | 0.005 |
| 其他参数 | 完全一致 | 完全一致 |

## 改动说明

只修改 `lambda_jepa_neg` 从 0.01 → 0.005，其他所有参数保持不变。
AG-JEPA 负样本 loss 权重减半，减轻负样本约束，让模型更自由地学习。
