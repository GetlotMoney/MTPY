# TUNE-031 实验框架

## Baseline

- 基于: TUNE-024 (H=74.09, seed=5)
- 框架: topo=0.1 框架 (局部补丁选择 + 几何感知编码 + 双向视觉-文本交互 + AG-JEPA 辅助训练)
- 不改代码，只改 config.yaml

## 与 baseline 的关系

| 参数 | TUNE-024 (baseline) | TUNE-031 (本实验) |
|---|---|---|
| lambda_topo_pearson | 0.10 | 0.15 |
| conditional_text_ratio | 0.010 | 0.010 |
| lambda_jepa_neg | 0.01 | 0.01 |
| 其他参数 | 完全一致 | 完全一致 |

## 改动说明

只修改 `lambda_topo_pearson` 从 0.10 → 0.15，其他所有参数保持不变。
Topology-preserving Pearson loss 权重增加，加强约束 Adapter 后的 200 类文本原型拓扑。
