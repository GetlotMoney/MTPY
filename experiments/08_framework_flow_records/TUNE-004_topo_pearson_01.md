# TUNE-004 topo Pearson weight baseline update

## 说明

本记录说明 TUNE-004 如何把当前 baseline 的文本拓扑保持 loss 权重从 `lambda_topo_pearson=0.05` 调到 `0.1`，并按用户确认提升为新的 CUB GZSL baseline。

## 流程位置

```text
CLIP 文本原型
  -> seen Adapter 训练
  -> 200 类文本原型拓扑约束
  -> Pearson topology loss
  -> 与 CE / consistency / MSDN / AG-JEPA 共同优化
```

TUNE-004 不改模型结构、不改训练脚本、不改数据集、不改 seed，只改变配置副本中的：

```yaml
lambda_topo_pearson:
  value: 0.1
```

## 结果

| 实验 | seed | U | S | H | ZS | 最佳 epoch | 日志 |
|---|---:|---:|---:|---:|---:|---:|---|
| old baseline | 5 | 73.30 | 72.53 | 72.91 | 81.72 | 23 | 历史 baseline |
| TUNE-004 | 5 | 73.10 | 73.61 | 73.35 | 81.44 | 51 | `experiments/04_hyperparameter_tuning/TUNE-004_topo_pearson_01/logs/TUNE-004_CUB_seed5_2026-06-09_17-50-43.txt` |

## 结论

`lambda_topo_pearson=0.1` 在 seed=5 下比旧 baseline 提升 `+0.44 H`。该结果主要提升 seen accuracy，同时 U 仅小幅下降，因此按用户确认提升为当前 baseline。后续需要补多 seed 复核，确认该提升不是单 seed 波动。
