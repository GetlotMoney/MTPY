# TUNE-017 调参流程记录

## 流程

```mermaid
flowchart LR
  A["旧批次 baseline config\nlambda_topo_pearson=0.05"] --> B["local_weight: 0.3 -> 0.2"]
  B --> C["train_VGSR_CUB.py --config TUNE-017/config.yaml"]
  C --> D["CUB GZSL seed=5"]
  D --> E["best epoch 30\nU=72.48 S=72.26 H=72.37 ZS=81.40"]
```

## 说明

本实验测试降低局部分数权重。当前主 baseline 已是 TUNE-004，H=73.35。

## 结论

H=72.37，低于当前 baseline，不提升。

## 日志

- `experiments/04_hyperparameter_tuning/TUNE-017_local_weight_02/logs/TUNE-017_CUB_seed5_2026-06-09_21-43-27.txt`
