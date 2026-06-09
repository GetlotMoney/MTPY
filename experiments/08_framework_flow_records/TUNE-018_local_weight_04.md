# TUNE-018 调参流程记录

## 流程

```mermaid
flowchart LR
  A["旧批次 baseline config\nlambda_topo_pearson=0.05"] --> B["local_weight: 0.3 -> 0.4"]
  B --> C["train_VGSR_CUB.py --config TUNE-018/config.yaml"]
  C --> D["CUB GZSL seed=5"]
  D --> E["best epoch 51\nU=73.16 S=71.42 H=72.28 ZS=81.65"]
```

## 说明

本实验测试提高局部分数权重。当前主 baseline 已是 TUNE-004，H=73.35。

## 结论

H=72.28，低于当前 baseline，不提升。

## 日志

- `experiments/04_hyperparameter_tuning/TUNE-018_local_weight_04/logs/TUNE-018_CUB_seed5_2026-06-09_21-52-41.txt`
