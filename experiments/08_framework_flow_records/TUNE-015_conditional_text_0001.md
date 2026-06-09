# TUNE-015 调参流程记录

## 流程

```mermaid
flowchart LR
  A["旧批次 baseline config\nlambda_topo_pearson=0.05"] --> B["conditional_text_ratio: 0.005 -> 0.001"]
  B --> C["train_VGSR_CUB.py --config TUNE-015/config.yaml"]
  C --> D["CUB GZSL seed=5"]
  D --> E["best epoch 30\nU=73.07 S=71.59 H=72.32 ZS=81.45"]
```

## 说明

本实验测试降低 conditional text ratio 是否更稳。当前主 baseline 已是 TUNE-004，H=73.35。

## 结论

H=72.32，低于当前 baseline，不提升。

## 日志

- `experiments/04_hyperparameter_tuning/TUNE-015_conditional_text_0001/logs/TUNE-015_CUB_seed5_2026-06-09_21-24-49.txt`
