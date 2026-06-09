# TUNE-014 调参流程记录

## 流程

```mermaid
flowchart LR
  A["旧批次 baseline config\nlambda_topo_pearson=0.05"] --> B["conditional_text_ratio: 0.005 -> 0.0"]
  B --> C["train_VGSR_CUB.py --config TUNE-014/config.yaml"]
  C --> D["CUB GZSL seed=5"]
  D --> E["best epoch 30\nU=72.09 S=73.00 H=72.54 ZS=81.75"]
```

## 说明

本实验测试关闭 conditional text ratio 是否更稳。当前主 baseline 已是 TUNE-004，H=73.35。

## 结论

H=72.54，低于当前 baseline，不提升。

## 日志

- `experiments/04_hyperparameter_tuning/TUNE-014_conditional_text_0/logs/TUNE-014_CUB_seed5_2026-06-09_21-15-27.txt`
