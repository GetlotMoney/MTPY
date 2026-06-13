# Review Packet - TUNE-001

范式: TUNE-LITE
Claude: 不需要，除非 Codex 自审发现代码或逻辑风险。

## 实验目的

baseline 配置复跑校准

## Baseline

main 9533b0d 当前 baseline 代码，seed=5，H=72.91。

## 改动范围

- 只复制 baseline 配置到本实验目录。
- 只改 none 从 baseline 到 baseline。
- 不改模型代码、训练脚本、数据加载、评估逻辑。

## 运行计划

``powershell
python train_VGSR_CUB.py --config experiments/04_hyperparameter_tuning/TUNE-001_baseline_replay/config.yaml
``

## 记录要求

记录 U/S/H/ZS、最佳 epoch、原始日志路径、实验日志副本路径。
