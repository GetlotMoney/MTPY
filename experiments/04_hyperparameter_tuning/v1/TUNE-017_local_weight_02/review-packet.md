# Review Packet - TUNE-017

范式: TUNE-LITE
Claude: 不需要，除非 Codex 自审发现代码或逻辑风险。

## 实验目的

降低局部分数权重到 0.2

## Baseline

main 9533b0d 原始调参批次 baseline 代码，seed=5，H=72.91。当前主框架 baseline 已是 TUNE-004，H=73.35；本实验用于补齐旧批次，不替代当前 baseline。

## 改动范围

- 只复制 baseline 配置到本实验目录。
- 只改 local_weight 从 0.3 到 0.2。
- 不改模型代码、训练脚本、数据加载、评估逻辑。

## 运行计划

``powershell
python train_VGSR_CUB.py --config experiments/04_hyperparameter_tuning/TUNE-017_local_weight_02/config.yaml
``

## 记录要求

记录 U/S/H/ZS、最佳 epoch、原始日志路径、实验日志副本路径。
