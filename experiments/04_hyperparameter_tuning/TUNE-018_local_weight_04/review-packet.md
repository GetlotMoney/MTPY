# Review Packet - TUNE-018

范式: TUNE-LITE
Claude: 不需要，除非 Codex 自审发现代码或逻辑风险。

## 实验目的

提高局部分数权重到 0.4

## Baseline

main 9533b0d 当前 baseline 代码，seed=5，H=72.91。

## 改动范围

- 只复制 baseline 配置到本实验目录。
- 只改 local_weight 从 0.3 到 0.4。
- 不改模型代码、训练脚本、数据加载、评估逻辑。

## 运行计划

``powershell
python train_VGSR_CUB.py --config experiments/04_hyperparameter_tuning/TUNE-018_local_weight_04/config.yaml
``

## 记录要求

记录 U/S/H/ZS、最佳 epoch、原始日志路径、实验日志副本路径。
