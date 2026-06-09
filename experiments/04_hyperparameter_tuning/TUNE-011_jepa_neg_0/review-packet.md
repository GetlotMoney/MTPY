# Review Packet - TUNE-011

范式: TUNE-LITE
Claude: 不需要，除非 Codex 自审发现代码或逻辑风险。

## 实验目的

关闭 AG-JEPA 负样本 loss

## Baseline

main 9533b0d 当前 baseline 代码，seed=5，H=72.91。

## 改动范围

- 只复制 baseline 配置到本实验目录。
- 只改 lambda_jepa_neg 从 0.02 到 0.0。
- 不改模型代码、训练脚本、数据加载、评估逻辑。

## 运行计划

``powershell
python train_VGSR_CUB.py --config experiments/04_hyperparameter_tuning/TUNE-011_jepa_neg_0/config.yaml
``

## 记录要求

记录 U/S/H/ZS、最佳 epoch、原始日志路径、实验日志副本路径。
