# TUNE-005 关闭双分支互蒸馏

类型: TUNE
范式: TUNE-LITE
状态: prepared

## Baseline

- 代码起点: main 9533b0d 当前 baseline 代码
- 数据集: CUB GZSL
- seed: 5
- baseline 指标: U=73.30, S=72.53, H=72.91, ZS=81.72
- 比较口径: 单 seed 候选调参结果，只与同 seed baseline 对照；不是最终多 seed 结论。

## 调参变量

| key | old | new |
|---|---:|---:|
| lambda_msdn | 0.05 | 0.0 |

调参口径下复核 MSDN 权重下界。

## 运行命令

``powershell
python train_VGSR_CUB.py --config experiments/04_hyperparameter_tuning/TUNE-005_msdn_0/config.yaml
``

## Codex 自审

见 codex-self-review.md。

## 结果

待运行后填写。

## 日志

- 原始日志: 待填写
- 实验日志副本: logs/

## 当前实验模块框架图

待运行后生成。
