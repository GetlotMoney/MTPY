# TUNE-016 提高条件文本扰动到 0.01

类型: TUNE
范式: TUNE-LITE
状态: completed

## Baseline

- 代码起点: main 9533b0d 原始调参批次 baseline 代码
- 数据集: CUB GZSL
- seed: 5
- 原始批次 baseline 指标: U=73.30, S=72.53, H=72.91, ZS=81.72
- 当前主框架 baseline: TUNE-004，`lambda_topo_pearson=0.1`，H=73.35
- 比较口径: 单 seed 候选调参结果；本实验 config 仍为旧批次口径 `lambda_topo_pearson=0.05`，只用于补齐中断批次，不替代当前 baseline。

## 调参变量

| key | old | new |
|---|---:|---:|
| conditional_text_ratio | 0.005 | 0.01 |

测试更强条件文本扰动。

## 运行命令

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'; F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/04_hyperparameter_tuning/TUNE-016_conditional_text_001/config.yaml
```

## Codex 自审

见 codex-self-review.md。

## 结果

| 数据集 | seed | U | S | H | ZS | 最佳 epoch | 对当前 baseline H=73.35 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 71.70 | 73.97 | 72.82 | 81.37 | 30 | -0.53 |

结论：`conditional_text_ratio=0.01` 是 TUNE-013 到 TUNE-020 中最高 H，但仍低于当前 TUNE-004 baseline H=73.35；不提升为 baseline。相对原始批次 baseline H=72.91 也未胜出。

## 日志

- 原始日志: `train_log/CUB/training_log_CUB_2026-06-09_21-34-00.txt`
- 实验日志副本: `logs/TUNE-016_CUB_seed5_2026-06-09_21-34-00.txt`

## 当前实验模块框架图

框架记录: `experiments/08_framework_flow_records/TUNE-016_conditional_text_001.md`
