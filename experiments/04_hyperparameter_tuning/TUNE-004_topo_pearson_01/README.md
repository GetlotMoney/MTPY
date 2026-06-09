# TUNE-004 提高文本拓扑保持 loss 到 0.10

类型: TUNE
范式: TUNE-LITE
状态: completed，已提升为当前 baseline

## Baseline

- 代码起点: main 9533b0d 当前 baseline 代码
- 数据集: CUB GZSL
- seed: 5
- baseline 指标: U=73.30, S=72.53, H=72.91, ZS=81.72
- 比较口径: 单 seed 候选调参结果，只与同 seed baseline 对照；不是最终多 seed 结论。

## 调参变量

| key | old | new |
|---|---:|---:|
| lambda_topo_pearson | 0.05 | 0.1 |

测试更强拓扑约束是否保护 unseen 语义。

## 运行命令

```powershell
F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/04_hyperparameter_tuning/TUNE-004_topo_pearson_01/config.yaml
```

## Codex 自审

见 codex-self-review.md。

## 结果

| 数据集 | seed | U | S | H | ZS | 最佳 epoch | 对旧 baseline H=72.91 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 73.10 | 73.61 | 73.35 | 81.44 | 51 | +0.44 |

结论：TUNE-004 将 `lambda_topo_pearson` 从 0.05 提高到 0.1 后，seed=5 的 H 从旧 baseline 72.91 提升到 73.35。按用户确认，本实验配置提升为当前 baseline。后续建议先做多 seed 复核，再继续调参或创新模块实验。

## 日志

- 原始日志: `train_log/CUB/training_log_CUB_2026-06-09_17-50-43.txt`
- 实验日志副本: `logs/TUNE-004_CUB_seed5_2026-06-09_17-50-43.txt`

## 当前实验模块框架图

框架记录: `experiments/08_framework_flow_records/TUNE-004_topo_pearson_01.md`
