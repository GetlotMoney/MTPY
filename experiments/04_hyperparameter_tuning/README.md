# 调参实验索引

本目录存放 `TUNE-xxx` 调参实验。调参实验默认使用 `TUNE-LITE` 范式：只改实验目录里的 `config.yaml`，每个候选只改变一个主变量，Codex 自审通过后运行，默认不调用 Claude 三轮审查。

## 查看方式

- 固定框架全量参数矩阵: `PARAMETER_TUNING_MATRIX.md`
- 批次汇总: `TUNE-BATCH-20260609.md`
- 单个实验记录: `TUNE-xxx_<slug>/README.md`
- 单个实验配置: `TUNE-xxx_<slug>/config.yaml`
- 单个实验日志: `TUNE-xxx_<slug>/logs/`

## 2026-06-09 baseline 调参批次

旧 baseline:

```text
main 9533b0d
CUB seed=5
U=73.30, S=72.53, H=72.91, ZS=81.72
```

当前已提升 baseline:

```text
TUNE-004_topo_pearson_01
lambda_topo_pearson: 0.05 -> 0.10
CUB seed=5
U=73.10, S=73.61, H=73.35, ZS=81.44
```

## 具体调参清单

| ID | 调参变量 | old | new | 状态 | H | 说明 |
|---|---|---:|---:|---|---:|---|
| TUNE-001 | 无 | baseline | baseline | done | 72.41 | baseline 复跑，用于检查运行口径。 |
| TUNE-002 | `lambda_topo_pearson` | 0.05 | 0.00 | done | 71.86 | 关闭文本拓扑保持 loss。 |
| TUNE-003 | `lambda_topo_pearson` | 0.05 | 0.02 | done | 72.43 | 降低文本拓扑保持 loss。 |
| TUNE-004 | `lambda_topo_pearson` | 0.05 | 0.10 | promoted_baseline | 73.35 | 提高文本拓扑保持 loss，已作为当前 baseline。 |
| TUNE-005 | `lambda_msdn` | 0.05 | 0.00 | done | 72.61 | 关闭 MSDN loss。 |
| TUNE-006 | `lambda_msdn` | 0.05 | 0.025 | done | 72.58 | 降低 MSDN loss。 |
| TUNE-007 | `lambda_msdn` | 0.05 | 0.10 | done | 72.35 | 提高 MSDN loss。 |
| TUNE-008 | `lambda_jepa` | 0.05 | 0.00 | done | 72.80 | 关闭 JEPA loss。 |
| TUNE-009 | `lambda_jepa` | 0.05 | 0.025 | done | 72.44 | 降低 JEPA loss。 |
| TUNE-010 | `lambda_jepa` | 0.05 | 0.10 | done | 72.48 | 提高 JEPA loss。 |
| TUNE-011 | `lambda_jepa_neg` | 0.02 | 0.00 | done | 72.36 | 关闭 JEPA negative loss。 |
| TUNE-012 | `lambda_jepa_neg` | 0.02 | 0.01 | done | 72.81 | 降低 JEPA negative loss。 |
| TUNE-013 | `lambda_jepa_neg` | 0.02 | 0.05 | done | 72.71 | 提高 JEPA negative loss，低于当前 TUNE-004 baseline。 |
| TUNE-014 | `conditional_text_ratio` | 0.005 | 0.00 | done | 72.54 | 关闭 conditional text ratio，低于当前 TUNE-004 baseline。 |
| TUNE-015 | `conditional_text_ratio` | 0.005 | 0.001 | done | 72.32 | 降低 conditional text ratio，低于当前 TUNE-004 baseline。 |
| TUNE-016 | `conditional_text_ratio` | 0.005 | 0.01 | done | 72.82 | 提高 conditional text ratio，是 13-20 中最高，但仍低于当前 TUNE-004 baseline。 |
| TUNE-017 | `local_weight` | 0.3 | 0.2 | done | 72.37 | 降低局部分支权重，低于当前 TUNE-004 baseline。 |
| TUNE-018 | `local_weight` | 0.3 | 0.4 | done | 72.28 | 提高局部分支权重，低于当前 TUNE-004 baseline。 |
| TUNE-019 | `lastvit_select_k` | 32 | 16 | done | 72.71 | 减少 patch 选择数量，低于当前 TUNE-004 baseline。 |
| TUNE-020 | `lastvit_select_k` | 32 | 64 | done | 72.37 | 增加 patch 选择数量，低于当前 TUNE-004 baseline。 |

注意：TUNE-013 到 TUNE-020 是补跑 `TUNE-BATCH-20260609` 的旧批次中断项，实验 config 仍为原始批次口径 `lambda_topo_pearson=0.05`。当前已提升 baseline 为 TUNE-004，`lambda_topo_pearson=0.1`，H=73.35；因此这些补跑结果只用于补全旧批次记录，不替代当前 baseline。

## 记录规范

之后每个 `TUNE-xxx` 必须在本 README 或批次汇总中写清楚：

- 调参变量名。
- old value 和 new value。
- 代码起点，也就是当时的 `main` baseline commit。
- 数据集、seed、运行命令。
- U/S/H/ZS、最佳 epoch、日志路径。
- 是否提升为新 baseline，或是否进入复核。

如果只跑单 seed，结论只能写成候选结论，不能写成最终结论。
