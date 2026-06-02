# DVSR Experiments

这个目录专门放后续实验管理材料，避免把实验计划、配置副本、结果记录散落在根目录。

## 目录说明

| 目录 | 用途 |
|---|---|
| `00_templates/` | 实验记录模板 |
| `01_module_replacement/` | 替换模块实验，例如换 patch selector、pooling、gate、loss 结构 |
| `02_ablation/` | 消融实验，例如关掉某个模块或 loss |
| `03_hyperparam_tuning/` | 调参实验，例如扫 K、lambda、学习率、sigma |
| `04_cross_dataset/` | 跨数据集实验，例如 AWA2 / SUN |
| `05_final_runs/` | 最终正式多 seed 或 warm-restart 结果 |
| `99_archive/` | 不再继续但需要保留的旧实验计划 |

## 命名规范

每个实验单独建一个文件夹：

```text
experiments/02_ablation/2026-06-02_disable_ag_jepa/
```

建议包含：

```text
README.md          # 实验目的、配置、结果、结论
config.yaml        # 该实验用的配置副本
notes.md           # 训练中临时观察，可选
```

原始训练日志仍放在 `train_log/`，不要复制大日志到这里，只在实验 README 里写日志路径。

## 基本原则

- 一次只改一个主要变量。
- 消融实验用 strict schedule，不能混 warm-restart。
- 多 seed 结果才作为正式主结果。
- 单次 H 小幅下降不直接判失败，先标为待补消融。
- 每个实验跑完必须写结论：保留 / 放弃 / 需要补跑。
