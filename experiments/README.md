# DVSR 实验管理区

这个目录专门放后续实验管理材料，避免把实验计划、配置副本、结果记录散落在根目录。

长期实验总表见 `EXPERIMENT_REGISTRY.md`。所有实验必须先在注册表里获得稳定 ID，再创建实验文件夹。

当前 Git 分支按实验大模块拆分：

| 分支 | 用途 |
|---|---|
| `main` | 当前 baseline 代码、主配置和总说明 |
| `experiment/ablation` | 消融实验记录 |
| `experiment/innovation` | 新增创新模块和模块替换 |
| `experiment/tuning` | 超参数扫描 |
| `experiment/cross-dataset` | 跨数据集实验 |
| `experiment/final-runs` | 最终复核和正式结果 |

## 目录说明

| 目录 | 用途 |
|---|---|
| `EXPERIMENT_REGISTRY.md` | 长期实验注册表，记录 ID、状态、优先级、路径和结果 |
| `00_templates/` | 实验记录模板 |
| `01_module_replacement/` | 替换模块实验，例如换补丁选择器、池化方法、门控结构、损失结构 |
| `02_ablation/` | 消融实验，例如关掉某个模块或 loss |
| `03_hyperparam_tuning/` | 调参实验，例如扫 K、lambda、学习率、sigma |
| `04_cross_dataset/` | 跨数据集实验，例如 AWA2 / SUN |
| `05_final_runs/` | 最终正式 seed 候选池或热重启结果 |
| `06_framework_flows/` | 每个实验完成后的代码框架图、实验流程图、指标数据和日志来源 |
| `99_archive/` | 不再继续但需要保留的旧实验计划 |

## 命名规范

每个实验单独建一个文件夹，文件夹名必须以实验 ID 开头：

```text
experiments/02_ablation/ABL-002_disable_ag_jepa/
```

建议包含：

```text
README.md          # 实验目的、配置、结果、结论
config.yaml        # 该实验用的配置副本
notes.md           # 训练中临时观察，可选
```

原始训练日志仍放在 `train_log/`，不要复制大日志到这里，只在实验 README 里写日志路径。

每个实验完成并分析后，还必须在 `06_framework_flows/` 中新增或更新一个文件：

```text
experiments/06_framework_flows/<EXP-ID>_<slug>.md
```

该文件必须包含：

- 当前实验对应的代码框架图或数据流图。
- 这张图说明的流程是什么。
- 本实验改变了流程图中的哪一段。
- 关键数据表：seed、U、S、H、ZS、最佳轮次、原始日志、实验日志副本。
- 结论：该实验对当前代码框架的理解有什么影响。

## 基本原则

- 一次只改一个主要变量。
- 消融实验用严格连续训练流程，不能混用热重启。
- 多 seed 不是取平均值；正式主结果按主指标 H 取候选 seed 中的最大值。
- 每个候选 seed 的 U / S / H / ZS / 最佳轮次 / 日志路径都必须记录，便于复查最高值来源。
- 单次 H 小幅下降不直接判失败，先标为待补消融。
- 每个实验跑完必须写结论：保留 / 放弃 / 需要补跑。
- 每个实验跑完必须生成对应的 `06_framework_flows/<EXP-ID>_<slug>.md`，用于长期沉淀代码框架图和实验数据。
