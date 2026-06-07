# DVSR 实验管理区

这个目录专门放后续实验管理材料，避免把实验计划、配置副本、结果记录散落在根目录。

长期实验总表见 `EXPERIMENT_REGISTRY.md`。所有实验必须先在注册表里获得稳定 ID，再创建实验文件夹。

当前 Git 分支按实验大模块拆分：

| 分支 | 用途 |
|---|---|
| `main` | 当前 baseline 代码、主配置和总说明 |
| `experiment/mod` | `MOD-xxx` 单模块创新实验 |
| `experiment/combo` | `COMBO-xxx` 组合模块实验 |
| `experiment/rev-mod` | `REV-MOD-xxx` 单模块复核 |
| `experiment/tune` | `TUNE-xxx` 调参实验 |
| `experiment/abl` | `ABL-xxx` 消融实验 |
| `experiment/xds` | `XDS-xxx` 跨数据集实验 |
| `experiment/final` | `FINAL-xxx` 最终复核和正式结果 |

历史兼容分支 `experiment/innovation`、`experiment/ablation`、`experiment/tuning`、`experiment/cross-dataset`、`experiment/final-runs` 只作为旧实验记录入口；新实验优先使用上表 7 类分支。

## 目录说明

| 目录 | 用途 |
|---|---|
| `EXPERIMENT_REGISTRY.md` | 长期实验注册表，记录 ID、状态、优先级、路径和结果 |
| `00_templates/` | 实验记录模板 |
| `01_module_replacement/` | 单模块创新、模块替换、组合模块和单模块复核，例如换补丁选择器、池化方法、门控结构、损失结构 |
| `02_ablation/` | 消融实验，例如关掉某个模块或 loss |
| `03_hyperparam_tuning/` | 调参实验，例如扫 K、lambda、学习率、sigma |
| `04_cross_dataset/` | 跨数据集实验，例如 AWA2 / SUN |
| `05_final_runs/` | 最终正式 seed 候选池或热重启结果 |
| `06_framework_flows/` | 每个实验完成后的代码框架图、实验流程图、指标数据和日志来源 |
| `99_archive/` | 不再继续但需要保留的旧实验计划 |

## 实验 ID 类型

| 前缀 | 含义 | 默认目录 |
|---|---|---|
| `MOD-xxx` | 单模块创新实验，判断单个机制是否有独立贡献 | `01_module_replacement/` |
| `COMBO-xxx` | 组合模块实验，判断两个或多个机制是否存在协同贡献 | `01_module_replacement/`，若以后单独建目录则为 `07_combo/` |
| `REV-MOD-xxx` | 单模块复核，用于 near_tie、win 或用户指定模块的多 seed / 同配置复查 | `01_module_replacement/` |
| `TUNE-xxx` | 调参实验，只扫超参、loss 权重、top-K、margin、训练日程等 | `03_hyperparam_tuning/` |
| `ABL-xxx` | 消融实验，验证已有模块或 loss 是否必须保留 | `02_ablation/` |
| `XDS-xxx` | 跨数据集实验，例如 AWA2 / SUN 迁移 | `04_cross_dataset/` |
| `FINAL-xxx` | 最终复核，服务论文或正式汇报结果 | `05_final_runs/` |

## 命名规范

每个实验单独建一个文件夹，文件夹名必须以实验 ID 开头：

```text
experiments/01_module_replacement/MOD-007_attribute_prompt_router/
experiments/01_module_replacement/COMBO-001_attr_ot_plus_cf_negative/
experiments/01_module_replacement/REV-MOD-004-001_attr_patch_ot_seed_review/
experiments/02_ablation/ABL-002_disable_ag_jepa/
experiments/03_hyperparam_tuning/TUNE-001_ag_jepa_weight_sweep/
experiments/04_cross_dataset/XDS-001_awa2_main_framework/
experiments/05_final_runs/FINAL-001_cub_main_framework_review/
```

建议包含：

```text
README.md          # 实验目的、配置、结果、结论
config.yaml        # 该实验用的配置副本
notes.md           # 训练中临时观察，可选
```

原始训练日志仍放在 `train_log/`，同时把本实验对应日志复制到实验目录的 `logs/` 子目录。实验日志副本文件名必须包含实验 ID、数据集、seed 和时间戳，例如：

```text
experiments/01_module_replacement/MOD-007_attribute_prompt_router/logs/MOD-007_CUB_seed5_20260607-153012.txt
```

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

## 创新树反馈闭环

每个实验都必须和创新树建立双向连接：

- 实验开始前，在实验 README 写清关联的创意树节点 ID、节点标题、分类和当前状态。
- 实验结束后，把本地结果反馈回创新树，更新节点的 `metrics`、`source_materials`、`history`、状态和权重。
- 如果实验影响了一个类别的整体判断，也要更新 `category_weights` 的说明或下一步动作。
- 创新树必须给出下一步意见：`REV-MOD`、`COMBO`、继续新 `MOD`、转 `TUNE`、补 `ABL`、做 `XDS/FINAL`，或暂停该方向。
- 实验 README 的“创新树反馈意见”必须和创新树、分类队列、`backlog.md` 保持一致。

反馈规则：

| 结果分级 | 创新树动作 |
|---|---|
| `win` | 节点状态改为 `validated`，提高 `own_gain` 和 `confidence`，生成 `REV-MOD` 或小范围 `TUNE`。 |
| `near_tie` | 节点状态改为 `weakened` 或保留 `testing`，生成 `REV-MOD`；如果有互补解释，可生成 `COMBO`。 |
| `soft_negative` | 节点状态多为 `weakened`，降低 `confidence` 或 `compatibility`，保留失败原因和可改良方向。 |
| `hard_negative` | 节点状态改为 `rejected` 或大幅降权；除非用户指定，不继续消耗训练预算。 |
| `blocked` | 节点状态保持 `testing` 或 `candidate`，在 history 写清阻塞原因，不用结果更新权重。 |

## 基本原则

- 一次只改一个主要变量。
- 消融实验用严格连续训练流程，不能混用热重启。
- 多 seed 不是取平均值；正式主结果按主指标 H 取候选 seed 中的最大值。
- 每个候选 seed 的 U / S / H / ZS / 最佳轮次 / 日志路径都必须记录，便于复查最高值来源。
- 单次 H 小幅下降不直接判失败，先标为待补消融。
- 每个实验跑完必须写结论：保留 / 放弃 / 需要补跑。
- 每个实验跑完必须生成对应的 `06_framework_flows/<EXP-ID>_<slug>.md`，用于长期沉淀代码框架图和实验数据。
