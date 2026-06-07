# 实验注册表

更新时间: 2026-06-07

这个文件是长期维护的实验总表，不是临时待办。每个实验必须有稳定 ID，并且实验文件夹名称必须以该 ID 开头。

## 1. 实验 ID 规则

| 前缀 | 类型 | 目录 |
|---|---|---|
| `MOD` | 单模块创新实验 / 模块替换 | `01_module_replacement/` |
| `COMBO` | 组合模块实验 | `01_module_replacement/`，若以后单独建目录则为 `07_combo/` |
| `REV-MOD` | 单模块复核 | `01_module_replacement/` |
| `TUNE` | 调参实验 | `03_hyperparam_tuning/` |
| `ABL` | 消融实验 | `02_ablation/` |
| `XDS` | 跨数据集实验 | `04_cross_dataset/` |
| `FINAL` | 最终复核 / 正式结果 | `05_final_runs/` |

命名格式：

```text
<TYPE>-<NUMBER>_<short_slug>
```

例子：

```text
experiments/01_module_replacement/MOD-007_attribute_prompt_router/
experiments/01_module_replacement/COMBO-001_attr_ot_plus_cf_negative/
experiments/01_module_replacement/REV-MOD-004-001_attr_patch_ot_seed_review/
experiments/02_ablation/ABL-001_disable_patch_selection/
experiments/03_hyperparam_tuning/TUNE-001_ag_jepa_weight_sweep/
experiments/04_cross_dataset/XDS-001_awa2_main_framework/
experiments/05_final_runs/FINAL-001_cub_main_framework_review/
```

`COMBO` 实验必须登记组成模块、协同假设、冲突分析和对照表。对照表至少包含 `baseline`、每个组成模块单独结果和组合结果，并计算：

```text
combo_gain = COMBO_H - max(baseline_H, module_A_H, module_B_H, ...)
```

只有 `combo_gain > 0`，才说明组合超过 baseline 和各单模块最好结果。

`REV-MOD` 用于复核 `near_tie`、`win` 或用户指定的单模块候选。复核必须写清原始 `MOD` ID、继承的代码/config、预注册 seed 列表和补跑范围，不能事后临时挑 seed。

状态字段：

| 状态 | 含义 |
|---|---|
| `计划中` | 已设计，未开始 |
| `进行中` | 正在训练或正在准备 |
| `已完成` | 已完成并记录结果 |
| `受阻` | 暂时不能做 |
| `已放弃` | 不继续做，但保留原因 |

## 2. 当前基线

项目实验比较口径：单个创新模块实验默认固定 `seed=5`，与同 seed baseline 对照。若某个实验进入多 seed 复查，必须预先写明 seed 列表并保留所有 seed 的完整结果；最高 H 可以作为项目选型参考，但不能隐藏其它 seed，也不能把事后临时挑 seed 当成单个实验的正式结论。

| 名称 | 框架 | H |
|---|---|---:|
| 当前主框架 | 局部补丁选择 + 几何感知编码 + 双向视觉-文本交互 + AG-JEPA 辅助训练，当前固定 seed=5 对照值 | 72.91 |
| 最高值来源 | 同上，seed=5 | 72.91 |

## 3. 实验分支总表

`main` 只维护当前 baseline 代码、主配置和总说明。具体实验记录按大模块分支维护：

| 分支 | 类型 | 目录 | 当前内容 |
|---|---|---|---|
| `experiment/single-module-innovation` | `MOD-xxx` 单模块创新 | `01_module_replacement/` | 判断单个机制是否有独立贡献 |
| `experiment/module-combination` | `COMBO-xxx` 组合模块 | `01_module_replacement/` | 判断两个或多个模块是否有协同收益 |
| `experiment/single-module-review` | `REV-MOD-xxx` 单模块复核 | `01_module_replacement/` | near_tie、win 或用户指定模块的多 seed / 同配置复核 |
| `experiment/hyperparameter-tuning` | `TUNE-xxx` 调参 | `03_hyperparam_tuning/` | AG-JEPA 权重、负文本 margin、patch 数量等扫描 |
| `experiment/ablation` | `ABL-xxx` 消融 | `02_ablation/`、`06_framework_flows/` | 验证已有模块或 loss 是否必须保留 |
| `experiment/cross-dataset` | `XDS-xxx` 跨数据集 | `04_cross_dataset/` | AWA2 / SUN 主框架迁移 |
| `experiment/final-review` | `FINAL-xxx` 最终复核 | `05_final_runs/` | 严格多 seed 复核、热重启上限、正式结果表 |

历史兼容分支：`experiment/innovation`、`experiment/tuning`、`experiment/final-runs` 保留旧实验记录，不作为新实验首选分支。

## 4. main 分支规则

- `main` 是当前 baseline，不承载具体实验目录。
- 已完成消融记录和新消融实验都使用 `experiment/ablation`。
- 新增实验必须先选择对应 7 类总控分支，再创建实验目录和配置副本。
- 如果某个分支结果成为新主框架，先在该分支复核，再按用户确认更新 `main`。

## 5. 单个实验文件夹规范

每个实验文件夹必须包含：

```text
README.md
config.yaml
```

可选：

```text
notes.md
```

`README.md` 从模板复制：

```text
experiments/00_templates/experiment_README_template.md
```

原始日志继续保留在 `train_log/`，同时把本实验对应日志复制到实验目录 `logs/`，文件名必须包含实验 ID、seed 和时间戳；实验 README 同时记录原始日志路径和实验日志副本路径。

## 5.1 创新模块实验

| ID | 状态 | 目录 | 核心改动 | 当前结果 |
|---|---|---|---|---|
| MOD-001 | 已完成 | `experiments/01_module_replacement/MOD-001_geometry_attribute_routing/` | FAE 后增加几何感知属性路由辅助 loss，默认主配置关闭，实验配置打开 | seed=5: U=73.24, S=71.68, H=72.45, ZS=81.49；低于 baseline H=72.91，当前版本不保留 |
| MOD-002 | 已完成 | `experiments/01_module_replacement/MOD-002_topology_aware_text_reservoir/` | 在 200 类文本原型上增加受限 CUB 属性 reservoir 低秩残差，并继续用 Pearson topology loss 约束 | seed=5: U=75.39, S=64.02, H=69.24, ZS=80.78；低于 baseline H=72.91，当前版本不保留 |
| MOD-003 | 已完成 | `experiments/01_module_replacement/MOD-003_uncertainty_gated_branch_distillation/` | 给现有 MSDN 双分支互蒸馏 loss 增加不确定性/分歧门控，默认主配置关闭 | seed=5: U=72.83, S=72.39, H=72.61, ZS=81.15；低于 baseline H=72.91，当前版本不保留 |
| MOD-004 | 已完成 | `experiments/01_module_replacement/MOD-004_attribute_guided_patch_ot/` | FAE 后局部 patch 与当前类别 top-K 属性文本原型做 Sinkhorn 软 OT 辅助对齐 | seed=5: U=72.86, S=72.95, H=72.90, ZS=81.61；基本持平但低于 baseline H=72.91，当前版本不保留 |
| MOD-005 | 已完成 | `experiments/01_module_replacement/MOD-005_semantic_patch_jepa_v2/` | 在现有 AG-JEPA 中新增默认关闭的 v2 目标构造：用类别文本和近邻类别原型选择判别 patch，并用 hard neighbor 作为负文本 | seed=5: U=73.10, S=72.03, H=72.56, ZS=81.55；低于 baseline H=72.91，当前版本不保留 |
| MOD-006 | 已完成 | `experiments/01_module_replacement/MOD-006_counterfactual_negative_text_mining/` | 在 GPT 文本原型上挖 seen 近邻负类，对 `logits_200` 加轻量反事实 margin loss，默认主配置关闭 | seed=5: U=72.65, S=72.13, H=72.39, ZS=81.44；低于 baseline H=72.91，当前版本不保留 |

## 6. 更新规则

每次实验前：

1. 在本注册表里把状态从 `计划中` 改成 `进行中`。
2. 创建对应实验文件夹。
3. 复制配置到实验文件夹的 `config.yaml`。

每次实验后：

1. 把每个 seed 的 U / S / H / ZS / 最佳轮次 / 日志路径写入该实验的 `README.md`。
2. 默认固定 seed=5 做单次正式对照；如果预注册多 seed 复查，必须保留所有 seed 的完整结果，并在结论中单独说明最高 H 是否仅作为项目选型参考。
3. 在 `experiments/06_framework_flows/` 中生成或更新该实验的框架图记录文件，包含流程图、流程说明、结果数据和日志路径。
4. 更新本注册表的状态和当前结果。
5. 如果结论影响主叙事，再更新 `DVSR_标准项目汇报.md`。
