# 实验注册表

更新时间: 2026-06-05

这个文件是长期维护的实验总表，不是临时待办。每个实验必须有稳定 ID，并且实验文件夹名称必须以该 ID 开头。

## 1. 实验 ID 规则

| 前缀 | 类型 | 目录 |
|---|---|---|
| `MOD` | 替换模块 | `01_module_replacement/` |
| `ABL` | 消融实验 | `02_ablation/` |
| `TUNE` | 调参实验 | `03_hyperparam_tuning/` |
| `XDS` | 跨数据集实验 | `04_cross_dataset/` |
| `FINAL` | 最终正式结果 | `05_final_runs/` |

命名格式：

```text
<TYPE>-<NUMBER>_<short_slug>
```

例子：

```text
experiments/02_ablation/ABL-001_disable_patch_selection/
experiments/03_hyperparam_tuning/TUNE-001_ag_jepa_weight_sweep/
experiments/04_cross_dataset/XDS-001_awa2_main_framework/
```

状态字段：

| 状态 | 含义 |
|---|---|
| `计划中` | 已设计，未开始 |
| `进行中` | 正在训练或正在准备 |
| `已完成` | 已完成并记录结果 |
| `受阻` | 暂时不能做 |
| `已放弃` | 不继续做，但保留原因 |

## 2. 当前基线

项目正式比较口径：同一设置可以测试多个 seed，但主结果不取平均值；按主指标 H 取候选 seed 中的最大值。其它 seed 只作为复查记录保留。

| 名称 | 框架 | H |
|---|---|---:|
| 当前主框架 | 局部补丁选择 + 几何感知编码 + 双向视觉-文本交互 + AG-JEPA 辅助训练，seed 候选池取最高 H | 72.91 |
| 最高值来源 | 同上，seed=5 | 72.91 |

## 3. 实验总表

| ID | 状态 | 优先级 | 类型 | 实验名称 | 核心问题 | 实验目录 | 当前结果 |
|---|---|---:|---|---|---|---|---|
| `ABL-001` | 已完成 | 1 | 消融 | 去掉局部补丁选择 | 32 个局部补丁信息瓶颈是否是核心贡献 | `02_ablation/ABL-001_disable_patch_selection/` | seed=5: U=74.22, S=69.07, H=71.55, ZS=81.84；较主基线 H=72.91 下降 1.36 |
| `ABL-002` | 进行中 | 2 | 消融 | 去掉 AG-JEPA 辅助训练 | AG-JEPA 是否带来真实增益 | `02_ablation/ABL-002_disable_ag_jepa/` | 已创建实验配置，等待审查放行 |
| `ABL-003` | 计划中 | 3 | 消融 | 去掉文本拓扑保持 | 文本语义结构约束是否有效 | `02_ablation/ABL-003_disable_text_topology/` | - |
| `ABL-004` | 计划中 | 4 | 消融 | 去掉双分支互蒸馏 | 两条视觉-文本分支是否需要互相约束 | `02_ablation/ABL-004_disable_branch_distillation/` | - |
| `ABL-005` | 计划中 | 5 | 消融 | 去掉条件文本扰动 | 图像条件化文本是否帮助 GZSL | `02_ablation/ABL-005_disable_conditional_text/` | - |
| `ABL-006` | 计划中 | 6 | 消融 | 去掉几何感知编码 | 选中补丁后是否还需要位置关系建模 | `02_ablation/ABL-006_disable_geometry_encoding/` | - |
| `TUNE-001` | 计划中 | 7 | 调参 | AG-JEPA 权重扫描 | 辅助训练权重是否最优 | `03_hyperparam_tuning/TUNE-001_ag_jepa_weight_sweep/` | - |
| `TUNE-002` | 计划中 | 8 | 调参 | 负文本 margin 权重扫描 | 负类文本约束是否过强或过弱 | `03_hyperparam_tuning/TUNE-002_ag_jepa_negative_weight_sweep/` | - |
| `TUNE-003` | 计划中 | 9 | 调参 | 局部补丁数量补充扫描 | 32 是否优于 24 / 48 / 96 | `03_hyperparam_tuning/TUNE-003_patch_count_extended_sweep/` | - |
| `XDS-001` | 计划中 | 10 | 跨数据集 | AWA2 主框架迁移 | 当前框架是否迁移到属性数据集 | `04_cross_dataset/XDS-001_awa2_main_framework/` | - |
| `XDS-002` | 计划中 | 11 | 跨数据集 | SUN 主框架迁移 | 当前框架是否迁移到场景数据集 | `04_cross_dataset/XDS-002_sun_main_framework/` | - |
| `MOD-001` | 计划中 | 12 | 替换模块 | 动态频域尺度 | 每张图是否需要自适应补丁平滑尺度 | `01_module_replacement/MOD-001_dynamic_frequency_scale/` | - |
| `MOD-002` | 计划中 | 13 | 替换模块 | 类别条件补丁选择 | 补丁选择是否应由类别文本引导 | `01_module_replacement/MOD-002_class_conditioned_patch_selection/` | - |
| `FINAL-001` | 计划中 | 14 | 最终结果 | 严格连续 seed 候选池复核 | 多个 seed 中寻找严格连续最高 H | `05_final_runs/FINAL-001_strict_multiseed_verification/` | - |
| `FINAL-002` | 计划中 | 15 | 最终结果 | 热重启 seed 候选池 | 多个 seed 中寻找热重启性能上限 | `05_final_runs/FINAL-002_warm_restart_multiseed/` | - |

## 4. 第一批建议执行

只维护最近要跑的 3 到 5 个实验。每完成一个实验，就从这里移除，并从实验总表里选择下一个“计划中”且优先级最高的实验补进来。

当前先跑这 4 个：

| 顺序 | ID | 实验 |
|---:|---|---|
| 1 | `ABL-002` | 去掉 AG-JEPA 辅助训练 |
| 2 | `ABL-003` | 去掉文本拓扑保持 |
| 3 | `ABL-004` | 去掉双分支互蒸馏 |
| 4 | `ABL-005` | 去掉条件文本扰动 |

原因：

- `ABL-001` 已完成，结果支持保留局部补丁选择。
- `ABL-002` 验证当前最强辅助训练信号。
- `ABL-003` / `ABL-004` 补齐论文 Table 2 需要的 loss-level 消融。
- `ABL-005` 验证图像条件化文本扰动是否是当前增益来源之一。

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

原始日志继续保留在 `train_log/`，不要复制到实验目录。实验 README 只写日志路径。

## 6. 更新规则

每次实验前：

1. 在本注册表里把状态从 `计划中` 改成 `进行中`。
2. 创建对应实验文件夹。
3. 复制配置到实验文件夹的 `config.yaml`。

每次实验后：

1. 把每个 seed 的 U / S / H / ZS / 最佳轮次 / 日志路径写入该实验的 `README.md`。
2. 如果同一设置有多个 seed，按 H 选择最大值作为该实验的正式比较结果。
3. 在 `experiments/06_framework_flows/` 中生成或更新该实验的框架图记录文件，包含流程图、流程说明、结果数据和日志路径。
4. 更新本注册表的状态和当前结果。
5. 如果结论影响主叙事，再更新 `DVSR_标准项目汇报.md`。
