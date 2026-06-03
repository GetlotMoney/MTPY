# 实验注册表

更新时间: 2026-06-02

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
| `planned` | 已设计，未开始 |
| `running` | 正在训练 |
| `done` | 已完成并记录结果 |
| `blocked` | 暂时不能做 |
| `dropped` | 不继续做，但保留原因 |

## 2. 当前基线

| 名称 | 框架 | H |
|---|---|---:|
| 当前主框架 | 局部补丁选择 + 几何感知编码 + 双向视觉-文本交互 + AG-JEPA 辅助训练 | 72.65 +/- 0.19 |
| 单点最高 | 同上，seed=5 | 72.91 |

## 3. 实验总表

| ID | 状态 | 优先级 | 类型 | 实验名称 | 核心问题 | 实验目录 | 当前结果 |
|---|---|---:|---|---|---|---|---|
| `ABL-001` | planned | 1 | 消融 | 去掉局部补丁选择 | 32 个局部 patch 信息瓶颈是否是核心贡献 | `02_ablation/ABL-001_disable_patch_selection/` | - |
| `ABL-002` | planned | 2 | 消融 | 去掉 AG-JEPA 辅助训练 | AG-JEPA 是否带来真实增益 | `02_ablation/ABL-002_disable_ag_jepa/` | 已有单 seed 证据: -0.46 H |
| `ABL-003` | planned | 3 | 消融 | 去掉文本拓扑保持 | 文本语义结构约束是否有效 | `02_ablation/ABL-003_disable_text_topology/` | - |
| `ABL-004` | planned | 4 | 消融 | 去掉双分支互蒸馏 | 两条视觉-文本分支是否需要互相约束 | `02_ablation/ABL-004_disable_branch_distillation/` | - |
| `ABL-005` | planned | 5 | 消融 | 去掉条件文本扰动 | 图像条件化文本是否帮助 GZSL | `02_ablation/ABL-005_disable_conditional_text/` | - |
| `ABL-006` | planned | 6 | 消融 | 去掉几何感知编码 | 选中 patch 后是否还需要位置关系建模 | `02_ablation/ABL-006_disable_geometry_encoding/` | - |
| `TUNE-001` | planned | 7 | 调参 | AG-JEPA 权重扫描 | 辅助训练权重是否最优 | `03_hyperparam_tuning/TUNE-001_ag_jepa_weight_sweep/` | - |
| `TUNE-002` | planned | 8 | 调参 | 负文本 margin 权重扫描 | 负类文本约束是否过强或过弱 | `03_hyperparam_tuning/TUNE-002_ag_jepa_negative_weight_sweep/` | - |
| `TUNE-003` | planned | 9 | 调参 | 局部 patch 数量补充扫描 | 32 是否优于 24 / 48 / 96 | `03_hyperparam_tuning/TUNE-003_patch_count_extended_sweep/` | - |
| `XDS-001` | planned | 10 | 跨数据集 | AWA2 主框架迁移 | 当前框架是否迁移到属性数据集 | `04_cross_dataset/XDS-001_awa2_main_framework/` | - |
| `XDS-002` | planned | 11 | 跨数据集 | SUN 主框架迁移 | 当前框架是否迁移到场景数据集 | `04_cross_dataset/XDS-002_sun_main_framework/` | - |
| `MOD-001` | planned | 12 | 替换模块 | 动态频域尺度 | 每张图是否需要自适应 patch 平滑尺度 | `01_module_replacement/MOD-001_dynamic_frequency_scale/` | - |
| `MOD-002` | planned | 13 | 替换模块 | 类别条件 patch 选择 | patch 选择是否应由类别文本引导 | `01_module_replacement/MOD-002_class_conditioned_patch_selection/` | - |
| `FINAL-001` | planned | 14 | 最终结果 | strict 多 seed 复核 | 复核当前主结果可复现性 | `05_final_runs/FINAL-001_strict_multiseed_verification/` | - |
| `FINAL-002` | planned | 15 | 最终结果 | warm-restart 多 seed | 评估最终性能上限 | `05_final_runs/FINAL-002_warm_restart_multiseed/` | - |

## 4. 第一批建议执行

先跑这 4 个：

| 顺序 | ID | 实验 |
|---:|---|---|
| 1 | `ABL-001` | 去掉局部补丁选择 |
| 2 | `ABL-002` | 去掉 AG-JEPA 辅助训练 |
| 3 | `ABL-003` | 去掉文本拓扑保持 |
| 4 | `ABL-004` | 去掉双分支互蒸馏 |

原因：

- `ABL-001` 验证核心结构。
- `ABL-002` 验证当前最强辅助训练信号。
- `ABL-003` / `ABL-004` 补齐论文 Table 2 需要的 loss-level 消融。

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

1. 在本注册表里把状态从 `planned` 改成 `running`。
2. 创建对应实验文件夹。
3. 复制配置到实验文件夹的 `config.yaml`。

每次实验后：

1. 把 U / S / H / ZS / best epoch / log path 写入该实验的 `README.md`。
2. 更新本注册表的状态和当前结果。
3. 如果结论影响主叙事，再更新 `DVSR_标准项目汇报.md`。
