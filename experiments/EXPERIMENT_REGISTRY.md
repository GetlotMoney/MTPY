# 实验注册表

更新时间: 2026-06-06

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

## 3. 实验分支总表

`main` 只维护当前 baseline 代码、主配置和总说明。具体实验记录按大模块分支维护：

| 分支 | 类型 | 目录 | 当前内容 |
|---|---|---|---|
| `experiment/ablation` | 消融 | `02_ablation/`、`06_framework_flows/` | `ABL-001` 到 `ABL-006` 完整记录 |
| `experiment/innovation` | 新增模块 / 替换模块 | `01_module_replacement/` | 后续创新模块实验 |
| `experiment/tuning` | 调参 | `03_hyperparam_tuning/` | AG-JEPA 权重、负文本 margin、patch 数量等扫描 |
| `experiment/cross-dataset` | 跨数据集 | `04_cross_dataset/` | AWA2 / SUN 主框架迁移 |
| `experiment/final-runs` | 最终结果 | `05_final_runs/` | 严格多 seed 复核、热重启上限、正式结果表 |

## 4. main 分支规则

- `main` 是当前 baseline，不承载具体实验目录。
- 已完成消融记录见 `experiment/ablation`。
- 新增实验必须先选择对应大模块分支，再创建实验目录和配置副本。
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
