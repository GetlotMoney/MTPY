# 实验配置管理流程

本项目把根目录配置作为可复用基线。所有受管理的实验都必须使用实验文件夹内的配置副本运行。

## 规则

- 根目录 `config/` 里的配置只作为基线参考。
- 每个受管理实验都必须把对应的基线 YAML 复制到自己的 `config.yaml`。
- 训练时必须显式传入 `--config` 参数。
- 实验记录里必须写清楚命令、配置路径、随机种子、训练流程、日志路径、最佳 U/S/H/ZS。
- 不要为了单个实验直接修改根目录配置；除非该实验的目的就是更新基线配置。

## 命令

CUB:

```powershell
python train_VGSR_CUB.py --config experiments/02_ablation/ABL-001_disable_patch_selection/config.yaml
```

AWA2:

```powershell
python train_VGSR_AWA2.py --config experiments/04_cross_dataset/XDS-001_awa2_main_framework/config.yaml
```

SUN:

```powershell
python train_VGSR_SUN.py --config experiments/04_cross_dataset/XDS-002_sun_main_framework/config.yaml
```

## 严格连续训练流程

消融实验默认使用严格连续训练流程，这样一次只改变一个变量。

CUB 消融实验设置：

```yaml
lr_stages:
  value: null
```

消融配置里不要使用 `restart_from_best: True`，除非该实验明确研究热重启或分段训练。
