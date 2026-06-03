# Experiment Config Workflow

This project keeps root configs as reusable baselines and runs tracked experiments from experiment-local config copies.

## Rules

- Keep root configs in `config/` as baseline references.
- Each tracked experiment must copy the relevant baseline YAML to its own `config.yaml`.
- Run training with an explicit `--config` argument.
- Record the exact command, config path, seed, schedule, log path, and best U/S/H/ZS in the experiment README.
- Do not edit root configs just to run one experiment unless the experiment is explicitly updating the baseline.

## Commands

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

## Strict Schedule

Ablation experiments default to strict schedule so only one variable changes at a time.

For CUB ablations, set:

```yaml
lr_stages:
  value: null
```

Do not use `restart_from_best: True` in ablation configs unless the experiment is explicitly testing warm-restart or staged training.
