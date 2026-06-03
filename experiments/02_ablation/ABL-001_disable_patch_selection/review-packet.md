# Review Packet: ABL-001

## Context

- Experiment ID: `ABL-001`
- Experiment file: `experiments/02_ablation/ABL-001_disable_patch_selection/README.md`
- Experiment config: `experiments/02_ablation/ABL-001_disable_patch_selection/config.yaml`
- Baseline config: `config/VGSR_cub_gzsl.yaml`
- Checkpoint commit before experiment: `f39c0eb`
- Current branch: `experiment/v3-baseline-72.48`

## Request

Review the current experiment implementation. Do not edit code.

Return a Markdown review beginning with exactly one top-level decision:

```text
Decision: ACCEPTED
```

or:

```text
Decision: REJECTED
```

## Intended Experiment

ABL-001 tests whether the LaSt-ViT local patch selector is a core contributor to the CUB GZSL result.

The experiment should change one main variable:

- Baseline: `lastvit_select_k.value = 32`
- ABL-001: `lastvit_select_k.value = 0`

The experiment config also disables staged warm-restart by setting:

```yaml
lr_stages:
  value: null
```

This is intended to follow the project rule that ablations use strict schedule.

## Validation Plan

Run only after Codex self-review and Claude Code review both accept:

```powershell
conda activate dassl_clip
python train_VGSR_CUB.py --config experiments/02_ablation/ABL-001_disable_patch_selection/config.yaml
```

Primary metric: CUB GZSL H.

Secondary metrics: U, S, ZS, best epoch, log path.

## Review Focus

- Is the experiment-local config used instead of mutating the root baseline config?
- Does `lastvit_select_k: 0` correctly disable local patch selection in `model/MyModel.py`?
- Does `lr_stages.value: null` represent strict schedule for this codebase?
- Are unrelated model or training changes avoided?
- Is the command reproducible and recorded?
