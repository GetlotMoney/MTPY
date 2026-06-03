# ABL-001: Disable Patch Selection

Date: 2026-06-04

Type: Ablation

## 1. Purpose

Test whether the LaSt-ViT local patch selector is a core contributor to the current CUB GZSL result.

## 2. Change

| Item | Baseline | ABL-001 |
|---|---|---|
| `lastvit_select_k` | `32` | `0` |
| Patch selector | enabled, selects 32 patches | disabled, FAE sees all 576 patches |
| `lr_stages.value` | staged warm-restart | `null` strict schedule |

## 3. Training Config

| Item | Value |
|---|---|
| Dataset | CUB |
| Seed | 5 |
| Schedule | strict |
| Command | `python train_VGSR_CUB.py --config experiments/02_ablation/ABL-001_disable_patch_selection/config.yaml` |
| Config | `experiments/02_ablation/ABL-001_disable_patch_selection/config.yaml` |
| Device | `cuda:0` |

## 4. Hypothesis

If the 32-patch local selector is important, disabling it should reduce CUB GZSL H compared with the current baseline.

## 5. Baseline

| Comparison | H |
|---|---:|
| Current main baseline | 72.65 +/- 0.19 |
| Current single seed=5 best | 72.91 |

## 6. Review

| Gate | Result | Notes |
|---|---|---|
| Codex self-review | ACCEPTED | Config-only ablation. Root baseline config is unchanged; `lastvit_select_k=0` disables selector; `lr_stages.value=null` uses strict schedule. |
| Claude Code review | ACCEPTED | Clean config-only ablation. Verified `lastvit_select_k=0` disables patch selection and `lr_stages.value=null` uses strict schedule. |

## 7. Results

| U | S | H | ZS | Best Epoch | Log |
|---:|---:|---:|---:|---:|---|
|  |  |  |  |  |  |

## 8. Conclusion

Status: running

Decision: pending

## 9. Follow-Up

- [ ] Parse training log after run.
- [ ] Update `experiments/EXPERIMENT_REGISTRY.md`.
