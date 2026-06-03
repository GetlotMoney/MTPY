Decision: ACCEPTED

## Code Review: ABL-001

**Request:** `ABL-001.round-1.71fe0fc4`
**Status:** done
**Reviewed by:** Claude Code session

---

### 🔴 Critical

None.

### 🟡 Warning

None.

### 🔵 Note

1. **Config diff is clean** — diff vs baseline `config/VGSR_cub_gzsl.yaml` shows exactly 2 changes:

    ```diff
    - lastvit_select_k.value: 32
    + lastvit_select_k.value: 0
    
    - lr_stages.value: [staged warm-restart]
    + lr_stages.value: null
    ```

    No unrelated parameters were changed.

2. **`lastvit_select_k: 0` correctly disables patch selection** — In `model/MyModel.py:578`, the guard is:
    ```python
    if lastvit_select_k > 0 and lastvit_select_k < patches.size(1):
    ```
    With `k=0`, the condition is `False`, so `topk_indices_v5` stays `None`, and `patches` stays as `[B, 576, 768]` (all 576 patches). The FAE path at line 595 then takes the `patches.size(1) == 576` branch (standard full-grid geometric encoding). This is correct — the model degrades gracefully to no patch selection.

3. **`lr_stages.value: null` correctly disables staged warm-restart** — In `train_VGSR_CUB.py:536`:
    ```python
    lr_stages = getattr(config, 'lr_stages', None) or []
    ```
    `None` is falsy, so `lr_stages` becomes `[]`. The multi-stage block is skipped; training falls back to `config.epochs=30` with standard cosine schedule. This is the project-defined "strict schedule" for ablations.

4. **Experiment follows project conventions**:
    - Uses `--config` argument (added in recent commit) to reference experiment-local `config.yaml`. Root `config/VGSR_cub_gzsl.yaml` is untouched.
    - README matches template structure, has experiment table, hypothesis, baseline H values, and review gate status.
    - Training command is recorded and reproducible.
    - Registered in `EXPERIMENT_REGISTRY.md` (ABL-001).

5. **No model code was modified** — the experiment is purely config-driven. This is the correct approach for a clean ablation.

### All Review Focus Questions

| Question | Answer |
|---|---|
| Experiment-local config used instead of mutating root? | ✅ Yes — `config.yaml` in experiment folder, diff confirms only 2 intended changes |
| Does `lastvit_select_k: 0` correctly disable patch selection? | ✅ Yes — code guard `k > 0` ensures 576-patch fallback |
| Does `lr_stages.value: null` represent strict schedule? | ✅ Yes — `None or []` skips multi-stage, uses standard epochs + cosine |
| Unrelated model or training changes avoided? | ✅ Yes — diff is exactly 2 lines |
| Reproducible command recorded? | ✅ Yes — in README and review packet |

---

**Verdict:** OK — the experiment is clean, minimal, and correctly implemented. Safe to run.
