---
name: cv-experiment
description: Execute the DVSR computer-vision experiment workflow. Use when running MOD, COMBO, REV-MOD, TUNE, ABL, XDS, or FINAL experiments; coordinating Codex implementation, Git checkpoint branches, risk-tiered review, training, result recording, framework diagrams, and idea-tree feedback.
---

# CV Experiment Skill

This repository copy is the GitHub-synced specification for the local `$cv实验` skill.

When used inside DVSR, first read:

```text
docs/experiment_workflow/README.md
```

Then load only the specific reference needed:

- `docs/experiment_workflow/experiment_taxonomy.md` for experiment IDs, COMBO, REV-MOD, module replacement, and conflict rules.
- `docs/experiment_workflow/github_branching.md` for branches, commits, push, and merge behavior.
- `docs/experiment_workflow/idea_tree_feedback.md` for updating `idea_tree.json`, `创意树.md`, queues, and backlog.
- `docs/experiment_workflow/claude_review.md` for Codex self-review and risk-tiered Claude review.
- `docs/experiment_workflow/skill_cv_experiment.md` for the concrete execution sequence.
- `docs/experiment_workflow/runbook.md` for user-facing start/check/sync commands.

Core rule:

```text
GitHub docs are the project truth. The local skill is only the executor.
```

Use the 7 official controller branches:

```text
MOD      -> experiment/single-module-innovation
COMBO    -> experiment/module-combination
REV-MOD  -> experiment/single-module-review
TUNE     -> experiment/hyperparameter-tuning
ABL      -> experiment/ablation
XDS      -> experiment/cross-dataset
FINAL    -> experiment/final-review
```

Do not run training until:

- Git status and branch are checked.
- A local checkpoint exists.
- The experiment has a directory, README, config, and review packet.
- The experiment paradigm is selected:
  - `TUNE-LITE`: config-only TUNE or no-code REV-MOD; Codex self-review `ACCEPTED`; Claude not required.
  - `STANDARD`: ABL, XDS, or config/switch-only FINAL; Codex self-review and Claude single pass `ACCEPTED`.
  - `STRICT`: MOD, COMBO, code-changing FINAL, or any forward/loss/data/eval change; Codex self-review and Claude three-pass review all `ACCEPTED`.

For baseline hyperparameter tuning:

- Use `TUNE-LITE` by default.
- Do not call Claude unless the user asks or the self-review finds code/logic risk.
- Only edit experiment-local `config.yaml`.
- Change one primary hyperparameter per candidate.
- User-specified TUNE batches may run up to 20 serial candidates.

After training, always update:

- Experiment README.
- Logs copy.
- `experiments/EXPERIMENT_REGISTRY.md`.
- `experiments/08_framework_flow_records/<EXP-ID>_<slug>.md`.
- External idea tree and human-readable view.
- Category queue and `backlog.md`.
