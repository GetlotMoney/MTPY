---
name: cv-experiment
description: Execute the DVSR computer-vision experiment workflow. Use when running MOD, COMBO, REV-MOD, TUNE, ABL, XDS, or FINAL experiments; coordinating Codex implementation, Git checkpoint branches, Claude review, training, result recording, framework diagrams, and idea-tree feedback.
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
- `docs/experiment_workflow/claude_review.md` for Codex self-review and Claude fixed three-pass review.
- `docs/experiment_workflow/skill_cv_experiment.md` for the concrete execution sequence.
- `docs/experiment_workflow/runbook.md` for user-facing start/check/sync commands.

Core rule:

```text
GitHub docs are the project truth. The local skill is only the executor.
```

Do not run training until:

- Git status and branch are checked.
- A local checkpoint exists.
- The experiment has a directory, README, config, and review packet.
- Codex self-review returns `ACCEPTED`.
- Claude review pass 1, pass 2, and pass 3 all return `ACCEPTED`.

After training, always update:

- Experiment README.
- Logs copy.
- `experiments/EXPERIMENT_REGISTRY.md`.
- `experiments/06_framework_flows/<EXP-ID>_<slug>.md`.
- External idea tree and human-readable view.
- Category queue and `backlog.md`.

