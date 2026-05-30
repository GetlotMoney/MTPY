---
name: paper-idea-tree
description: Maintain a weighted research idea tree while reading academic papers. Use when the user asks to summarize a paper into reusable research ideas, record paper innovations, compare modules, capture reported metric gains, maintain a "创意树"/idea tree, update node weights from experiment results, or rank candidate ideas/modules for future research.
---

# Paper Idea Tree

## Overview

Use this skill to turn paper reading into a persistent, ranked idea tree. Each paper contributes innovation nodes, useful modules, evidence from reported results, and later experiment feedback that updates node weights.

The skill itself is a self-contained folder at `C:\Users\Administrator\.codex\skills\paper-idea-tree`.

The default project binding is `DVSR`:

- Project root: `C:\Users\Administrator\Desktop\项目\DVSR`
- Idea-tree folder: `C:\Users\Administrator\Desktop\项目\DVSR\创新指导清单\paper-idea-tree`

The default artifacts inside the bound project folder are:

- `idea_tree.json`: source of truth
- `创意树.md`: human-readable ranked view
- `idea_tree.md`: ASCII filename mirror for preview panes
- `bindings.json`: project binding metadata

Use `scripts/idea_tree.py` to initialize, update weights, and render. When the user mentions DVSR or does not specify a project, use the `DVSR` binding. Read `references/idea_tree_schema.md` only when field details or scoring rules are needed.

## Workflow

1. Locate or create the tree.
   - Run `python <skill>/scripts/idea_tree.py init --project DVSR` if the DVSR tree does not exist.
   - Prefer the bound DVSR idea-tree folder unless the user names another project directory.

2. Read the paper.
   - Extract title, venue/year if present, task/setting, datasets, metrics, baseline comparisons, ablation findings, and limitations.
   - Do not record vague claims like "improves performance" without either a metric, ablation, or clear mechanism.

3. Add or merge idea nodes.
   - Create one node per reusable idea, module, loss, prompt strategy, alignment method, data trick, or evaluation insight.
   - Merge into an existing node when the mechanism is essentially the same, and add the new paper as additional evidence.
   - Keep nodes implementation-oriented: "local region-attribute OT alignment" is better than "better interpretability".

4. Score each node.
   - Use paper evidence for `paper_gain`.
   - Use the user's own experiment reports for `own_gain`.
   - Estimate `novelty`, `compatibility`, `cost`, and `confidence` conservatively.
   - Re-render after weight updates so `创意树.md` is sorted.

5. Report back.
   - Summarize newly added or changed nodes.
   - Mention the highest-ranked ideas and why they moved.
   - Flag missing evidence that would change the ranking.

## Node Scoring

Use 0-5 scores:

- `paper_gain`: strength of reported paper improvement, including ablations.
- `own_gain`: user's reproduction or new experiment improvement.
- `novelty`: how non-obvious the idea is relative to the user's current tree.
- `compatibility`: how easily it can combine with the user's current method.
- `cost`: implementation/training/inference cost; higher is worse.
- `confidence`: reliability of the evidence.

Default weight:

`0.30 * own_gain + 0.25 * paper_gain + 0.15 * novelty + 0.15 * compatibility + 0.15 * confidence - 0.15 * cost`

If `own_gain` is unknown, score it as `0` and mark the node `candidate`. Increase it only after the user reports experiments.

## Commands

Initialize or repair the tree:

```powershell
python C:\Users\Administrator\.codex\skills\paper-idea-tree\scripts\idea_tree.py init --root .
```

Render the ranked Markdown view:

```powershell
python C:\Users\Administrator\.codex\skills\paper-idea-tree\scripts\idea_tree.py render --root .
```

For the bound DVSR project, omit `--root` or pass `--project DVSR`:

```powershell
python C:\Users\Administrator\.codex\skills\paper-idea-tree\scripts\idea_tree.py render --project DVSR
```

Add a paper record:

```powershell
python C:\Users\Administrator\.codex\skills\paper-idea-tree\scripts\idea_tree.py add-paper --root . --title "Paper title" --file "paper.pdf" --venue "ICCV" --year 2025 --summary "One-sentence summary"
```

Add a node:

```powershell
python C:\Users\Administrator\.codex\skills\paper-idea-tree\scripts\idea_tree.py add-node --root . --id "local-attribute-alignment" --title "Local attribute alignment" --category "alignment" --summary "Align image regions with attributes." --mechanism "Use OT or attention to match local visual tokens to attribute text." --why-good "Improves interpretability and fine-grained discrimination." --paper "Paper title" --tags "ZSL,CLIP,attribute"
```

Update score after paper evidence or user's experiment:

```powershell
python C:\Users\Administrator\.codex\skills\paper-idea-tree\scripts\idea_tree.py update-node --root . --id "local-attribute-alignment" --paper-gain 3.5 --own-gain 4 --novelty 4 --compatibility 4 --cost 2 --confidence 4 --status "validated" --note "User experiment improved CUB H by 1.8."
```

## Extraction Rules

For each paper, capture:

- Innovation: what is new compared with prior work.
- Good module: reusable component and why it works.
- Evidence: dataset, setting, metric, baseline, value, gain, and ablation when available.
- Transfer idea: how the module could combine with existing nodes.
- Risk: assumptions, cost, dependence on annotations, extra models, or weak evaluation.

When exact numbers are not available, write `unknown` in notes instead of inventing values. If a paper reports many tables, prioritize the main result table and the ablation table that supports the node.

## Status Labels

- `candidate`: paper-only idea; no local experiment yet.
- `testing`: user is running or planning experiments.
- `validated`: user's experiment supports the node.
- `weakened`: user's experiment is weaker than paper evidence.
- `rejected`: evidence is poor or the idea conflicts with project constraints.
- `merged`: node was folded into another node.
