# Idea Tree Schema

`idea_tree.json` is the source of truth. `创意树.md` is generated from it.

## Top Level

```json
{
  "version": 1,
  "updated_at": "ISO timestamp",
  "project": {},
  "papers": [],
  "nodes": []
}
```

## Project Object

When the tree is bound to an experiment project, include:

- `name`: project binding name, e.g. `DVSR`.
- `project_root`: local project path.
- `tree_root`: folder where the idea tree files live.
- `notes`: short description of the binding.

## Paper Object

Required or recommended fields:

- `title`: paper title.
- `file`: local file path or filename.
- `venue`: conference/journal if known.
- `year`: publication year if known.
- `summary`: one-sentence contribution summary.
- `tasks`: list such as `["ZSL", "GZSL", "fine-grained"]`.
- `datasets`: list such as `["CUB", "SUN", "AWA2"]`.
- `notes`: caveats, limitations, or reading notes.

## Node Object

Recommended fields:

- `id`: stable lowercase id, e.g. `local-region-attribute-ot`.
- `title`: short readable name.
- `category`: `alignment`, `prompt`, `loss`, `architecture`, `data`, `evaluation`, `training`, or `other`.
- `summary`: what the idea is.
- `mechanism`: how it works.
- `why_good`: why it may improve results.
- `source_papers`: list of paper evidence objects.
- `metrics`: list of metric evidence objects.
- `tags`: topic labels.
- `connections`: related node ids.
- `status`: `candidate`, `testing`, `validated`, `weakened`, `rejected`, or `merged`.
- `weight_factors`: scores from 0 to 5.
- `weight`: computed ranking score.
- `history`: notes about weight changes.

## Metric Evidence

Use this shape when possible:

```json
{
  "dataset": "CUB",
  "setting": "GZSL",
  "metric": "H",
  "baseline": "method name or unknown",
  "value": "72.3",
  "gain": "+1.8",
  "source": "paper",
  "note": "main table or ablation detail"
}
```

For user experiments, set `source` to `user-experiment` and include the date or run id in `note`.
