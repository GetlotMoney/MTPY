# DVSR 实验工作流总览

本目录是 DVSR 项目的实验自动化“真规范”。本机 Codex skill 只是执行器；当 skill、对话记忆和本目录冲突时，以本目录为准。

## 目标

当前研究目标：

```text
先通过创新模块把 CUB GZSL 主指标推到 H >= 74，再进入系统调参。
```

当前严格连续 baseline：

| 数据集 | seed | U | S | H | ZS |
|---|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 73.30 | 72.53 | 72.91 | 81.72 |

项目选型口径：

- 不取多 seed 平均值作为主结果。
- 同一配置可以预注册多个 seed。
- 主结果按候选 seed 中主指标 `H` 的最大值记录。
- 所有 seed 的 U / S / H / ZS / 日志都必须完整保留。

## 闭环流程

```text
外部创新指导清单 / 创意树
  ↓
分类队列 queues/*.md
  ↓
backlog.md 当前执行窗口
  ↓
GitHub 实验分支
  ↓
本地 Git checkpoint
  ↓
Codex 实现最小代码或配置改动
  ↓
Codex 自审
  ↓
Claude 固定三轮审查
  ↓
审查通过后运行实验
  ↓
记录 README / logs / EXPERIMENT_REGISTRY / framework flow
  ↓
反馈 idea_tree.json / 创意树.md
  ↓
创新树给下一步意见
  ↓
更新分类队列 / backlog / registry
  ↓
提交并按用户要求 push
```

## 规范文件

| 文件 | 用途 |
|---|---|
| `experiment_taxonomy.md` | 实验 ID 类型：`MOD`、`COMBO`、`REV-MOD`、`TUNE`、`ABL`、`XDS`、`FINAL`。 |
| `github_branching.md` | GitHub 分支、提交、push、merge 规则。 |
| `idea_tree_feedback.md` | 实验结果如何反馈创新树并生成下一步意见。 |
| `claude_review.md` | Codex 自审与 Claude 固定三轮审查规范。 |
| `skill_cv_experiment.md` | `$cv实验` skill 应该如何执行本项目工作流。 |
| `runbook.md` | 用户实际怎么启动、检查和收尾。 |

## 项目内外文件边界

DVSR 仓库内维护：

- `README.md`
- `backlog.md`
- `experiments/`
- `docs/experiment_workflow/`
- `codex_skills/cv-experiment/`

外部创新指导清单当前位于：

```text
C:\Users\Administrator\Desktop\项目\创新指导清单
```

其中维护：

- `paper-idea-tree/idea_tree.json`
- `paper-idea-tree/创意树.md`
- `papers/`
- `queues/*.md`

如果以后需要 GitHub 完整同步创新指导清单，应把它迁入本仓库、建单独 repo，或作为 submodule 管理；不能只依赖本机路径。

