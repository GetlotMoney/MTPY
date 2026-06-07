# 使用手册

## 我现在要开始实验

常用指令：

```text
使用 $cv实验 开始创新模块实验
```

批量：

```text
使用 $cv实验 跑 5 个创新模块实验
```

指定实验：

```text
使用 $cv实验 执行 MOD-007
```

## 我想检查当前状态

让 Codex 汇报：

```text
先看 git 状态、当前分支、backlog、分类队列和最近实验记录，告诉我下一步做什么。
```

Codex 应该检查：

- 当前分支。
- 是否干净。
- 最近 commit。
- `backlog.md`。
- 对应 `queues/*.md`。
- `experiments/EXPERIMENT_REGISTRY.md`。
- 创新树下一步建议。

## 我想同步 GitHub

明确说：

```text
提交并 push 当前分支。
```

Codex 应该：

1. 汇报当前分支和远端。
2. 检查 staged / unstaged 内容。
3. 提交。
4. push 到 `origin/<current-branch>`。

不会自动 push `main`。

## 我想知道 Claude 审查在哪里

每个实验目录内：

```text
experiments/<group>/<EXP-ID>_<slug>/claude-review.md
```

review packet：

```text
experiments/<group>/<EXP-ID>_<slug>/review-packet.md
```

## 我想知道日志在哪里

原始训练日志仍在：

```text
train_log/
```

实验日志副本在：

```text
experiments/<group>/<EXP-ID>_<slug>/logs/<EXP-ID>_<dataset>_<seed>_<timestamp>.txt
```

## 我想知道创新树有没有更新

检查：

```text
C:\Users\Administrator\Desktop\项目\创新指导清单\paper-idea-tree\idea_tree.json
C:\Users\Administrator\Desktop\项目\创新指导清单\paper-idea-tree\创意树.md
```

实验 README 也必须有“创新树反馈意见”。

