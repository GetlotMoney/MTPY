# GitHub 分支与提交规范

## 分支角色

```text
main
  当前稳定 baseline，只放用户确认后的主框架。

experiment/innovation
  历史兼容分支：旧创新实验记录入口。

experiment/single-module-innovation
  MOD-xxx 单模块创新实验总控分支，只做分类记录和结果汇总，不作为新实验代码起点。

experiment/module-combination
  COMBO-xxx 组合模块实验总控分支，只做分类记录和结果汇总，不作为新实验代码起点。

experiment/single-module-review
  REV-MOD-xxx 单模块复核总控分支，只做分类记录和结果汇总，不作为新实验代码起点。

experiment/hyperparameter-tuning
  TUNE-xxx 调参实验总控分支，只做分类记录和结果汇总，不作为新实验代码起点。

experiment/ablation
  ABL-xxx 消融实验总控分支，只做分类记录和结果汇总，不作为新实验代码起点。

experiment/cross-dataset
  XDS-xxx 跨数据集实验总控分支，只做分类记录和结果汇总，不作为新实验代码起点。

experiment/final-review
  FINAL-xxx 最终复核和正式结果总控分支，只做分类记录和结果汇总，不作为新实验代码起点。

experiment/workflow-spec
  实验工作流规范分支。

exp/<EXP-ID>_<slug>
  单个实验代码分支。
```

历史兼容分支 `experiment/tuning`、`experiment/final-runs` 保留旧记录；新实验的代码起点统一以 `main` 为准，再创建对应的 `exp/<EXP-ID>_<slug>` 分支。`experiment/*` 只负责分类记录和结果汇总，不作为新实验代码起点。

## 实验代码起点

默认规则：

```text
main = 当前用户认可的 baseline
新实验 = 从 main 派生 exp/<EXP-ID>_<slug>
```

7 类 `experiment/*` 总控分支只是分类管理和结果汇总层，不作为新实验的代码起点。换句话说：

```text
正确: main -> exp/TUNE-021_xxx
错误: experiment/hyperparameter-tuning -> exp/TUNE-021_xxx
```

例外情况：

- `REV-MOD` 或用户明确指定复核某个旧实验时，可以从该实验对应 commit 派生，但必须在 README 和 review packet 写清“继承自哪个实验/commit”。
- 如果某个实验被确认提升为新主框架，必须先合入 `main`；之后的新实验再从更新后的 `main` 派生。
- 不允许从多个实验分支互相叠加后直接开新实验，除非这是明确登记的 `COMBO` 实验。

## 单个实验的分支流程

```text
确认 main 是当前 baseline
  ↓
从 main 新建 exp/<EXP-ID>_<slug>
  ↓
创建本地 checkpoint commit
  ↓
Codex 实现
  ↓
审查通过后运行实验
  ↓
提交实验代码、config、README、日志副本和框架图
  ↓
push exp/<EXP-ID>_<slug>
  ↓
把实验结果摘要同步回对应总控分支
```

## 成功与失败的处理

失败实验：

- 保留实验分支和记录。
- 不合并进 `main`。
- 把结果、失败原因和创新树反馈同步回对应 7 类总控分支。

成功实验：

- 先生成 `REV-MOD` 复核。
- 复核通过后，再考虑进入 `main`。
- 更新 `DVSR_标准项目汇报.md` 前必须得到用户确认。

## 提交规则

每个实验至少有三类提交：

```text
Checkpoint before <EXP-ID>
Implement <EXP-ID> <slug>
Record <EXP-ID> results
```

规范或队列更新可以单独提交：

```text
Update experiment workflow docs
Update idea tree feedback for <EXP-ID>
```

## push 规则

- 实验本体完成后，实验分支应提交并 push，保证本地与 GitHub 同步。
- Codex 不自动 push `main`，除非用户明确要求。
- 如果用户要求“同步 GitHub”，默认 push 当前分支到 `origin/<current-branch>`。
- 不自动 merge。
- 不自动 push `main`。
- 不自动删除远端实验分支。

## 为什么 GitHub 要保存规范

本机 skill 会变，聊天会被压缩，外部目录也可能不在另一台机器上。GitHub 仓库中的 `docs/experiment_workflow/` 是项目级真相来源，保证以后任何 agent 都能按同一套实验制度工作。
