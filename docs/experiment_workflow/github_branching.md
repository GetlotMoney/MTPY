# GitHub 分支与提交规范

## 分支角色

```text
main
  当前稳定 baseline，只放用户确认后的主框架。

experiment/innovation
  历史兼容分支：旧创新实验记录入口。

experiment/mod
  MOD-xxx 单模块创新实验总控分支。

experiment/combo
  COMBO-xxx 组合模块实验总控分支。

experiment/rev-mod
  REV-MOD-xxx 单模块复核总控分支。

experiment/tune
  TUNE-xxx 调参实验总控分支。

experiment/abl
  ABL-xxx 消融实验总控分支。

experiment/xds
  XDS-xxx 跨数据集实验总控分支。

experiment/final
  FINAL-xxx 最终复核和正式结果分支。

experiment/workflow-spec
  实验工作流规范分支。

exp/<EXP-ID>_<slug>
  单个实验代码分支。
```

历史兼容分支 `experiment/ablation`、`experiment/tuning`、`experiment/cross-dataset`、`experiment/final-runs` 保留旧记录；新实验按 `experiment/mod`、`experiment/combo`、`experiment/rev-mod`、`experiment/tune`、`experiment/abl`、`experiment/xds`、`experiment/final` 执行。

## 单个实验的分支流程

```text
确认 baseline 分支
  ↓
新建 exp/<EXP-ID>_<slug>
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

- Codex 不自动 push，除非用户明确要求。
- 如果用户要求“同步 GitHub”，默认 push 当前分支到 `origin/<current-branch>`。
- 不自动 merge。
- 不自动 push `main`。
- 不自动删除远端实验分支。

## 为什么 GitHub 要保存规范

本机 skill 会变，聊天会被压缩，外部目录也可能不在另一台机器上。GitHub 仓库中的 `docs/experiment_workflow/` 是项目级真相来源，保证以后任何 agent 都能按同一套实验制度工作。
