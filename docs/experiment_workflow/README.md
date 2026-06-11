# DVSR 实验工作流总览

本目录是 DVSR 项目的实验自动化“真规范”。本机 Codex skill 只是执行器；当 skill、对话记忆和本目录冲突时，以本目录为准。

## 目标

当前研究目标：

```text
先通过创新模块把 CUB GZSL 主指标推到 H >= 76，再进入系统调参。
```

当前严格连续 baseline：

| 数据集 | seed | U | S | H | ZS |
|---|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 72.33 | 75.95 | 74.09 | 81.75 |

项目选型口径：

- 不取多 seed 平均值作为主结果。
- 同一配置可以预注册多个 seed。
- 主结果按候选 seed 中主指标 `H` 的最大值记录。
- 所有 seed 的 U / S / H / ZS / 日志都必须完整保留。

## 训练策略（不可违反）

**所有实验使用 SGDR 风格的三段式余弦退火（带重启，不到零）。**

```yaml
lr_stages:
  value:
    - {lr: 0.001,  epochs: 20, eta_min: 1e-5}
    - {lr: 0.0001, epochs: 20, eta_min: 1e-6}
    - {lr: 0.00001, epochs: 10, eta_min: 1e-7}
```

三段 20+20+10=50ep，每段 lr 从初始值余弦退火到 eta_min（不到 0），段切换时 lr 跳跃重启，权重不回滚。

设计原理：
- Stage1 20ep 内 lr 从 1e-3 自然退到 1e-5，在辅助 loss 开始压垮 S 之前收束。
- 每段末尾留底（eta_min ≠ 0），避免权重"冻住"。
- 段切换时 lr 跳跃充当"暖重启"。

> 2026-06-11: 30+30+30 SGDR 导致 S 崩盘（Stage1 过长，辅助 loss 推偏模型），改为 20+20+10。

红线：
- `restart_from_best` 必须为 `False`（或不写）。不允许根据测试集 H 回滚 checkpoint。
- 不允许在训练过程中读取测试集指标做任何决策（早停、回滚、调 lr 均不可）。
- 最终结果报告 best epoch 的 H（GZSL 社区惯例），但不得用此做训练内决策。
- 每个 stage 必须显式声明 `eta_min` > 0。

> 2026-06-11 发现 TUNE-024/031 使用了 `restart_from_best: True` + 旧三段式（到 0），属于测试集泄露。
> 自即日起切换为 SGDR 风格（eta_min 不到 0），历史数据作废。旧策略的实验存档保留于 experiments/04_hyperparameter_tuning/ 原目录。

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

## GitHub 总控分支

| 实验类型 | 分支 |
|---|---|
| `MOD-xxx` 单模块创新实验 | `experiment/single-module-innovation` |
| `COMBO-xxx` 组合模块实验 | `experiment/module-combination` |
| `REV-MOD-xxx` 单模块复核 | `experiment/single-module-review` |
| `TUNE-xxx` 调参实验 | `experiment/hyperparameter-tuning` |
| `ABL-xxx` 消融实验 | `experiment/ablation` |
| `XDS-xxx` 跨数据集实验 | `experiment/cross-dataset` |
| `FINAL-xxx` 最终复核 | `experiment/final-review` |

具体单次实验再从对应总控分支派生 `exp/<EXP-ID>_<slug>`。

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

- `paper-idea-tree/idea_tree.json` — 创新树（只包含创新模块 MOD）
- `paper-idea-tree/创意树.md` — 创新树人类可读视图（只包含创新模块 MOD）
- `papers/`
- `queues/*.md`

如果以后需要 GitHub 完整同步创新指导清单，应把它迁入本仓库、建单独 repo，或作为 submodule 管理；不能只依赖本机路径。

> **创新树（创意树）只放创新模块（MOD）候选。** TUNE/ABL/COMBO/XDS/REV-MOD/FINAL 不进入创新树，各分支有独立候选池启动。

---

## OpenClaw 多代理兼容映射

本项目同时支持 **Codex** 和 **OpenClaw** 两个运行时执行工作流。两者共享同一套项目规范（本目录），互不冲突。

### 核心原则

- **项目规范唯一真源**：`docs/experiment_workflow/` 目录对所有执行器统一。
- **审查门不变**：无论用 Codex 还是 OpenClaw，实验运行前必须通过同样的审查门。
- **输出格式不变**：`claude-review.md`、`EXPERIMENT_REGISTRY.md`、`backlog.md` 等文件格式一致。
- **MCP 可用**：`claude_code_worker` MCP 服务可通过 OpenClaw 的 `mcporter` 技能调用，但审查优先使用子代理。

### 角色映射表

| 原有规范角色 | 原有执行者 | OpenClaw 执行者 | 说明 |
|---|---|---|---|
| 代码实现/关键决策 | Codex | `glm-5.1` 子代理 | 最小 diff 实现 + **最重要的决策（调参方向、实验设计、模块选择）** |
| Codex 自审 | Codex | 协调者 `qwen3.7-max` | 检查项与原有规范完全一致 |
| Claude 审查第 1 轮 | Claude CLI (MCP) | `qwen3.7-max` 子代理 | 单代理连续三轮，输出格式不变 |
| Claude 审查第 2 轮 | Claude CLI (MCP) | `qwen3.7-max` 子代理 | 同一会话内连续执行 |
| Claude 审查第 3 轮 | Claude CLI (MCP) | `qwen3.7-max` 子代理 | 同一会话内连续执行 |
| 项目读取 | Codex 自己读 | `deepseek-v4-pro` 子代理 | 长上下文文件读取，新增独立角色 |
| 结果分析 | Codex 自己分析 | `qwen3.7-max` 子代理 | 训练日志分析，新增独立角色 |

### 审查门对照

**原有规范（Codex 模式）**：
```text
Codex 自审: ACCEPTED
Claude 第 1 轮: ACCEPTED
Claude 第 2 轮: ACCEPTED
Claude 第 3 轮: ACCEPTED
```

**OpenClaw 模式**：
```text
协调者自审: ACCEPTED
审查子代理 第 1 轮: ACCEPTED
审查子代理 第 2 轮: ACCEPTED
审查子代理 第 3 轮: ACCEPTED
```

输出文件 `claude-review.md` 格式不变，仍包含：
```markdown
Pass 1 Decision: ACCEPTED
Pass 2 Decision: ACCEPTED
Pass 3 Decision: ACCEPTED
Overall Decision: ACCEPTED
```

### 使用方式

- **Codex 用户**：继续用 `$cv实验` 指令，MCP 自动调用 Claude CLI 审查。
- **OpenClaw 用户**：审查由 `qwen3.7-max` 子代理执行，不经过 MCP。
- **混合使用**：两种方式不会同时运行同一实验，不存在冲突。

### 新增角色的定位

`deepseek-v4-pro`（项目读取）是 OpenClaw 的**增强角色**，不替代原有审查门，只让流程更专业：

- 读取创意树、backlog、分类队列 → 独立子代理，避免长上下文污染主会话
- 审查仍由 `qwen3.7-max` 子代理连续三轮完成，保持上下文连续性
- 结果分析由 `qwen3.7-max` 子代理执行

### 规范文件加载顺序

无论使用哪种执行器，启动后必须先读取：

```text
docs/experiment_workflow/README.md
├── experiment_taxonomy.md
├── github_branching.md
├── idea_tree_feedback.md
├── claude_review.md
├── skill_cv_experiment.md
└── runbook.md
```

当 skill、对话记忆和本目录冲突时，**以本目录为准**。
