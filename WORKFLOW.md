# DVSR 工作流文件索引

本文档是工作流文件的总入口。所有规范、工具、注册表和手册的索引都在这里，不移动任何文件位置，只负责让你知道东西在哪。

---

## 规范文件

| 文件 | 用途 | 优先级 |
|---|---|---|
| [docs/experiment_workflow/README.md](docs/experiment_workflow/README.md) | 工作流总览：目标、流程图、文件索引、代理映射 | **第一** |
| [docs/experiment_workflow/experiment_taxonomy.md](docs/experiment_workflow/experiment_taxonomy.md) | 实验分类：MOD、COMBO、REV-MOD、TUNE、ABL、XDS、FINAL | 启动时读 |
| [docs/experiment_workflow/github_branching.md](docs/experiment_workflow/github_branching.md) | GitHub 分支规则：总控分支、实验分支、checkpoint、push | 启动时读 |
| [docs/experiment_workflow/claude_review.md](docs/experiment_workflow/claude_review.md) | 审查规范：Codex 自审 + Claude 三轮审查 | 启动时读 |
| [docs/experiment_workflow/idea_tree_feedback.md](docs/experiment_workflow/idea_tree_feedback.md) | 创新树反馈：如何更新 idea_tree.json、队列、backlog | 实验后读 |
| [docs/experiment_workflow/skill_cv_experiment.md](docs/experiment_workflow/skill_cv_experiment.md) | Codex 执行顺序：$cv实验 启动后必须加载的文件列表 | 启动时读 |
| [docs/experiment_workflow/runbook.md](docs/experiment_workflow/runbook.md) | 用户操作手册：启动指令、检查状态、查看日志、同步 GitHub | 运行时参考 |

> 当 skill、对话记忆和本目录冲突时，以 `docs/experiment_workflow/` 为准。

---

## 实验注册与执行

| 文件 | 用途 | 更新时机 |
|---|---|---|
| [backlog.md](backlog.md) | 当前执行窗口：排序好的待跑实验列表 | 每轮实验后 |
| [experiments/EXPERIMENT_REGISTRY.md](experiments/EXPERIMENT_REGISTRY.md) | 实验注册表：所有实验状态、结果、分类 | 每轮实验后 |
| [experiments/08_framework_flow_records/](experiments/08_framework_flow_records/) | 框架流程记录：每个实验的模块框架图 | 实验完成后 |
| [experiments/01_single_module_innovation/](experiments/01_single_module_innovation/) | 单模块创新实验数据 | 运行时自动生成 |
| [experiments/02_module_combination/](experiments/02_module_combination/) | 组合模块实验数据 | 运行时自动生成 |
| [experiments/03_single_module_review/](experiments/03_single_module_review/) | 单模块复核数据 | 运行时自动生成 |
| [experiments/04_hyperparameter_tuning/](experiments/04_hyperparameter_tuning/) | 调参实验数据 | 运行时自动生成 |
| [experiments/05_ablation/](experiments/05_ablation/) | 消融实验数据 | 运行时自动生成 |
| [experiments/06_cross_dataset/](experiments/06_cross_dataset/) | 跨数据集实验数据 | 运行时自动生成 |
| [experiments/07_final_review/](experiments/07_final_review/) | 最终复核数据 | 运行时自动生成 |

---

## 工具与执行器

| 工具 | 文件 | 说明 |
|---|---|---|
| **Codex Skill** | [codex_skills/cv-experiment/SKILL.md](codex_skills/cv-experiment/SKILL.md) | `$cv实验` 的本地执行规范 |
| **MCP 服务** | [mcp/claude-code-worker-mcp/README.md](mcp/claude-code-worker-mcp/README.md) | Claude 审查工作器 MCP 服务，OpenClaw 可通过 `mcporter` 技能调用 |
| **OpenClaw 工作流** | [.openclaw-workspace/OPENCLAW_WORKFLOW.md](.openclaw-workspace/OPENCLAW_WORKFLOW.md) | OpenClaw 多代理执行流程 |
| **OpenClaw 代理** | [.openclaw-workspace/AGENTS.md](.openclaw-workspace/AGENTS.md) | 项目上下文与启动规范 |

---

## 项目汇报

| 文件 | 用途 |
|---|---|
| [DVSR_标准项目汇报.md](DVSR_标准项目汇报.md) | 当前唯一项目汇报，日常接手、开实验、写论文优先读 |
| [README.md](README.md) | 项目主 README，核心思想、实验结果、项目结构 |

---

## 外部创新指导清单

```
C:\Users\Administrator\Desktop\项目\创新指导清单
```

其中维护：
- `paper-idea-tree/idea_tree.json` — 创意树数据（只包含创新模块 MOD）
- `paper-idea-tree/创意树.md` — 人类可读视图（只包含创新模块 MOD）
- `papers/` — 论文资料
- `queues/*.md` — 分类队列

> 创意树（创新树）只放创新模块（MOD）候选。TUNE/ABL/COMBO/XDS/REV-MOD/FINAL 不进入创意树。
> 如果以后需要 GitHub 同步，应迁入仓库、建单独 repo 或作为 submodule；不能只依赖本机路径。

---

## 各分支候选池

| 分支 | 候选池来源 | 是否需要创意树 |
|------|-----------|-------------|
| **MOD** | 创意树（创新树） | ✅ 需要 |
| **TUNE** | 调参策略矩阵 | ❌ 不需要 |
| **ABL** | 现有模块列表（逐个关闭） | ❌ 不需要 |
| **COMBO** | 单模块结果（两个 near_tie/win 组合） | ❌ 不需要 |
| **REV-MOD** | 注册表标记（已跑的 win/near_tie） | ❌ 不需要 |
| **XDS** | 数据集迁移策略 | ❌ 不需要 |
| **FINAL** | 用户指定或注册表标记 | ❌ 不需要 |

### 启动流程

```
用户指令
  ├── "跑创新实验" → 读创意树 → 选 MOD 候选
  ├── "跑调参实验" → 读调参矩阵 → 选 TUNE 候选
  ├── "跑消融实验" → 读模块列表 → 选 ABL 候选
  ├── "跑组合实验" → 读单模块结果 → 选 COMBO 候选
  ├── "跑复核" → 读注册表 → 选 REV-MOD 候选
  └── "跑跨数据集" → 读迁移策略 → 选 XDS 候选
```

各分支独立启动，不互相占用创意树。

---

## 环境要求

| 项目 | 配置 |
|------|------|
| 操作系统 | Windows |
| Python | 3.x |
| conda 环境 | `dvsr_gpu` |
| CUDA | cuda:0 (CUB/SUN), cuda:1 (AWA2) |
| 主要依赖 | `torch`, `clip` (OpenAI), `numpy`, `scipy`, `yaml`, `Pillow` |

## 数据准备

| 数据 | 路径 |
|------|------|
| xlsa17 标准划分 | `data/xlsa17/data/{CUB,AWA2,SUN}/` |
| CUB 原始图片 | `data/CUB/images/` |
| GPT-4 类描述 | `data/gpt4_data/cub.pt` |
| CLIP 特征缓存 | `data/cache/CUB_*.pt`（自动生成） |

## 运行命令

```bash
# 预提取 CLIP 特征（加速）
python tools/extract_features.py

# 训练
python train_VGSR_CUB.py      # CUB
python train_VGSR_AWA2.py     # AWA2
python train_VGSR_SUN.py      # SUN
```

## 加载顺序

无论使用哪种执行器，启动后优先加载：

```
1. docs/experiment_workflow/README.md
2. docs/experiment_workflow/experiment_taxonomy.md
3. docs/experiment_workflow/github_branching.md
4. docs/experiment_workflow/claude_review.md
5. docs/experiment_workflow/skill_cv_experiment.md
6. backlog.md
7. experiments/EXPERIMENT_REGISTRY.md
```

---

*本索引不替代任何规范文件，只负责指路。当索引和实际文件内容冲突时，以实际文件为准。*
