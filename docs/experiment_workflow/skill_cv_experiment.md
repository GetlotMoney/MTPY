# `$cv实验` skill 行为规范

## 定位

`$cv实验` 是本机 Codex 执行器，不是项目规范本身。项目规范以本目录为准：

```text
docs/experiment_workflow/
```

本机 skill 启动后必须先读取本目录，再执行流程。

## 启动语句

用户可以说：

```text
使用 $cv实验 开始创新模块实验
使用 $cv实验 跑 5 个创新模块实验
使用 $cv实验 执行 MOD-007
使用 $cv实验 把消融实验全部跑完
```

用户不需要额外说“按 backlog.md 串行执行”。skill 必须自己知道：

- 先读分类队列。
- 同步近期候选到 `backlog.md`。
- 从 `backlog.md` 选择最高优先级 open 项。

## 实验范式

`$cv实验` 不再把所有实验都套同一个重流程。先根据实验类型和风险选择范式，再执行。

| 范式 | 适用类型 | 审查要求 | 运行条件 |
|---|---|---|---|
| `TUNE-LITE` 轻量调参 | `TUNE`、不改代码的 `REV-MOD` | Codex 自审 1 次即可；默认不需要 Claude | 只改配置副本；每次只改一个主变量；记录 old/new value |
| `STANDARD` 标准实验 | `ABL`、`XDS`、只改配置或开关的 `FINAL` | Codex 自审 + Claude 单轮审查 | 不改核心 forward/loss/评估逻辑，或只关闭已有路径 |
| `STRICT` 严格实验 | `MOD`、`COMBO`、改核心代码的 `FINAL`、任何改 forward/loss/数据流/评估逻辑的实验 | Codex 自审 + Claude 三轮审查 | 代码改动已最小化；默认关闭可回退；审查全部 `ACCEPTED` |

范式优先级规则：

- 只要改模型 forward、loss、数据加载、评估逻辑或训练脚本行为，就升级到 `STRICT`。
- 只改实验目录里的 `config.yaml`，且不改变代码，保持 `TUNE-LITE`。
- `TUNE` 不允许同时改多个主变量；如果要扫多个候选值，必须拆成多个 `TUNE-xxx` 或在同一个 `TUNE` README 中预注册完整网格。
- `COMBO` 永远不能走轻量范式，因为它必须检查模块冲突和协同假设。
- 用户明确要求更严格审查时，以用户要求为准。

## 执行顺序

1. 读取 `docs/experiment_workflow/README.md`。
2. 读取相关规范文件：taxonomy、branching、idea tree feedback、Claude review。
3. 检查 Git 状态和当前分支。
4. 读取外部创新指导清单和分类队列。
5. 同步近期候选到 `backlog.md`。
6. 选择最高优先级实验。
7. 根据实验 ID 创建或切换 `exp/<EXP-ID>_<slug>` 分支。
8. 创建本地 checkpoint。
9. 准备实验目录、`config.yaml`、README、review packet。
10. 根据范式实现最小代码或配置改动。
11. Codex 自审。
12. 按范式执行审查门：
    - `TUNE-LITE`: Codex 自审 `ACCEPTED` 后直接运行。
    - `STANDARD`: Claude 单轮 `ACCEPTED` 后运行。
    - `STRICT`: Claude 三轮全部 `ACCEPTED` 后运行。
13. 审查通过后运行实验。
14. 记录结果、日志副本、registry、framework flow。
15. 反馈创新树。
16. 更新分类队列和 `backlog.md`。
17. 提交；只有用户明确要求时 push。

## 批量实验

默认一次只跑 1 个实验。用户指定次数时：

- `batch_limit` 等于用户给出的次数。
- `STRICT` 和 `STANDARD` 最大为 10。
- `TUNE-LITE` 最大为 20；超过 20 必须拆批。
- 串行执行，一个实验完整结束后才选择下一个。
- 每完成一个实验，重新读取创新树、分类队列和 `backlog.md`。
- 如果出现审查失败、训练失败、环境失败、Git 无关改动或用户中断，停止批量。

批量 `TUNE-LITE` 规则：

- 默认从当前 `main` baseline 代码快照派生。
- 每个候选配置必须有独立 ID、独立实验目录、独立日志副本。
- 每个候选只改一个主超参；例如 `lambda_topo_pearson`、`lambda_msdn`、`lambda_jepa`、`conditional_text_ratio`、`local_weight`、`lr_stages`。
- 如果用户要求“跑 20 个调参实验”，skill 自动生成两批或一批 20 个串行候选，不需要 Claude 三轮审查。

## 分支选择

默认规则：

- `MOD` 从 `experiment/single-module-innovation` 派生 `exp/<EXP-ID>_<slug>`。
- `COMBO` 从 `experiment/module-combination` 派生 `exp/<EXP-ID>_<slug>`。
- `REV-MOD` 从 `experiment/single-module-review` 派生 `exp/<EXP-ID>_<slug>`。
- `TUNE` 从 `experiment/hyperparameter-tuning` 派生 `exp/<EXP-ID>_<slug>`。
- `ABL` 从 `experiment/ablation` 派生 `exp/<EXP-ID>_<slug>`。
- `XDS` 从 `experiment/cross-dataset` 派生 `exp/<EXP-ID>_<slug>`。
- `FINAL` 从 `experiment/final-review` 派生 `exp/<EXP-ID>_<slug>`。

若当前分支已有用户未提交改动，先汇报，不静默纳入实验。

## 交付要求

每个实验结束后，至少更新：

- 实验 README。
- 实验日志副本。
- `experiments/EXPERIMENT_REGISTRY.md`。
- `experiments/08_framework_flow_records/<EXP-ID>_<slug>.md`。
- 创新树 `idea_tree.json` 和 `创意树.md`。
- 对应分类队列。
- `backlog.md`。
