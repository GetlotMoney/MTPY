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
10. Codex 实现最小代码或配置改动。
11. Codex 自审。
12. Claude 三轮审查。
13. 审查通过后运行实验。
14. 记录结果、日志副本、registry、framework flow。
15. 反馈创新树。
16. 更新分类队列和 `backlog.md`。
17. 提交；只有用户明确要求时 push。

## 批量实验

默认一次只跑 1 个实验。用户指定次数时：

- `batch_limit` 等于用户给出的次数。
- 最大为 10。
- 串行执行，一个实验完整结束后才选择下一个。
- 每完成一个实验，重新读取创新树、分类队列和 `backlog.md`。
- 如果出现审查失败、训练失败、环境失败、Git 无关改动或用户中断，停止批量。

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
