# 创新树反馈规范

## 原则

实验不是只产出分数，还要改变创新树的判断。每次实验结束后，都必须回答：

```text
这个结果支持还是削弱了哪个创意节点？
节点状态怎么变？
权重怎么变？
类别权重要不要变？
下一步应该做什么？
```

## 文件位置

外部创新指导清单当前位于：

```text
C:\Users\Administrator\Desktop\项目\创新指导清单
```

关键文件：

```text
paper-idea-tree/idea_tree.json
paper-idea-tree/创意树.md
queues/01_module_replacement.md
queues/02_ablation.md
queues/03_hyperparam_tuning.md
queues/04_cross_dataset.md
queues/05_final_runs.md
```

## 实验前连接

实验 README 或 review packet 必须写：

- 关联创意树节点 ID。
- 节点标题。
- 节点类别。
- 当前节点状态。
- 该节点为什么被选中。
- 实验类型：`MOD`、`COMBO`、`REV-MOD`、`TUNE`、`ABL`、`XDS` 或 `FINAL`。

如果没有明确节点：

```text
临时用户指定实验，待回填创意树
```

但实验结束后必须补回创意树。

## 实验后回写

必须更新：

- `idea_tree.json` 的 `metrics`。
- `idea_tree.json` 的 `source_materials`。
- `idea_tree.json` 的 `history`。
- `创意树.md`。
- 对应分类队列。
- `backlog.md`。

实验 README 必须新增“创新树反馈意见”。

## 结果分级到动作

| 结果分级 | 节点反馈 | 下一步意见 |
|---|---|---|
| `win` | `validated`，提高 `own_gain` 和 `confidence` | 优先生成 `REV-MOD`，必要时小范围 `TUNE`。 |
| `near_tie` | `weakened` 或继续 `testing`，小幅提高或保持权重 | 先生成 `REV-MOD`；有互补机制时生成 `COMBO`。 |
| `soft_negative` | `weakened`，降低 `confidence` 或 `compatibility` | 保留改良方向，通常不马上继续；互补明确可低优先级 `COMBO`。 |
| `hard_negative` | `rejected` 或大幅降权 | 除非用户指定，暂停该方向。 |
| `blocked` | 不用指标更新权重，在 history 写阻塞原因 | 先修环境或实现问题，再决定是否重跑。 |

## 创新树给出的下一步意见

允许动作：

- `REV-MOD`：单模块复核。
- `COMBO`：组合模块实验。
- `MOD`：继续新单模块初筛。
- `TUNE`：小范围调参或系统调参。
- `ABL`：补消融定位原因。
- `XDS`：跨数据集验证。
- `FINAL`：最终复核。
- `暂停`：该方向暂不继续。

下一步意见必须同步到分类队列；只有近期 3 到 10 个最高优先级项进入 `backlog.md`。

