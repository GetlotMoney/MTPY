# MOD-002 Codex 自审

日期: 2026-06-07

结论: ACCEPTED

## 1. 代码与配置正确性

决策: ACCEPTED

- 新模块由 `use_text_attr_reservoir` 和 `text_attr_reservoir_ratio` 控制，主配置默认关闭。
- 模块打开时要求 `class_attr` 和 `attr_text_embeds`，二者由 CUB dataloader 传入。
- 低秩投影最后一层 zero init，训练初始点等价于未加 reservoir。
- reservoir 只改 200 类文本原型，不改 patch 选择、FAE、MSDN、AG-JEPA、评估函数。

## 2. 实验设计与可复现性

决策: ACCEPTED

- 实验配置独立存放在 `experiments/01_single_module_innovation/MOD-002_topology_aware_text_reservoir/config.yaml`。
- 训练命令显式使用 `--config`，不会覆盖根配置。
- 来源复核已写入 README：TPR 本地 PDF 下载不完整，官方代码 clone 失败，因此标记 `未完整参考论文原文` 和 `未参考官方代码`。
- 本次只改变一个主要变量：新增受限 text reservoir 残差；保留当前 `lambda_topo_pearson=0.05`。

## 3. 轻量验证

已执行:

```text
F:\Anaconda\envs\dvsr_gpu\python.exe -m py_compile model\MyModel.py train_VGSR_CUB.py
```

待执行 dummy forward/loss:

- 打开 `use_text_attr_reservoir=True`: logits `[2,150]`，loss=5.1998，`loss_topo=0.0029`，loss 可计算。
- 关闭 `use_text_attr_reservoir=False`: logits `[2,150]`，loss=5.2046，reservoir 路径不启用。

## 4. 风险

- 论文 PDF 当前无法完整解析，官方代码也未能访问，本实验是保守迁移而不是 TPR 复现。
- reservoir 会同时影响 seen/unseen 文本原型；虽然有 Pearson topology loss 约束，仍可能降低 unseen 识别。
- `text_attr_reservoir_ratio=0.05` 是起点值，若训练稳定但无提升，后续再决定是否调参或放弃。
