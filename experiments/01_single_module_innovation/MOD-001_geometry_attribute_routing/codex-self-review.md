# MOD-001 Codex 自审

日期: 2026-06-07

结论: ACCEPTED

## 1. 代码与配置正确性

决策: ACCEPTED

- 新模块由 `use_geo_attr_routing` 和 `lambda_geo_attr_routing` 控制，主配置默认关闭。
- 属性矩阵和属性文本原型作为非持久 buffer 进入模型，不写入 checkpoint，降低旧 checkpoint 加载风险。
- 关闭模块时 `attr_route_score=None`，`loss_geo_attr=0.0`，不会改变主 logits 计算。
- 训练脚本只新增材料传递和日志打印，不改变数据采样、评估函数和主训练循环。

## 2. 实验设计与可复现性

决策: ACCEPTED

- 实验配置独立存放在 `experiments/01_single_module_innovation/MOD-001_geometry_attribute_routing/config.yaml`。
- 训练命令显式使用 `--config`，不会覆盖根配置。
- 来源复核已写入 README：论文已读；官方仓库存在但官方代码 clone 超时，本次标记为 `未参考官方代码`。
- 本次只改变一个主要变量：新增几何感知属性路由辅助 loss。

## 3. 轻量验证

决策: ACCEPTED

已执行:

```text
F:\Anaconda\envs\dvsr_gpu\python.exe -m py_compile model\MyModel.py train_VGSR_CUB.py
```

结果: 通过。

已执行 dummy forward/loss:

- 打开 `use_geo_attr_routing=True`: `clip_S_pp` 形状 `[2,150]`，`attr_route_score` 形状 `[2,312]`，`loss_geo_attr` 可计算。
- 关闭 `use_geo_attr_routing=False`: `attr_route_score=None`，`loss_geo_attr=0.0`。

## 4. 风险

- 官方代码未能完整访问，本次迁移基于论文机制和本项目现有结构，需在 review packet 中保留风险标记。
- 第一版没有实现 OT/Sinkhorn，只实现 top-K 正属性多正例对比 loss；这是有意的最小验证，不应被解释为完整 LaZSL 复现。
- `lambda_geo_attr_routing=0.05` 是起点值，若训练稳定但无提升，后续可进入调参队列。

## 5. Round 1 Claude 拒绝后的修复

结论: ACCEPTED

Claude Round 1 拒绝点是实验 README 中的“多 seed 取最大 H”口径。为避免本次 MOD-001 结果解释出现 seed 选择偏差，本实验已改为固定 `seed=5` 单次正式对照，只与同 seed baseline 比较；其它 seed 若后续补跑，只作为补充稳定性记录，不进入 MOD-001 主结论。
