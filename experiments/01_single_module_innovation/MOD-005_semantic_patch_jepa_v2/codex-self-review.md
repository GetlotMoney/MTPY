# MOD-005 Codex 自审

日期: 2026-06-07

结论: ACCEPTED

## 1. 代码与配置正确性

决策: ACCEPTED

- 新模块由 `use_ag_jepa_v2` 控制，主配置默认关闭。
- `use_ag_jepa_v2=False` 时仍使用原 AG-JEPA：类别文本选 top-K patch，`context + class_text` 预测 target，cyclic next seen class 作为负文本。
- `use_ag_jepa_v2=True` 时，先在 seen 类文本中找 nearest neighbor，排除当前类别，再用 `sim(patch, class_text) - w * sim(patch, neighbor_text)` 选择判别 patch。
- v2 的 hard negative 使用 nearest neighbor text，不改变 `lambda_jepa`、`lambda_jepa_neg`、`jepa_topk`、seed、epoch 或评估口径。

## 2. 来源复核与实验设计

决策: ACCEPTED

- 来源是本项目现有 AG-JEPA 和 ABL-002，本次不是外部论文迁移。
- ABL-002 记录显示关闭 AG-JEPA 后 H=71.08，相对 baseline H=72.91 下降 1.83，因此升级目标构造有本地实验依据。
- 本次只改变 AG-JEPA 目标构造；权重固定为 baseline 配置。

## 3. 轻量验证

已执行:

```text
F:\Anaconda\envs\dvsr_gpu\python.exe -m py_compile model\MyModel.py train_VGSR_CUB.py
```

已执行 dummy `_ag_jepa_loss`:

- `use_ag_jepa_v2=False`: `loss_jepa=0.9008`, `loss_jepa_neg=0.1378`
- `use_ag_jepa_v2=True`: `loss_jepa=0.9845`, `loss_jepa_neg=0.2267`

结论：关闭路径和打开路径均能返回正常标量，未发现语法错误或明显维度错误。
