# MOD-006 Codex 自审

日期: 2026-06-07

结论: ACCEPTED

## 1. 代码与配置正确性

决策: ACCEPTED

- 新模块由 `use_cf_neg_text` 和 `lambda_cf_neg_text` 控制，主配置默认关闭。
- 关闭时 `loss_cf_neg_text=0.0`，不改变 baseline loss。
- 打开时从 `all_text` 中按当前训练标签查找 seen 类文本近邻，排除当前类别，再在 `logits_200` 上计算 margin loss。
- 负类挖掘只使用 seen classes，不使用 unseen 标签，不改变评估流程。

## 2. 来源复核与实验设计

决策: ACCEPTED

- 数据来源是当前项目已有 GPT 描述：`cub_gpt55.pt` 和 `cub_merge_readable.json`。
- `cub_merge_readable.json` 包含 200 类、每类 14 条部位/外观描述。
- 离线近邻样例合理：`Laysan_Albatross` 的近邻包含其它 albatross，`Pine_Grosbeak` 的近邻包含相似 finch / tanager / waxwing。
- 本次只新增轻量训练 loss，不改主干、AG-JEPA、seed、epoch 或评估口径。

## 3. 轻量验证

已执行:

```text
F:\Anaconda\envs\dvsr_gpu\python.exe -m py_compile model\MyModel.py train_VGSR_CUB.py
```

已执行 dummy `compute_loss`:

- `use_cf_neg_text=False`: `loss=6.0708`, `loss_cf_neg_text=0.0000`
- `use_cf_neg_text=True`: `loss=5.2193`, `loss_cf_neg_text=0.9919`

结论：关闭路径和打开路径均能返回正常标量，未发现语法错误或明显维度错误。
