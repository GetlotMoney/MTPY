# MOD-006 审查包：反事实负文本挖掘

## 1. 实验假设

当前 CUB 文本描述已经包含细粒度部位属性。若对每个训练样本挖掘语义相似的 seen 类作为负文本，并要求当前类 logit 超过这些负类一个 margin，可能增强细粒度边界并提升 H。

## 2. 来源复核

- 来源节点: `counterfactual-negative-text-mining`
- 数据: `data/gpt4_data/cub_gpt55.pt`、`data/gpt4_data/cub_merge_readable.json`
- 离线检查: 可读 JSON 有 200 类，每类 14 条描述；近邻样例中同属/同形态鸟类能被排到前列。
- 论文原文: 不适用，本模块来自本项目 GPT 描述和用户 idea。
- 官方代码: 不适用。

## 3. 最小 diff

- `model/MyModel.py`: 新增 `loss_cf_neg_text`，只在 `use_cf_neg_text=True` 且 `lambda_cf_neg_text>0` 时计算。
- `train_VGSR_CUB.py`: 日志打印开关和 `CFNeg` loss。
- `config/VGSR_cub_gzsl.yaml`: 新增默认关闭配置。
- `experiments/01_single_module_innovation/MOD-006_counterfactual_negative_text_mining/config.yaml`: 打开 `use_cf_neg_text=True`，设置 `lambda_cf_neg_text=0.03`。

## 4. 审查重点

1. 关闭 `use_cf_neg_text=False` 时是否不改变 baseline。
2. 文本近邻是否只从 seen classes 中挖，是否排除当前类别。
3. `labels` 和 `logits_200` 是否都使用全局类别 ID。
4. 是否只新增轻量训练 loss，没有改主干、seed、epoch、评估口径。
5. margin loss 是否存在明显数值或维度风险。

## 5. 计划运行命令

```powershell
$env:PYTHONIOENCODING='utf-8'
F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/01_single_module_innovation/MOD-006_counterfactual_negative_text_mining/config.yaml
```
