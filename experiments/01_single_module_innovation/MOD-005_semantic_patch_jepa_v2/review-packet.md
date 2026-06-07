# MOD-005 审查包：语义补丁 JEPA v2

## 1. 实验假设

现有 AG-JEPA 已被 ABL-002 验证为有效训练信号。MOD-005 不调 AG-JEPA 权重，只把目标构造从“类别文本直接选 top-K patch + cyclic negative”替换为“类别文本相对近邻类别原型的判别 patch + nearest-neighbor hard negative”。

## 2. 来源复核

- 来源节点: `semantic-patch-jepa-v2`
- 本地证据: ABL-002 关闭 AG-JEPA 后 H=71.08，相对 baseline H=72.91 下降 1.83。
- 论文原文: 不适用，本模块来自本项目本地实验和用户 idea。
- 官方代码: 不适用。

## 3. 最小 diff

- `model/MyModel.py`: 新增 `use_ag_jepa_v2`、`jepa_v2_neighbor_topk`、`jepa_v2_neighbor_weight`；关闭时原 AG-JEPA 逻辑不变，打开时使用近邻类别语义构造判别 patch score 和 hard negative。
- `train_VGSR_CUB.py`: 训练日志打印 `use_ag_jepa_v2`。
- `config/VGSR_cub_gzsl.yaml`: 新增默认关闭配置。
- `experiments/01_single_module_innovation/MOD-005_semantic_patch_jepa_v2/config.yaml`: 打开 `use_ag_jepa_v2=True`。

## 4. 审查重点

1. 关闭 `use_ag_jepa_v2=False` 时是否退回原 AG-JEPA。
2. `labels`、`seenclass` 和 `all_text` 的索引是否仍然使用全局类别 ID。
3. nearest seen class 计算是否排除当前类别。
4. 是否只改变 AG-JEPA 目标构造，没有混入权重、seed、epoch 或评估口径变化。
5. 是否存在显著数值风险或静默维度错误。

## 5. 计划运行命令

```powershell
$env:PYTHONIOENCODING='utf-8'
F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/01_single_module_innovation/MOD-005_semantic_patch_jepa_v2/config.yaml
```
