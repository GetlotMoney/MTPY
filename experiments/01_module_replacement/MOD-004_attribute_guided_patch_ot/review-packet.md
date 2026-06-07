# MOD-004 Claude 审查包

请对本实验做固定三轮审查。每一轮必须返回 `ACCEPTED` 或 `REJECTED`，最终给出 Overall 决策。只审查，不修改代码。

## 实验

- ID: MOD-004
- 名称: 属性引导的局部补丁 OT 对齐
- 目录: `experiments/01_module_replacement/MOD-004_attribute_guided_patch_ot/`
- 目标: 验证 FAE 后局部 patch memory 与当前类别 top-K 属性文本原型的 Sinkhorn 软 OT 辅助 loss 是否提升 CUB GZSL H。
- 正式比较口径: 固定 `seed=5` 单次对照。

## 关键文件

- `model/MyModel.py`
- `train_VGSR_CUB.py`
- `config/VGSR_cub_gzsl.yaml`
- `experiments/01_module_replacement/MOD-004_attribute_guided_patch_ot/config.yaml`
- `experiments/01_module_replacement/MOD-004_attribute_guided_patch_ot/README.md`
- `experiments/01_module_replacement/MOD-004_attribute_guided_patch_ot/codex-self-review.md`

## 来源复核

- 已参考 LaZSL 本地论文原文。
- 官方代码仓库存在，但之前 clone/API/raw 访问失败或超时。
- 标记: `未参考官方代码`。

## 迁移设计

```text
attr_patch_sim: [B, N_patch, 312]
target class_attr top-K: [B, K]
sim = gather(attr_patch_sim, top-K)
transport = Sinkhorn(exp(sim * temp))
loss_attr_patch_ot = 1 - sum(transport * sim)
```

关闭等价性:

- 根配置默认 `use_attr_patch_ot=False`
- 根配置默认 `lambda_attr_patch_ot=0.0`
- 关闭时不传属性文本矩阵、不返回 `attr_patch_sim`、不计算 OT loss。

## 已执行验证

```text
F:\Anaconda\envs\dvsr_gpu\python.exe -m py_compile model\MyModel.py train_VGSR_CUB.py
```

Dummy forward/loss:

- 打开模块: logits `[2,150]`，`loss_attr_patch_ot=0.8877`，返回 `attr_patch_sim`。
- 关闭模块: logits `[2,150]`，`loss_attr_patch_ot=0.0`，不返回 `attr_patch_sim`。

## 输出要求

请把审查结果写入:

```text
experiments/01_module_replacement/MOD-004_attribute_guided_patch_ot/claude-review.md
```
