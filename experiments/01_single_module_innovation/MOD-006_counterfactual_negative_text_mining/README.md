# MOD-006 反事实负文本挖掘

日期: 2026-06-07

类型: 创新模块 / 新增代码模块

状态: 已完成

## 1. 实验目的

验证从 GPT 类别文本原型中挖掘相似 seen 类作为反事实负文本，并对当前类 logit 加轻量 margin 约束，是否能提升 CUB GZSL 主指标 H。

当前 baseline:

| 数据集 | seed | U | S | H | ZS |
|---|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 73.30 | 72.53 | 72.91 | 81.72 |

## 2. 来源复核与迁移规格

| 项 | 内容 |
|---|---|
| 创意来源 | `counterfactual-negative-text-mining` 创意树节点 |
| 数据来源 | `data/gpt4_data/cub_gpt55.pt`、`data/gpt4_data/cub_merge_readable.json` |
| 外部论文 | 无直接论文迁移要求 |
| 本次是否参考论文原文 | 不适用，本模块来自本项目 GPT 描述和用户 idea |
| 本次是否参考官方代码 | 不适用 |

离线检查:

- `cub_merge_readable.json` 包含 200 类，每类 14 条部位/外观描述。
- 近邻样例:
  - `Laysan_Albatross` -> `Sooty_Albatross`、`Black_footed_Albatross`、`Frigatebird`
  - `Spotted_Catbird` -> `Gray_Catbird`、`American_Three_toed_Woodpecker`、`Northern_Flicker`
  - `Pine_Grosbeak` -> `Gray_crowned_Rosy_Finch`、`Summer_Tanager`、`Bohemian_Waxwing`

迁移设计:

```text
all_text from current text_source=gpt55
  -> normalize text prototypes
  -> for each training label, find top-K nearest seen classes excluding itself
  -> positive logit = logits_200[label]
  -> negative logits = logits_200[nearest seen negatives]
  -> loss_cf_neg_text = mean ReLU(neg_logit - pos_logit + margin)
```

关闭等价性:

- `use_cf_neg_text=False`
- `lambda_cf_neg_text=0.0`
- 关闭时不计算该 margin loss，不改变主干、AG-JEPA、评估或日志以外行为。
- 主配置默认关闭，只有本实验配置打开。

## 3. 改动内容

| 配置项 | 主配置默认 | MOD-006 实验值 |
|---|---:|---:|
| `use_cf_neg_text` | `False` | `True` |
| `lambda_cf_neg_text` | `0.0` | `0.03` |
| `cf_neg_topk` | `5` | `5` |
| `cf_neg_margin` | `0.2` | `0.2` |

最小 diff 文件:

- `model/MyModel.py`
- `train_VGSR_CUB.py`
- `config/VGSR_cub_gzsl.yaml`
- `experiments/01_single_module_innovation/MOD-006_counterfactual_negative_text_mining/config.yaml`

## 4. 训练配置

| 项 | 值 |
|---|---|
| 数据集 | CUB |
| 随机种子 | 5 |
| 命令 | `F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/01_single_module_innovation/MOD-006_counterfactual_negative_text_mining/config.yaml` |
| 正式比较口径 | 固定 seed=5，与同 seed baseline 对照 |

## 5. 审查记录

Codex 自审: ACCEPTED，见 `codex-self-review.md`

Claude 三轮审查: ACCEPTED，见 `claude-review.md`

## 6. 结果

| 数据集 | seed | U | S | H | ZS | 最佳 epoch | baseline H | ΔH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 72.65 | 72.13 | 72.39 | 81.44 | 51 | 72.91 | -0.52 |

日志与模型产物:

- 原始日志: `train_log/CUB/training_log_CUB_2026-06-07_20-50-24.txt`
- 实验日志副本: `experiments/01_single_module_innovation/MOD-006_counterfactual_negative_text_mining/logs/MOD-006_CUB_seed5_20260607-205024.txt`
- 最佳模型: `train_log/CUB/best_model_CUB_2026-06-07_20-50-24_H7239.pth`
- 完整 checkpoint: `train_log/CUB/ckpt_full_CUB_2026-06-07_20-50-24.pth`

## 7. 结论

MOD-006 没有提升主指标。H 从 72.91 降到 72.39，下降 0.52；U 从 73.30 降到 72.65，S 从 72.53 降到 72.13，ZS 从 81.72 降到 81.44。

结论：当前训练期 `lambda_cf_neg_text=0.03` 的反事实负文本 margin 不纳入主框架。它可能把 already-seen 的相似类边界压得过硬，使文本语义空间更偏 seen 分类而不是 GZSL 泛化。后续如果继续做负文本方向，建议改成推理期重排、离线描述筛选，或只对高置信混淆类启用更小权重。

## 8. 后续动作

- [x] Codex 自审。
- [x] Claude 三轮审查。
- [x] 审查通过后运行训练。
- [x] 复制并命名实验日志。
- [x] 更新 `experiments/EXPERIMENT_REGISTRY.md`。
- [x] 更新 `backlog.md` 和 `创新指导清单/queues/01_module_replacement.md`。
- [x] 生成 `experiments/08_framework_flow_records/MOD-006_counterfactual_negative_text_mining.md`。
- [x] 根据结果更新创意树权重。
