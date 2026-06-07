# MOD-002 拓扑感知的自适应文本属性库

日期: 2026-06-07

类型: 创新模块 / 新增代码模块

状态: 已完成 / 不保留

## 1. 实验目的

验证一个受限 text reservoir 是否能在保留当前 Pearson 文本拓扑约束的前提下，提升 CUB GZSL 主指标 H。

当前 baseline:

| 数据集 | seed | U | S | H | ZS |
|---|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 73.30 | 72.53 | 72.91 | 81.72 |

本次正式比较口径: 固定 `seed=5` 单次对照，只与同 seed 当前主框架 baseline 比较。

## 2. 来源复核与迁移规格

### 2.1 原始来源

| 项 | 内容 |
|---|---|
| 论文 | TPR: Topology-Preserving Reservoirs for Generalized Zero-Shot Learning |
| 本地 PDF | `创新指导清单/papers/TPR Topology-Preserving Reservoirs for Generalized Zero-Shot Learning.pdf` |
| PDF 状态 | 已尝试下载到本地，但当前文件解析报 `STREAM_TRUNCATED_PREMATURELY`，不能作为完整论文原文读取 |
| 在线来源 | OpenReview / NeurIPS Proceedings 论文页 |
| 官方代码 | `https://github.com/AllenML-Chen/TPR` |
| 官方代码状态 | 仓库存在；`git clone --depth 1` 因连接重置失败；raw/API 访问不稳定 |
| 本次是否参考论文原文 | 部分参考：只参考在线摘要、论文元信息和已有创意树摘录；标记为 `未完整参考论文原文` |
| 本次是否参考官方代码 | 否，标记为 `未参考官方代码` |

### 2.2 原方法要点

TPR 的核心启发是：在 GZSL 中，seen-only 训练容易破坏 seen/unseen 的原始语义拓扑，因此需要让可学习的 reservoir 在增强类别语义表达时，同时保持原始 CLIP 类别间的 pairwise angle / topology。

本项目已经有 `lambda_topo_pearson=0.05`，用于约束增强后类别文本 pairwise cosine 与原始 CLIP 文本拓扑一致。MOD-002 在此基础上增加一个受限 text reservoir：用 CUB 专家属性和属性文本原型形成每个类别的属性语义原型，再通过低秩投影生成小幅文本残差。

### 2.3 迁移到 DVSR 的取舍

可迁移部分:

- reservoir 只作为类别文本的受限残差，不替换主视觉分支。
- 用已有 CUB `class_attr` 和 `clip_att` 构造属性库，避免引入额外数据。
- 继续使用当前 Pearson topology loss 约束增强后的 200 类文本。

不迁移部分:

- 不复现完整 TPR 训练框架和双空间对齐。
- 不改评估协议和 bias 调整。
- 不改 patch 选择、FAE、MSDN、AG-JEPA 权重。
- 不把 reservoir 设为默认主框架。

当前项目插入点:

```text
seen/unseen CLIP 类文本
  -> seen Adapter
  -> 200 类 all_text
  -> MOD-002: CUB 属性 text reservoir 低秩残差
  -> base_logits / cross_tf
  -> Pearson topology loss 约束增强后 all_text
```

关闭等价性:

- `use_text_attr_reservoir=False`
- `text_attr_reservoir_ratio=0.0`
- 关闭时不创建/使用 reservoir 残差，`all_text` 与当前 baseline 一致。
- 主配置默认关闭，只有本实验配置打开。

## 3. 改动内容

| 项 | 原设置 | 新设置 |
|---|---|---|
| 主配置默认 | 无 reservoir 键 | 新增默认关闭键 |
| 实验配置 | 无 reservoir | `use_text_attr_reservoir=True` |
| 残差比例 | 无 | `text_attr_reservoir_ratio=0.05` |
| 属性 top-K | 无 | `text_attr_reservoir_topk=32` |
| 属性温度 | 无 | `text_attr_reservoir_temp=10.0` |
| 低秩 hidden | 无 | `text_attr_reservoir_hidden=256` |

最小 diff 文件:

- `model/MyModel.py`
- `train_VGSR_CUB.py`
- `config/VGSR_cub_gzsl.yaml`
- `experiments/01_module_replacement/MOD-002_topology_aware_text_reservoir/config.yaml`

## 4. 训练配置

| 项 | 值 |
|---|---|
| 数据集 | CUB |
| 随机种子 | 5 |
| 训练流程 | 当前 CUB 多段 cosine 配置，沿用主配置 |
| 命令 | `F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/01_module_replacement/MOD-002_topology_aware_text_reservoir/config.yaml` |
| 配置文件 | `experiments/01_module_replacement/MOD-002_topology_aware_text_reservoir/config.yaml` |
| 正式比较口径 | 固定 seed=5，与同 seed baseline 对照 |

## 5. 审查记录

Codex 自审: ACCEPTED，见 `codex-self-review.md`

Claude 三轮审查: ACCEPTED，见 `claude-review.md`

## 6. 结果

| seed | U | S | H | ZS | 最佳轮次 | 原始日志 | 实验日志副本 |
|---:|---:|---:|---:|---:|---:|---|---|
| 5 | 75.39 | 64.02 | 69.24 | 80.78 | 22 | `train_log/CUB/training_log_CUB_2026-06-07_19-10-33.txt` | `experiments/01_module_replacement/MOD-002_topology_aware_text_reservoir/logs/MOD-002_CUB_seed5_20260607-191033.txt` |

## 7. 结论

保留 / 放弃 / 待补跑: 放弃当前版本

理由: 固定 seed=5 下 H=69.24，低于同 seed baseline 72.91，下降 3.67。MOD-002 明显提升了 U 到 75.39，但 S 掉到 64.02，说明这一版 text reservoir 把语义空间推向 unseen 友好但牺牲 seen 判别，当前 `ratio=0.05 + topK 属性原型残差` 过强或方向不稳。后续如果继续 TPR 思路，应优先做更保守的 residual gate、只作用 seen 或只作用 Adapter 内部，而不是直接改 200 类 all_text。

## 8. 后续动作

- [x] Codex 自审。
- [x] Claude 三轮审查。
- [x] 审查通过后运行训练。
- [x] 复制并命名实验日志。
- [x] 更新 `experiments/EXPERIMENT_REGISTRY.md`。
- [x] 更新 `backlog.md` 和 `创新指导清单/queues/01_module_replacement.md`。
- [x] 生成 `experiments/06_framework_flows/MOD-002_topology_aware_text_reservoir.md`。
- [x] 根据结果更新创意树权重。
