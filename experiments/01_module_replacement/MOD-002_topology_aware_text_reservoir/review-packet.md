# MOD-002 Claude 审查包

请对本实验做固定三轮审查。每一轮必须返回 `ACCEPTED` 或 `REJECTED`，最终给出 Overall 决策。只审查，不修改代码。

## 实验

- ID: MOD-002
- 名称: 拓扑感知的自适应文本属性库
- 目录: `experiments/01_module_replacement/MOD-002_topology_aware_text_reservoir/`
- 目标: 验证受限 text reservoir 是否能在保留当前 Pearson 文本拓扑约束的前提下，提升 CUB GZSL H。
- 正式比较口径: 固定 `seed=5` 单次对照，只与同 seed 当前主框架 baseline 比较。

## 关键文件

- `model/MyModel.py`
- `train_VGSR_CUB.py`
- `config/VGSR_cub_gzsl.yaml`
- `experiments/01_module_replacement/MOD-002_topology_aware_text_reservoir/config.yaml`
- `experiments/01_module_replacement/MOD-002_topology_aware_text_reservoir/README.md`
- `experiments/01_module_replacement/MOD-002_topology_aware_text_reservoir/codex-self-review.md`

## 来源复核

- 论文: TPR: Topology-Preserving Reservoirs for Generalized Zero-Shot Learning
- 本地 PDF: `创新指导清单/papers/TPR Topology-Preserving Reservoirs for Generalized Zero-Shot Learning.pdf`
- PDF 状态: 已尝试下载到本地，但当前文件解析报 `STREAM_TRUNCATED_PREMATURELY`，不能作为完整论文原文读取。
- 在线来源: OpenReview / NeurIPS Proceedings 论文页。
- 官方代码: `https://github.com/AllenML-Chen/TPR`
- 官方代码状态: 仓库存在；`git clone --depth 1` 因连接重置失败；raw/API 访问不稳定。
- 标记: `未完整参考论文原文`；`未参考官方代码`。

## 迁移设计

TPR 的迁移只采用“受限 reservoir + topology preserving”的保守机制，不复现完整 TPR。

```text
seen/unseen CLIP 类文本
  -> seen Adapter
  -> 200 类 all_text
  -> class_attr top-K 加权 CUB 属性文本原型
  -> 低秩投影生成 text reservoir residual
  -> all_text = normalize(all_text + ratio * residual)
  -> base_logits / cross_tf
  -> lambda_topo_pearson 约束增强后 all_text
```

关闭等价性:

- 根配置默认 `use_text_attr_reservoir=False`
- 根配置默认 `text_attr_reservoir_ratio=0.0`
- 关闭时不调用 reservoir，`all_text` 与 baseline 一致。
- 实验配置才打开 `use_text_attr_reservoir=True` 且 `text_attr_reservoir_ratio=0.05`。

## 已执行验证

```text
F:\Anaconda\envs\dvsr_gpu\python.exe -m py_compile model\MyModel.py train_VGSR_CUB.py
```

Dummy forward/loss:

- 打开模块: `clip_S_pp` 形状 `[2,150]`，loss=5.1998，`loss_topo=0.0029`，loss 可计算。
- 关闭模块: `clip_S_pp` 形状 `[2,150]`，loss=5.2046，reservoir 路径不启用。

## 三轮审查要求

Pass 1: 代码/配置正确性

- 是否默认关闭且关闭时不改变 baseline 路径。
- 是否存在 checkpoint 兼容性风险。
- 是否存在张量形状、device、dtype、梯度或显存风险。
- 是否存在无关改动。

Pass 2: 实验设计和可复现性

- 来源复核是否足够。
- `未完整参考论文原文` 和 `未参考官方代码` 风险是否标清。
- 迁移是否符合“最小验证一个主要变量”。
- config、seed、命令、日志位置是否可复现。
- 是否保持当前 `lambda_topo_pearson=0.05`，避免和调参混在一起。

Pass 3: 运行计划和放行

- 是否可以开始训练。
- 训练命令是否明确。
- 需要记录哪些结果和日志副本。
- 如果拒绝，请列出必须修复项。

## 输出要求

请把审查结果写入:

```text
experiments/01_module_replacement/MOD-002_topology_aware_text_reservoir/claude-review.md
```

格式必须包含:

- Overall: `ACCEPTED` 或 `REJECTED`
- Pass 1: `ACCEPTED` 或 `REJECTED`
- Pass 2: `ACCEPTED` 或 `REJECTED`
- Pass 3: `ACCEPTED` 或 `REJECTED`
- 主要发现
- 是否允许 Codex 开始训练
