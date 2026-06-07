# MOD-001 Claude 审查包

请对本实验做固定三轮审查。每一轮必须返回 `ACCEPTED` 或 `REJECTED`，最终给出 Overall 决策。只审查，不修改代码。

## 实验

- ID: MOD-001
- 名称: 几何感知属性路由
- 目录: `experiments/01_module_replacement/MOD-001_geometry_attribute_routing/`
- 目标: 验证 FAE 后的几何局部表示对齐 CUB 属性原型后，是否能提升 CUB GZSL H。
- 正式比较口径: 固定 `seed=5` 单次对照，只与同 seed 当前主框架 baseline 比较；如后续补跑其它 seed，全部作为补充稳定性记录，不用于本次主结论。

## 关键文件

- `model/MyModel.py`
- `train_VGSR_CUB.py`
- `config/VGSR_cub_gzsl.yaml`
- `experiments/01_module_replacement/MOD-001_geometry_attribute_routing/config.yaml`
- `experiments/01_module_replacement/MOD-001_geometry_attribute_routing/README.md`
- `experiments/01_module_replacement/MOD-001_geometry_attribute_routing/codex-self-review.md`

## 来源复核

- 论文: Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model
- 本地 PDF: `创新指导清单/papers/Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model.pdf`
- 官方代码: `https://github.com/shiming-chen/LaZSL`
- 官方代码状态: `git ls-remote` 已确认仓库存在；GitHub API/raw 访问失败，`git clone --depth 1` 超时。
- 标记: 已参考论文原文；`未参考官方代码`。

## 迁移设计

原 LaZSL 做局部视觉区域与语义属性集合的对齐。本实验不复现完整 LaZSL，不引入随机多尺度裁剪和 OT/Sinkhorn，而是做最小迁移:

```text
CLIP patches
  -> LaSt patch selection
  -> embed_cv
  -> FAE geometry-aware memory
  -> attr_route_score = max_patch cos(memory, attr_text)
  -> loss_geo_attr = top-K class attributes multi-positive contrastive loss
```

关闭等价性:

- 根配置默认 `use_geo_attr_routing=False`
- 根配置默认 `lambda_geo_attr_routing=0.0`
- 关闭时 `attr_route_score=None`
- 关闭时 `loss_geo_attr=0.0`
- 属性材料是非持久 buffer，不进入 checkpoint

## 已执行验证

```text
F:\Anaconda\envs\dvsr_gpu\python.exe -m py_compile model\MyModel.py train_VGSR_CUB.py
```

结果: 通过。

Dummy forward/loss:

- 打开模块: logits `[2,150]`，属性路由 `[2,312]`，loss 可计算。
- 关闭模块: 属性路由为 `None`，`loss_geo_attr=0.0`。

## 审查修复记录

- Round 1 Claude 审查: `REJECTED`。
- 拒绝原因: README 中原先写了“同一设置可测试多个 seed，主结果取候选 seed 中 H 的最大值”，Claude 判定存在 seed 选择偏差。
- 已修复: 本实验正式口径改为固定 `seed=5` 单次对照；其它 seed 只作为补充稳定性记录，不用于 MOD-001 主结论。

## 三轮审查要求

Pass 1: 代码/配置正确性

- 是否默认关闭且关闭时不改变 baseline 路径。
- 是否存在 checkpoint 兼容性风险。
- 是否存在张量形状、device、dtype、梯度或显存风险。
- 是否存在无关改动。

Pass 2: 实验设计和可复现性

- 来源复核是否足够。
- 未参考官方代码的风险是否标清。
- 迁移是否符合“最小验证一个主要变量”。
- config、seed、命令、日志位置是否可复现。

Pass 3: 运行计划和放行

- 是否可以开始训练。
- 训练命令是否明确。
- 需要记录哪些结果和日志副本。
- 如果拒绝，请列出必须修复项。

## 输出要求

请把审查结果写入:

```text
experiments/01_module_replacement/MOD-001_geometry_attribute_routing/claude-review.md
```

格式必须包含:

- Overall: `ACCEPTED` 或 `REJECTED`
- Pass 1: `ACCEPTED` 或 `REJECTED`
- Pass 2: `ACCEPTED` 或 `REJECTED`
- Pass 3: `ACCEPTED` 或 `REJECTED`
- 主要发现
- 是否允许 Codex 开始训练
