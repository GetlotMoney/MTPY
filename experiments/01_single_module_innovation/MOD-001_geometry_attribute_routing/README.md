# MOD-001 几何感知属性路由

日期: 2026-06-07

类型: 创新模块 / 替换模块

状态: 已完成 / 不保留

## 1. 实验目的

验证一个低侵入的属性路由辅助目标是否能提升 CUB GZSL 主指标 H。

当前主框架已经有 FAE 几何感知局部视觉表示、双向视觉-文本交互和 AG-JEPA 辅助训练。MOD-001 不替换主分类器，而是在 FAE 后增加一个可关闭的辅助 loss：让局部视觉 memory 对齐 CUB 的 312 个属性文本原型，并优先把概率质量路由到当前类别的 top-K 专家属性上。

目标基线:

| 数据集 | seed | U | S | H | ZS |
|---|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 73.30 | 72.53 | 72.91 | 81.72 |

本次 MOD-001 正式比较口径: 固定 `seed=5` 单次对照，只与同 seed 的当前主框架基线比较。若后续补跑其它 seed，全部作为补充稳定性记录，不用于本次 MOD-001 的主结论。

## 2. 来源复核与迁移规格

### 2.1 原始来源

| 项 | 内容 |
|---|---|
| 论文 | Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model |
| 方法简称 | LaZSL |
| 本地 PDF | `创新指导清单/papers/Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model.pdf` |
| 官方代码 | `https://github.com/shiming-chen/LaZSL` |
| 官方代码状态 | `git ls-remote` 已确认仓库存在；GitHub API/raw 访问失败，`git clone --depth 1` 超时 |
| 本次是否参考论文原文 | 是 |
| 本次是否参考官方代码 | 否，标记为 `未参考官方代码` |

### 2.2 原方法要点

LaZSL 的核心是局部视觉-语义对齐：

- 用 LLM 构造类别语义属性集合。
- 用多尺度随机裁剪构造局部视觉区域集合。
- 用 CLIP 编码局部视觉区域和语义属性。
- 用 OT/Sinkhorn 做局部视觉-语义匹配。
- 用视觉选择和局部-全局混合相似度生成类别分数。

### 2.3 迁移到 DVSR 的取舍

可迁移部分:

- “局部视觉集合 ↔ 属性语义集合”的对齐思想。
- 用属性原型解释细粒度类别判别。
- 用辅助约束提高局部视觉 memory 的属性可读性。

不迁移部分:

- 不引入 LaZSL 的随机多尺度裁剪，因为本项目已有 CLIP ViT patch 和 FAE 几何编码。
- 不直接替换主 logits，因为这会同时改变分类路径、loss 和评估口径，风险太大。
- 不实现 OT/Sinkhorn 主路径，第一版先用 top-K 正属性的多正例对比 loss 验证方向。

当前项目插入点:

```text
CLIP patches
  -> LaSt patch selection
  -> embed_cv
  -> FAE geometry-aware memory
  -> MOD-001 attr_route_score: max_patch cos(memory, attr_text)
  -> loss_geo_attr: top-K 类别属性多正例对比
```

关闭等价性:

- `use_geo_attr_routing=False`
- `lambda_geo_attr_routing=0.0`
- 关闭后 `attr_route_score=None`，`loss_geo_attr=0`，主 logits 路径不改变。
- 属性材料作为非持久 buffer，不进入 checkpoint。

## 3. 改动内容

| 项 | 原设置 | 新设置 |
|---|---|---|
| 主配置默认 | 无属性路由键 | 新增默认关闭键 |
| 实验配置 | 无属性路由 | `use_geo_attr_routing=True` |
| 辅助 loss | 无 | `lambda_geo_attr_routing=0.05` |
| 正属性数量 | 无 | `geo_attr_route_topk=16` |
| 属性温度 | 无 | `geo_attr_route_temp=14.28` |

最小 diff 文件:

- `model/MyModel.py`
- `train_VGSR_CUB.py`
- `config/VGSR_cub_gzsl.yaml`
- `experiments/01_single_module_innovation/MOD-001_geometry_attribute_routing/config.yaml`

## 4. 训练配置

| 项 | 值 |
|---|---|
| 数据集 | CUB |
| 随机种子 | 5 |
| 训练流程 | 当前 CUB 多段 cosine 配置，沿用主配置 |
| 命令 | `F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/01_single_module_innovation/MOD-001_geometry_attribute_routing/config.yaml` |
| 配置文件 | `experiments/01_single_module_innovation/MOD-001_geometry_attribute_routing/config.yaml` |
| 原始日志 | `train_log/CUB/training_log_CUB_2026-06-07_18-23-26.txt` |
| 实验日志副本 | `experiments/01_single_module_innovation/MOD-001_geometry_attribute_routing/logs/MOD-001_seed5_2026-06-07_18-23-26_training_log.txt` |
| 正式比较口径 | 固定 seed=5，与同 seed baseline 对照 |

## 5. 审查记录

Codex 自审: ACCEPTED，见 `codex-self-review.md`

Claude 三轮审查:

- Round 1: REJECTED。Pass 1 接受；Pass 2/3 因 README 中写了“多 seed 取最大 H 作为主结果”而拒绝。
- 修复: 本 README 已改为固定 seed=5 单次正式对照；其它 seed 仅作为补充稳定性记录。
- Round 2: ACCEPTED。Pass 1/2/3 全部接受，允许开始训练。

## 6. 结果

| seed | U | S | H | ZS | 最佳轮次 | 原始日志 | 实验日志副本 |
|---:|---:|---:|---:|---:|---:|---|---|
| 5 | 73.24 | 71.68 | 72.45 | 81.49 | 51 | `train_log/CUB/training_log_CUB_2026-06-07_18-23-26.txt` | `experiments/01_single_module_innovation/MOD-001_geometry_attribute_routing/logs/MOD-001_seed5_2026-06-07_18-23-26_training_log.txt` |

## 7. 对比基线

| 对比对象 | H | 差值 |
|---|---:|---:|
| MOD-001 | 72.45 | -0.46 |
| 当前主框架基线 | 72.91 |  |

## 8. 结论

保留 / 放弃 / 待补跑: 放弃当前版本

理由: 固定 seed=5 下 H=72.45，低于同 seed baseline 72.91，下降 0.46。属性路由 loss 能正常收敛并趋近很小，但没有带来 GZSL-H 提升，说明第一版“max patch 到属性原型 + top-K 多正例概率质量”更像额外约束，未改善当前双向交互主路径。

## 9. 后续动作

- [x] Codex 自审。
- [x] Claude 三轮审查。
- [x] 审查通过后运行训练。
- [x] 复制并命名实验日志。
- [x] 更新 `experiments/EXPERIMENT_REGISTRY.md`。
- [x] 更新 `backlog.md` 和 `创新指导清单/queues/01_module_replacement.md`。
- [x] 生成 `experiments/08_framework_flow_records/MOD-001_geometry_attribute_routing.md`。
- [x] 根据结果更新创意树权重。
