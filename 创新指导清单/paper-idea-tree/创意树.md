# 创意树

更新时间：2026-05-24T02:31:58+08:00

绑定项目：DVSR
项目路径：`C:/Users/Administrator/Desktop/项目/DVSR`
说明：DVSR 实验项目的论文创意树；根据论文创新、模块可用性和本地实验效果更新节点权重。

## 排名

| 排名 | 权重 | 状态 | 节点 | 类别 | 核心理由 |
|---:|---:|---|---|---|---|
| 1 | 2.76 | candidate | 类别角度拓扑保持正则 | 损失函数 | 直接缓解 seen 类过拟合和 unseen 类性能坍塌问题，且可以和 prompt tuning、adapter tuning 等方法组合。 |
| 2 | 2.6 | candidate | 无 base/novel 先验的真实 GZSL 评估 | 评估设置 | 能避免方法只在已知 base/novel 划分时表现好，更适合作为论文汇报和实验设计中的真实泛化标准。 |
| 3 | 2.57 | candidate | 属性库双空间对齐 | 对齐模块 | 可以补足单一隐空间难以表达细粒度视觉-语言模式的问题，适合作为 CLIP-GZSL 的辅助对齐分支。 |
| 4 | 2.5 | candidate | 局部区域-属性最优传输对齐 | 对齐模块 | 适合细粒度 ZSL，因为关键判别信息常出现在局部部件；LaZSL 在 CUB、Place365 等细粒度/复杂数据集上提升更明显，且不需要额外训练。 |
| 5 | 2.45 | candidate | 免训练的可解释 VLM 分类器 | 评估/推理策略 | 适合训练数据少或想做轻量分支时使用；也能帮助区分模型提升来自结构调参还是更好的语义 grounding。 |
| 6 | 2.195 | candidate | 分支间互语义蒸馏 | 损失函数 | 是一个可迁移的多分支 ZSL 训练损失；当每个分支只能捕捉部分证据时，可以稳定视觉-语义对齐。 |
| 7 | 2.065 | candidate | 双向因果属性-视觉注意力 | 结构模块 | 当数据集提供属性标注时很有价值；它直接针对语义错位和注意力伪相关问题。 |

## 节点详情

### 类别角度拓扑保持正则

- ID: `topology-preserving-class-angle-regularization`
- 权重: `2.76`
- 状态: `candidate`
- 类别: `损失函数`
- 摘要: 在适配 seen/base 类时，尽量保留 CLIP 原始类别间的两两角度拓扑结构。
- 机制: 加入拓扑保持目标，使微调后的特征仍接近 CLIP 对 base 和 novel 类共同形成的语义结构。
- 好在哪里: 直接缓解 seen 类过拟合和 unseen 类性能坍塌问题，且可以和 prompt tuning、adapter tuning 等方法组合。
- 评分因子: paper_gain=4.5, own_gain=0, novelty=4.4, compatibility=4, cost=2, confidence=4.5
- 标签: 广义零样本学习, CLIP, 拓扑保持, 正则化, seen-unseen 平衡
- 来源论文:
  - TPR: Topology-Preserving Reservoirs for Generalized Zero-Shot Learning；证据: TPR Table 1: best H in 11/12 datasets; ablation section reports dual-space alignment and topology-preserving objective are complementary across seen, unseen, and harmonic metrics.
- 更新记录:
  - 2026-05-24T02:08:51+08:00: Added from TPR.
  - 2026-05-24T02:12:31+08:00: ????????
  - 2026-05-23T18:16:56.561Z: 改为中文表述。

### 无 base/novel 先验的真实 GZSL 评估

- ID: `practical-gzsl-without-base-novel-oracle`
- 权重: `2.6`
- 状态: `candidate`
- 类别: `评估设置`
- 摘要: 在测试样本未预先划分 seen/base 与 unseen/novel 的情况下评估零样本适配能力。
- 机制: 使用统一测试池，同时报告 seen 准确率、unseen 准确率和调和平均 H，而不是只做 base-to-novel 分开评估。
- 好在哪里: 能避免方法只在已知 base/novel 划分时表现好，更适合作为论文汇报和实验设计中的真实泛化标准。
- 评分因子: paper_gain=3.5, own_gain=0, novelty=3, compatibility=5, cost=1, confidence=4.5
- 标签: 广义零样本学习, 评估协议, seen-unseen, 调和平均
- 来源论文:
  - TPR: Topology-Preserving Reservoirs for Generalized Zero-Shot Learning；证据: TPR motivation: base-to-novel assumes test samples are already divided; paper evaluates GZSL with S/U/H and reports TPR best H on 11/12 datasets.
- 更新记录:
  - 2026-05-24T02:08:51+08:00: Added from TPR as an evaluation idea.
  - 2026-05-24T02:12:31+08:00: ????????
  - 2026-05-23T18:16:56.561Z: 改为中文表述。

### 属性库双空间对齐

- ID: `attribute-reservoir-dual-space-alignment`
- 权重: `2.57`
- 状态: `candidate`
- 类别: `对齐模块`
- 摘要: 在常规 CLIP 隐空间对齐之外，引入由静态词表和可学习 token 组成的属性空间。
- 机制: 构建 attribute reservoir 控制语义粒度，同时在 CLIP 隐空间和属性空间中对齐图像/文本特征。
- 好在哪里: 可以补足单一隐空间难以表达细粒度视觉-语言模式的问题，适合作为 CLIP-GZSL 的辅助对齐分支。
- 评分因子: paper_gain=4.7, own_gain=0, novelty=4.2, compatibility=3.6, cost=3, confidence=4.5
- 标签: 广义零样本学习, CLIP, 属性库, 双空间对齐, 细粒度识别
- 来源论文:
  - TPR: Topology-Preserving Reservoirs for Generalized Zero-Shot Learning；证据: Table 1: TPR obtains best harmonic mean H on 11/12 GZSL datasets; paper reports average absolute gains over prompt learning methods of +12.46 S, +9.46 U, +11.94 H.
- 更新记录:
  - 2026-05-24T02:08:51+08:00: Added from TPR.
  - 2026-05-24T02:12:31+08:00: ????????
  - 2026-05-23T18:16:56.561Z: 改为中文表述。

### 局部区域-属性最优传输对齐

- ID: `local-region-attribute-ot`
- 权重: `2.5`
- 状态: `candidate`
- 类别: `对齐模块`
- 摘要: 不只匹配整图和类别名称，而是把图像局部区域与类别属性进行细粒度对齐。
- 机制: 构建图像局部区域 token 和属性文本 token，使用最优传输计算局部视觉-语义匹配关系，并得到可解释相似度。
- 好在哪里: 适合细粒度 ZSL，因为关键判别信息常出现在局部部件；LaZSL 在 CUB、Place365 等细粒度/复杂数据集上提升更明显，且不需要额外训练。
- 评分因子: paper_gain=4, own_gain=0, novelty=4, compatibility=4, cost=2, confidence=4
- 标签: 零样本学习, CLIP, 属性, 最优传输, 可解释性, 细粒度识别
- 来源论文:
  - Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model；证据: Table 1: LaZSL average accuracy 66.8/69.7/74.0 with ViT-B/32, ViT-B/16, ViT-L/14; gains over DCLIP +2.0/+1.6/+1.2 average, CUB gains +3.8/+3.1/+2.6.
- 更新记录:
  - 2026-05-24T02:08:12+08:00: Added from LaZSL.
  - 2026-05-24T02:12:31+08:00: ????????
  - 2026-05-23T18:16:56.557Z: 改为中文表述。

### 免训练的可解释 VLM 分类器

- ID: `training-free-interpretable-vlm-classifier`
- 权重: `2.45`
- 状态: `candidate`
- 类别: `评估/推理策略`
- 摘要: 用离散属性作为 CLIP 类视觉语言模型的分类接口，在避免 prompt/model tuning 的同时保留可解释性。
- 机制: 从类别属性构建语义集合，再通过局部对齐与图像证据比较，而不是学习额外 prompt 参数。
- 好在哪里: 适合训练数据少或想做轻量分支时使用；也能帮助区分模型提升来自结构调参还是更好的语义 grounding。
- 评分因子: paper_gain=3.5, own_gain=0, novelty=3.5, compatibility=4.5, cost=1, confidence=3.5
- 标签: 零样本学习, CLIP, 可解释性, 免训练, 属性
- 来源论文:
  - Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model；证据: LaZSL compares against interpretable ZSL and trained prompt-learning methods; paper emphasizes state-of-the-art average performance among interpretable methods without additional training.
- 更新记录:
  - 2026-05-24T02:08:12+08:00: Added from LaZSL.
  - 2026-05-24T02:12:31+08:00: ????????
  - 2026-05-23T18:16:56.561Z: 改为中文表述。

### 分支间互语义蒸馏

- ID: `mutual-semantic-distillation-between-branches`
- 权重: `2.195`
- 状态: `candidate`
- 类别: `损失函数`
- 摘要: 让互补的视觉-语义分支在共享语义空间中互相教学、互相约束。
- 机制: 把属性引导的视觉特征和视觉引导的属性特征映射到同一语义空间，通过语义蒸馏让两个分支协同学习。
- 好在哪里: 是一个可迁移的多分支 ZSL 训练损失；当每个分支只能捕捉部分证据时，可以稳定视觉-语义对齐。
- 评分因子: paper_gain=3.8, own_gain=0, novelty=3.5, compatibility=3.5, cost=2.5, confidence=3.8
- 标签: 零样本学习, 广义零样本学习, 蒸馏, 语义空间, 损失函数
- 来源论文:
  - Mutually Causal Semantic Distillation Network for Zero-Shot Learning；证据: MSDN++ reports SOTA-style improvements and FLO GZSL U/S/H = 69.2/80.7/74.5, above the conference MSDN H 70.3.
- 更新记录:
  - 2026-05-24T02:08:28+08:00: Added from MSDN++.
  - 2026-05-24T02:12:31+08:00: ????????
  - 2026-05-23T18:16:56.561Z: 改为中文表述。

### 双向因果属性-视觉注意力

- ID: `bidirectional-causal-attribute-visual-attention`
- 权重: `2.065`
- 状态: `candidate`
- 类别: `结构模块`
- 摘要: 使用两个因果注意力分支，让属性指导视觉定位，同时让视觉证据反向修正属性表示。
- 机制: 训练 attribute-to-visual 和 visual-to-attribute 两个因果注意力子网，减少单向注意力带来的伪相关，学习更可靠的属性关联。
- 好在哪里: 当数据集提供属性标注时很有价值；它直接针对语义错位和注意力伪相关问题。
- 评分因子: paper_gain=4, own_gain=0, novelty=3.8, compatibility=3, cost=3.5, confidence=3.8
- 标签: 零样本学习, 广义零样本学习, 属性, 因果注意力, 属性定位
- 来源论文:
  - Mutually Causal Semantic Distillation Network for Zero-Shot Learning；证据: MSDN++ Table 1 reports strong CZSL/GZSL results on CUB, SUN, AWA2; extracted row includes CUB acc 78.5 and H 70.6, SUN H 42.1, AWA2 H 72.5.
- 更新记录:
  - 2026-05-24T02:08:28+08:00: Added from MSDN++.
  - 2026-05-24T02:12:31+08:00: ????????
  - 2026-05-23T18:16:56.561Z: 改为中文表述。

## 论文索引

- Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model (ICCV 2025): LaZSL 通过最优传输进行局部视觉-语义对齐，让 CLIP/VLM 的零样本预测在无需额外训练的情况下具备可解释性。
- Mutually Causal Semantic Distillation Network for Zero-Shot Learning (IJCV): MSDN++ 通过双向因果注意力和互语义蒸馏，学习更可靠的属性-视觉因果关联，用于 CZSL/GZSL。
- TPR: Topology-Preserving Reservoirs for Generalized Zero-Shot Learning (NeurIPS 2024): TPR 面向更真实的 GZSL 场景，通过属性库双空间对齐和拓扑保持目标增强 CLIP 的 seen/unseen 平衡。
