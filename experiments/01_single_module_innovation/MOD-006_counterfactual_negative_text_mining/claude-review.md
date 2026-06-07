Pass 1 Decision: ACCEPTED  
Pass 2 Decision: ACCEPTED  
Pass 3 Decision: ACCEPTED  
Overall Decision: ACCEPTED  

# MOD-006 审查结果：反事实负文本挖掘

审查对象：`experiments/01_single_module_innovation/MOD-006_counterfactual_negative_text_mining/review-packet.md`  
审查方式：基于审查包，补充读取 `model/MyModel.py`、`train_VGSR_CUB.py`、默认配置与实验配置。未修改代码，未运行长实验。

---

## Pass 1：来源、实验边界与可复现性审查

**Decision: ACCEPTED**

### 结论

MOD-006 的实验假设、数据来源和运行入口描述基本清晰；实验配置通过独立 `config.yaml` 打开模块，默认配置保持关闭，具备可复现实验入口。

### 证据

1. 审查包说明来源节点为 `counterfactual-negative-text-mining`，且明确不是论文复现，而是基于项目 GPT 描述和用户 idea 的模块化实验。
2. 审查包列出使用数据：
   - `data/gpt4_data/cub_gpt55.pt`
   - `data/gpt4_data/cub_merge_readable.json`
3. 实验配置 `experiments/.../config.yaml` 中：
   - `text_source: gpt55`
   - `use_cf_neg_text: True`
   - `lambda_cf_neg_text: 0.03`
   - `cf_neg_topk: 5`
   - `cf_neg_margin: 0.2`
4. 默认配置 `config/VGSR_cub_gzsl.yaml` 中新增项默认关闭：
   - `use_cf_neg_text: False`
   - `lambda_cf_neg_text: 0.0`

### 风险评估

- 未发现新模块引入外部未说明数据、测试标签或测试图像监督。
- 模块使用文本原型进行 seen 类负类挖掘，不使用评估标签。
- 审查包未提供实际运行结果，但本轮主要审查实现与实验包是否可启动；不因未跑长实验而拒绝。

---

## Pass 2：实现正确性、数据泄漏与维度/数值审查

**Decision: ACCEPTED**

### 结论

`loss_cf_neg_text` 的实现符合审查重点：仅在开关打开且权重大于 0 时生效；负类只从 seen classes 中选；排除当前类；`labels` 与 `logits_200` 均按全局类别 ID 使用；margin loss 维度广播合理，未发现明显数值风险。

### 关键实现位置

`model/MyModel.py` 中 `compute_loss()` 的 MOD-006 逻辑：

```python
loss_cf_neg_text = torch.tensor(0.0, device=logits.device)
lambda_cf_neg_text = self.config.__dict__.get('lambda_cf_neg_text', 0)
if self.config.__dict__.get('use_cf_neg_text', False) and lambda_cf_neg_text > 0:
    logits_200_cf = in_package.get('logits_200', None)
    all_text_cf = in_package.get('all_text', None)
    if logits_200_cf is not None and all_text_cf is not None:
        seen = self.seenclass.to(device=logits.device, dtype=torch.long)
        labels_cf = labels.to(device=logits.device, dtype=torch.long)
        with torch.no_grad():
            text_n = F.normalize(all_text_cf.float(), dim=1)
            label_text = text_n[labels_cf]
            seen_text = text_n[seen]
            neg_score = label_text @ seen_text.T
            same_class = labels_cf.unsqueeze(1).eq(seen.unsqueeze(0))
            neg_score = neg_score.masked_fill(same_class, -float("inf"))
            neg_k = int(self.config.__dict__.get('cf_neg_topk', 5))
            neg_k = max(1, min(neg_k, seen.numel() - 1))
            neg_idx = neg_score.topk(k=neg_k, dim=1).indices
            neg_cls = seen[neg_idx]
        pos_logits = logits_200_cf.gather(1, labels_cf.unsqueeze(1))
        neg_logits = logits_200_cf.gather(1, neg_cls)
        margin = float(self.config.__dict__.get('cf_neg_margin', 0.2))
        loss_cf_neg_text = F.relu(neg_logits - pos_logits + margin).mean()
        loss = loss + lambda_cf_neg_text * loss_cf_neg_text
```

### 审查点逐项确认

#### 1. 关闭 `use_cf_neg_text=False` 时不改变 baseline

通过条件：

```python
if self.config.__dict__.get('use_cf_neg_text', False) and lambda_cf_neg_text > 0:
```

默认配置为：

```yaml
use_cf_neg_text:
  value: False
lambda_cf_neg_text:
  value: 0.0
```

因此默认情况下只会创建一个 0 张量并返回日志字段，不改变训练 loss、forward、模型结构或评估逻辑。

#### 2. 文本近邻只从 seen classes 中挖，且排除当前类别

实现中：

```python
seen = self.seenclass.to(...)
seen_text = text_n[seen]
neg_score = label_text @ seen_text.T
same_class = labels_cf.unsqueeze(1).eq(seen.unsqueeze(0))
neg_score = neg_score.masked_fill(same_class, -float("inf"))
```

负类候选明确限制为 `self.seenclass`，并用 mask 排除当前类别。符合要求。

#### 3. `labels` 与 `logits_200` 都使用全局类别 ID

训练中 `batch_label` 来自数据加载或缓存标签，`compute_loss()` 里原有 CE 逻辑将全局 seen label 映射为局部 seen label；而 MOD-006 直接使用全局 label：

```python
pos_logits = logits_200_cf.gather(1, labels_cf.unsqueeze(1))
neg_logits = logits_200_cf.gather(1, neg_cls)
```

`logits_200` 是 `[B, 200]` 全局类别空间，`labels_cf` 和 `neg_cls` 也是全局类别 ID，因此索引一致。

#### 4. margin loss 维度合理

- `pos_logits`: `[B, 1]`
- `neg_logits`: `[B, K]`
- `neg_logits - pos_logits + margin`: 通过广播得到 `[B, K]`
- `.mean()` 得到标量 loss

维度无明显风险。

#### 5. 数值风险

- 文本相似度近邻计算放在 `torch.no_grad()` 中，不给文本近邻选择反传梯度，符合“挖掘负类”设计。
- 当前类 masked 为 `-inf` 后再 `topk`，且 `neg_k <= seen.numel()-1`，在 CUB seen 类数量为 150 的设定下安全。
- margin 使用 `ReLU(neg - pos + margin)`，是标准 hinge/margin 形式。
- 权重 `lambda_cf_neg_text=0.03` 较小，属于轻量辅助 loss。

未发现正确性 bug 或明显数值/维度风险。

---

## Pass 3：训练脚本、配置、日志与评估口径审查

**Decision: ACCEPTED**

### 结论

训练脚本只新增了配置摘要和 loss 日志项；未发现 MOD-006 改动主干结构、seed、epoch、评估函数或 GZSL 指标口径。实验配置通过独立 config 打开模块，默认配置关闭，符合模块替换实验要求。

### 证据

#### 1. 训练脚本仅增加日志

`train_VGSR_CUB.py` 中模块配置摘要新增：

```python
print_log(f"│  use_cf_neg_text: {getattr(config, 'use_cf_neg_text', False)}")
print_log(f"│  lambda_cf_neg_text: {getattr(config, 'lambda_cf_neg_text', 0.0)}")
```

训练 step 日志新增：

```python
cf_v = loss_pack.get('loss_cf_neg_text', torch.tensor(0.)).item()
...
f"JEPA: {jp_v:.4f}  JNeg: {jn_v:.4f}  CFNeg: {cf_v:.4f}  "
```

未看到训练主流程、优化器、调度器、评估调用发生与 MOD-006 相关的改变。

#### 2. 评估口径未改变

评估仍调用：

```python
acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(
    dataloader, clip_model, model, config.device,
    bias_unseen=gzsl_bias)
```

MOD-006 没有修改 `forward()` 的评估分支，也没有修改 `eval_zs_gzsl` 调用方式。该 loss 仅训练期通过 `compute_loss()` 生效。

#### 3. 实验配置打开项符合审查包

实验配置中：

```yaml
use_cf_neg_text:
  value: True
lambda_cf_neg_text:
  value: 0.03
cf_neg_topk:
  value: 5
cf_neg_margin:
  value: 0.2
```

默认配置中同位置为关闭状态，符合“默认关闭、实验配置开启”的要求。

### 注意事项

- 当前实验配置继承了已有 recipe 中的其他模块，例如 AG-JEPA、MSDN、topology、conditional text 等；这不是 MOD-006 本身新增的问题，但后续解释实验结果时应说明该实验是在当前项目强 baseline/组合 recipe 上叠加 MOD-006，而不是纯净只含单一 loss 的最小 baseline。
- 审查未运行长实验，因此不评价指标提升是否成立，只确认实现和实验口径未发现阻断性问题。

---

## 总体结论

MOD-006 的实现满足审查包列出的核心要求：

- 默认关闭，不改变 baseline loss 路径；
- 开启时只增加轻量训练 loss；
- 负文本从 seen 类近邻中挖掘；
- 排除当前类别；
- `labels`、`neg_cls`、`logits_200` 均使用全局类别 ID；
- margin loss 维度与数值处理合理；
- 训练脚本只增加日志，不改变评估口径。

因此三轮审查均通过，Overall Decision 为 **ACCEPTED**。
Pass 1 Decision: ACCEPTED  
Pass 2 Decision: ACCEPTED  
Pass 3 Decision: ACCEPTED  
Overall Decision: ACCEPTED  

# MOD-006 审查结果：反事实负文本挖掘

审查对象：`experiments/01_single_module_innovation/MOD-006_counterfactual_negative_text_mining/review-packet.md`  
审查方式：基于审查包，补充读取 `model/MyModel.py`、`train_VGSR_CUB.py`、默认配置与实验配置。未修改代码，未运行长实验。

---

## Pass 1：来源、实验边界与可复现性审查

**Decision: ACCEPTED**

### 结论

MOD-006 的实验假设、数据来源和运行入口描述基本清晰；实验配置通过独立 `config.yaml` 打开模块，默认配置保持关闭，具备可复现实验入口。

### 证据

1. 审查包说明来源节点为 `counterfactual-negative-text-mining`，且明确不是论文复现，而是基于项目 GPT 描述和用户 idea 的模块化实验。
2. 审查包列出使用数据：
   - `data/gpt4_data/cub_gpt55.pt`
   - `data/gpt4_data/cub_merge_readable.json`
3. 实验配置 `experiments/.../config.yaml` 中：
   - `text_source: gpt55`
   - `use_cf_neg_text: True`
   - `lambda_cf_neg_text: 0.03`
   - `cf_neg_topk: 5`
   - `cf_neg_margin: 0.2`
4. 默认配置 `config/VGSR_cub_gzsl.yaml` 中新增项默认关闭：
   - `use_cf_neg_text: False`
   - `lambda_cf_neg_text: 0.0`

### 风险评估

- 未发现新模块引入外部未说明数据、测试标签或测试图像监督。
- 模块使用文本原型进行 seen 类负类挖掘，不使用评估标签。
- 审查包未提供实际运行结果，但本轮主要审查实现与实验包是否可启动；不因未跑长实验而拒绝。

---

## Pass 2：实现正确性、数据泄漏与维度/数值审查

**Decision: ACCEPTED**

### 结论

`loss_cf_neg_text` 的实现符合审查重点：仅在开关打开且权重大于 0 时生效；负类只从 seen classes 中选；排除当前类；`labels` 与 `logits_200` 均按全局类别 ID 使用；margin loss 维度广播合理，未发现明显数值风险。

### 关键实现位置

`model/MyModel.py` 中 `compute_loss()` 的 MOD-006 逻辑：

```python
loss_cf_neg_text = torch.tensor(0.0, device=logits.device)
lambda_cf_neg_text = self.config.__dict__.get('lambda_cf_neg_text', 0)
if self.config.__dict__.get('use_cf_neg_text', False) and lambda_cf_neg_text > 0:
    logits_200_cf = in_package.get('logits_200', None)
    all_text_cf = in_package.get('all_text', None)
    if logits_200_cf is not None and all_text_cf is not None:
        seen = self.seenclass.to(device=logits.device, dtype=torch.long)
        labels_cf = labels.to(device=logits.device, dtype=torch.long)
        with torch.no_grad():
            text_n = F.normalize(all_text_cf.float(), dim=1)
            label_text = text_n[labels_cf]
            seen_text = text_n[seen]
            neg_score = label_text @ seen_text.T
            same_class = labels_cf.unsqueeze(1).eq(seen.unsqueeze(0))
            neg_score = neg_score.masked_fill(same_class, -float("inf"))
            neg_k = int(self.config.__dict__.get('cf_neg_topk', 5))
            neg_k = max(1, min(neg_k, seen.numel() - 1))
            neg_idx = neg_score.topk(k=neg_k, dim=1).indices
            neg_cls = seen[neg_idx]
        pos_logits = logits_200_cf.gather(1, labels_cf.unsqueeze(1))
        neg_logits = logits_200_cf.gather(1, neg_cls)
        margin = float(self.config.__dict__.get('cf_neg_margin', 0.2))
        loss_cf_neg_text = F.relu(neg_logits - pos_logits + margin).mean()
        loss = loss + lambda_cf_neg_text * loss_cf_neg_text
```

### 审查点逐项确认

#### 1. 关闭 `use_cf_neg_text=False` 时不改变 baseline

通过条件：

```python
if self.config.__dict__.get('use_cf_neg_text', False) and lambda_cf_neg_text > 0:
```

默认配置为：

```yaml
use_cf_neg_text:
  value: False
lambda_cf_neg_text:
  value: 0.0
```

因此默认情况下只会创建一个 0 张量并返回日志字段，不改变训练 loss、forward、模型结构或评估逻辑。

#### 2. 文本近邻只从 seen classes 中挖，且排除当前类别

实现中：

```python
seen = self.seenclass.to(...)
seen_text = text_n[seen]
neg_score = label_text @ seen_text.T
same_class = labels_cf.unsqueeze(1).eq(seen.unsqueeze(0))
neg_score = neg_score.masked_fill(same_class, -float("inf"))
```

负类候选明确限制为 `self.seenclass`，并用 mask 排除当前类别。符合要求。

#### 3. `labels` 与 `logits_200` 都使用全局类别 ID

训练中 `batch_label` 来自数据加载或缓存标签，`compute_loss()` 里原有 CE 逻辑将全局 seen label 映射为局部 seen label；而 MOD-006 直接使用全局 label：

```python
pos_logits = logits_200_cf.gather(1, labels_cf.unsqueeze(1))
neg_logits = logits_200_cf.gather(1, neg_cls)
```

`logits_200` 是 `[B, 200]` 全局类别空间，`labels_cf` 和 `neg_cls` 也是全局类别 ID，因此索引一致。

#### 4. margin loss 维度合理

- `pos_logits`: `[B, 1]`
- `neg_logits`: `[B, K]`
- `neg_logits - pos_logits + margin`: 通过广播得到 `[B, K]`
- `.mean()` 得到标量 loss

维度无明显风险。

#### 5. 数值风险

- 文本相似度近邻计算放在 `torch.no_grad()` 中，不给文本近邻选择反传梯度，符合“挖掘负类”设计。
- 当前类 masked 为 `-inf` 后再 `topk`，且 `neg_k <= seen.numel()-1`，在 CUB seen 类数量为 150 的设定下安全。
- margin 使用 `ReLU(neg - pos + margin)`，是标准 hinge/margin 形式。
- 权重 `lambda_cf_neg_text=0.03` 较小，属于轻量辅助 loss。

未发现正确性 bug 或明显数值/维度风险。

---

## Pass 3：训练脚本、配置、日志与评估口径审查

**Decision: ACCEPTED**

### 结论

训练脚本只新增了配置摘要和 loss 日志项；未发现 MOD-006 改动主干结构、seed、epoch、评估函数或 GZSL 指标口径。实验配置通过独立 config 打开模块，默认配置关闭，符合模块替换实验要求。

### 证据

#### 1. 训练脚本仅增加日志

`train_VGSR_CUB.py` 中模块配置摘要新增：

```python
print_log(f"│  use_cf_neg_text: {getattr(config, 'use_cf_neg_text', False)}")
print_log(f"│  lambda_cf_neg_text: {getattr(config, 'lambda_cf_neg_text', 0.0)}")
```

训练 step 日志新增：

```python
cf_v = loss_pack.get('loss_cf_neg_text', torch.tensor(0.)).item()
...
f"JEPA: {jp_v:.4f}  JNeg: {jn_v:.4f}  CFNeg: {cf_v:.4f}  "
```

未看到训练主流程、优化器、调度器、评估调用发生与 MOD-006 相关的改变。

#### 2. 评估口径未改变

评估仍调用：

```python
acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(
    dataloader, clip_model, model, config.device,
    bias_unseen=gzsl_bias)
```

MOD-006 没有修改 `forward()` 的评估分支，也没有修改 `eval_zs_gzsl` 调用方式。该 loss 仅训练期通过 `compute_loss()` 生效。

#### 3. 实验配置打开项符合审查包

实验配置中：

```yaml
use_cf_neg_text:
  value: True
lambda_cf_neg_text:
  value: 0.03
cf_neg_topk:
  value: 5
cf_neg_margin:
  value: 0.2
```

默认配置中同位置为关闭状态，符合“默认关闭、实验配置开启”的要求。

### 注意事项

- 当前实验配置继承了已有 recipe 中的其他模块，例如 AG-JEPA、MSDN、topology、conditional text 等；这不是 MOD-006 本身新增的问题，但后续解释实验结果时应说明该实验是在当前项目强 baseline/组合 recipe 上叠加 MOD-006，而不是纯净只含单一 loss 的最小 baseline。
- 审查未运行长实验，因此不评价指标提升是否成立，只确认实现和实验口径未发现阻断性问题。

---

## 总体结论

MOD-006 的实现满足审查包列出的核心要求：

- 默认关闭，不改变 baseline loss 路径；
- 开启时只增加轻量训练 loss；
- 负文本从 seen 类近邻中挖掘；
- 排除当前类别；
- `labels`、`neg_cls`、`logits_200` 均使用全局类别 ID；
- margin loss 维度与数值处理合理；
- 训练脚本只增加日志，不改变评估口径。

因此三轮审查均通过，Overall Decision 为 **ACCEPTED**。
