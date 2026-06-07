Pass 1 Decision: ACCEPTED  
Pass 2 Decision: ACCEPTED  
Pass 3 Decision: ACCEPTED  
Overall Decision: ACCEPTED  

# MOD-005 审查结果：ACCEPTED

## 审查范围

审查包：`experiments/01_module_replacement/MOD-005_semantic_patch_jepa_v2/review-packet.md`

重点核查文件：

- `model/MyModel.py`
- `train_VGSR_CUB.py`
- `config/VGSR_cub_gzsl.yaml`
- `experiments/01_module_replacement/MOD-005_semantic_patch_jepa_v2/config.yaml`

本次只审查，未修改代码，未运行长实验。

---

## Pass 1：实现正确性审查

**Decision: ACCEPTED**

### 1. `use_ag_jepa_v2=False` 时可退回原 AG-JEPA

`model/MyModel.py` 中新增参数：

```python
self.use_ag_jepa_v2 = bool(getattr(config, 'use_ag_jepa_v2', False))
self.jepa_v2_neighbor_topk = int(getattr(config, 'jepa_v2_neighbor_topk', 5))
self.jepa_v2_neighbor_weight = float(getattr(config, 'jepa_v2_neighbor_weight', 0.2))
```

在 `_ag_jepa_loss()` 内，v2 逻辑完全由：

```python
if self.use_ag_jepa_v2:
```

控制。

关闭时仍走原逻辑：

- patch score：`patch_score = patch · class_text`
- negative：seen class cyclic next negative

```python
seen = self.seenclass.to(device=device)
label_pos = torch.zeros_like(labels)
for i, cls_idx in enumerate(seen):
    label_pos[labels == cls_idx] = i
neg_pos = (label_pos + 1) % seen.numel()
neg_text = all_text[seen[neg_pos]]
```

因此默认关闭时原 AG-JEPA 行为保持。

### 2. labels / seenclass / all_text 索引使用全局类别 ID

`all_text` 在 forward 中按全局类别 ID 构造：

```python
all_text[self.seenclass] = seen_text
all_text[self.unseenclass] = self.unseen_text_embeds
```

AG-JEPA 中：

```python
class_text = all_text[labels]
```

训练 labels 是全局类别 ID，`seenclass` 也是全局类别 ID。v2 中：

```python
seen = self.seenclass.to(device=device, dtype=torch.long)
seen_text = all_text[seen]
```

索引体系一致，未发现局部 ID / 全局 ID 混用问题。

### 3. nearest seen class 计算排除当前类别

v2 中：

```python
neighbor_score = text_n @ seen_text_n.T
same_class = labels.unsqueeze(1).eq(seen.unsqueeze(0))
neighbor_score = neighbor_score.masked_fill(same_class, -float("inf"))
```

当前类别会被显式 mask 为 `-inf`，之后再 `topk`：

```python
neighbor_idx = neighbor_score.topk(k=n_neighbor, dim=1).indices
```

满足“nearest seen class 计算排除当前类别”的要求。

### 4. hard negative 构造合理

开启 v2 时，negative 使用 nearest seen neighbor prototype：

```python
neg_text = neighbor_text
```

关闭时仍为 cyclic negative。该改动局限在 AG-JEPA 辅助目标内部，未直接改变主分类 logits 或评估路径。

---

## Pass 2：实验隔离与指标有效性审查

**Decision: ACCEPTED**

### 1. 默认配置关闭 v2

`config/VGSR_cub_gzsl.yaml` 中：

```yaml
use_ag_jepa_v2:
  value: False
jepa_v2_neighbor_topk:
  value: 5
jepa_v2_neighbor_weight:
  value: 0.2
```

默认主配置保持 v2 关闭，符合“关闭时原逻辑不变”的实验隔离要求。

### 2. 实验配置仅开启 v2

实验配置：

`experiments/01_module_replacement/MOD-005_semantic_patch_jepa_v2/config.yaml`

对应位置：

```yaml
use_ag_jepa_v2:
  value: True
jepa_v2_neighbor_topk:
  value: 5
jepa_v2_neighbor_weight:
  value: 0.2
```

从审查到的配置片段看，实验配置与默认配置在 MOD-005 关键点上的差异是开启 `use_ag_jepa_v2=True`，未发现同时修改 seed、epoch、评估口径、主 loss 权重等与本模块无关的实验变量。

### 3. 训练日志记录新增开关

`train_VGSR_CUB.py` 中新增打印：

```python
print_log(f"│  use_ag_jepa_v2: {getattr(config, 'use_ag_jepa_v2', False)}")
```

该项有助于复现实验和确认运行时配置，满足日志可追踪性要求。

### 4. 未发现数据泄漏

v2 的 nearest neighbor 限定在 seen classes：

```python
seen = self.seenclass.to(...)
seen_text = all_text[seen]
```

没有使用 unseen label，也没有在训练中基于 unseen 测试标签构造目标。`all_text` 中虽包含 unseen 原型，但 v2 nearest negative 只从 seen 中选取；主模型原有 GZSL 语义原型机制不属于本次新增泄漏。

---

## Pass 3：数值稳定性、维度安全与可复现性审查

**Decision: ACCEPTED**

### 1. 维度检查

关键张量维度可对齐：

- `patches`: `[B, N, 768]`
- `all_text`: `[200, 768]`
- `labels`: `[B]`
- `class_text = all_text[labels]`: `[B, 768]`
- `seen_text = all_text[seen]`: `[n_seen, 768]`
- `neighbor_score = text_n @ seen_text_n.T`: `[B, n_seen]`
- `neighbor_idx`: `[B, n_neighbor]`
- `neighbor_cls = seen[neighbor_idx]`: `[B, n_neighbor]`
- `neighbor_text`: `[B, 768]`

patch score：

```python
patch_score = torch.einsum('bnd,bd->bn', patch_n, text_n)
neighbor_patch_score = torch.einsum('bnd,bd->bn', patch_n, neighbor_n)
```

输出 `[B, N]`，`topk(dim=1)` 合法。

target 构造：

```python
target = patch_z[mask].view(B, k, -1).mean(dim=1).detach()
```

由于每个 batch 样本 mask 中固定有 `k` 个 True，`view(B, k, -1)` 成立。

### 2. 数值风险可接受

当前类别 mask 为 `-inf` 后做 `topk`。由于训练 seen class 数量远大于 1，且代码中：

```python
n_neighbor = max(1, min(int(self.jepa_v2_neighbor_topk), seen.numel() - 1))
```

避免了 topk 超过可选类别数的问题。CUB seen class 数量约 150，不存在 `seen.numel() <= 1` 的实际风险。

`neighbor_text` 作为 top-k 文本均值后参与：

```python
text_condition = class_text - self.jepa_v2_neighbor_weight * neighbor_text
```

虽然没有再次 normalize，但后续进入 `embed_text`，且权重默认 `0.2`，数值幅度风险较低；不构成拒绝理由。

### 3. 可复现性

配置中保留：

```yaml
random_seed:
  value: 5
```

训练脚本设置：

```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
```

实验包给出明确运行命令：

```powershell
$env:PYTHONIOENCODING='utf-8'
F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/01_module_replacement/MOD-005_semantic_patch_jepa_v2/config.yaml
```

未发现缺失关键复现实验配置的问题。

---

## 结论

MOD-005 的改动符合审查包描述：

- 默认关闭时保留原 AG-JEPA；
- 开启时仅替换 AG-JEPA 的 patch target / negative 构造；
- labels、seenclass、all_text 使用全局类别 ID 一致；
- nearest seen class 显式排除当前类别；
- 实验配置未发现混入 seed、epoch、评估口径等无关变化；
- 未发现数据泄漏、明显维度错误或严重数值风险。

**Overall Decision: ACCEPTED**
Pass 1 Decision: ACCEPTED  
Pass 2 Decision: ACCEPTED  
Pass 3 Decision: ACCEPTED  
Overall Decision: ACCEPTED  

# MOD-005 审查结果：ACCEPTED

## 审查范围

审查包：`experiments/01_module_replacement/MOD-005_semantic_patch_jepa_v2/review-packet.md`

重点核查文件：

- `model/MyModel.py`
- `train_VGSR_CUB.py`
- `config/VGSR_cub_gzsl.yaml`
- `experiments/01_module_replacement/MOD-005_semantic_patch_jepa_v2/config.yaml`

本次只审查，未修改代码，未运行长实验。

---

## Pass 1：实现正确性审查

**Decision: ACCEPTED**

### 1. `use_ag_jepa_v2=False` 时可退回原 AG-JEPA

`model/MyModel.py` 中新增参数：

```python
self.use_ag_jepa_v2 = bool(getattr(config, 'use_ag_jepa_v2', False))
self.jepa_v2_neighbor_topk = int(getattr(config, 'jepa_v2_neighbor_topk', 5))
self.jepa_v2_neighbor_weight = float(getattr(config, 'jepa_v2_neighbor_weight', 0.2))
```

在 `_ag_jepa_loss()` 内，v2 逻辑完全由：

```python
if self.use_ag_jepa_v2:
```

控制。

关闭时仍走原逻辑：

- patch score：`patch_score = patch · class_text`
- negative：seen class cyclic next negative

```python
seen = self.seenclass.to(device=device)
label_pos = torch.zeros_like(labels)
for i, cls_idx in enumerate(seen):
    label_pos[labels == cls_idx] = i
neg_pos = (label_pos + 1) % seen.numel()
neg_text = all_text[seen[neg_pos]]
```

因此默认关闭时原 AG-JEPA 行为保持。

### 2. labels / seenclass / all_text 索引使用全局类别 ID

`all_text` 在 forward 中按全局类别 ID 构造：

```python
all_text[self.seenclass] = seen_text
all_text[self.unseenclass] = self.unseen_text_embeds
```

AG-JEPA 中：

```python
class_text = all_text[labels]
```

训练 labels 是全局类别 ID，`seenclass` 也是全局类别 ID。v2 中：

```python
seen = self.seenclass.to(device=device, dtype=torch.long)
seen_text = all_text[seen]
```

索引体系一致，未发现局部 ID / 全局 ID 混用问题。

### 3. nearest seen class 计算排除当前类别

v2 中：

```python
neighbor_score = text_n @ seen_text_n.T
same_class = labels.unsqueeze(1).eq(seen.unsqueeze(0))
neighbor_score = neighbor_score.masked_fill(same_class, -float("inf"))
```

当前类别会被显式 mask 为 `-inf`，之后再 `topk`：

```python
neighbor_idx = neighbor_score.topk(k=n_neighbor, dim=1).indices
```

满足“nearest seen class 计算排除当前类别”的要求。

### 4. hard negative 构造合理

开启 v2 时，negative 使用 nearest seen neighbor prototype：

```python
neg_text = neighbor_text
```

关闭时仍为 cyclic negative。该改动局限在 AG-JEPA 辅助目标内部，未直接改变主分类 logits 或评估路径。

---

## Pass 2：实验隔离与指标有效性审查

**Decision: ACCEPTED**

### 1. 默认配置关闭 v2

`config/VGSR_cub_gzsl.yaml` 中：

```yaml
use_ag_jepa_v2:
  value: False
jepa_v2_neighbor_topk:
  value: 5
jepa_v2_neighbor_weight:
  value: 0.2
```

默认主配置保持 v2 关闭，符合“关闭时原逻辑不变”的实验隔离要求。

### 2. 实验配置仅开启 v2

实验配置：

`experiments/01_module_replacement/MOD-005_semantic_patch_jepa_v2/config.yaml`

对应位置：

```yaml
use_ag_jepa_v2:
  value: True
jepa_v2_neighbor_topk:
  value: 5
jepa_v2_neighbor_weight:
  value: 0.2
```

从审查到的配置片段看，实验配置与默认配置在 MOD-005 关键点上的差异是开启 `use_ag_jepa_v2=True`，未发现同时修改 seed、epoch、评估口径、主 loss 权重等与本模块无关的实验变量。

### 3. 训练日志记录新增开关

`train_VGSR_CUB.py` 中新增打印：

```python
print_log(f"│  use_ag_jepa_v2: {getattr(config, 'use_ag_jepa_v2', False)}")
```

该项有助于复现实验和确认运行时配置，满足日志可追踪性要求。

### 4. 未发现数据泄漏

v2 的 nearest neighbor 限定在 seen classes：

```python
seen = self.seenclass.to(...)
seen_text = all_text[seen]
```

没有使用 unseen label，也没有在训练中基于 unseen 测试标签构造目标。`all_text` 中虽包含 unseen 原型，但 v2 nearest negative 只从 seen 中选取；主模型原有 GZSL 语义原型机制不属于本次新增泄漏。

---

## Pass 3：数值稳定性、维度安全与可复现性审查

**Decision: ACCEPTED**

### 1. 维度检查

关键张量维度可对齐：

- `patches`: `[B, N, 768]`
- `all_text`: `[200, 768]`
- `labels`: `[B]`
- `class_text = all_text[labels]`: `[B, 768]`
- `seen_text = all_text[seen]`: `[n_seen, 768]`
- `neighbor_score = text_n @ seen_text_n.T`: `[B, n_seen]`
- `neighbor_idx`: `[B, n_neighbor]`
- `neighbor_cls = seen[neighbor_idx]`: `[B, n_neighbor]`
- `neighbor_text`: `[B, 768]`

patch score：

```python
patch_score = torch.einsum('bnd,bd->bn', patch_n, text_n)
neighbor_patch_score = torch.einsum('bnd,bd->bn', patch_n, neighbor_n)
```

输出 `[B, N]`，`topk(dim=1)` 合法。

target 构造：

```python
target = patch_z[mask].view(B, k, -1).mean(dim=1).detach()
```

由于每个 batch 样本 mask 中固定有 `k` 个 True，`view(B, k, -1)` 成立。

### 2. 数值风险可接受

当前类别 mask 为 `-inf` 后做 `topk`。由于训练 seen class 数量远大于 1，且代码中：

```python
n_neighbor = max(1, min(int(self.jepa_v2_neighbor_topk), seen.numel() - 1))
```

避免了 topk 超过可选类别数的问题。CUB seen class 数量约 150，不存在 `seen.numel() <= 1` 的实际风险。

`neighbor_text` 作为 top-k 文本均值后参与：

```python
text_condition = class_text - self.jepa_v2_neighbor_weight * neighbor_text
```

虽然没有再次 normalize，但后续进入 `embed_text`，且权重默认 `0.2`，数值幅度风险较低；不构成拒绝理由。

### 3. 可复现性

配置中保留：

```yaml
random_seed:
  value: 5
```

训练脚本设置：

```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
```

实验包给出明确运行命令：

```powershell
$env:PYTHONIOENCODING='utf-8'
F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/01_module_replacement/MOD-005_semantic_patch_jepa_v2/config.yaml
```

未发现缺失关键复现实验配置的问题。

---

## 结论

MOD-005 的改动符合审查包描述：

- 默认关闭时保留原 AG-JEPA；
- 开启时仅替换 AG-JEPA 的 patch target / negative 构造；
- labels、seenclass、all_text 使用全局类别 ID 一致；
- nearest seen class 显式排除当前类别；
- 实验配置未发现混入 seed、epoch、评估口径等无关变化；
- 未发现数据泄漏、明显维度错误或严重数值风险。

**Overall Decision: ACCEPTED**
