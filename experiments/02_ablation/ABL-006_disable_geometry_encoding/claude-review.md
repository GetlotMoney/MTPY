Pass 1 Decision: ACCEPTED  
Pass 2 Decision: ACCEPTED  
Pass 3 Decision: ACCEPTED  
Overall Decision: ACCEPTED  

# ABL-006 Claude Code 三轮审查结果

审查对象：`experiments/02_ablation/ABL-006_disable_geometry_encoding/review-packet.md`  
实验：ABL-006 去掉几何感知编码  
审查目的：判断是否允许进入训练阶段。  
执行限制：未修改代码，未运行训练或长实验；仅基于审查包和必要文件读取完成审查。

---

## Pass 1：代码 / 配置正确性、改动最小性、无关改动风险

**Decision: ACCEPTED**

### 审查结论

当前实验配置满足 ABL-006 的核心要求：实验目录内配置将 `use_fae.value` 从基线的 `True` 改为 `False`，并保留主框架其它关键模块设置。未发现需要阻止开跑的代码正确性问题或无关模型改动证据。

### 关键证据

已检查实验配置：

- `experiments/02_ablation/ABL-006_disable_geometry_encoding/config.yaml`
  - `dataset.value: CUB`
  - `device.value: cuda:0`
  - `epochs.value: 30`
  - `random_seed.value: 5`
  - `lastvit_select_k.value: 32`
  - `use_fae.value: False`
  - `use_conditional_text.value: True`
  - `conditional_text_ratio.value: 0.005`
  - `lambda_msdn.value: 0.05`
  - `lambda_topo_pearson.value: 0.05`
  - `use_ag_jepa.value: True`
  - `lr_stages.value: null`

与根配置 `config/VGSR_cub_gzsl.yaml` 对比，关键消融点符合预期：

- 根配置：`use_fae.value: True`
- ABL-006：`use_fae.value: False`

同时，`lr_stages.value: null` 是审查包明确说明的流程控制例外，用于关闭多段训练，改为严格连续训练流程；因此不视为无关变量污染。

### `use_fae=False` 代码路径确认

在 `model/MyModel.py` 中：

- `VGSR.__init__` 从配置读取：
  - `use_fae = getattr(config, 'use_fae', True)`
- `CrossModalTransformer.__init__` 中：
  - `self.use_fae = use_fae`
  - 若 `self.use_fae` 为 `True`，才初始化：
    - `BoxRelationalEmbedding`
    - `FAELayer`
  - 若为 `False`：
    - `self.box_emb = None`
    - `self.fae = None`
- `CrossModalTransformer.forward` 中：
  - 若 `self.use_fae` 为 `False`，执行：
    - `memory = vis`
  - 即直接使用线性投影后的 patch 表示，跳过 `BoxRelationalEmbedding + FAELayer`。

这与实验定义“去掉几何感知编码，但保留双向 cross-attention 结构”一致。

### 无关改动风险

审查前可见 Git 状态仅显示：

```text
D experiments/02_ablation/ABL-006_disable_geometry_encoding/claude-review.md
```

这是审查输出文件相关状态，不是模型、训练脚本、根基线配置或数据处理逻辑改动。当前审查未发现无关代码改动证据。

---

## Pass 2：实验设计、指标有效性、基线可比性、seed/config 可复现性、数据泄漏风险

**Decision: ACCEPTED**

### 审查结论

实验设计清晰，主变量明确，指标口径与项目约定一致，seed 和配置路径可复现，未发现数据泄漏或指标无效问题。

### 实验变量控制

ABL-006 的主要消融变量为：

| 项目 | 基线 | ABL-006 |
|---|---:|---:|
| `use_fae` | `True` | `False` |

其它关键主框架参数保持为当前主设置：

- `lastvit_select_k=32`
- `use_conditional_text=True`
- `conditional_text_ratio=0.005`
- `lambda_msdn=0.05`
- `lambda_topo_pearson=0.05`
- `use_ag_jepa=True`
- `random_seed=5`

该设置能够隔离评估“选中局部补丁后，几何感知视觉编码是否仍有贡献”。

### 训练流程可比性

审查包说明当前项目正式主基线 H=72.91，来源为严格连续训练 seed=5。本实验通过：

```yaml
lr_stages:
  value: null
```

关闭多段训练。训练脚本 `train_VGSR_CUB.py` 中逻辑为：

```python
lr_stages = getattr(config, 'lr_stages', None) or []
if lr_stages:
    ...
```

因此 `lr_stages: null` 会被解析为 `None`，再转为空列表 `[]`，不会启用多段训练，符合严格连续训练流程。

### 指标有效性

计划记录指标包括：

- CUB GZSL H，主指标
- U
- S
- ZS
- 最佳轮次
- 原始日志路径
- 实验日志副本路径

项目正式比较口径为“主指标 H 最大值”，不是多 seed 平均值。本实验第一轮只跑 `seed=5`，与当前主基线来源 seed 对齐，具备第一轮可比性。

### 可复现性

配置中已固定：

```yaml
random_seed:
  value: 5
```

训练脚本中设置：

```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
```

同时训练命令明确指定实验目录配置：

```powershell
F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/02_ablation/ABL-006_disable_geometry_encoding/config.yaml
```

这避免误用根目录默认配置。

### 数据泄漏风险

训练脚本中 unseen 类文本嵌入可来自 GPT/GPT-style 文本描述，但审查包和代码语义均表明这是类别语义描述，不使用 unseen 测试图像监督信号。当前未发现新增数据路径、测试标签参与训练、或基于测试集调参的证据。

---

## Pass 3：最终运行计划、环境、命令、日志命名、回滚安全性、是否允许开跑

**Decision: ACCEPTED**

### 审查结论

运行命令、环境、日志计划和后续记录计划符合 workflow 要求，可以进入训练阶段。

### 计划命令检查

审查包指定命令：

```powershell
$env:PYTHONIOENCODING='utf-8'; F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/02_ablation/ABL-006_disable_geometry_encoding/config.yaml
```

检查结果：

- 使用指定 Python：
  - `F:\Anaconda\envs\dvsr_gpu\python.exe`
- 明确传入实验配置：
  - `--config experiments/02_ablation/ABL-006_disable_geometry_encoding/config.yaml`
- `PYTHONIOENCODING=utf-8` 仅影响 Windows 控制台编码，不改变训练配置。

### 日志命名计划

实验日志副本命名规则符合要求：

```text
experiments/02_ablation/ABL-006_disable_geometry_encoding/logs/ABL-006_CUB_seed5_<YYYYMMDD-HHMMSS>.txt
```

训练脚本自身会写入原始日志：

```text
train_log/CUB/training_log_CUB_<timestamp>.txt
```

README 中也已预留结果表字段：

- seed
- U
- S
- H
- ZS
- 最佳轮次
- 原始日志
- 实验日志副本

### 后续记录计划

README 中已列出训练后动作：

- 复制训练日志到实验 `logs/` 目录
- 生成：
  - `experiments/06_framework_flows/ABL-006_disable_geometry_encoding.md`
- 更新：
  - `experiments/EXPERIMENT_REGISTRY.md`
  - `backlog.md`

其中 framework flow 文件要求包含流程图、流程说明、指标数据和日志路径；该计划符合实验 workflow。

### 回滚与安全性

当前无需要修改根配置或模型代码的计划。训练命令只读取实验本地配置，不会覆盖根基线配置。未发现不安全权限操作、删除数据、覆盖基线配置或自动污染其它实验记录的计划。

---

## 最终结论

ABL-006 当前准备充分，配置变量控制合理，代码路径支持 `use_fae=False` 正确跳过 `BoxRelationalEmbedding + FAELayer`，严格连续训练设置有效，运行命令和日志记录计划符合 workflow。

**允许进入训练阶段。**
Pass 1 Decision: ACCEPTED  
Pass 2 Decision: ACCEPTED  
Pass 3 Decision: ACCEPTED  
Overall Decision: ACCEPTED  

# ABL-006 Claude Code 三轮审查结果

审查对象：`experiments/02_ablation/ABL-006_disable_geometry_encoding/review-packet.md`  
实验：ABL-006 去掉几何感知编码  
审查目的：判断是否允许进入训练阶段。  
执行限制：未修改代码，未运行训练或长实验；仅基于审查包和必要文件读取完成审查。

---

## Pass 1：代码 / 配置正确性、改动最小性、无关改动风险

**Decision: ACCEPTED**

### 审查结论

当前实验配置满足 ABL-006 的核心要求：实验目录内配置将 `use_fae.value` 从基线的 `True` 改为 `False`，并保留主框架其它关键模块设置。未发现需要阻止开跑的代码正确性问题或无关模型改动证据。

### 关键证据

已检查实验配置：

- `experiments/02_ablation/ABL-006_disable_geometry_encoding/config.yaml`
  - `dataset.value: CUB`
  - `device.value: cuda:0`
  - `epochs.value: 30`
  - `random_seed.value: 5`
  - `lastvit_select_k.value: 32`
  - `use_fae.value: False`
  - `use_conditional_text.value: True`
  - `conditional_text_ratio.value: 0.005`
  - `lambda_msdn.value: 0.05`
  - `lambda_topo_pearson.value: 0.05`
  - `use_ag_jepa.value: True`
  - `lr_stages.value: null`

与根配置 `config/VGSR_cub_gzsl.yaml` 对比，关键消融点符合预期：

- 根配置：`use_fae.value: True`
- ABL-006：`use_fae.value: False`

同时，`lr_stages.value: null` 是审查包明确说明的流程控制例外，用于关闭多段训练，改为严格连续训练流程；因此不视为无关变量污染。

### `use_fae=False` 代码路径确认

在 `model/MyModel.py` 中：

- `VGSR.__init__` 从配置读取：
  - `use_fae = getattr(config, 'use_fae', True)`
- `CrossModalTransformer.__init__` 中：
  - `self.use_fae = use_fae`
  - 若 `self.use_fae` 为 `True`，才初始化：
    - `BoxRelationalEmbedding`
    - `FAELayer`
  - 若为 `False`：
    - `self.box_emb = None`
    - `self.fae = None`
- `CrossModalTransformer.forward` 中：
  - 若 `self.use_fae` 为 `False`，执行：
    - `memory = vis`
  - 即直接使用线性投影后的 patch 表示，跳过 `BoxRelationalEmbedding + FAELayer`。

这与实验定义“去掉几何感知编码，但保留双向 cross-attention 结构”一致。

### 无关改动风险

审查前可见 Git 状态仅显示：

```text
D experiments/02_ablation/ABL-006_disable_geometry_encoding/claude-review.md
```

这是审查输出文件相关状态，不是模型、训练脚本、根基线配置或数据处理逻辑改动。当前审查未发现无关代码改动证据。

---

## Pass 2：实验设计、指标有效性、基线可比性、seed/config 可复现性、数据泄漏风险

**Decision: ACCEPTED**

### 审查结论

实验设计清晰，主变量明确，指标口径与项目约定一致，seed 和配置路径可复现，未发现数据泄漏或指标无效问题。

### 实验变量控制

ABL-006 的主要消融变量为：

| 项目 | 基线 | ABL-006 |
|---|---:|---:|
| `use_fae` | `True` | `False` |

其它关键主框架参数保持为当前主设置：

- `lastvit_select_k=32`
- `use_conditional_text=True`
- `conditional_text_ratio=0.005`
- `lambda_msdn=0.05`
- `lambda_topo_pearson=0.05`
- `use_ag_jepa=True`
- `random_seed=5`

该设置能够隔离评估“选中局部补丁后，几何感知视觉编码是否仍有贡献”。

### 训练流程可比性

审查包说明当前项目正式主基线 H=72.91，来源为严格连续训练 seed=5。本实验通过：

```yaml
lr_stages:
  value: null
```

关闭多段训练。训练脚本 `train_VGSR_CUB.py` 中逻辑为：

```python
lr_stages = getattr(config, 'lr_stages', None) or []
if lr_stages:
    ...
```

因此 `lr_stages: null` 会被解析为 `None`，再转为空列表 `[]`，不会启用多段训练，符合严格连续训练流程。

### 指标有效性

计划记录指标包括：

- CUB GZSL H，主指标
- U
- S
- ZS
- 最佳轮次
- 原始日志路径
- 实验日志副本路径

项目正式比较口径为“主指标 H 最大值”，不是多 seed 平均值。本实验第一轮只跑 `seed=5`，与当前主基线来源 seed 对齐，具备第一轮可比性。

### 可复现性

配置中已固定：

```yaml
random_seed:
  value: 5
```

训练脚本中设置：

```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
```

同时训练命令明确指定实验目录配置：

```powershell
F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/02_ablation/ABL-006_disable_geometry_encoding/config.yaml
```

这避免误用根目录默认配置。

### 数据泄漏风险

训练脚本中 unseen 类文本嵌入可来自 GPT/GPT-style 文本描述，但审查包和代码语义均表明这是类别语义描述，不使用 unseen 测试图像监督信号。当前未发现新增数据路径、测试标签参与训练、或基于测试集调参的证据。

---

## Pass 3：最终运行计划、环境、命令、日志命名、回滚安全性、是否允许开跑

**Decision: ACCEPTED**

### 审查结论

运行命令、环境、日志计划和后续记录计划符合 workflow 要求，可以进入训练阶段。

### 计划命令检查

审查包指定命令：

```powershell
$env:PYTHONIOENCODING='utf-8'; F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/02_ablation/ABL-006_disable_geometry_encoding/config.yaml
```

检查结果：

- 使用指定 Python：
  - `F:\Anaconda\envs\dvsr_gpu\python.exe`
- 明确传入实验配置：
  - `--config experiments/02_ablation/ABL-006_disable_geometry_encoding/config.yaml`
- `PYTHONIOENCODING=utf-8` 仅影响 Windows 控制台编码，不改变训练配置。

### 日志命名计划

实验日志副本命名规则符合要求：

```text
experiments/02_ablation/ABL-006_disable_geometry_encoding/logs/ABL-006_CUB_seed5_<YYYYMMDD-HHMMSS>.txt
```

训练脚本自身会写入原始日志：

```text
train_log/CUB/training_log_CUB_<timestamp>.txt
```

README 中也已预留结果表字段：

- seed
- U
- S
- H
- ZS
- 最佳轮次
- 原始日志
- 实验日志副本

### 后续记录计划

README 中已列出训练后动作：

- 复制训练日志到实验 `logs/` 目录
- 生成：
  - `experiments/06_framework_flows/ABL-006_disable_geometry_encoding.md`
- 更新：
  - `experiments/EXPERIMENT_REGISTRY.md`
  - `backlog.md`

其中 framework flow 文件要求包含流程图、流程说明、指标数据和日志路径；该计划符合实验 workflow。

### 回滚与安全性

当前无需要修改根配置或模型代码的计划。训练命令只读取实验本地配置，不会覆盖根基线配置。未发现不安全权限操作、删除数据、覆盖基线配置或自动污染其它实验记录的计划。

---

## 最终结论

ABL-006 当前准备充分，配置变量控制合理，代码路径支持 `use_fae=False` 正确跳过 `BoxRelationalEmbedding + FAELayer`，严格连续训练设置有效，运行命令和日志记录计划符合 workflow。

**允许进入训练阶段。**
