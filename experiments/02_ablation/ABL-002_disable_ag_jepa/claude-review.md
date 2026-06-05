Pass 1 Decision: ACCEPTED  
Pass 2 Decision: ACCEPTED  
Pass 3 Decision: ACCEPTED  
Overall Decision: ACCEPTED  

# Claude Code 三轮审查结果：ABL-002 去掉 AG-JEPA 辅助训练

## 审查范围与限制

- 仅审查实验准备是否允许进入训练阶段。
- 未修改代码。
- 未运行训练或长实验。
- 主要依据审查包；为补足证据，读取了：
  - `experiments/02_ablation/ABL-002_disable_ag_jepa/config.yaml`
  - `config/VGSR_cub_gzsl.yaml`
  - `experiments/02_ablation/ABL-002_disable_ag_jepa/README.md`
  - `model/MyModel.py`
  - `train_VGSR_CUB.py`

---

## Pass 1：代码 / 配置正确性、改动最小性、无关改动风险

**Decision: ACCEPTED**

### 审查结论

实验配置符合 ABL-002 的预期消融目标：关闭 AG-JEPA，并清零相关损失权重，同时使用实验目录下的独立配置文件，没有证据显示修改了模型核心代码或根目录基线配置。

### 关键证据

实验配置 `experiments/02_ablation/ABL-002_disable_ag_jepa/config.yaml` 中关键字段为：

```yaml
dataset:
  value: CUB
device:
  value: cuda:0
epochs:
  value: 30
random_seed:
  value: 5
lastvit_select_k:
  value: 32
use_ag_jepa:
  value: False
lambda_jepa:
  value: 0.0
lambda_jepa_neg:
  value: 0.0
lr_stages:
  value: null
```

与根目录基线配置相比，AG-JEPA 相关字段确实从：

```yaml
use_ag_jepa:
  value: True
lambda_jepa:
  value: 0.05
lambda_jepa_neg:
  value: 0.02
```

变为本实验的：

```yaml
use_ag_jepa:
  value: False
lambda_jepa:
  value: 0.0
lambda_jepa_neg:
  value: 0.0
```

`lr_stages.value` 也由根配置中的多段训练列表变为 `null`，符合审查包声明的“消融实验使用严格连续训练流程”。

### AG-JEPA 关闭逻辑检查

`model/MyModel.py` 中：

- `self.use_ag_jepa = bool(getattr(config, 'use_ag_jepa', False))`
- 仅当 `self.use_ag_jepa` 为真时创建 `self.jepa_predictor`
- `_ag_jepa_loss()` 开头对 `not self.use_ag_jepa` 直接返回零损失
- `compute_loss()` 中仅当：

```python
if self.use_ag_jepa and (lambda_jepa > 0 or lambda_jepa_neg > 0):
```

才计算并加入 JEPA 损失

因此本实验配置下：

- 不创建 AG-JEPA predictor
- 不计算 AG-JEPA positive loss
- 不计算 AG-JEPA negative loss
- 不向总 loss 注入残留 JEPA 项

### 最小性与无关改动风险

审查前可见变更仅为：

```text
D experiments/02_ablation/ABL-002_disable_ag_jepa/claude-review.md
```

结合审查包和已读配置，未发现需要拒绝的无关模型改动或训练脚本改动证据。

---

## Pass 2：实验设计、指标有效性、基线可比性、seed/config 可复现性、数据泄漏风险

**Decision: ACCEPTED**

### 审查结论

实验设计与 ABL-002 目标匹配，主指标、辅助指标、seed、配置路径和基线比较口径清晰。未发现数据泄漏或指标无效问题。

### 实验设计有效性

实验目标是验证 AG-JEPA 辅助训练是否是当前 CUB GZSL 结果增益来源。实验变量集中在：

1. `use_ag_jepa: True → False`
2. `lambda_jepa: 0.05 → 0.0`
3. `lambda_jepa_neg: 0.02 → 0.0`
4. `lr_stages.value` 改为 `null`，以匹配严格连续训练规范

该设计可以隔离“是否使用 AG-JEPA 辅助目标”对 CUB GZSL 主指标 H 的影响。

### 指标与比较口径

审查包和 README 均明确：

- 主指标：CUB GZSL H
- 辅助指标：U、S、ZS、最佳轮次、原始日志路径、实验日志副本路径
- 项目正式比较值取主指标 H 的最大值，不取多 seed 平均
- 当前主基线 H = 72.91，来源为严格连续训练 seed=5
- 本实验第一轮测试只跑 seed=5

该口径清楚，指标有效。

### 可复现性

配置文件中明确：

```yaml
random_seed:
  value: 5
device:
  value: cuda:0
epochs:
  value: 30
batch_size:
  value: 64
```

训练脚本 `train_VGSR_CUB.py` 中读取 `--config`，并设置：

```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
```

虽然脚本没有强制完整 CUDA deterministic 模式，但这与现有项目训练流程一致，不构成本次实验准备阶段的拒绝理由。

### 数据泄漏风险

训练脚本中 unseen 文本嵌入可能来自 GPT 描述缓存或类名文本，但这是零样本 / 广义零样本任务中使用类别语义描述的常规输入，且脚本注释也说明 GPT 描述为纯文本语义，不含视觉监督信号。未发现使用测试图像标签或测试图像特征参与训练的证据。

---

## Pass 3：最终运行计划、环境、命令、日志命名、回滚安全性、是否允许开跑

**Decision: ACCEPTED**

### 审查结论

运行计划、命令、日志命名、结果记录和后续产物要求明确。允许在 Codex 自审和 Claude 三轮审查均通过后启动训练。

### 运行命令检查

计划命令为：

```powershell
$env:PYTHONIOENCODING='utf-8'; F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/02_ablation/ABL-002_disable_ag_jepa/config.yaml
```

符合要求：

- 使用指定 Python：`F:\Anaconda\envs\dvsr_gpu\python.exe`
- 使用实验目录配置：`experiments/02_ablation/ABL-002_disable_ag_jepa/config.yaml`
- `PYTHONIOENCODING=utf-8` 仅影响 Windows 控制台输出编码，不改变训练配置

### 日志与结果记录

README 已要求训练完成后：

- 复制训练日志到实验目录 `logs/`
- 命名规则：

```text
experiments/02_ablation/ABL-002_disable_ag_jepa/logs/ABL-002_CUB_seed5_<YYYYMMDD-HHMMSS>.txt
```

- 在结果表中记录：

```text
seed / U / S / H / ZS / 最佳轮次 / 原始日志 / 实验日志副本
```

### 后续框架流文档

README 后续动作中已包含：

```text
生成 experiments/06_framework_flows/ABL-002_disable_ag_jepa.md，记录代码框架图、流程说明和本实验数据。
```

满足审查包对实验完成后产物的要求。

### 回滚安全性

README 记录实验前 checkpoint：

```text
3d828ad Prepare ABL-002 workflow checkpoint
```

审查包也列出该 checkpoint。当前实验使用独立配置文件，不需要修改根目录基线配置，回滚风险较低。

---

## 最终结论

ABL-002 当前实验准备满足进入训练阶段的条件。

允许在本审查结果写入后，按计划命令启动 seed=5 的 CUB GZSL 训练，并在训练完成后按 workflow 记录指标、日志路径和框架流文档。
Pass 1 Decision: ACCEPTED  
Pass 2 Decision: ACCEPTED  
Pass 3 Decision: ACCEPTED  
Overall Decision: ACCEPTED  

# Claude Code 三轮审查结果：ABL-002 去掉 AG-JEPA 辅助训练

## 审查范围与限制

- 仅审查实验准备是否允许进入训练阶段。
- 未修改代码。
- 未运行训练或长实验。
- 主要依据审查包；为补足证据，读取了：
  - `experiments/02_ablation/ABL-002_disable_ag_jepa/config.yaml`
  - `config/VGSR_cub_gzsl.yaml`
  - `experiments/02_ablation/ABL-002_disable_ag_jepa/README.md`
  - `model/MyModel.py`
  - `train_VGSR_CUB.py`

---

## Pass 1：代码 / 配置正确性、改动最小性、无关改动风险

**Decision: ACCEPTED**

### 审查结论

实验配置符合 ABL-002 的预期消融目标：关闭 AG-JEPA，并清零相关损失权重，同时使用实验目录下的独立配置文件，没有证据显示修改了模型核心代码或根目录基线配置。

### 关键证据

实验配置 `experiments/02_ablation/ABL-002_disable_ag_jepa/config.yaml` 中关键字段为：

```yaml
dataset:
  value: CUB
device:
  value: cuda:0
epochs:
  value: 30
random_seed:
  value: 5
lastvit_select_k:
  value: 32
use_ag_jepa:
  value: False
lambda_jepa:
  value: 0.0
lambda_jepa_neg:
  value: 0.0
lr_stages:
  value: null
```

与根目录基线配置相比，AG-JEPA 相关字段确实从：

```yaml
use_ag_jepa:
  value: True
lambda_jepa:
  value: 0.05
lambda_jepa_neg:
  value: 0.02
```

变为本实验的：

```yaml
use_ag_jepa:
  value: False
lambda_jepa:
  value: 0.0
lambda_jepa_neg:
  value: 0.0
```

`lr_stages.value` 也由根配置中的多段训练列表变为 `null`，符合审查包声明的“消融实验使用严格连续训练流程”。

### AG-JEPA 关闭逻辑检查

`model/MyModel.py` 中：

- `self.use_ag_jepa = bool(getattr(config, 'use_ag_jepa', False))`
- 仅当 `self.use_ag_jepa` 为真时创建 `self.jepa_predictor`
- `_ag_jepa_loss()` 开头对 `not self.use_ag_jepa` 直接返回零损失
- `compute_loss()` 中仅当：

```python
if self.use_ag_jepa and (lambda_jepa > 0 or lambda_jepa_neg > 0):
```

才计算并加入 JEPA 损失

因此本实验配置下：

- 不创建 AG-JEPA predictor
- 不计算 AG-JEPA positive loss
- 不计算 AG-JEPA negative loss
- 不向总 loss 注入残留 JEPA 项

### 最小性与无关改动风险

审查前可见变更仅为：

```text
D experiments/02_ablation/ABL-002_disable_ag_jepa/claude-review.md
```

结合审查包和已读配置，未发现需要拒绝的无关模型改动或训练脚本改动证据。

---

## Pass 2：实验设计、指标有效性、基线可比性、seed/config 可复现性、数据泄漏风险

**Decision: ACCEPTED**

### 审查结论

实验设计与 ABL-002 目标匹配，主指标、辅助指标、seed、配置路径和基线比较口径清晰。未发现数据泄漏或指标无效问题。

### 实验设计有效性

实验目标是验证 AG-JEPA 辅助训练是否是当前 CUB GZSL 结果增益来源。实验变量集中在：

1. `use_ag_jepa: True → False`
2. `lambda_jepa: 0.05 → 0.0`
3. `lambda_jepa_neg: 0.02 → 0.0`
4. `lr_stages.value` 改为 `null`，以匹配严格连续训练规范

该设计可以隔离“是否使用 AG-JEPA 辅助目标”对 CUB GZSL 主指标 H 的影响。

### 指标与比较口径

审查包和 README 均明确：

- 主指标：CUB GZSL H
- 辅助指标：U、S、ZS、最佳轮次、原始日志路径、实验日志副本路径
- 项目正式比较值取主指标 H 的最大值，不取多 seed 平均
- 当前主基线 H = 72.91，来源为严格连续训练 seed=5
- 本实验第一轮测试只跑 seed=5

该口径清楚，指标有效。

### 可复现性

配置文件中明确：

```yaml
random_seed:
  value: 5
device:
  value: cuda:0
epochs:
  value: 30
batch_size:
  value: 64
```

训练脚本 `train_VGSR_CUB.py` 中读取 `--config`，并设置：

```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
```

虽然脚本没有强制完整 CUDA deterministic 模式，但这与现有项目训练流程一致，不构成本次实验准备阶段的拒绝理由。

### 数据泄漏风险

训练脚本中 unseen 文本嵌入可能来自 GPT 描述缓存或类名文本，但这是零样本 / 广义零样本任务中使用类别语义描述的常规输入，且脚本注释也说明 GPT 描述为纯文本语义，不含视觉监督信号。未发现使用测试图像标签或测试图像特征参与训练的证据。

---

## Pass 3：最终运行计划、环境、命令、日志命名、回滚安全性、是否允许开跑

**Decision: ACCEPTED**

### 审查结论

运行计划、命令、日志命名、结果记录和后续产物要求明确。允许在 Codex 自审和 Claude 三轮审查均通过后启动训练。

### 运行命令检查

计划命令为：

```powershell
$env:PYTHONIOENCODING='utf-8'; F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/02_ablation/ABL-002_disable_ag_jepa/config.yaml
```

符合要求：

- 使用指定 Python：`F:\Anaconda\envs\dvsr_gpu\python.exe`
- 使用实验目录配置：`experiments/02_ablation/ABL-002_disable_ag_jepa/config.yaml`
- `PYTHONIOENCODING=utf-8` 仅影响 Windows 控制台输出编码，不改变训练配置

### 日志与结果记录

README 已要求训练完成后：

- 复制训练日志到实验目录 `logs/`
- 命名规则：

```text
experiments/02_ablation/ABL-002_disable_ag_jepa/logs/ABL-002_CUB_seed5_<YYYYMMDD-HHMMSS>.txt
```

- 在结果表中记录：

```text
seed / U / S / H / ZS / 最佳轮次 / 原始日志 / 实验日志副本
```

### 后续框架流文档

README 后续动作中已包含：

```text
生成 experiments/06_framework_flows/ABL-002_disable_ag_jepa.md，记录代码框架图、流程说明和本实验数据。
```

满足审查包对实验完成后产物的要求。

### 回滚安全性

README 记录实验前 checkpoint：

```text
3d828ad Prepare ABL-002 workflow checkpoint
```

审查包也列出该 checkpoint。当前实验使用独立配置文件，不需要修改根目录基线配置，回滚风险较低。

---

## 最终结论

ABL-002 当前实验准备满足进入训练阶段的条件。

允许在本审查结果写入后，按计划命令启动 seed=5 的 CUB GZSL 训练，并在训练完成后按 workflow 记录指标、日志路径和框架流文档。
