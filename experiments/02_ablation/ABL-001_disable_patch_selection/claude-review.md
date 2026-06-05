Pass 1 Decision: ACCEPTED  
Pass 2 Decision: ACCEPTED  
Pass 3 Decision: ACCEPTED  
Overall Decision: ACCEPTED  

# ABL-001 Claude Code 固定三轮审查结果

## 总结

本次审查基于审查包，并补充读取了以下文件作为必要证据：

- `experiments/02_ablation/ABL-001_disable_patch_selection/config.yaml`
- `config/VGSR_cub_gzsl.yaml`
- `model/MyModel.py`
- `experiments/02_ablation/ABL-001_disable_patch_selection/README.md`

结论：当前实验准备可以进入训练阶段。未发现会阻止开跑的正确性 bug、数据泄漏、指标无效、不可复现缺口、无关模型改动风险或不安全权限问题。

---

## Pass 1：代码 / 配置正确性、改动最小性、无关改动风险

**Decision: ACCEPTED**

### 审查结论

通过。

### 证据

1. **实验配置使用实验目录内配置，而不是根目录基线配置**

   计划命令为：

   ```powershell
   F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/02_ablation/ABL-001_disable_patch_selection/config.yaml
   ```

   该命令明确指向实验目录配置文件，未要求修改或运行根目录 `config/VGSR_cub_gzsl.yaml`。

2. **关键变量变更符合预期**

   实验配置中：

   ```yaml
   lastvit_select_k:
     value: 0
   ```

   根目录基线配置中：

   ```yaml
   lastvit_select_k:
     value: 32
   ```

   这符合 ABL-001 的核心消融变量：关闭局部补丁选择器。

3. **`lr_stages.value: null` 已正确设置**

   实验配置中：

   ```yaml
   lr_stages:
     value: null
   ```

   根目录基线配置中存在多阶段训练配置。实验配置将其置空，符合审查包中“消融实验使用严格连续训练流程”的要求。

4. **关键基础配置符合审查包要求**

   实验配置中已确认：

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
     value: 0
   lr_stages:
     value: null
   ```

5. **`lastvit_select_k = 0` 在模型代码中具有明确关闭语义**

   `model/MyModel.py` 中相关逻辑为：

   ```python
   self.lastvit_select_k = int(getattr(config, 'lastvit_select_k', 0))
   ```

   并在 `CrossModalTransformer.forward` 中：

   ```python
   if lastvit_select_k > 0 and lastvit_select_k < patches.size(1):
       ...
       patches = torch.gather(...)
       topk_indices_v5 = topk_indices
   ```

   因此当 `lastvit_select_k = 0` 时，不会进入 patch top-k 选择分支，后续 `patches.size(1) == 576` 时走完整 576 patch 的标准 FAE 路径。

6. **未发现需要拒绝的无关代码改动证据**

   审查前可见变更仅显示：

   ```text
   D experiments/02_ablation/ABL-001_disable_patch_selection/claude-review.md
   ```

   该文件为审查输出文件，会由 MCP 写入最终 Markdown。该删除状态本身不构成模型或训练代码的无关改动风险。

---

## Pass 2：实验设计、指标有效性、基线可比性、seed/config 可复现性、数据泄漏风险

**Decision: ACCEPTED**

### 审查结论

通过。

### 证据与评价

1. **实验问题清晰**

   实验目的为验证“局部补丁选择器是否是当前 CUB GZSL 结果的核心贡献来源”。核心对照为：

   | 项目 | 基线 | 本实验 |
   |---|---:|---:|
   | `lastvit_select_k` | 32 | 0 |

   该消融设计直接对应目标问题。

2. **指标口径有效**

   审查包明确主指标为：

   ```text
   CUB GZSL H
   ```

   辅助记录项包括：

   - U
   - S
   - ZS
   - 最佳轮次
   - 原始日志路径
   - 实验日志副本路径

   README 中也预留了结果表格：

   ```markdown
   | seed | U | S | H | ZS | 最佳轮次 | 原始日志 | 实验日志副本 |
   ```

   指标记录口径完整，主指标明确。

3. **基线比较口径已说明**

   审查包说明项目正式比较值取主指标 H 的最大值，而不是多 seed 平均值；当前主基线 H=72.91，来源为严格连续训练 seed=5。

   本实验第一轮只跑 seed=5，与当前主基线来源 seed 一致，适合作为第一轮消融验证。

4. **随机种子与配置可复现性满足当前开跑要求**

   实验配置中：

   ```yaml
   random_seed:
     value: 5
   ```

   README 也记录 seed=5。命令和配置路径明确，能够复现实验启动条件。

5. **`lr_stages: null` 与严格连续训练口径一致**

   实验配置将 `lr_stages.value` 设为 `null`，符合审查包给出的项目规范要求：消融实验使用严格连续训练流程。虽然根目录基线配置中存在多阶段训练设置，但本实验审查包明确当前正式比较基线为严格连续训练 seed=5，因此该设置不构成拒绝理由。

6. **未发现新增数据泄漏风险**

   实验没有引入新的数据源、测试集调参逻辑或评估偏置搜索。配置中 `gzsl_bias.value: 0.0`，未见测试集调参风险。文本来源、AG-JEPA 等设置与基线配置一致，核心变量变更集中在 patch selection 关闭与训练流程规范化。

---

## Pass 3：最终运行计划、环境、命令、日志命名、回滚安全性、是否允许开跑

**Decision: ACCEPTED**

### 审查结论

通过，允许在 Codex 自审和 Claude 三轮审查均通过后开跑。

### 运行计划核对

1. **Python 环境正确**

   计划命令使用：

   ```powershell
   F:\Anaconda\envs\dvsr_gpu\python.exe
   ```

   符合审查包要求。

2. **训练命令正确**

   允许运行的命令为：

   ```powershell
   F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/02_ablation/ABL-001_disable_patch_selection/config.yaml
   ```

   该命令使用实验目录配置，符合实验隔离要求。

3. **日志副本命名规则明确**

   审查包要求：

   ```text
   experiments/02_ablation/ABL-001_disable_patch_selection/logs/ABL-001_CUB_seed5_<YYYYMMDD-HHMMSS>.txt
   ```

   README 后续动作中也包含对应日志复制与命名要求。

4. **结果记录要求明确**

   README 已预留结果表，并列明后续需要记录：

   - U
   - S
   - H
   - ZS
   - 最佳轮次
   - 原始日志
   - 实验日志副本

5. **后续框架流文档要求已列入计划**

   README 后续动作中包含：

   ```markdown
   - [ ] 生成 `experiments/06_framework_flows/ABL-001_disable_patch_selection.md`，记录代码框架图、流程说明和本实验数据。
   ```

   符合审查包要求。

6. **回滚安全性可接受**

   审查包和 README 均记录实验前 checkpoint：

   ```text
   0874c5d Prepare ABL-001 workflow checkpoint
   ```

   当前可见工作区变更仅涉及审查输出文件删除状态，最终会由 MCP 写入审查 Markdown。未发现需要阻止训练的不安全权限或不可回滚改动。

### 建议但非阻塞

- 训练前可以再创建一个包含最终审查文档的 Git checkpoint，以便完整记录“审查通过后开跑”的状态；但已有实验前 checkpoint `0874c5d`，因此这不是阻止开跑的条件。

---

# 最终结论

ABL-001 当前实验准备满足进入训练阶段的要求。  
在 Codex 自审和本 Claude 三轮审查结果均为 `ACCEPTED` 后，可以按计划运行训练命令。
Pass 1 Decision: ACCEPTED  
Pass 2 Decision: ACCEPTED  
Pass 3 Decision: ACCEPTED  
Overall Decision: ACCEPTED  

# ABL-001 Claude Code 固定三轮审查结果

## 总结

本次审查基于审查包，并补充读取了以下文件作为必要证据：

- `experiments/02_ablation/ABL-001_disable_patch_selection/config.yaml`
- `config/VGSR_cub_gzsl.yaml`
- `model/MyModel.py`
- `experiments/02_ablation/ABL-001_disable_patch_selection/README.md`

结论：当前实验准备可以进入训练阶段。未发现会阻止开跑的正确性 bug、数据泄漏、指标无效、不可复现缺口、无关模型改动风险或不安全权限问题。

---

## Pass 1：代码 / 配置正确性、改动最小性、无关改动风险

**Decision: ACCEPTED**

### 审查结论

通过。

### 证据

1. **实验配置使用实验目录内配置，而不是根目录基线配置**

   计划命令为：

   ```powershell
   F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/02_ablation/ABL-001_disable_patch_selection/config.yaml
   ```

   该命令明确指向实验目录配置文件，未要求修改或运行根目录 `config/VGSR_cub_gzsl.yaml`。

2. **关键变量变更符合预期**

   实验配置中：

   ```yaml
   lastvit_select_k:
     value: 0
   ```

   根目录基线配置中：

   ```yaml
   lastvit_select_k:
     value: 32
   ```

   这符合 ABL-001 的核心消融变量：关闭局部补丁选择器。

3. **`lr_stages.value: null` 已正确设置**

   实验配置中：

   ```yaml
   lr_stages:
     value: null
   ```

   根目录基线配置中存在多阶段训练配置。实验配置将其置空，符合审查包中“消融实验使用严格连续训练流程”的要求。

4. **关键基础配置符合审查包要求**

   实验配置中已确认：

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
     value: 0
   lr_stages:
     value: null
   ```

5. **`lastvit_select_k = 0` 在模型代码中具有明确关闭语义**

   `model/MyModel.py` 中相关逻辑为：

   ```python
   self.lastvit_select_k = int(getattr(config, 'lastvit_select_k', 0))
   ```

   并在 `CrossModalTransformer.forward` 中：

   ```python
   if lastvit_select_k > 0 and lastvit_select_k < patches.size(1):
       ...
       patches = torch.gather(...)
       topk_indices_v5 = topk_indices
   ```

   因此当 `lastvit_select_k = 0` 时，不会进入 patch top-k 选择分支，后续 `patches.size(1) == 576` 时走完整 576 patch 的标准 FAE 路径。

6. **未发现需要拒绝的无关代码改动证据**

   审查前可见变更仅显示：

   ```text
   D experiments/02_ablation/ABL-001_disable_patch_selection/claude-review.md
   ```

   该文件为审查输出文件，会由 MCP 写入最终 Markdown。该删除状态本身不构成模型或训练代码的无关改动风险。

---

## Pass 2：实验设计、指标有效性、基线可比性、seed/config 可复现性、数据泄漏风险

**Decision: ACCEPTED**

### 审查结论

通过。

### 证据与评价

1. **实验问题清晰**

   实验目的为验证“局部补丁选择器是否是当前 CUB GZSL 结果的核心贡献来源”。核心对照为：

   | 项目 | 基线 | 本实验 |
   |---|---:|---:|
   | `lastvit_select_k` | 32 | 0 |

   该消融设计直接对应目标问题。

2. **指标口径有效**

   审查包明确主指标为：

   ```text
   CUB GZSL H
   ```

   辅助记录项包括：

   - U
   - S
   - ZS
   - 最佳轮次
   - 原始日志路径
   - 实验日志副本路径

   README 中也预留了结果表格：

   ```markdown
   | seed | U | S | H | ZS | 最佳轮次 | 原始日志 | 实验日志副本 |
   ```

   指标记录口径完整，主指标明确。

3. **基线比较口径已说明**

   审查包说明项目正式比较值取主指标 H 的最大值，而不是多 seed 平均值；当前主基线 H=72.91，来源为严格连续训练 seed=5。

   本实验第一轮只跑 seed=5，与当前主基线来源 seed 一致，适合作为第一轮消融验证。

4. **随机种子与配置可复现性满足当前开跑要求**

   实验配置中：

   ```yaml
   random_seed:
     value: 5
   ```

   README 也记录 seed=5。命令和配置路径明确，能够复现实验启动条件。

5. **`lr_stages: null` 与严格连续训练口径一致**

   实验配置将 `lr_stages.value` 设为 `null`，符合审查包给出的项目规范要求：消融实验使用严格连续训练流程。虽然根目录基线配置中存在多阶段训练设置，但本实验审查包明确当前正式比较基线为严格连续训练 seed=5，因此该设置不构成拒绝理由。

6. **未发现新增数据泄漏风险**

   实验没有引入新的数据源、测试集调参逻辑或评估偏置搜索。配置中 `gzsl_bias.value: 0.0`，未见测试集调参风险。文本来源、AG-JEPA 等设置与基线配置一致，核心变量变更集中在 patch selection 关闭与训练流程规范化。

---

## Pass 3：最终运行计划、环境、命令、日志命名、回滚安全性、是否允许开跑

**Decision: ACCEPTED**

### 审查结论

通过，允许在 Codex 自审和 Claude 三轮审查均通过后开跑。

### 运行计划核对

1. **Python 环境正确**

   计划命令使用：

   ```powershell
   F:\Anaconda\envs\dvsr_gpu\python.exe
   ```

   符合审查包要求。

2. **训练命令正确**

   允许运行的命令为：

   ```powershell
   F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/02_ablation/ABL-001_disable_patch_selection/config.yaml
   ```

   该命令使用实验目录配置，符合实验隔离要求。

3. **日志副本命名规则明确**

   审查包要求：

   ```text
   experiments/02_ablation/ABL-001_disable_patch_selection/logs/ABL-001_CUB_seed5_<YYYYMMDD-HHMMSS>.txt
   ```

   README 后续动作中也包含对应日志复制与命名要求。

4. **结果记录要求明确**

   README 已预留结果表，并列明后续需要记录：

   - U
   - S
   - H
   - ZS
   - 最佳轮次
   - 原始日志
   - 实验日志副本

5. **后续框架流文档要求已列入计划**

   README 后续动作中包含：

   ```markdown
   - [ ] 生成 `experiments/06_framework_flows/ABL-001_disable_patch_selection.md`，记录代码框架图、流程说明和本实验数据。
   ```

   符合审查包要求。

6. **回滚安全性可接受**

   审查包和 README 均记录实验前 checkpoint：

   ```text
   0874c5d Prepare ABL-001 workflow checkpoint
   ```

   当前可见工作区变更仅涉及审查输出文件删除状态，最终会由 MCP 写入审查 Markdown。未发现需要阻止训练的不安全权限或不可回滚改动。

### 建议但非阻塞

- 训练前可以再创建一个包含最终审查文档的 Git checkpoint，以便完整记录“审查通过后开跑”的状态；但已有实验前 checkpoint `0874c5d`，因此这不是阻止开跑的条件。

---

# 最终结论

ABL-001 当前实验准备满足进入训练阶段的要求。  
在 Codex 自审和本 Claude 三轮审查结果均为 `ACCEPTED` 后，可以按计划运行训练命令。
