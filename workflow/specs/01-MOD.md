# MOD-xxx 创新模块实验规范

## 定位

**唯一需要创意树的实验类型。** 从外部创意树（创新指导清单/paper-idea-tree/）读取候选，生成新代码，验证单个创新机制的独立贡献。

---

## 候选池

| 来源 | 说明 |
|------|------|
| 创意树节点 | `idea_tree.json` 中状态为 `candidate` 或 `validated` 的节点 |
| 用户指定 | 用户直接说"做 MOD-xxx" |

**选择规则**（按优先级）：
1. 选择分 = 模块权重 × 类别权重 / 5
2. 先按类别权重挑方向，再按模块权重挑节点
3. 一次只改一个主要机制
4. 达到 H >= 76 之前优先换创新模块，不先做大规模调参

---

## 启动条件

- [ ] 有明确创意树节点或用户指定原因
- [ ] 当前 baseline 已确定
- [ ] Git 工作区干净
- [ ] 实验目录 `experiments/01_single_module_innovation/<EXP-ID>/` 未存在

---

## 专用流程

```
STEP 0: 读取创意树
  └── 读 idea_tree.json + 创意树.md
  └── 选择最高优先级候选节点
  └── 确定实验 ID: MOD-xxx_<slug>

STEP 1: Git 分支
  └── 从 experiment/single-module-innovation 派生 exp/MOD-xxx_<slug>
  └── 创建 checkpoint

STEP 2: 准备实验文件
  └── 新建目录: experiments/01_single_module_innovation/MOD-xxx_<slug>/
  └── 复制 config/VGSR_cub_gzsl.yaml → config.yaml
  └── 写 review-packet.md:
      ├── 实验假设（来自创意树节点）
      ├── 来源复核（论文/代码/本地实验）
      ├── 模块关系分析（互补/重叠/替代/冲突）
      ├── 最小 diff 设计
      ├── 新模块配置键（默认关闭）
      └── 关闭后退回 baseline 的路径
  └── 写 README.md 模板
  └── 写实验框架图（framework.md）:
      ├── 当前实验模块框架图
      ├── 模块插入点
      ├── 与 baseline 的关系
      └── 关闭后退回 baseline 的路径
  └── 写创意档案（可选，实验后补）

STEP 3: 代码实现
  └── 实现最小代码改动
  └── 新模块默认关闭，配置键设为 False/0.0
  └── 确保关闭开关后完全退回 baseline
  └── 只改一个主要变量
  └── 输出: diff + config.yaml + README.md 初稿

STEP 4: Codex 自审
  └── 检查 diff 是否只服务当前实验
  └── 检查关闭后是否退回 baseline
  └── 检查配置键默认值是否安全
  └── 检查是否无意改变 seed/学习率/epoch
  └── 检查 README 是否绑定创意树节点
  └── 结果: ACCEPTED / REJECTED

STEP 5: 审查（STRICT 范式）
  └── 第 1 轮: 代码/配置正确性
  └── 第 2 轮: 实验设计和可复现性
  └── 第 3 轮: 最终运行计划
  └── 输出: claude-review.md
  └── 任一轮 REJECTED → 修复 → 重新审查（最多 10 轮）

STEP 6: 运行实验
  └── 条件: 所有审查门 ACCEPTED
  └── python train_VGSR_CUB.py --config <experiment-config.yaml>
  └── 监控训练，记录最佳 epoch

STEP 7: 结果分析
  └── 解析 U/S/H/ZS
  └── 结果分级:
      ├── win:         H >= baseline + 0.30
      ├── near_tie:   baseline - 0.15 <= H < baseline + 0.30
      ├── soft_negative: baseline - 0.80 <= H < baseline - 0.15
      ├── hard_negative: H < baseline - 0.80
      └── blocked:     未跑通
  └── 判断是否需要多 seed 复核

STEP 8: 记录与反馈
  └── 更新实验 README
  └── 更新 EXPERIMENT_REGISTRY.md
  └── 写 framework flow 记录
  └── 更新创意树:
      ├── 节点权重更新
      ├── 状态更新（candidate → tested_positive/negative）
      ├── 实验结果记录
      └── 下一步建议
  └── 更新 backlog.md
  └── Git 提交
  └── 用户明确要求时 push
```

---

## 审查范式

**默认 STRICT**：Codex 自审 + 审查子代理连续三轮。

---

## 输出规范

| 文件 | 位置 | 说明 |
|------|------|------|
| README.md | `experiments/01_single_module_innovation/MOD-xxx_<slug>/` | 实验结果、日志路径、创意树反馈 |
| config.yaml | `experiments/01_single_module_innovation/MOD-xxx_<slug>/` | 实验配置 |
| claude-review.md | `experiments/01_single_module_innovation/MOD-xxx_<slug>/` | 审查记录 |
| review-packet.md | `experiments/01_single_module_innovation/MOD-xxx_<slug>/` | 审查包 |
| framework.md | `experiments/01_single_module_innovation/MOD-xxx_<slug>/` | 当前实验模块框架图（与 baseline 对照） |
| environment.md | `experiments/01_single_module_innovation/MOD-xxx_<slug>/` | 运行环境记录（必须记录） |
| 训练日志 | `experiments/01_single_module_innovation/MOD-xxx_<slug>/logs/` | 日志副本 |
| framework flow | `experiments/08_framework_flow_records/MOD-xxx_<slug>.md` | 全局框架图记录 |
| 创意档案 | `experiments/09_idea_records/MOD-xxx_<slug>.md` | 创意来源深度记录 |

> 实验目录下的 `framework.md` 是**当前实验视角**，说明这个实验在 baseline 基础上改动了什么、插入点在哪、关闭后怎么退回。即使 baseline 换过，这个文件里的对照是准确的。

> `environment.md` **必须记录**运行环境，确保可复现。内容见下方模板。

---

## 特殊规则

1. **新模块默认关闭** — 只在实验 config.yaml 打开
2. **关闭后退回 baseline** — 必须验证
3. **一次只改一个主要变量** — 不堆叠
4. **创意树绑定** — README 必须标注创意树节点 ID
5. **seed 策略** — 初筛用 seed=5，候选复核用 1, 5, 10

---

## 执行环境

- **Host**: 本地 Windows 机器 (`PC-20260227YXOR`)
- **工作目录**: `C:\Users\Administrator\Desktop\项目\DVSR`
- **Shell**: PowerShell
- **conda 环境**: `dvsr_gpu`（需提前激活）
- **CUDA**: cuda:0 (CUB/SUN), cuda:1 (AWA2)
- **Python**: `python`（conda 环境中的 Python 3.x）

## 运行命令

```powershell
# 激活环境
conda activate dvsr_gpu

# 进入工作目录
cd C:\Users\Administrator\Desktop\项目\DVSR

# 运行训练
python train_VGSR_CUB.py --config experiments/01_single_module_innovation/MOD-007_class_angle_topology/config.yaml

# 日志位置
# 原始: train_log/CUB/training_log_CUB_2026-06-11_HH-MM-SS.txt
# 副本: experiments/01_single_module_innovation/MOD-007/logs/
```

## environment.md 模板

```markdown
# 运行环境

## 硬件

| 项目 | 值 |
|------|---|
| Host | PC-20260227YXOR |
| GPU | NVIDIA RTX 4090 |
| CUDA | 12.1 |
| CPU | Intel i9-13900K |
| RAM | 64GB |

## 软件

| 项目 | 值 |
|------|---|
| 操作系统 | Windows 11 22H2 |
| Python | 3.10.13 |
| conda 环境 | dvsr_gpu |
| PyTorch | 2.1.2+cu121 |
| clip | OpenAI 1.0.1 |

## 运行命令

```powershell
conda activate dvsr_gpu
cd C:\Users\Administrator\Desktop\项目\DVSR
python train_VGSR_CUB.py --config experiments/01_single_module_innovation/MOD-xxx/config.yaml
```

## 数据完整性

| 数据 | 路径 | 状态 |
|------|------|------|
| xlsa17 | data/xlsa17/ | ✅ 已存在 |
| CUB images | data/CUB/images/ | ✅ 已存在 |
| GPT-4 描述 | data/gpt4_data/cub.pt | ✅ 已存在 |
| CLIP 缓存 | data/cache/ | ✅ 已存在 |
```

## 相关文件

- `docs/experiment_workflow/experiment_taxonomy.md` — 实验分类总览
- `docs/experiment_workflow/claude_review.md` — 审查规范
- `WORKFLOW.md` — 全局索引
- `创新指导清单/paper-idea-tree/创意树.md` — 创意树
