# REV-MOD-xxx 单模块复核实验规范

## 定位

**不进入创意树。** 从注册表读取已验证的 `win` 或 `near_tie` 模块，做多 seed 稳定性验证。

---

## 候选池

| 来源 | 说明 |
|------|------|
| 注册表标记 | 状态为 `win` 或 `near_tie` 的 MOD |
| 用户指定 | 用户明确说"复核 MOD-xxx" |

---

## 启动条件

- [ ] 原始 MOD 已完成
- [ ] 原始 MOD 状态为 `win` 或 `near_tie`
- [ ] 已明确要补跑的 seed 列表
- [ ] Git 工作区干净

---

## 专用流程

```
STEP 0: 读取注册表
  └── 从 EXPERIMENT_REGISTRY.md 读取 win/near_tie 模块
  └── 选择要复核的模块
  └── 确定 seed 列表（已跑过的 seed 可复用）
  └── 确定实验 ID: REV-MOD-xxx_<original_MOD>_<seed_list>

STEP 1: Git 分支
  └── 从 experiment/single-module-review 派生 exp/REV-MOD-xxx_<original>
  └── 创建 checkpoint

STEP 2: 准备实验文件
  └── 新建目录: experiments/03_single_module_review/REV-MOD-xxx/
  └── 继承原始 MOD 的代码和 config
  └── 写 review-packet.md:
      ├── 原始 MOD ID
      ├── 继承的代码/config
      ├── seed 列表
      ├── 需要补跑的 seed
      └── 判断口径: 仍看候选 seed 最大 H
  └── 写 README.md 模板
  └── 写实验框架图（framework.md）:
      ├── 继承的模块框架图
      ├── 与当前 baseline 的关系
      └── 模块插入点

STEP 3: 配置继承
  └── 复用原始 MOD 的代码改动
  └── 复用原始 MOD 的实验配置
  └── 只改 seed

STEP 4: 自审
  └── 检查是否继承原始 MOD 完整配置
  └── 检查 seed 列表是否已记录
  └── 检查是否只改 seed
  └── 结果: ACCEPTED / REJECTED

STEP 5: 审查（STANDARD 范式）
  └── 审查: 继承是否正确、复现性
  └── 输出: claude-review.md

STEP 6: 运行实验
  └── 条件: 审查通过
  └── 串行跑所有 seed
  └── 记录每个 seed 的 U/S/H/ZS

STEP 7: 结果分析
  └── 判断多 seed 稳定性:
      ├── 稳定: 所有 seed H 都 >= baseline
      ├── 波动: 部分 seed 低于 baseline
      └── 不稳定: 多数 seed 低于 baseline
  └── 更新候选 seed 池

STEP 8: 记录与反馈
  └── 更新实验 README
  └── 更新 EXPERIMENT_REGISTRY.md
  └── 更新 backlog.md
  └── Git 提交
  └── 用户明确要求时 push
```

---

## 审查范式

**默认 STANDARD**：Codex 自审 + 审查子代理单轮。

---

## 输出规范

| 文件 | 位置 | 说明 |
|------|------|------|
| README.md | `experiments/03_single_module_review/REV-MOD-xxx/` | 原始 MOD、seed 列表、各 seed 结果、稳定性判断 |
| config.yaml | `experiments/03_single_module_review/REV-MOD-xxx/` | 继承配置 |
| claude-review.md | `experiments/03_single_module_review/REV-MOD-xxx/` | 审查记录 |
| framework.md | `experiments/03_single_module_review/REV-MOD-xxx/` | 继承模块框架图（与当前 baseline 对照） |
| 训练日志 | `experiments/03_single_module_review/REV-MOD-xxx/logs/` | 日志副本 |

> 实验目录下的 `framework.md` 是**当前实验视角**，说明继承的模块在 baseline 中的位置。即使 baseline 换过，这个文件里的对照是准确的。

---

## 特殊规则

1. **继承原始 MOD** — 不复现代码，直接复用
2. **只改 seed** — 其他参数不动
3. **串行跑** — 一个 seed 跑完再跑下一个
4. **候选 seed 池取最大 H** — 不取平均
5. **不进入创意树** — 复核结果不反馈到创新树

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
python train_VGSR_CUB.py --config experiments/03_single_module_review/REV-MOD-001/config.yaml

# 日志位置
# 原始: train_log/CUB/
# 副本: experiments/03_single_module_review/REV-MOD-001/logs/
```

## 相关文件

- `docs/experiment_workflow/experiment_taxonomy.md` — 实验分类总览
- `WORKFLOW.md` — 全局索引
