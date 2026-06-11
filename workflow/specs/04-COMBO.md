# COMBO-xxx 组合模块实验规范

## 定位

**不进入创意树。** 从单模块实验结果读取候选，验证两个或多个模块是否有协同贡献。

---

## 候选池

| 来源 | 说明 |
|------|------|
| 单模块结果 | 两个模块都是 `win` 或 `near_tie` |
| 理论互补 | 一个模块提升 U，另一个提升 S |
| 创意树节点关联 | 两个节点来自同一论文、同一机制链 |
| 用户指定 | 用户明确指定组合尝试 |

**禁止组合**：
- 两个模块都是 `hard_negative`
- 两个模块改同一个位置、同一分支或同一 loss 目标
- 没有明确互补假设
- 没有写组成模块和冲突分析

---

## 启动条件

- [ ] 组成模块已有单模块实验结果
- [ ] 有明确互补假设
- [ ] 已做冲突分析
- [ ] Git 工作区干净

---

## 专用流程

```
STEP 0: 读取单模块结果
  └── 从注册表读取组成模块的单模块结果
  └── 确定组合假设: 为什么互补
  └── 确定实验 ID: COMBO-xxx_<moduleA>_<moduleB>

STEP 1: Git 分支
  └── 从 experiment/module-combination 派生 exp/COMBO-xxx_<moduleA>_<moduleB>
  └── 创建 checkpoint

STEP 2: 准备实验文件
  └── 新建目录: experiments/02_module_combination/COMBO-xxx/
  └── 复制 config/VGSR_cub_gzsl.yaml → config.yaml
  └── 写 review-packet.md:
      ├── 组成模块: 原始实验 ID、模块名、单模块结果
      ├── 协同假设: 为什么互补
      ├── 冲突分析: 是否存在同一位置/loss/梯度冲突
      └── 对照表: baseline、各模块单独结果、组合结果
  └── 写 README.md 模板
  └── 写实验框架图（framework.md）:
      ├── 当前实验模块组合图
      ├── 组成模块插入点
      ├── 与 baseline 的关系
      └── 关闭后退回 baseline 的路径

STEP 3: 配置修改
  └── 继承各组成模块的实验配置
  └── 确保所有模块开关都打开
  └── 检查是否有冲突（同一位置改两次）

STEP 4: 自审
  └── 检查是否组成模块都已验证
  └── 检查冲突分析是否完整
  └── 检查对照表是否列出
  └── 结果: ACCEPTED / REJECTED

STEP 5: 审查（STANDARD 范式）
  └── 审查: 组合逻辑是否正确、是否可复现
  └── 输出: claude-review.md

STEP 6: 运行实验
  └── 条件: 审查通过
  └── python train_VGSR_CUB.py --config <experiment-config.yaml>
  └── 记录 U/S/H/ZS

STEP 7: 结果分析
  └── 计算组合收益:
      combo_gain = COMBO_H - max(baseline_H, moduleA_H, moduleB_H, ...)
  └── 分级:
      ├── 协同: combo_gain > 0.30
      ├── 叠加: 0 < combo_gain <= 0.30
      ├── 冲突: combo_gain < 0
      └── 无协同: combo_gain = 0

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
| README.md | `experiments/02_module_combination/COMBO-xxx/` | 组成模块、协同假设、对照表、组合收益 |
| config.yaml | `experiments/02_module_combination/COMBO-xxx/` | 组合配置 |
| claude-review.md | `experiments/02_module_combination/COMBO-xxx/` | 审查记录 |
| framework.md | `experiments/02_module_combination/COMBO-xxx/` | 当前实验模块组合图（与 baseline 对照） |
| 训练日志 | `experiments/02_module_combination/COMBO-xxx/logs/` | 日志副本 |

> 实验目录下的 `framework.md` 是**当前实验视角**，说明这个组合在 baseline 基础上打开了哪些模块、各模块位置、冲突点在哪。即使 baseline 换过，这个文件里的对照是准确的。

---

## 特殊规则

1. **必须有互补假设** — 不能"堆一下"
2. **必须有对照表** — baseline、各模块单独、组合结果
3. **组合收益计算** — 必须超过 baseline 和各单模块最好结果
4. **不进入创意树** — 组合结果不反馈到创新树

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
python train_VGSR_CUB.py --config experiments/02_module_combination/COMBO-001/config.yaml

# 日志位置
# 原始: train_log/CUB/
# 副本: experiments/02_module_combination/COMBO-001/logs/
```

## 相关文件

- `docs/experiment_workflow/experiment_taxonomy.md` — 实验分类总览
- `WORKFLOW.md` — 全局索引
