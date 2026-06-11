# TUNE-xxx 调参实验规范

## 定位

**不进入创意树。** 从调参策略矩阵读取候选，扫描超参数、loss 权重、训练日程，优化已有框架的性能。

---

## 候选池

| 来源 | 说明 |
|------|------|
| 调参策略矩阵 | `PARAMETER_TUNING_MATRIX.md` 或 `04_hyperparameter_tuning/` 目录下已有配置 |
| 用户指定 | 用户直接说"扫 topo 权重" |
| 创新模块验证后 | 某 MOD 验证通过，需要稳定性调参 |

**选择规则**（按优先级）：
1. 当前 baseline 已稳定，需要性能提升
2. 创新模块验证后，需要参数稳定
3. 用户明确指定参数方向
4. 达到 H >= 74 前，调参不是主路径（除非用户指定）

---

## 启动条件

- [ ] 当前 baseline 已确定（至少 1 个 seed 跑通）
- [ ] 明确调参方向和范围
- [ ] 不改代码，只改 config.yaml
- [ ] Git 工作区干净

---

## 专用流程

```
STEP 0: 读取调参矩阵
  └── 读 PARAMETER_TUNING_MATRIX.md 或已有 TUNE 记录
  └── 确定调参方向: 权重 / 学习率 / margin / top-K / epoch / 其他
  └── 确定实验 ID: TUNE-xxx_<param>_<range>

STEP 1: Git 分支
  └── 从 experiment/hyperparameter-tuning 派生 exp/TUNE-xxx_<param>
  └── 创建 checkpoint

STEP 2: 准备实验文件
  └── 新建目录: experiments/04_hyperparameter_tuning/TUNE-xxx_<param>/
  └── 复制 config/VGSR_cub_gzsl.yaml → config.yaml
  └── 写 review-packet.md:
      ├── 调参假设（为什么调这个参数）
      ├── 参数范围（grid 或范围）
      ├── 单变量原则（只改一个参数）
      └── 运行命令
  └── 写 README.md 模板
  └── 写实验框架图（framework.md）:
      ├── 基于哪个 baseline
      ├── 改了什么参数
      ├── 参数范围
      └── 与 baseline 的关系（只改 config，不改代码）

STEP 3: 配置修改
  └── 不改任何代码，只改 config.yaml
  └── 明确参数名、范围、步长
  └── 输出: config.yaml 变动 + README.md

STEP 4: 自审
  └── 检查是否只改一个参数
  └── 检查是否无意中改动了其他参数
  └── 检查 seed 是否一致
  └── 检查运行命令是否正确
  └── 结果: ACCEPTED / REJECTED

STEP 5: 审查（TUNE-LITE 范式）
  └── 审查: 参数范围是否合理、是否可复现
  └── 输出: claude-review.md
  └── 任一轮 REJECTED → 修复 → 重新审查

STEP 6: 运行实验
  └── 条件: 审查通过
  └── 批量运行多个参数组合（串行或并行）
  └── 记录每个组合的 U/S/H/ZS

STEP 7: 结果分析
  └── 找出最优参数组合
  └── 判断最优值是否超过 baseline
  └── 分级:
      ├── win:         最优 H >= baseline + 0.30
      ├── near_tie:   baseline - 0.15 <= 最优 H < baseline + 0.30
      ├── soft_negative: 最优 H < baseline - 0.15
      └── 无提升:     所有参数都 < baseline

STEP 8: 记录与反馈
  └── 更新实验 README
  └── 更新 EXPERIMENT_REGISTRY.md
  └── 更新调参矩阵（记录最优参数）
  └── 更新 backlog.md
  └── Git 提交
  └── 用户明确要求时 push
```

---

## 审查范式

**默认 TUNE-LITE**：Codex 自审 1 次，不强制审查子代理。

若调参涉及改动核心代码（如 loss 函数），升级到 STANDARD。

---

## 输出规范

| 文件 | 位置 | 说明 |
|------|------|------|
| README.md | `experiments/04_hyperparameter_tuning/TUNE-xxx/` | 参数范围、最优结果、参数对比表 |
| config.yaml | `experiments/04_hyperparameter_tuning/TUNE-xxx/` | 最优参数配置 |
| claude-review.md | `experiments/04_hyperparameter_tuning/TUNE-xxx/` | 审查记录（如适用） |
| framework.md | `experiments/04_hyperparameter_tuning/TUNE-xxx/` | 基于哪个 baseline、改了什么参数 |
| environment.md | `experiments/04_hyperparameter_tuning/TUNE-xxx/` | 运行环境记录（必须记录） |
| 参数对比表 | `experiments/04_hyperparameter_tuning/TUNE-xxx/` | 所有参数组合结果 |
| 训练日志 | `experiments/04_hyperparameter_tuning/TUNE-xxx/logs/` | 日志副本 |

> 实验目录下的 `framework.md` 记录**基于哪个 baseline** 和**改了什么参数**。即使 baseline 换过，这个文件明确说明起点，避免对照混乱。

> `environment.md` **必须记录**运行环境，确保可复现。内容见下方模板。

---

## 特殊规则

1. **只改 config，不改代码** — 例外: 需改动核心代码时升级审查范式
2. **单变量原则** — 一次只调一个参数（或其组合）
3. **seed 一致** — 所有参数组合用相同 seed
4. **批量限制** — 最大 20 个参数组合
5. **不进入创意树** — 调参结果不反馈到创新树

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
python train_VGSR_CUB.py --config experiments/04_hyperparameter_tuning/TUNE-025/config.yaml

# 日志位置
# 原始: train_log/CUB/
# 副本: experiments/04_hyperparameter_tuning/TUNE-025/logs/
```

## 相关文件

- `docs/experiment_workflow/experiment_taxonomy.md` — 实验分类总览
- `WORKFLOW.md` — 全局索引
