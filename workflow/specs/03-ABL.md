# ABL-xxx 消融实验规范

## 定位

**不进入创意树。** 从现有模块列表读取候选，逐个关闭已有模块，验证模块是否必须保留。

---

## 候选池

| 来源 | 说明 |
|------|------|
| 现有模块列表 | 当前框架中所有可开关的模块（patch selection、geometry encoding、conditional text 等） |
| 用户指定 | 用户直接说"关闭 xxx 模块看看" |
| 创新模块验证后 | 新模块加入后，需要验证是否和已有模块冲突 |

**选择规则**（按优先级）：
1. 核心模块优先（对 H 影响大的模块）
2. 新加入模块优先（验证必要性）
3. 用户明确指定

---

## 启动条件

- [ ] 当前 baseline 已确定
- [ ] 明确要关闭的模块
- [ ] 模块有配置开关（可关闭）
- [ ] Git 工作区干净

---

## 专用流程

```
STEP 0: 读取模块列表
  └── 列出当前框架所有可开关模块
  └── 选择要关闭的模块
  └── 确定实验 ID: ABL-xxx_disable_<module>

STEP 1: Git 分支
  └── 从 experiment/ablation 派生 exp/ABL-xxx_disable_<module>
  └── 创建 checkpoint

STEP 2: 准备实验文件
  └── 新建目录: experiments/05_ablation/ABL-xxx_disable_<module>/
  └── 复制 config/VGSR_cub_gzsl.yaml → config.yaml
  └── 修改 config: 关闭目标模块（设为 False/0.0）
  └── 写 review-packet.md:
      ├── 关闭理由（为什么验证这个模块）
      ├── 关闭方式（配置键设为 False/0.0）
      └── 预期结果（模块必须时 H 应下降）
  └── 写 README.md 模板
  └── 写实验框架图（framework.md）:
      ├── 基于哪个 baseline
      ├── 关闭哪个模块
      ├── 关闭方式
      └── 与 baseline 的关系（只改 config，不改代码）

STEP 3: 配置修改
  └── 不改任何代码，只改 config.yaml
  └── 关闭目标模块
  └── 保持其他所有参数不变

STEP 4: 自审
  └── 检查是否只关闭了一个模块
  └── 检查是否无意中关闭其他模块
  └── 检查是否保持其他参数不变
  └── 结果: ACCEPTED / REJECTED

STEP 5: 审查（TUNE-LITE 范式）
  └── 审查: 关闭方式是否正确、是否可复现
  └── 输出: claude-review.md

STEP 6: 运行实验
  └── 条件: 审查通过
  └── python train_VGSR_CUB.py --config <experiment-config.yaml>
  └── 记录 U/S/H/ZS

STEP 7: 结果分析
  └── 与 baseline 对比: delta H = baseline_H - ABL_H
  └── 判断模块必要性:
      ├── 关键模块: delta H >= 1.0（模块关闭后 H 大幅下降）
      ├── 有效模块: 0.3 <= delta H < 1.0
      ├── 弱模块: 0.0 <= delta H < 0.3
      └── 冗余/有害: delta H < 0（关闭后 H 反而上升）

STEP 8: 记录与反馈
  └── 更新实验 README
  └── 更新 EXPERIMENT_REGISTRY.md
  └── 更新模块必要性评级
  └── 更新 backlog.md
  └── Git 提交
  └── 用户明确要求时 push
```

---

## 审查范式

**默认 TUNE-LITE**：Codex 自审 1 次。

---

## 输出规范

| 文件 | 位置 | 说明 |
|------|------|------|
| README.md | `experiments/05_ablation/ABL-xxx/` | 关闭模块、结果对比、必要性评级 |
| config.yaml | `experiments/05_ablation/ABL-xxx/` | 关闭模块的配置 |
| claude-review.md | `experiments/05_ablation/ABL-xxx/` | 审查记录 |
| framework.md | `experiments/05_ablation/ABL-xxx/` | 基于哪个 baseline、关闭哪个模块 |
| environment.md | `experiments/05_ablation/ABL-xxx/` | 运行环境记录（必须记录） |
| 训练日志 | `experiments/05_ablation/ABL-xxx/logs/` | 日志副本 |

> 实验目录下的 `framework.md` 记录**基于哪个 baseline** 和**关闭哪个模块**。即使 baseline 换过，这个文件明确说明起点，避免对照混乱。

> `environment.md` **必须记录**运行环境，确保可复现。内容参考 MOD 规范模板。

---

## 特殊规则

1. **只关闭一个模块** — 不堆叠关闭
2. **不改代码** — 只改 config.yaml
3. **保持其他参数不变** — 只动目标模块开关
4. **seed 一致** — 与 baseline 相同 seed
5. **不进入创意树** — 消融结果不反馈到创新树

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
python train_VGSR_CUB.py --config experiments/05_ablation/ABL-005/config.yaml

# 日志位置
# 原始: train_log/CUB/
# 副本: experiments/05_ablation/ABL-005/logs/
```

## 相关文件

- `docs/experiment_workflow/experiment_taxonomy.md` — 实验分类总览
- `WORKFLOW.md` — 全局索引
