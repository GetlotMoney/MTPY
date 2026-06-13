# Agent 实验管理规范

> 适用于 OpenClaw Agent 自动执行实验的分支与记录管理

---

## 核心原则

**代码管理（分支）** + **实验记录（main 统一管理）** = 完整可回溯的实验历史

---

## 1. 代码管理（分支）

### 规则
- **每个实验从 `main` 切出独立分支**
- **分支名 = 实验名**，例如 `exp/TUNE-051_topo008_cond008`
- **分支包含代码修改**：config.yaml、模型代码等
- **实验分支就是代码快照**，可随时回滚到任意版本

### 流程
```
main (干净基线)
  ↓ 切出
exp/TUNE-051_topo008_cond008 (实验代码快照)
  ↓ 修改 config.yaml
  ↓ 运行实验
  ↓ 提交并推送分支
```

---

## 2. 实验记录（main 统一管理）

### 规则
- **`main` 的 `experiments/` 目录必须包含所有实验结果**
- **实验记录包含**：README.md、config.yaml、training_log.txt
- **不修改 main 的代码配置**：`config/VGSR_cub_gzsl.yaml` 保持干净
- **main 只作为实验记录的统一入口**，不是实验基线

### 同步时机
- 实验完成后，立即把实验记录合并到 `main`
- 使用 `git checkout <exp-branch> -- experiments/<type>/<exp-id>/`
- 不合并代码修改，只合并实验记录

### 流程
```
exp/TUNE-051 (实验完成)
  ↓  checkout 实验记录
main/experiments/04_hyperparameter_tuning/TUNE-051/ (统一记录)
  ↓ 提交并推送 main
```

---

## 3. 为什么这样设计

| 问题 | 解决方案 |
|------|---------|
| 想回滚到某个实验版本 | 直接 checkout 实验分支 |
| 想看所有实验结果 | 直接看 main 的 experiments/ 目录 |
| 想对比不同实验 | 实验分支对比 + main 记录对比 |
| 避免 main 代码混乱 | main 只记录不修改代码 |

---

## 4. 完整流程示例

### 执行 TUNE-051 实验

```bash
# 1. 从 main 切出实验分支
git checkout main
git checkout -b exp/TUNE-051_topo008_cond008

# 2. 修改实验配置
#    experiments/04_hyperparameter_tuning/TUNE-051/config.yaml
#    (修改 topo, cond, jepa_neg 等参数)

# 3. 运行实验
python train_VGSR_CUB.py --config experiments/04_hyperparameter_tuning/TUNE-051/config.yaml

# 4. 实验完成后，保存结果
#    experiments/04_hyperparameter_tuning/TUNE-051/README.md
#    experiments/04_hyperparameter_tuning/TUNE-051/TUNE-051_training_log.txt

# 5. 提交实验分支
git add experiments/04_hyperparameter_tuning/TUNE-051/
git commit -m "TUNE-051: topo=0.08, cond=0.008, H=73.70, -0.15 vs baseline"
git push origin exp/TUNE-051_topo008_cond008

# 6. 同步实验记录到 main
git checkout main
git checkout exp/TUNE-051_topo008_cond008 -- experiments/04_hyperparameter_tuning/TUNE-051/
git add experiments/04_hyperparameter_tuning/TUNE-051/
git commit -m "docs: add TUNE-051 experiment record"
git push origin main
```

---

## 5. 关键禁忌

❌ **不要在 main 上直接修改实验参数**（config/VGSR_cub_gzsl.yaml）
❌ **不要把实验代码合并到 main**（只合并实验记录）
❌ **不要删除已完成的实验分支**（保留回滚能力）
❌ **不要把 main 当作实验基线**（实验分支才是基线）

---

## 6. 目录结构

```
main/
├── config/VGSR_cub_gzsl.yaml          ← 干净基线配置（不修改）
├── experiments/
│   ├── 01_single_module_innovation/   ← MOD 实验记录
│   ├── 02_module_combination/         ← COMBO 实验记录
│   ├── 03_single_module_review/     ← REV 实验记录
│   ├── 04_hyperparameter_tuning/    ← TUNE 实验记录
│   │   ├── TUNE-001_baseline/
│   │   ├── TUNE-051_topo008_cond008/  ← 实验记录（README + config + log）
│   │   └── ...
│   ├── 05_ablation/                  ← ABL 实验记录
│   ├── 06_cross_dataset/             ← XDS 实验记录
│   └── 07_final_review/              ← FINAL 实验记录
└── ...

exp/TUNE-051_topo008_cond008/          ← 实验代码快照
├── config/VGSR_cub_gzsl.yaml          ← 实验时的修改
└── experiments/04_hyperparameter_tuning/TUNE-051/  ← 实验结果
```

---

## 7. Agent 执行规范

当 Agent 执行实验时：

1. **自动从 main 切出实验分支**
2. **自动修改配置并运行实验**
3. **自动保存实验结果到 experiments/目录**
4. **自动提交实验分支**
5. **自动同步实验记录到 main**
6. **main 保持代码干净**

---

## 更新记录

- 2026-06-13: 制定 Agent 实验管理规范
