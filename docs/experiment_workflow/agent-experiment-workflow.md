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

## 3. 框架版本管理（关键！）

### 问题
- 改了代码框架后，实验记录混在一起
- 不知道哪些实验是哪个框架版本跑的
- 回滚框架后，找不到对应框架的实验记录

### 解决方案

**方法1：按框架版本划分实验记录**
```
experiments/04_hyperparameter_tuning/
  ├── v1/          ← 框架版本1的实验
  │   ├── TUNE-001_baseline/
  │   ├── TUNE-038_topo008/
  │   └── ...
  ├── v2/          ← 框架版本2的实验
  │   ├── TUNE-052_topo008_cond012/
  │   └── ...
  └── v3/          ← 框架版本3的实验
      └── ...
```

**方法2：每个实验记录标注框架版本**
```
experiments/04_hyperparameter_tuning/TUNE-038_topo008/
  ├── README.md
  ├── config.yaml
  ├── TUNE-038_training_log.txt
  └── FRAMEWORK_VERSION.md  ← 标注框架版本
```

### 推荐方法
**方法1 + 方法2 结合**：
1. 每个框架版本开始时，在 `experiments/` 下创建 `vN/` 目录
2. 该版本的所有实验记录放在 `vN/` 下
3. 每个实验记录 README 中也标注框架版本

### 框架版本标记
```
# README.md 开头必须标注

**框架版本**: v1
**框架基线**: main@6da66323 (2026-06-10)
**框架改动**: 无改动，标准框架

```

---

## 4. 为什么这样设计

| 问题 | 解决方案 |
|------|---------|
| 想回滚到某个实验版本 | 直接 checkout 实验分支 |
| 想看所有实验结果 | 直接看 main 的 experiments/ 目录 |
| 想对比不同实验 | 实验分支对比 + main 记录对比 |
| 避免 main 代码混乱 | main 只记录不修改代码 |
| **框架改了，找不到之前实验** | 按框架版本划分实验记录 |
| **回滚框架后，实验记录对应不上** | 每个实验记录标注框架版本 |

---

## 5. 完整流程示例

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

## 6. 关键禁忌

❌ **不要在 main 上直接修改实验参数**（config/VGSR_cub_gzsl.yaml）
❌ **不要把实验代码合并到 main**（只合并实验记录）
❌ **不要删除已完成的实验分支**（保留回滚能力）
❌ **不要把 main 当作实验基线**（实验分支才是基线）
❌ **不要把不同框架版本的实验记录混在一起**（按版本划分）

---

## 7. 目录结构

### 旧结构（有问题）
```
experiments/04_hyperparameter_tuning/
  ├── TUNE-001_baseline/
  ├── TUNE-051_topo008_cond008/   ← 框架v1
  ├── TUNE-052_topo008_cond012/   ← 框架v2（改了代码！）
  └── ...
  ← 混在一起，不知道哪个是哪个框架
```

### 新结构（推荐）
```
experiments/04_hyperparameter_tuning/
  ├── v1/                           ← 框架版本1
  │   ├── TUNE-001_baseline/
  │   ├── TUNE-051_topo008_cond008/
  │   └── ...
  ├── v2/                           ← 框架版本2（改了代码）
  │   ├── TUNE-052_topo008_cond012/
  │   └── ...
  └── v3/                           ← 框架版本3
      └── ...
```

---

## 8. 框架版本切换流程

```
当前框架: v1
  ↓ 用户修改了框架代码
创建 v2 目录
  ↓ 继续实验
v2 下的实验记录
  ↓ 用户想回滚到 v1
checkout v1 的 main 版本
  ↓ 在 v1 下继续实验
v1 下的实验记录
```

---

## 9. Agent 执行规范

当 Agent 执行实验时：

1. **自动从 main 切出实验分支**
2. **自动修改配置并运行实验**
3. **自动保存实验结果到 experiments/<version>/<exp-id>/**
4. **自动提交实验分支**
5. **自动同步实验记录到 main**
6. **main 保持代码干净**

---

## 10. 迁移说明

### 当前实验记录
- TUNE-001 ~ TUNE-037: 框架版本未明确，需要确认
- TUNE-038 ~ TUNE-051: 框架版本 v1（当前标准框架）

### 迁移建议
1. 确认 TUNE-001 ~ TUNE-037 的框架版本
2. 按版本划分到 `v1/` 或 `v2/` 目录
3. 每个实验记录补充 `FRAMEWORK_VERSION.md`

---

## 更新记录

- 2026-06-13: 制定 Agent 实验管理规范
- 2026-06-13: 增加框架版本管理（解决不同框架版本实验记录混淆问题）
