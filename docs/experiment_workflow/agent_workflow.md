# Agent 工作流规范

## 核心原则

**GitHub 分支管理 = 项目管理快照**
- `main` 保持干净，只放稳定基线
- 每个实验从 `main` 切出，实验分支 = 代码快照
- 实验分支命名 = 实验命名

## 分支结构

```
main
  ├── exp/TUNE-043_cond008      ← 实验代码快照
  ├── exp/TUNE-051_topo008_cond008
  └── ...
```

**规则：**
1. `main` 不修改实验参数
2. 每个实验从 `main` 切出
3. 实验分支只保留代码快照（config + 必要修改）
4. 实验完成后提交推送分支

## 实验目录结构

```
experiments/
  ├── 01_single_module_innovation/   ← MOD 实验
  ├── 02_module_combination/          ← COMBO 实验
  ├── 03_single_module_review/        ← REV 实验
  ├── 04_hyperparameter_tuning/       ← TUNE 实验
  ├── 05_ablation/                    ← ABL 实验
  ├── 06_cross_dataset/               ← XDS 实验
  └── 07_final_review/                ← FINAL 实验
```

**规则：**
1. 实验结果放在对应类型目录下
2. 每个实验目录包含：README.md, config.yaml, training_log.txt
3. 实验目录命名 = 实验命名

## 实验流程

```
1. 从 main 切出分支 exp/<EXP-ID>
2. 创建 experiments/<type>/<EXP-ID>/ 目录
3. 修改 config.yaml（如有代码修改也包含）
4. 运行实验
5. 保存结果到 experiments/<type>/<EXP-ID>/
6. 提交实验分支（包含代码 + 实验结果）
7. 推送实验分支
8. main 保持干净
```

## 关键规则

- **main 永远干净**：不在 main 上修改实验参数
- **实验分支 = 快照**：包含实验时的代码状态
- **实验目录 = 记录**：包含实验结果、配置、日志
- **不合并到 main**：除非用户明确确认
- **不修改总控分支**：不需要总控分支汇总
