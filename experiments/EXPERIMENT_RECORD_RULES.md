# 实验记录规范

## 核心原则

1. experiments/ 目录只属于 main 分支
2. 所有实验记录（config、日志、README、图表）全部写回 main，不留在实验分支上
3. 实验分支只放代码改动，不放记录

## 流程

1. 从 main 创建实验分支 exp/<EXP-ID>_<slug>
2. 在实验分支上改代码
3. 跑实验：实验结果写回 main 的 experiments/<分类>/<EXP-ID>_<slug>/
4. 代码改动提交在实验分支上，实验记录提交在 main 上

## 好处

- 在任何分支上都能看到全部实验记录
- 实验记录只增不减，永远不丢失
- 实验分支可以删除，但 main 上的记录永远在

## 分支使用原则

- **有代码改动** → 从 main 开 exp/<EXP-ID>_<slug>，改代码、提交、审查，实验记录写回 main
- **无代码改动**（纯调参 TUNE-LITE、纯配置修改） → 直接在 main 上创建实验目录和 config，跑完后记录也提交到 main
- 实验记录永远只提交到 main，不留在实验分支上

## 各种实验类型与分支的关系

| 实验类型 | 范式 | 是否需要新分支 |
|---|---:|---|
| TUNE-LITE（调参，只改 config） | Codex 自审 | ❌ 不需要，直接 main 上做 |
| TUNE（调参，改代码加参数） | Codex 自审 + Claude | ✅ 需要开 exp/TUNE-xxx |
| MOD（创新模块） | 标准审查 | ✅ 需要开 exp/MOD-xxx |
| COMBO（组合模块） | 标准审查 | ✅ 需要开 exp/COMBO-xxx |
| REV-MOD（单模块复核） | Claude 审查 | ✅ 复核不改代码，直接 main 上做 |
| ABL（消融实验） | 标准审查 | ✅ 需要开 exp/ABL-xxx |
