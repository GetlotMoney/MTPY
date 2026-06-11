# FINAL-xxx 最终复核实验规范

## 定位

**不进入创意树。** 为正式汇报、论文表格或最终主结果做最终复核，确保所有指标可复现。

---

## 候选池

| 来源 | 说明 |
|------|------|
| 用户指定 | 用户明确说"最终复核"或"论文表格" |
| 注册表标记 | 当前最佳框架，需要多 seed 最终确认 |
| 里程碑要求 | 达到某个目标后需要最终报告 |

---

## 启动条件

- [ ] 框架已确定（最佳 baseline 或 win 模块组合）
- [ ] 所有组成实验已完成
- [ ] 用户明确要求最终复核

---

## 专用流程

```
STEP 0: 确定最终框架
  └── 从注册表读取当前最佳框架
  └── 确定实验 ID: FINAL-xxx_<description>

STEP 1: Git 分支
  └── 从 experiment/final-review 派生 exp/FINAL-xxx_<description>
  └── 创建 checkpoint

STEP 2: 准备实验文件
  └── 新建目录: experiments/07_final_review/FINAL-xxx/
  └── 确定最终配置
  └── 确定最终 seed 列表（通常 1, 3, 5, 7, 10）
  └── 写 review-packet.md:
      ├── 最终框架组成
      ├── 最终配置
      ├── seed 列表
      └── 论文表格要求
  └── 写 README.md 模板
  └── 写实验框架图（framework.md）:
      ├── 基于哪个最终框架
      ├── 最终配置说明
      ├── seed 列表
      └── 与 baseline 的关系（最终确认）

STEP 3: 配置确定
  └── 复用最佳框架配置
  └── 不做任何新改动
  └── 确保所有模块开关正确

STEP 4: 自审
  └── 检查配置是否和最佳框架一致
  └── 检查是否无意改动
  └── 检查 seed 列表是否完整
  └── 结果: ACCEPTED / REJECTED

STEP 5: 审查（STRICT 范式）
  └── 审查: 配置一致性、可复现性
  └── 输出: claude-review.md

STEP 6: 运行实验
  └── 条件: 审查通过
  └── 串行跑所有 seed
  └── 记录每个 seed 的 U/S/H/ZS

STEP 7: 结果分析
  └── 整理所有 seed 结果
  └── 按论文要求报告:
      ├── 最高 H（候选池取最大）
      ├── 各 seed 详细结果
      └── 与 SOTA 对比

STEP 8: 记录与反馈
  └── 更新实验 README
  └── 更新 EXPERIMENT_REGISTRY.md
  └── 生成论文表格
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
| README.md | `experiments/07_final_review/FINAL-xxx/` | 最终框架、所有 seed 结果、论文表格 |
| config.yaml | `experiments/07_final_review/FINAL-xxx/` | 最终配置 |
| claude-review.md | `experiments/07_final_review/FINAL-xxx/` | 审查记录 |
| framework.md | `experiments/07_final_review/FINAL-xxx/` | 基于哪个最终框架、最终配置说明 |
| 训练日志 | `experiments/07_final_review/FINAL-xxx/logs/` | 日志副本 |
| 论文表格 | `experiments/07_final_review/FINAL-xxx/` | 格式化表格 |

> 实验目录下的 `framework.md` 记录**基于哪个最终框架** 和**最终配置说明**。即使框架更新，这个文件明确说明最终确认的基准，避免对照混乱。

---

## 特殊规则

1. **不做任何新改动** — 只复用已验证框架
2. **多 seed 完整记录** — 所有 seed 结果必须保留
3. **配置一致性** — 必须和最佳框架完全一致
4. **不进入创意树** — 最终复核不反馈到创新树

---

## 相关文件

- `docs/experiment_workflow/experiment_taxonomy.md` — 实验分类总览
- `WORKFLOW.md` — 全局索引
