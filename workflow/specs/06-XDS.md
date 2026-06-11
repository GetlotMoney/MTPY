# XDS-xxx 跨数据集实验规范

## 定位

**不进入创意树。** 验证已验证框架在其他数据集（AWA2 / SUN）上的泛化能力。

---

## 候选池

| 来源 | 说明 |
|------|------|
| 已验证框架 | 当前 baseline 或 win/near_tie 模块组合 |
| 用户指定 | 用户明确说"验证 AWA2" |
| 论文要求 | 需要多数据集结果 |

---

## 启动条件

- [ ] 框架在 CUB 上已验证
- [ ] 目标数据集配置文件存在（`config/VGSR_awa2_gzsl.yaml` 或 `config/VGSR_sun_gzsl.yaml`）
- [ ] 数据集已下载
- [ ] Git 工作区干净

---

## 专用流程

```
STEP 0: 选择框架和数据集
  └── 确定要验证的框架（baseline 或 win 模块组合）
  └── 确定目标数据集（AWA2 / SUN）
  └── 确定实验 ID: XDS-xxx_<dataset>_<framework>

STEP 1: Git 分支
  └── 从 experiment/cross-dataset 派生 exp/XDS-xxx_<dataset>
  └── 创建 checkpoint

STEP 2: 准备实验文件
  └── 新建目录: experiments/06_cross_dataset/XDS-xxx/
  └── 复制对应数据集 config → config.yaml
  └── 应用框架配置（如有 win 模块）
  └── 写 review-packet.md:
      ├── 迁移框架
      ├── 目标数据集
      ├── 预期指标
      └── 已知差异（与 CUB 的不同）
  └── 写 README.md 模板
  └── 写实验框架图（framework.md）:
      ├── 基于哪个 CUB 框架
      ├── 迁移到哪个数据集
      ├── 改了什么参数
      └── 与 CUB baseline 的关系

STEP 3: 配置迁移
  └── 复用 CUB 上的框架配置
  └── 调整数据集特定参数（类数、描述文件路径等）
  └── 不改模型结构

STEP 4: 自审
  └── 检查数据集配置是否正确
  └── 检查模型结构是否未改
  └── 检查数据路径是否存在
  └── 结果: ACCEPTED / REJECTED

STEP 5: 审查（TUNE-LITE 范式）
  └── 审查: 迁移是否正确
  └── 输出: claude-review.md

STEP 6: 运行实验
  └── 条件: 审查通过
  └── python train_VGSR_<DATASET>.py --config <config.yaml>
  └── 记录 U/S/H/ZS

STEP 7: 结果分析
  └── 与 CUB 结果对比
  └── 判断泛化能力:
      ├── 强泛化: 目标数据集 H >= 论文 SOTA
      ├── 中等泛化: 目标数据集 H 接近 CUB 相对提升
      └── 弱泛化: 目标数据集 H 显著低于预期

STEP 8: 记录与反馈
  └── 更新实验 README
  └── 更新 EXPERIMENT_REGISTRY.md
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
| README.md | `experiments/06_cross_dataset/XDS-xxx/` | 迁移框架、数据集、结果、与 CUB 对比 |
| config.yaml | `experiments/06_cross_dataset/XDS-xxx/` | 数据集配置 |
| claude-review.md | `experiments/06_cross_dataset/XDS-xxx/` | 审查记录 |
| framework.md | `experiments/06_cross_dataset/XDS-xxx/` | 基于哪个 CUB 框架、迁移到哪个数据集 |
| 训练日志 | `experiments/06_cross_dataset/XDS-xxx/logs/` | 日志副本 |

> 实验目录下的 `framework.md` 记录**基于哪个 CUB 框架** 和**迁移到哪个数据集**。即使 CUB 框架更新，这个文件明确说明起点，避免对照混乱。

---

## 特殊规则

1. **不改模型结构** — 只换数据集配置
2. **复用 CUB 框架** — 不重新设计
3. **数据集特定参数** — 类数、描述文件路径等必须调整
4. **不进入创意树** — 跨数据集结果不反馈到创新树

---

## 相关文件

- `docs/experiment_workflow/experiment_taxonomy.md` — 实验分类总览
- `WORKFLOW.md` — 全局索引
