# DVSR / VGSR 项目快照 (2026-06-01 更新)

> **新 agent 必读**: 这个文件是项目当前完整状态。读完这一份 + 三件套 (TODO_ROADMAP / EXPERIMENT_LEDGER / INNOVATION_RECORD), 就能从这继续。

---

## 一、项目背景 30 秒速览

**DVSR (VGSR)** 是 Generalized Zero-Shot Learning (GZSL) 研究项目, 用 CLIP + Adapter + GPT 描述做细粒度图像分类。

- **Backbone**: CLIP ViT-L/14@336px (frozen, 不训练)
- **任务**: 给图分类到 200 类 CUB 鸟种, 其中 150 seen / 50 unseen
- **核心指标**: H (Harmonic mean of seen acc & unseen acc)
- **数据集**: CUB-200-2011 (主), AWA2/SUN (跨数据集验证待做)
- **环境**: Windows + `F:\Anaconda\envs\dassl_clip_v2\python.exe`, GPU `cuda:0`

---

## 二、当前最佳结果

### ⭐ 新 Main Baseline (锁定, 论文 main result, 2026-06-01)

**P3.10 配置 3-seed 60ep [20+30+10]**: H = **72.65 ± 0.19** (U=73.07±0.85, S=72.26±0.85, ZS=81.54±0.16)

- seed=5 (P3.10): H=72.91 ⭐ 单点最高 (U=73.30, S=72.53, ZS=81.72, best@ep23)
- seed=42 (P3.12): H=72.53 (U=71.93, S=73.14, ZS=81.56, best@ep21)
- seed=2024 (P3.13): H=72.50 (U=73.97, S=71.10, ZS=81.34, best@ep34)

**关键配置**: K=32 + LaSt v5 平稳前景 + AG-JEPA train ON + PriorCorrection OFF + 60ep [20+30+10]

### 旧 Main Baseline (50ep, 已替换)

**v5+FAE 3-seed 50ep**: H = 72.52 ± 0.11 (U=74.13±0.62, S=70.99±0.40, ZS=81.56±0.18)
- seed=5: H=72.53 / seed=42: H=72.65 / seed=2024: H=72.39

### 单点最高

**Phase 3.10 (seed=5, 60ep)**: H = **72.91** (U=73.30, S=72.53, ZS=81.72, best@ep23)

### 纯 CLIP zero-shot (Table 1 第一行)

H = 61.28 (U=60.88, S=61.69, ZS=78.07)

→ **主方法 vs 纯 CLIP**: H +11.37, U +12.19, S +10.57, ZS +3.47

---

## 三、当前模型框架 (15 个组件)

代码位置: `model/MyModel.py`

| # | 组件 | yaml 字段 | 当前值 | 说明 |
|---|---|---|---|---|
| 1 | CLIP backbone | (固定) | ViT-L/14@336px frozen | 768 维, 24×24=576 patches |
| 2 | Adapter (text) | `adapter_ratio` | 0.2 | bottleneck 768→192→768 |
| 3 | text_source | `text_source` | gpt55 | GPT-5.5 风格描述 7 句/类 |
| 4 | CrossModalTransformer | `tf_common_dim/heads` | 512 / 4 | 双向 v2s+s2v |
| 5 | FAE | `use_fae` | True | 几何感知自注意力 |
| 6 | weight_s2v | `weight_s2v` | 0.5 | 两路融合系数 |
| 7 | local_weight (β) | `local_weight` | 0.3 | base + β·local |
| 8 | pool_method | `pool_method` | mean | s2v 出口池化 |
| 9 | score_mode | `score_mode` | add | base + β·local |
| 10 | **LaSt v5** patch 选择器 | `lastvit_select_k` | 64/32 (扫描中) | ⭐ 主创新 |
| 11 | LaSt sigma | `lastvit_select_sigma` | 0.0 (auto √D) | 频域核宽度 |
| 12 | **G1 Topo Pearson** | `lambda_topo_pearson` | 0.05 | 文本拓扑约束 |
| 13 | **G2 MSDN** | `lambda_msdn` | 0.05 | 双分支互蒸馏 |
| 14 | **G3 Conditional Text** | `use_conditional_text + conditional_text_ratio` | True / 0.005 | CoCoOp-style |
| 15 | **G3 Cons KL** | `lambda_consist + consist_dynamic` | 0.05 + 动态衰减 | 一致性约束 |

**已关闭模块** (失败或不适用):
- D-α 动态门控 (`gating_dynamic=fixed`)
- D-w 双分支动态 (`weight_s2v_mode=fixed`)
- DMP 动态池化 (`pool_dynamic=fixed`)
- Calibration loss (`lambda_cal=0.0`)
- L2-SP (`lambda_l2sp=0.0`)
- cosine_only (`score_mode=add` 不切)
- LaSt-CLS 增强 (`use_lastvit_cls=False`)
- L_bias hinge (`lambda_bias=0.0`, 待修)
- Adapter L2-SP / Distill / v_anchor / t_unseen_anchor / aux_s2v/v2s 全 0

---

## 四、训练配置

```yaml
# 当前 60ep 三段 schedule (50ep main 等价 + 10ep 1e-5 微调)
lr_stages:
  - {lr: 0.001,   epochs: 20}   # Stage 1
  - {lr: 0.0001,  epochs: 30}   # Stage 2 (前 50ep 等价 main baseline)
  - {lr: 0.00001, epochs: 10}   # Stage 3 (1e-5 微调)
# 总 60ep, 段切换严格连续 (默认 restart_from_best=False)

batch_size: 64
random_seed: 5  (3-seed: 5/42/2024)
optimizer: Adam, weight_decay=1e-4
use_amp: False
```

**重要规则**:
- main baseline = strict 50ep `[1e-3×20, 1e-4×30]`
- 当前实验 = 60ep `[1e-3×20, 1e-4×30, 1e-5×10]` (前 50ep 等价 main)
- **段切换默认严格连续** (代码 `train_VGSR_CUB.py` 行 ~714, `restart_from_best=False`)
- Phase 8 最后用 `restart_from_best=True` 收尾压榨

---

## 五、Loss 函数

```
L = CE(seen)                                           ← 主损失
  + 0.05 · L_topo_pearson                              ← G1 文本拓扑
  + 0.05 · L_msdn (KL_pair, T=2)                       ← G2 双分支互蒸馏
  + 0.05/(1+0.1·cons) · L_consist (KL base↔local)      ← G3 一致性 + 动态防爆
```

---

## 六、关键已确认结论 (新 agent 必看, 不要再撞坑)

### ✅ 健康的方向
- **v5+FAE patch 选择器**: K=32 H=72.70, K=64 H=72.54, **K 越小越好** (前景去噪)
- **G1+G2+G3 三创新**: 各 +0.1~0.4 H
- **60ep [20+30+10] schedule**: 跟 50ep main 持平, 1e-5 微调无显著涨

### ❌ 已确认失败 (不要再做)
| 项 | H | 原因 |
|---|---|---|
| Cosine-only v1-v8 | 43-50 | 流形错位 |
| LaSt v1 (`pool_method=lastvit`) | 71.18 | 特征值进梯度链路, frozen CLIP 切断 |
| LaSt v2 (`use_lastvit_cls`) | 70.87 | 同 v1 |
| LaSt v3 (`+lastvit_proj+LN`) | 21.11 | LayerNorm 量级冲突 |
| **D-α (gating_dynamic=mlp)** | **70.28** | frozen CLIP+seen-only CE → α 退化常数 sigmoid≈0.33 |
| **D-α + Cons** | **66.22** final | Cons 跟 D-α 反向打架, Cons 8→11.57 无界增长 |
| **90ep schedule** | **66.49** final | Stage 1 lr=1e-3×30 过长, 软正则推飞模型 |
| 50ep 三合一 | 34→18 | Loss 量级失控 |
| F5 P1 per-class pool | 68 | 拖累 seen |
| CGC C5 hinge | 40 (ep1) | seen-only batch 跟 CE 反向打架 |

### ⚠️ 重要勘误 (2026-05-31, 不要讲反)
**LaSt patch 选择器实际是"前景同质性筛选"**, 不是 part-aware:
- `topk(largest=True)` 选**通道维度上最平稳的 token** (滤掉 high-norm artifact / 背景 lazy token)
- 公式 `diff = x / (|x_lp - x| + ε)`: 比值大 → 高频残差小 → patch 平稳
- CUB 鸟眼/喙/翅纹 (通道波动大) 反而被丢掉
- 实测验证: K=32 (72.70) > K=64 (72.54) → K 越小越严格筛选越好

---

## 七、当前正在跑 / 下一步

### 当前实验状态
- **P3.2 K=16 已完成**: H=72.23 (vs K=32 72.70, 跌 0.47), **K=32 是最优**
- 不再跑 K=8 (信号已明确, 单峰)

### 立刻可执行 (按优先级)

| 优先级 | 任务 | 改动 | 时间 |
|---|---|---|---|
| ⭐⭐⭐ | **P3.6 反向对照** (K=32, largest=False) | 改 `lastvit_select_patches` 加 `largest` 参数, yaml 加 `lastvit_select_largest: False` | 改代码 5min + 训练 35min |
| ⭐⭐ | P3.4 K=128 ablation | yaml `lastvit_select_k: 128` | 35min |
| ⭐⭐ | P3.5 K=256 ablation | yaml `lastvit_select_k: 256` | 40min |
| ⭐ | A1-A6 单变量 ablation (Phase 2.4-2.9) | 见 ROADMAP | 各 35min |
| ⭐ | AWA2 跨数据集 (Phase 2.2) | `python train_VGSR_AWA2.py` | ~1h |
| ⭐ | SUN 跨数据集 (Phase 2.3) | `python train_VGSR_SUN.py` | ~1h |
| 终点 | Phase 8 warm-restart (锁定最优后) | yaml 加 `restart_from_best: True` | 多 seed 2-3h |

### P3.6 改代码模板 (留给新 agent)

`model/MyModel.py` 行 ~104 函数 `lastvit_select_patches`:

```python
def lastvit_select_patches(F_p, K=64, sigma=None, largest=True):  # ← 加 largest 参数
    ...
    _, topk_indices = torch.topk(patch_score, k=K, dim=1, largest=largest)  # ← 用参数
    return topk_indices, patch_score
```

`VGSR.__init__` 行 ~810 加:
```python
self.lastvit_select_largest = bool(getattr(config, 'lastvit_select_largest', True))
```

`VGSR.forward` 透传 `largest=self.lastvit_select_largest` 到 cross_tf。

`CrossModalTransformer.forward` 入口加参数 `lastvit_select_largest=True`, 传给 `lastvit_select_patches`。

yaml 加:
```yaml
lastvit_select_largest:
  value: False  # P3.6 反向对照
```

---

## 八、文件位置 (重要)

### 核心代码
- `model/MyModel.py` — VGSR 主模型 + LaSt 函数
- `train_VGSR_CUB.py` — CUB 训练入口
- `tools/dataset.py` / `tools/helper_func.py` — 数据 + 评估
- `config/VGSR_cub_gzsl.yaml` — 当前实验配置

### 文档 (新 agent 必读三件套)
- `Guide/TODO_ROADMAP.md` — 待办路线图 (8 Phase)
- `report/EXPERIMENT_LEDGER.html` — 实验账本主表 (#1-#19+)
- `创新指导清单/INNOVATION_RECORD.md` — 创新方案 + 累计收益

### 历史归档
- `.kiro/docs/conversation_log.md` — 完整对话历史摘要 (hook 自动追加)
- `.kiro/docs/history_summary.md` — 早期摘要
- `Guide/COSINE_ONLY_REPORT.md` — cosine-only 系列失败记录
- `Guide/CHANGELOG_VDT_TransZero.md` — 早期架构演化

### 训练日志
- `train_log/CUB/training_log_CUB_*.txt` — 每次训练日志
- `train_log/CUB/best_model_CUB_*_H*.pth` — 训练 best ckpt
- `train_log/CUB/ckpt_full_CUB_*.pth` — 完整 ckpt (含 optimizer/scheduler)

### 缓存数据
- `data/cache/CUB_*_features.pt` — 预提取 CLIP 特征 (训练直接加载)
- `data/cache/CUB_class_text_embeds.pt` — 文本嵌入缓存
- `data/gpt4_data/cub_gpt55.pt` — GPT-5.5 描述

### Hooks
- `.kiro/hooks/save-conversation-summary.kiro.hook` — 每次回复后自动写日志
- `.kiro/hooks/sync-experiment-records.kiro.hook` — 用户触发后自动同步三件套

---

## 九、Git 备份

- 当前分支: `experiment/v3-baseline-72.48` (commit `9639ed7`)
- 远程: https://github.com/GetlotMoney/MTPY/tree/experiment/v3-baseline-72.48
- main 分支干净未动

---

## 十、新 Agent 接手 5 步

1. **读这份 SNAPSHOT** + `TODO_ROADMAP.md` (Phase 1-8) + `EXPERIMENT_LEDGER.html` 主表 (#1-#19+)
2. **读 `INNOVATION_RECORD.md`** 看 LaSt v5+FAE 完整方案 (尤其勘误段)
3. **读 `model/MyModel.py`** 行 100-170 (LaSt 函数) + 行 540-600 (cross_tf forward) + 行 1200-1240 (VGSR forward 透传)
4. **检查 yaml 当前状态**:
   - `lastvit_select_k` 当前值 (跑完哪一组)
   - `lr_stages` 60ep 三段
   - `gating_dynamic: fixed`, `pool_method: mean`
5. **跟用户确认下一步任务** (按 ROADMAP 优先级表)

---

## 十一、给新 Agent 的硬规则 (用户原话总结)

1. **不轻易宣告"失败"** — 单组实验 H 微跌 ≤ 2% 算"待补消融", 真失败只有"机制错"或"训练崩盘"
2. **数学/PyTorch 行为要程序验证**, 不要凭直觉
3. **多 seed 均值±方差**才是论文 main result
4. **schedule 一致性** — ablation 用同一 schedule 公平对比
5. **GZSL 领域共识** — 用 test 选 ckpt 是合规的 (没 val split)
6. **不能轻易追求 73**: H=73.05 / 73.20 是 warm-restart 离群值, main result 锁死 72.65±0.19 (60ep 新 main) / 72.52±0.11 (50ep 旧 main)
7. **修改计划必须同步 TODO_ROADMAP / EXPERIMENT_LEDGER / INNOVATION_RECORD 三件套**
8. **画图不要用 mermaid** (Kiro 聊天面板不渲染), 直接列表/表格
9. **每次回复后 hook 自动写 conversation_log.md 摘要**, 不需要 agent 手动写

---

**Snapshot 时间**: 2026-05-31 17:15
**最后实验**: P3.2 K=16 H=72.23 (vs K=32 72.70 跌 0.47, K=32 确认最优)
**下一步推荐**: P3.6 反向对照 (K=32 + largest=False) 验证 "前景去噪假设"
