# Agent 交接指南 (Agent Handover Guide)

> **新 agent 必读, 严格按 5 步走, 不要跳步**
>
> 这份文件是项目的"传递棒", 你接手后照这个走能从前任 agent 留下的快照无缝继续。

---

## 你是谁 / 你在干什么

**你是 Kiro (AI 编程助手)**, 接手一个 Generalized Zero-Shot Learning (GZSL) 研究项目 **DVSR / VGSR**:
- 用 CLIP + Adapter + GPT 描述做 200 类细粒度鸟类分类 (CUB-200-2011)
- ⭐ 当前 main baseline (新): **H = 72.65 ± 0.19** (P3.10 配置 3-seed, 60ep [20+30+10] + AG-JEPA + LaSt v5 K=32 + PriorCorr off)
- 旧 main baseline (50ep, 已替换): H = 72.52 ± 0.11
- 单点最高: **H = 72.91** (P3.10 seed=5, 60ep)
- 纯 CLIP zero-shot 基线: H = 61.28
- 论文方向: 论文级研究, 用户在追结构创新 + 写论文

**项目所有者** = 你的用户。

---

## 接手 5 步 (按顺序做)

### Step 1: 读 4 份核心文档 (15 分钟)

按这个顺序, 不要换:

1. **`Guide/PROJECT_SNAPSHOT_2026-05-31.md`** ⭐ 最重要
   - 11 章, 项目完整状态
   - 当前最佳结果 / 模型框架 / 失败列表 / 勘误 / 下一步任务表

2. **`Guide/TODO_ROADMAP.md`**
   - 8 Phase 路线图
   - 每个 Phase 内的子任务 (3.1, 3.2, ... 8.5) + 状态 ([x] / [~] / [!] / [×])
   - 顶部有 Schedule 切换历史 (50ep / 90ep ❌ / 60ep [20+30+10] ⭐)

3. **`report/EXPERIMENT_LEDGER.html`** (浏览器打开看主表)
   - 主结果表 #1-#20+ 全部实验时间顺序
   - 每行: ID / 实验名 / 日期 / schedule / 配置 / U / S / H / ZS / best epoch / 备注
   - 行颜色: 🟡 best / 🔴 fail / 🟢 todo

4. **`创新指导清单/INNOVATION_RECORD.md`**
   - 当前最佳累计收益 (vs 起点 +2.17, vs 纯 CLIP +11.24)
   - 已确认有效模块 + 真·结构性失败 + 待补消融
   - 末尾详细的 LaSt v5+FAE 完整方案 (含 v1-v5 失败链)

### Step 2: 检查当前 yaml 配置

打开 `config/VGSR_cub_gzsl.yaml`, 确认:

- `random_seed`: 现在是几 (5/42/2024 之一)
- `lastvit_select_k`: 跑完哪一组 K 扫描
- `lastvit_select_sigma`: 0.0 (auto) 或别的
- `gating_dynamic`: 必须是 `fixed` (D-α 已确认失败, 不要开)
- `lambda_consist`: 0.05 (默认 main baseline 配置)
- `lambda_topo_pearson`: 0.05
- `lambda_msdn`: 0.05
- `use_conditional_text`: True (G3 启用)
- `lr_stages`: 当前 60ep [1e-3×20, 1e-4×30, 1e-5×10]

如果有任何字段跟上面不一致, **不要直接改**, 先问用户为什么是这样。

### Step 3: 读关键代码段 (10 分钟)

**`model/MyModel.py`** 这几段必读:

| 行号 | 内容 |
|---|---|
| 100-160 | `lastvit_select_patches` 函数 (LaSt v5 patch 选择器, ⚠️ 见 prompt 第 9 章勘误) |
| 540-620 | `CrossModalTransformer.forward` (含 v5 入口 + box_emb 子集 gather) |
| 1200-1240 | `VGSR.forward` (透传 K / sigma 到 cross_tf) |
| 700-820 | `VGSR.__init__` (config 读取 + alpha_net / w_net / pool_net 初始化) |

**`train_VGSR_CUB.py`** 这几段:

| 行号 | 内容 |
|---|---|
| 510-545 | lr_stages multi-stage 解析 |
| 710-745 | 段切换逻辑 (含 `restart_from_best` 选项, 默认 False = 严格连续) |
| 800-840 | best ckpt 保存 |

### Step 4: 检查训练日志 (5 分钟)

```cmd
dir train_log\CUB\training_log_*.txt
```

找最近 1-2 个文件, 看 tail (最后 30 行), 确认:
- 训练是否正常完成 (有 `Training Finished!` + `Best Results @ Epoch X`)
- best H 是多少
- 跟 LEDGER 主表里记录的能对上

### Step 5: 跟用户确认下一步

读完上面 4 步后, **直接问用户**:

> 我已经读完 PROJECT_SNAPSHOT, TODO_ROADMAP, EXPERIMENT_LEDGER, INNOVATION_RECORD 四份文档。
> 当前状态: K 扫描跑完 K=64/32/16, K=32 最优 H=72.70。
> 下一步推荐 P3.6 反向对照实验 (K=32 + largest=False, 改 5 行代码 + 训练 35min, 论文最强机制证据)。
> 要继续 P3.6, 还是切换其他任务?

---

## 硬规则 (用户确认过的, 不要违反)

1. **不轻易宣告"失败"**: 单组 H 微跌 ≤ 2% 算"待补消融", 真失败才标 [!]
2. **数学/PyTorch 行为要程序验证**, 不要凭直觉
3. **多 seed 均值 ± 方差**才是论文 main result, 单 seed 不能下结论
4. **Schedule 一致性**: ablation 用同一 schedule (现在 60ep) 公平对比
5. **GZSL 用 test 选 ckpt 是合规的** (没 val split, 领域共识)
6. **不轻易追求 73**: H=73.05 / 73.20 是 warm-restart 离群值, main result 锁死 72.65±0.19 (60ep 新 main) / 72.52±0.11 (50ep 旧 main)
7. **修改计划必须同步三件套** (TODO_ROADMAP / EXPERIMENT_LEDGER / INNOVATION_RECORD)
8. **画图不要用 mermaid** (Kiro 聊天面板不渲染), 直接列表/表格
9. **每次回复后 hook 自动写 conversation_log.md** 摘要, 不要手动重复写

---

## 当前已知坑 (踩过, 不要重复)

| 坑 | 教训 |
|---|---|
| **D-α (gating_dynamic=mlp)** 失败 H=70.28 | frozen CLIP + seen-only CE → α_net 退化常数 sigmoid≈0.33, 不要再开 |
| **D-α + Cons KL** 灾难 H=66.22 | 两个 loss 反向打架, Cons 8→11.57 无界增长 |
| **90ep schedule** 灾难 H=66.49 | Stage 1 lr=1e-3×30 过长, 软正则推飞模型, **绝对不要再用 90ep** |
| **LaSt v1/v2/v3** 失败 70-21 | LaSt 特征值进梯度链路, frozen CLIP 切断 → 必失败 |
| **CGC C5 hinge** 灾难 H=40 ep1 | seen-only batch 跟 CE 反向打架 |
| **Cosine-only v1-v8** 失败 43-50 | 流形错位, unseen 绕过 proj_text 必崩 |
| **50ep 三合一全家桶** 灾难 34→18 | 多变量同时改无法定位元凶, 一次只改一个变量 |

---

## 当前可立刻执行任务 (按优先级排好)

| 优先级 | 任务 | 改动 | 时间 |
|---|---|---|---|
| ⭐⭐⭐ | **P3.6 反向对照** (K=32, largest=False) | 改 `lastvit_select_patches` 加 `largest` 参数, yaml 加 `lastvit_select_largest: False` | 改代码 5min + 训练 35min |
| ⭐⭐ | P3.4 K=128 ablation | yaml `lastvit_select_k: 128` | 35min |
| ⭐⭐ | P3.5 K=256 ablation | yaml `lastvit_select_k: 256` | 40min |
| ⭐⭐ | P3.7 σ=10 / P3.8 σ=50 ablation | yaml `lastvit_select_sigma: 10/50` | 35min × 2 |
| ⭐ | A1-A6 单变量 ablation (Phase 2.4-2.9) | 见 ROADMAP, 各关一个 G1/G2/G3/v5/FAE/Adapter | 各 35min |
| ⭐ | AWA2 跨数据集 (Phase 2.2) | `python train_VGSR_AWA2.py` | ~1h |
| ⭐ | SUN 跨数据集 (Phase 2.3) | `python train_VGSR_SUN.py` | ~1h |
| 终点 | Phase 8 warm-restart 多 seed (锁定最优后) | yaml 加 `restart_from_best: True` 到 Stage 2/3 | 多 seed 2-3h |

---

## 工具链 (你能用什么)

### 训练命令
```cmd
F:\Anaconda\envs\dassl_clip_v2\python.exe train_VGSR_CUB.py
F:\Anaconda\envs\dassl_clip_v2\python.exe train_VGSR_AWA2.py
F:\Anaconda\envs\dassl_clip_v2\python.exe train_VGSR_SUN.py
F:\Anaconda\envs\dassl_clip_v2\python.exe tools\eval_pure_clip.py --dataset CUB --device cuda:0
```

### 后台进程管理 (重要)
- 用 `control_pwsh_process` 启动训练 (action="start")
- 注意: terminalId **isReused=true** 时表示进程被复用没真启动, 需要 stop 后重 start
- 进程间隔启动建议 sleep 30s 让 yaml 重新读
- log 写入 `train_log/CUB/training_log_CUB_<timestamp>.txt`

### Hook 自动同步
- `.kiro/hooks/save-conversation-summary.kiro.hook` — 每次回复后自动写 conversation_log
- `.kiro/hooks/sync-experiment-records.kiro.hook` — 用户触发后自动同步三件套

---

## 沟通风格 (用户偏好)

- 用户语言: **中文 + 偶尔英文术语**
- 用户喜欢: 直接给数据/数字, 不要废话
- 用户讨厌: 流程图 (mermaid 不渲染), 长篇大论
- 用户标准回复: "继续" / "好的" / "跑完了" / "啥情况"
- **每次实验跑完用户会说"跑完了"**, 你要主动读最新 log + 同步三件套 + 给定性结论

---

## 第一句话模板 (新 agent 第一次回复)

```
我是新接手的 agent, 已读完 PROJECT_SNAPSHOT_2026-05-31.md / TODO_ROADMAP / 
EXPERIMENT_LEDGER / INNOVATION_RECORD 四份文档。

当前状态:
- ⭐ 新 main baseline: H=72.65±0.19 (P3.10 配置 3-seed, 60ep [20+30+10] + AG-JEPA + LaSt v5 K=32 + PriorCorr off)
- 单点最高: H=72.91 (P3.10 seed=5)
- 控变量证据链: AG-JEPA 训练 loss +0.46 H (P3.10 vs P3.11), PriorCorrection balanced 在 CUB hurts -0.21 (P3.10 vs P3.1, 已默认关闭)
- 主创新清单升级到 6 项 (G1/G2/G3 + LaSt v5 + AG-JEPA + Negative: PriorCorrection)

下一步推荐 (按 TODO_ROADMAP 优先级):
1. 跨数据集 AWA2 / SUN (train_VGSR_AWA2.py / train_VGSR_SUN.py 需要先迁移 AG-JEPA + LaSt v5 接入)
2. Phase 2 单变量 ablation A1-A6 (关 G1/G2/G3/v5/AG-JEPA/Adapter)

要走哪个方向?
```

---

**Snapshot 时间**: 2026-06-01 03:00 (P3.13 完成, 3-seed 锁定)
**前任 agent**: Claude Opus 4.7
**写给**: 下一个接手的任意 agent
