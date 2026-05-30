# Cosine-Only 框架与代码审查报告（汇报版）

**日期**: 2026-05-25  
**模型**: VGSR (CLIP ViT-L/14@336px frozen + Adapter + 双向 Transformer)  
**任务**: CUB-200-2011 GZSL  
**当前 baseline**: H = 72.24（add 模式，mean 池化，固定权重）

---

## 0. TL;DR — 一句话

我们正在把模型从 **`logits = base_logits + β·local_score`（add 模式）** 重构为 **`logits = cos(v_enh, t_enh) × scale`（cosine_only 模式）**，目标是统一两端表示的训练框架并提升可解释性。  
**关键风险已识别：cosine_only 把 CLIP `base_logits` 从主路径踢成了旁路 anchor，必须通过额外 loss 把它接回来；**否则会重演 v1-v4 失败（H 跌到 50 以下）。

---

## 1. 数据流：当前 cosine_only 框架（含残差与锚定）

```
                  CLIP ViT-L/14@336px (frozen)
                            │
                  ┌─────────┴──────────┐
              cls_token            patches
              [B, 768]             [B, 576, 768]
                  │                       │
                  │                ┌──────┴──────────┐
                  │                │  Adapter        │  (seen 150 → seen 150)
                  │                │  bottleneck MLP │  ~295K 可学参数
                  │                │  ratio=0.2      │  delta_residual
                  │                └──────┬──────────┘
                  │                       │ adapted_seen [150, 768]
                  │                       │
                  │                  拼成 200 类 all_text
                  │                  unseen 50 永远用 CLIP 原始
                  │                       │ all_text [200, 768]
                  │                       │
                  │                ┌──────┴───────────────────┐
                  │                │ CrossModalTransformer    │
                  │                │ (TransZero++ 风格)        │
                  │                │ ┌────────────────────┐   │
                  │                │ │ embed_cv (768→512) │   │
                  │                │ │ FAE 几何解耦自注意 │   │
                  │                │ │   ↓ memory         │   │
                  │                │ │ ┌────────────────┐ │   │
                  │                │ │ │ v2s decoder    │ │   │
                  │                │ │ │ Q=text KV=memory│ │   │
                  │                │ │ │ → F_p_v2s      │ │   │
                  │                │ │ │ → proj_text    │ │   │
                  │                │ │ │   t_enh_raw    │ │   │   [B, 200, 768]
                  │                │ │ └────────────────┘ │   │
                  │                │ │ ┌────────────────┐ │   │
                  │                │ │ │ s2v decoder    │ │   │
                  │                │ │ │ Q=memory KV=text│ │   │
                  │                │ │ │ → F_p_s2v      │ │   │
                  │                │ │ │ → mean pool    │ │   │
                  │                │ │ │ → proj_visual  │ │   │
                  │                │ │ │   v_enh_raw    │ │   │   [B, 768]
                  │                │ │ └────────────────┘ │   │
                  │                │ │ score_v2s, score_s2v│  │   [B, 200]
                  │                │ └────────────────────┘   │
                  │                └──────┬───────────────────┘
                  │                       │
                  │       ┌─── residual fusion ────┐
                  │       │                        │
                  │  v_enh = (1-vr)·v_enh_raw      t_enh = (1-tr)·t_enh_raw
                  │        + vr·cls_token                + tr·all_text
                  │       │                        │
                  │       ▼                        ▼
                  │   v_enh [B, 768]           t_enh [B, 200, 768]
                  │       │                        │
                  │       └──────────┬─────────────┘
                  │                  ▼
                  │           cos(v_enh, t_enh) × logit_scale
                  │                  │
                  │                  ▼
                  │            logits_200 [B, 200] ───────┐
                  │                                       │ 主 CE on seenclass
                  │                                       │
                  └──→ base_logits = cos(cls, all_text) × scale  ← 旁路 anchor
                       (不进主路径, 仅作为 distill teacher)
```

---

## 2. 关键改动（vs add 模式）

### 2.1 计分公式

| 项 | add 模式（旧） | cosine_only（新） |
|---|---|---|
| Logits | `base_logits + β·local_score` | `cos(v_enh, t_enh) × scale` |
| base_logits 角色 | **主路径**（永远在） | **旁路 teacher**（只在 distill 开时进梯度） |
| 视觉端 | cls_token 直接用 | v_enh = (1-vr)·v_enh_raw + vr·cls_token |
| 文本端 | adapted_seen + raw_unseen 拼接 | t_enh = (1-tr)·t_enh_raw + tr·all_text |

### 2.2 已识别的结构性风险

1. **base_logits 离开主路径**：默认 forward 不再保证 CLIP zero-shot 兜底
2. **Unseen 列共享 proj_text 权重**：seen-only CE 通过 proj_text 间接漂移 unseen 类输出（PyTorch 自动微分确保 unseen 列输入梯度=0，但权重梯度仍受 seen 列影响）
3. **学习目标只切 seen 150 类**：v_enh / t_enh 都是为 seen 优化的，评估时却要求全 200 类竞争，典型 seen-bias 来源

---

## 3. 损失函数全景（9 个，按物理含义分类）

> 主 CE 始终开启；其他通过 `lambda_*` 开关控制。

### 3.1 主分类损失

| ID | yaml 名 | 数学定义 | 物理含义 |
|----|---------|---------|---------|
| ① 主 CE | (恒开) | `CE(logits, seen_label_local_idx)` | seen 类硬分类，唯一直接梯度源 |

### 3.2 cosine_only 锚定三件套（修结构性丢锚）

| ID | yaml 名 | 数学定义 | 物理含义 |
|----|---------|---------|---------|
| ② v_anchor | `lambda_v_anchor` | `1 − cos(v_enh_raw, cls_token.detach())` | v_enh 锚回 CLIP CLS（视觉端） |
| ③ t_unseen_anchor | `lambda_t_unseen_anchor` | `1 − cos(t_enh_raw[:,unseen,:].mean(B), all_text[unseen].detach())` | t_enh 在 unseen 列锚回 CLIP 原始（修共享 proj_text 漂移） |
| ④ distill | `lambda_distill` | `KL(softmax(base/T) ‖ softmax(logits_200/T)) × T²` | 让 base_logits 通过软约束回到主路径 |

### 3.3 双分支辅助 CE（修双分类器失能）

| ID | yaml 名 | 数学定义 | 物理含义 |
|----|---------|---------|---------|
| ⑤ aux_s2v | `lambda_aux_s2v` | `CE(score_s2v[:,seen] × T_aux, seen_label)`，T_aux=14.28 | s2v 分支自身也是合格 seen 分类器 |
| ⑥ aux_v2s | `lambda_aux_v2s` | `CE(score_v2s[:,seen] × T_aux, seen_label)` | v2s 分支同上 |

### 3.4 几何 / 拓扑约束

| ID | yaml 名 | 数学定义 | 物理含义 |
|----|---------|---------|---------|
| ⑦ topo Pearson | `lambda_topo_pearson` | `1 − Pearson(pairwise_cos(t_enh), pairwise_cos(clip_text))` | 类与类相对角度拓扑保持（TPR 论文用法） |
| ⑧ consist | `lambda_consist` | `KL(softmax(base/T) ‖ softmax(local/T))` on seen | local_score 排序不能与 base 矛盾 |
| ⑨ l2sp | `lambda_l2sp` | `mean(adapter_delta²)` | adapter 输出 delta 别离 0 太远 |

KL 方向（PyTorch `F.kl_div(log_p, p)` = `KL(p ‖ exp(log_p))`）：base 当 teacher，logits_200 当 student。

---

## 4. 总损失公式（实验 E0 配置）

```
total_loss = loss_CE + 0.1 × loss_distill
```

E0 关闭其他全部辅助 loss，仅留 distill 让 base 弱回归主路径，作为论文里 cosine_only 的真正 baseline。

---

## 5. 历史失败实验回顾（v1-v5）

| 版本 | 设计 | H | 失败原因 |
|------|------|---|---------|
| v1 | unseen 列绕过 proj_text 直接用 all_text | 43.83 | seen/unseen 流形错位 |
| v2 | v1 + 事后 detach unseen | 45.05 | 同上 |
| v3 | 流形共享 + 事后 detach | 45.32 | unseen 仍间接被 seen-only CE 拉歪 |
| v4 | cross_tf 内 proj_text 前早期 detach | 50.40 | （v4 当时认为是核心修复，实际审稿人指正自动微分本就保证 unseen 输入梯度=0，detach 无效） |
| **v5** | residual=1.0/1.0 退化为 base_logits 等价 | **72.04** | 数学上其实就是 add 模式的 `cos(cls, all_text)×scale` 部分，**不是真 cosine_only 成绩** |

**结论**：v1-v4 已充分证明纯 cosine_only 不可行；v5 只是退化等价。**真正未验证的状态是 "cosine_only + 锚定/蒸馏 + residual<1.0"**，这是当前实验路线的目标。

---

## 6. 实验路线（E0-E8，每组 epoch=20）

```
阶段一 真 baseline (base 弱参与):
  [E0] cosine_only + residual=0.5/0.5 + lambda_distill=0.1            ← 当前 yaml 状态

阶段二 加辅助监督:
  [E1] E0 + lambda_aux_s2v=0.2 + lambda_aux_v2s=0.2
  [E2] E1 + lambda_topo_pearson=0.05

阶段三 加锚定 (修漂移):
  [E3] E2 + lambda_v_anchor=0.1
  [E4] E2 + lambda_t_unseen_anchor=0.2
  [E5] E2 + 双锚 (E3 + E4)

阶段四 base 强参与 (条件触发):
  [E6] E5 + lambda_distill=0.3                  仅当 E5 H<72 时
  [E7] cosine_base_blend=0.3 (硬混合)            仅当 E6 还不到 72 时, 需先实现

阶段五 动态 residual (论文创新点):
  [E8] residual_mode=learnable_split           需先实现
       tr_seen init=0 (sigmoid=0.5)
       tr_unseen init=3.0 (sigmoid≈0.95) 强保护
```

每组 epoch=20，CUB，random_seed=5。

---

## 7. 决策标准

| 触发条件 | 行动 |
|---------|------|
| E0 H < 60 | 放弃 cosine_only，回 add 模式 + P1（Class-aware Attention Pool） |
| E0 H ∈ [60, 65] | 调 distill_temp 或加重 distill |
| E0 H ≥ 65 | 走 E1（加 aux） |
| E5 H ≥ 72.24（add 基线） | **核心叙事完成可写论文**: "cosine_only + 锚定 + 辅助 CE 框架可匹敌 add 模式且具更好可解释性" |
| E5 H < 72.24 | 跑 E6/E7 兜底 |

---

## 8. 代码审查要点（请专家关注）

### 8.1 cosine_only 计分的实现位置

**文件**: `model/MyModel.py` 第 ~895-945 行

```python
if self.score_mode == 'cosine_only':
    v_enh_raw = cm_out['v_enh']                    # [B, 768]
    t_enh_raw = cm_out['t_enh']                    # [B, 200, 768]
    
    # 视觉端残差
    v_enh = (1 - self.visual_residual) * v_enh_raw + \
            self.visual_residual * cls_token       # [B, 768]
    
    # 文本端残差 (seen/unseen 同一比例, 后续可拆)
    t_enh_seen   = (1 - tr) * t_enh_raw[:, seenclass]   + tr * all_text[seenclass]
    t_enh_unseen = (1 - tr) * t_enh_raw[:, unseenclass] + tr * all_text[unseenclass]
    t_enh = torch.zeros_like(...)
    t_enh[:, seenclass] = t_enh_seen
    t_enh[:, unseenclass] = t_enh_unseen
    
    # 余弦计分
    v_n = F.normalize(v_enh, dim=-1).unsqueeze(1)   # [B, 1, 768]
    t_n = F.normalize(t_enh, dim=-1)                # [B, 200, 768]
    logits_200 = (v_n * t_n).sum(dim=-1) * logit_scale
```

**审查关注**:
- residual 当前是固定标量，未来要改成 `nn.Parameter` 经 sigmoid（E8 任务）
- t_enh 的 seen / unseen 拼接当前 in-place 写到 zeros 张量，**不影响梯度**（但风格上不优雅，可考虑用 index_copy 或直接拼接）
- logit_scale 取自 `self.logit_scale.exp().clamp(max=100)`，跟 CLIP 论文一致

### 8.2 五个新 loss 的实现

**文件**: `model/MyModel.py` 第 ~1080-1150 行

```python
# L1 v_anchor
if lambda_v_anchor > 0 and v_enh_raw is not None and cls_token is not None:
    loss_v_anchor = (1.0 - F.cosine_similarity(
        v_enh_raw, cls_token.detach(), dim=-1)).mean()

# L2 t_unseen_anchor
if lambda_t_unseen_anchor > 0 and t_enh_raw is not None and all_text is not None:
    t_enh_unseen_mean = t_enh_raw[:, self.unseenclass, :].mean(dim=0)
    target_unseen     = all_text[self.unseenclass].detach()
    loss_t_unseen_anchor = (1.0 - F.cosine_similarity(
        t_enh_unseen_mean, target_unseen, dim=-1)).mean()

# L3 distill
if lambda_distill > 0 and logits_200 is not None and base_logits is not None:
    base_p = F.softmax(base_logits.detach() / T_d, dim=-1)
    log_p  = F.log_softmax(logits_200 / T_d, dim=-1)
    loss_distill = F.kl_div(log_p, base_p, reduction='batchmean') * (T_d * T_d)

# L4/L5 aux CE
if lambda_aux_s2v > 0 and score_s2v is not None:
    loss_aux_s2v = F.cross_entropy(
        score_s2v[:, self.seenclass] * T_aux, seen_labels)
```

**审查关注**:
- `cls_token.detach()` / `all_text.detach()` / `base_logits.detach()` 都正确，确保锚 target 不接梯度
- L2 的 batch mean：用全 batch 平均 t_enh[:, unseen, :] 作为本 batch 共识的 unseen 表示，与单一 target 对齐。**潜在风险**：batch 内若 seen/unseen 比例严重失衡，平均不准确。是否要改成全数据集滑动平均（EMA）？
- L3 distill T_d=4.0 是常用值，可调
- L4/L5 aux_temp T_aux=14.28（≈1/0.07）让余弦×温度后量级与主 CE 相当

### 8.3 数据流校验（forward 末尾透传）

**文件**: `model/MyModel.py` 第 ~975-995 行

```python
out = {
    'logits':      logits,       # 训练 [B, 150] / 评估 [B, 200]
    'logits_200':  logits_200,   # 始终 [B, 200], 给 distill / cal 用
    'base_logits': base_logits,  # CLIP 旁路, 给 distill teacher
    'local_score': local_score,  # 双分支融合分, 给 consist 用
    'text_topology_features': topology_text,  # 给 topo 用
    ...
}
if self.score_mode == 'cosine_only':
    out['v_enh_raw'] = cm_out['v_enh']
    out['t_enh_raw'] = cm_out['t_enh']
    out['cls_token'] = cls_token
    out['all_text']  = all_text
    out['score_s2v'] = cm_out.get('score_s2v')
    out['score_v2s'] = cm_out.get('score_v2s')
```

**审查关注**:
- 仅 cosine_only 透传锚定字段，add 模式零副作用
- compute_loss 全部 `in_package.get(k, None)` 取值，缺失字段安全降级

### 8.4 与之前 v3/v4 的差异

v3/v4 的 "early detach" 已被审稿人证伪后删除：

```python
# OLD (v4, 已删):
F_p_v2s_for_text = F_p_v2s.clone()
F_p_v2s_for_text[:, unseen_idx, :] = F_p_v2s[:, unseen_idx, :].detach()
t_enh = self.proj_text(F_p_v2s_for_text)

# NEW (当前):
t_enh = self.proj_text(F_p_v2s)   # 自动微分保证 unseen 输入梯度=0, detach 无意义
```

这一点专家如果质疑可参考我们的 `verify_grad.py` 验证脚本（小例子证明 `Linear(x_all)` 后只对 seen 输出求 loss，unseen 输入梯度严格为 0）。

---

## 9. 已确认无效但保留代码的设计

| 设计 | 状态 | 保留原因 |
|------|------|---------|
| `lambda_cal` self-calibration | 实验证伪 (0.001 中性, 0.01 -6%) | 论文 ablation 引用 |
| `gating='cig'` | 实测 H=71.80 (-0.44) | 论文 ablation 引用 |
| `pool_method='lastvit'` | 5 次实验全失败 | 论文 negative result 章节 |
| `use_lastvit_cls` | 实验失败 | 同上 |

---

## 10. 给专家的核心问题

1. **L2 t_unseen_anchor 的 batch mean 是否合理**？还是应该改成 EMA？
2. **distill teacher 是否应该改用 base_logits + α·local_score（add 模式输出）而不是 base_logits**？这样让 student 学到的不仅是 CLIP zero-shot，还有 add 模式的成熟经验
3. **E8 动态 residual 的 unseen 初始化偏 0.95 是否过强**？是否应该让模型自己学到偏离值
4. **是否需要加 `cosine_base_blend` 硬混合开关（E7）作为 distill 失败的兜底**？

---

## 附录 A. 关键文件路径

- `model/MyModel.py` — 主模型 + cosine_only forward + 9 个 loss
- `config/VGSR_cub_gzsl.yaml` — 当前 E0 配置
- `train_VGSR_CUB.py` — 训练入口（含 E0-E8 都用同一脚本）
- `EXPERIMENT_RECORD.md` — 实验队列与历史记录

## 附录 B. 当前 yaml E0 状态

```yaml
score_mode:              cosine_only
epochs:                  20
random_seed:             5
pool_method:             mean
text_residual:           0.5
visual_residual:         0.5
adapter_ratio:           0.2

# E0 唯一开启的辅助 loss
lambda_distill:          0.1
distill_temp:            4.0

# 全部关闭
lambda_v_anchor:         0.0
lambda_t_unseen_anchor:  0.0
lambda_aux_s2v:          0.0
lambda_aux_v2s:          0.0
lambda_topo_pearson:     0.0
lambda_consist:          0.0
lambda_l2sp:             0.0
lambda_cal:              0.0
```
