"""
VGSR 主模型 (CLIP + Adapter + GPT + 双向 Transformer)
=========================================================
模块化说明：本文件内包含三个独立模块，每个模块可单独使用。
  1. Adapter                    — 文本特征轻量增强 (VDT-TransZero)
  2. CrossModalTransformer      — 视觉-语义双向 Transformer (TransZero++)
     包含子组件: BoxRelationalEmbedding, GeometryMultiHeadAttention, FAELayer
  3. VGSR                       — 主模型, 组合上述模块

接口契约 (不要破坏):
  - forward(clip_features, is_train=False) 返回字典, 必含键 'clip_S_pp'
  - is_train=True  → logits [B, n_seen]      (训练用)
  - is_train=False → logits [B, num_class]   (评估用)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==========================================================
#  模块 1: Adapter (VDT-TransZero 轻量语义增强)
# ==========================================================
class Adapter(nn.Module):
    """
    Bottleneck 结构的轻量 Adapter
    结构: Linear(d → d/r) → ReLU → Linear(d/r → d)
    作用: 对 seen 类文本特征做任务相关的语义微调
    """
    def __init__(self, c_in, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
        )

    def forward(self, x):
        return self.fc(x)


# ==========================================================
#  模块 2: CrossModalTransformer (TransZero++ 双向交互)
# ==========================================================

class BoxRelationalEmbedding(nn.Module):
    """
    Box Relational Embedding (来自 TransZero / TransZero++)
    预计算 24×24 grid 上两两 patch 之间的相对几何位置编码 [576, 576, dim_g]
    FAE 的核心：让注意力感知空间距离，消除相邻 patch 的几何纠缠
    """
    def __init__(self, grid_size=(24, 24), dim_g=64, wave_len=1000.0):
        super().__init__()
        self.grid_size = grid_size
        self.dim_g = dim_g
        self.wave_len = wave_len
        # 预计算后 register_buffer, 不参与训练也不需要梯度
        self.register_buffer('geometry_embedding', self._compute_embedding())

    def _compute_embedding(self):
        H, W = self.grid_size
        seq_len = H * W
        # 每个 patch 的 (x_min, y_min, x_max, y_max) 左上右下角
        x = torch.arange(H).float()
        y = torch.arange(W).float()
        px_min = x.view(-1, 1).expand(-1, W).contiguous().view(-1)  # [HW]
        py_min = y.view(1, -1).expand(H, -1).contiguous().view(-1)  # [HW]
        px_max = px_min + 1
        py_max = py_min + 1
        # 中心 + 宽高
        cx = (px_min + px_max) * 0.5
        cy = (py_min + py_max) * 0.5
        w = px_max - px_min + 1.0
        h = py_max - py_min + 1.0
        # 4 种相对几何量
        delta_x = cx.unsqueeze(0) - cx.unsqueeze(1)             # [HW, HW]
        delta_x = torch.clamp(torch.abs(delta_x / w.unsqueeze(0)), min=1e-3).log()
        delta_y = cy.unsqueeze(0) - cy.unsqueeze(1)
        delta_y = torch.clamp(torch.abs(delta_y / h.unsqueeze(0)), min=1e-3).log()
        delta_w = torch.log(w.unsqueeze(0) / w.unsqueeze(1))
        delta_h = torch.log(h.unsqueeze(0) / h.unsqueeze(1))
        # 堆成 [HW, HW, 4]
        pos_mat = torch.stack([delta_x, delta_y, delta_w, delta_h], dim=-1)
        # 正弦位置编码
        feat_range = torch.arange(self.dim_g / 8).float()
        dim_mat = 1.0 / (self.wave_len ** (feat_range / (self.dim_g / 8)))  # [dim_g/8]
        dim_mat = dim_mat.view(1, 1, 1, -1)
        pos_mat = pos_mat.unsqueeze(-1) * 100.0                  # [HW, HW, 4, 1]
        mul_mat = (pos_mat * dim_mat).view(seq_len, seq_len, -1) # [HW, HW, dim_g/2]
        embedding = torch.cat([mul_mat.sin(), mul_mat.cos()], dim=-1)  # [HW, HW, dim_g]
        return embedding.half()  # float16 存储，显存减半

    def forward(self, batch_size):
        """返回 [B, 576, 576, dim_g]"""
        return self.geometry_embedding.unsqueeze(0).expand(
            batch_size, -1, -1, -1)


class GeometryMultiHeadAttention(nn.Module):
    """
    带几何位置编码的多头自注意力 (FAE 的核心操作)
    注意力得分 = scaled_dot_product(Q, K) + geometry_weight
    """
    def __init__(self, dim_com, heads, dim_g=64, dropout=0.1):
        super().__init__()
        assert dim_com % heads == 0
        self.heads = heads
        self.d_k = dim_com // heads
        self.fc_q = nn.Linear(dim_com, dim_com)
        self.fc_k = nn.Linear(dim_com, dim_com)
        self.fc_v = nn.Linear(dim_com, dim_com)
        self.fc_o = nn.Linear(dim_com, dim_com)
        # 每个 head 独立的几何权重生成器
        self.WGs = nn.ModuleList([nn.Linear(dim_g, 1, bias=True)
                                  for _ in range(heads)])
        self.ln = nn.LayerNorm(dim_com)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, geometry_emb):
        # x: [B, N, dim_com]      geometry_emb: [B, N, N, dim_g] (float16 存储)
        B, N, D = x.shape
        q = self.fc_q(x).view(B, N, self.heads, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(x).view(B, N, self.heads, self.d_k).permute(0, 2, 1, 3)
        v = self.fc_v(x).view(B, N, self.heads, self.d_k).permute(0, 2, 1, 3)
        att = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)         # [B, h, N, N]
        # geometry_emb 是 float16，转 float32 再送入 Linear
        geo_flat = geometry_emb.float().reshape(-1, geometry_emb.shape[-1])    # [B*N*N, dim_g]
        geo_per_head = [layer(geo_flat).view(B, N, N, 1).permute(0, 3, 1, 2)
                        for layer in self.WGs]
        geo_weights = F.relu(torch.cat(geo_per_head, dim=1))                   # [B, h, N, N]
        att = att - geo_weights
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(B, N, D)
        out = self.fc_o(out)
        return self.ln(x + out)


class FAELayer(nn.Module):
    """Feature Augmentation Encoder 单层: Geometry Attention + FFN"""
    def __init__(self, dim_com, heads, dropout=0.1, dim_g=64):
        super().__init__()
        self.attn = GeometryMultiHeadAttention(dim_com, heads, dim_g, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim_com, dim_com * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim_com * 2, dim_com),
        )
        self.ln = nn.LayerNorm(dim_com)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, geometry_emb):
        x = self.attn(x, geometry_emb)
        x = self.ln(x + self.dropout(self.ffn(x)))
        return x


class CrossModalTransformer(nn.Module):
    """
    双向视觉-语义交互 Transformer (参考 TransZero++)
    ┌──────────────────────────────────────────────────────────┐
    │  输入:                                                    │
    │    patches: [B, 576, 768]   CLIP 的 24×24 patch 特征      │
    │    text:    [N_cls, 768]    (经 Adapter 增强后的) 文本特征 │
    │                                                          │
    │  1. 投影到公共维度 dim_com                                 │
    │  2. FAE: 视觉自注意力 + 几何位置编码                       │
    │  3. s2v 分支: text Query, visual KV  → 每类视觉定位嵌入    │
    │  4. v2s 分支: visual Query, text KV  → 视觉感知语义       │
    │  5. 两路余弦相似度加权融合 → logits [B, N_cls]             │
    └──────────────────────────────────────────────────────────┘
    """
    def __init__(self, dim_f=768, dim_com=512, heads=4, dropout=0.1,
                 weight_s2v=0.5, grid_size=(24, 24), dim_g=64,
                 text_residual=0.2, visual_residual=0.2):
        super().__init__()
        self.dim_f = dim_f
        self.dim_com = dim_com
        self.weight_s2v = weight_s2v
        self.text_residual   = text_residual    # v2s 文本残差系数 (保护 unseen 文本)
        self.visual_residual = visual_residual  # s2v 视觉残差系数 (保护 CLIP 原始视觉)

        # 输入投影 768 → dim_com (两路共享视觉投影, 文本投影独立)
        self.embed_cv = nn.Linear(dim_f, dim_com)       # 共享视觉投影
        self.embed_text = nn.Linear(dim_f, dim_com)     # 共享文本投影

        # 几何位置编码 (预计算, 不训练)
        self.box_emb = BoxRelationalEmbedding(grid_size=grid_size, dim_g=dim_g)

        # FAE: 共享, 两路看到一致的视觉理解 (减少参数 + 提升耦合)
        self.fae = FAELayer(dim_com, heads, dropout, dim_g=dim_g)

        # v2s 解码: 先跑, 用原始文本去 query 视觉, 得到文本增强
        self.decoder_v2s = nn.TransformerDecoderLayer(
            d_model=dim_com, nhead=heads, dim_feedforward=dim_com * 2,
            dropout=dropout, batch_first=True)

        # s2v 解码: 后跑, 用 v2s 增强过的文本 query 视觉, 得到视觉增强
        # 这里形成了 cross-connect: s2v 依赖 v2s 的输出
        self.decoder_s2v = nn.TransformerDecoderLayer(
            d_model=dim_com, nhead=heads, dim_feedforward=dim_com * 2,
            dropout=dropout, batch_first=True)

        # 投影回 768 维 (两路各自独立)
        self.proj_visual = nn.Linear(dim_com, dim_f)   # s2v 输出: 视觉特征
        self.proj_text   = nn.Linear(dim_com, dim_f)   # v2s 输出: 文本 delta

    def forward(self, patches, text, cls_token=None):
        """
        Args:
            patches:   [B, 576, 768]  CLIP patch 特征 (不含 CLS)
            text:      [N_cls, 768]   原始文本特征
            cls_token: [B, 768]       CLIP CLS token (保留接口兼容)
        Returns:
            local_score: [B, N_cls]   双分支产生的额外局部注意力分数

        真正双分支设计 (方向相反，并行独立):
            v2s 分支: 文本 Query 视觉 → 视觉感知的文本表示
            s2v 分支: 视觉 Query 文本 → 文本感知的视觉表示
            两路各自产出分数，加权融合
            FAE 同时接收两路梯度，被双向约束

        双向体现:
            v2s 梯度: loss → score_v2s → F_p_v2s → decoder_v2s → memory → FAE
            s2v 梯度: loss → score_s2v → F_p_s2v → decoder_s2v → memory → FAE
            FAE 参数同时被两路约束，学到更通用的视觉表示
        """
        B = patches.size(0)
        geo_emb = self.box_emb(B)                                # [B, 576, 576, dim_g]

        # ========== 共享视觉编码 ==========
        # 两路共用同一个 memory，视觉理解一致，节省参数
        vis    = self.embed_cv(patches)                          # [B, 576, dim_com]
        memory = self.fae(vis, geo_emb)                          # [B, 576, dim_com]

        txt_com   = self.embed_text(text)                        # [N_cls, dim_com]
        txt_batch = txt_com.unsqueeze(0).expand(B, -1, -1)       # [B, N_cls, dim_com]

        # ========== v2s 分支: 文本 Query 视觉 ==========
        # 每个类的文本去 576 个 patch 里找相关视觉区域
        # Q=txt_batch (文本), KV=memory (视觉)
        F_p_v2s = self.decoder_v2s(tgt=txt_batch,
                                    memory=memory)               # [B, N_cls, dim_com]
        # F_p_v2s[b,k,:] = 第b张图，第k类文本感知视觉后的表示

        # ========== s2v 分支: 视觉 Query 文本 (方向相反) ==========
        # 每个 patch 去文本里找语义信息
        # Q=memory (视觉), KV=txt_batch (文本)
        F_p_s2v = self.decoder_s2v(tgt=memory,
                                    memory=txt_batch)            # [B, 576, dim_com]
        # F_p_s2v[b,p,:] = 第b张图，第p个patch感知文本后的表示

        # ========== 两路各自产出分数 ==========
        # v2s 分数: 视觉感知的文本 vs 原始文本 → 每类的匹配分数
        v2s_n   = F.normalize(F_p_v2s, dim=-1)                  # [B, N_cls, dim_com]
        txt_n   = F.normalize(txt_batch, dim=-1)                 # [B, N_cls, dim_com]
        score_v2s = (v2s_n * txt_n).sum(dim=-1)                  # [B, N_cls]

        # s2v 分数: 文本感知的视觉（全局池化）vs 原始文本 → 每类的匹配分数
        s2v_pooled = F_p_s2v.mean(dim=1)                         # [B, dim_com] 576→全局
        s2v_n      = F.normalize(s2v_pooled, dim=-1)             # [B, dim_com]
        txt_single = F.normalize(txt_com, dim=-1)                # [N_cls, dim_com]
        score_s2v  = s2v_n @ txt_single.T                        # [B, N_cls]

        # ========== 加权融合 ==========
        local_score = self.weight_s2v * score_s2v + \
                      (1.0 - self.weight_s2v) * score_v2s        # [B, N_cls]

        return local_score


# ==========================================================
#  模块 3: 主模型 VGSR
# ==========================================================
class VGSR(nn.Module):
    """
    GZSL 主模型
    ┌──────────────────────────────────────────────────────────┐
    │  训练阶段 (is_train=True):                                │
    │    - 传入 [B, 577, 768] (CLIP spatial) 或 [B, 1, 768]    │
    │    - text = Adapter 增强的 seen 150 类                    │
    │    - CrossModalTransformer(patches, text) → [B, 150]     │
    │    - CE loss on seen 局部索引 (0~149)                     │
    │                                                          │
    │  评估阶段 (is_train=False):                               │
    │    - 传入 [B, 577, 768]                                  │
    │    - text = seen_adapted (150) + unseen_clip (50)        │
    │           按全局类别索引拼成 200 类                        │
    │    - CrossModalTransformer(patches, text) → [B, 200]     │
    │    - 评估时 argmax 在全 200 类中竞争                       │
    └──────────────────────────────────────────────────────────┘
    """
    def __init__(self, config, seenclass, unseenclass,
                 seen_text_embeds,       # [150, 768]
                 unseen_text_embeds):    # [50,  768]
        super().__init__()
        self.config = config
        self.nclass = config.num_class   # 200
        self.dim_f = config.dim_f_clip   # 768
        self.seenclass = seenclass       # [150] 全局索引
        self.unseenclass = unseenclass   # [50]  全局索引

        # ---- 冻结文本特征 ----
        self.seen_text_embeds   = nn.Parameter(
            F.normalize(seen_text_embeds,   dim=1), requires_grad=False)
        self.unseen_text_embeds = nn.Parameter(
            F.normalize(unseen_text_embeds, dim=1), requires_grad=False)

        # ---- Adapter (语义端轻量增强) ----
        self.adapter_ratio = getattr(config, 'adapter_ratio', 0.2)
        self.text_adapter  = Adapter(self.dim_f, reduction=4)

        # ---- CrossModalTransformer (双向视觉-语义交互) ----
        tf_common_dim = getattr(config, 'tf_common_dim', 512)
        tf_heads      = getattr(config, 'tf_heads', 4)
        tf_dropout    = getattr(config, 'tf_dropout', 0.1)
        weight_s2v    = getattr(config, 'weight_s2v', 0.5)
        text_residual   = getattr(config, 'text_residual', 0.2)
        visual_residual = getattr(config, 'visual_residual', 0.2)
        self.local_weight = getattr(config, 'local_weight', 0.3)  # 局部分数加权系数
        self.cross_tf = CrossModalTransformer(
            dim_f=self.dim_f,
            dim_com=tf_common_dim,
            heads=tf_heads,
            dropout=tf_dropout,
            weight_s2v=weight_s2v,
            grid_size=(24, 24),
            dim_g=64,
            text_residual=text_residual,
            visual_residual=visual_residual,
        )

        # ---- 温度系数 (可学习) ----
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_adapted_seen_text(self):
        """Adapter 残差增强的 seen 类文本 [150, 768]"""
        x = self.seen_text_embeds
        adapted = self.adapter_ratio * self.text_adapter(x) + \
                  (1.0 - self.adapter_ratio) * x
        return F.normalize(adapted, dim=1)

    def _prepare_patches(self, clip_features):
        """
        从 CLIP 特征中提取 patch 序列
        - [B, 577, 768] → 取 [:, 1:, :] 得到 [B, 576, 768]
        - [B,   1, 768] → 广播成 [B, 576, 768] (仅向后兼容 CLS 缓存模式)
        - [B,     768]  → 同上
        """
        if clip_features.dim() == 3:
            if clip_features.size(1) == 577:       # 完整 spatial
                return clip_features[:, 1:, :]
            elif clip_features.size(1) == 576:     # 纯 patch
                return clip_features
            elif clip_features.size(1) == 1:       # CLS 缓存兼容
                return clip_features.expand(-1, 576, -1)
        # [B, 768]
        return clip_features.unsqueeze(1).expand(-1, 576, -1)

    def forward(self, clip_features, is_train=False):
        """
        Args:
            clip_features: CLIP 输出, 支持 [B, 577, 768] / [B, 576, 768] / [B, 768]
            is_train: True  → 输出 [B, 150] seen 类 logits (loss 用)
                      False → 输出 [B, 200] 全类 logits (评估用)

        数据流 (对称双分支, 两端各自增强):
            视觉分支 (s2v): patches → enhanced_visual [B, 768]
            语义分支 (v2s): text    → enhanced_text [N_cls, 768]  (强残差保护 unseen)
            分类: enhanced_visual @ enhanced_text.T × logit_scale (基线方式)
        """
        # 分离 CLS token 和 patches
        if clip_features.dim() == 3 and clip_features.size(1) == 577:
            cls_token = clip_features[:, 0, :]                   # [B, 768]
            patches   = clip_features[:, 1:, :]                  # [B, 576, 768]
        else:
            patches = self._prepare_patches(clip_features)       # [B, 576, 768]
            cls_token = None

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        seen_text = self.get_adapted_seen_text()                 # [150, 768]

        # 构建全 200 类文本
        all_text = torch.zeros(self.nclass, self.dim_f,
                               device=patches.device, dtype=patches.dtype)
        all_text[self.seenclass]   = seen_text
        all_text[self.unseenclass] = self.unseen_text_embeds

        # ── 基线分数: CLIP 原始余弦相似度 (完全不动, 保留零样本能力) ──
        if cls_token is not None:
            vis_n = F.normalize(cls_token, dim=1)                # [B, 768]
        else:
            vis_n = F.normalize(patches.mean(dim=1), dim=1)
        text_n = F.normalize(all_text, dim=1)                    # [200, 768]
        base_logits = vis_n @ text_n.T * logit_scale             # [B, 200]

        # ── 局部分数: 双分支产生的额外注意力分数 ──
        local_score = self.cross_tf(patches, all_text, cls_token) # [B, 200]

        # ── 最终 logits = 基线 + β × 局部加分 ──
        # base_logits 保留 CLIP 全部能力, local_score 只补充局部细节
        logits_200 = base_logits + self.local_weight * local_score

        if is_train:
            logits = logits_200[:, self.seenclass]               # [B, 150]
        else:
            logits = logits_200                                  # [B, 200]

        return {
            'logits':     logits,
            'logits_200': logits_200,   # compute_loss 里 calibration loss 用
            'clip_S_pp':  logits,
            'clip_pred':  logits,
        }

    def compute_loss(self, in_package):
        """
        训练 Loss = CE loss + λ_cal × self-calibration loss
        
        CE loss: 监督 seen 类分类
        Self-calibration loss: 强制模型给 unseen 类保留一定概率
            -log(mean(P(unseen))) 越小说明 unseen 类概率越高
            防止模型完全忽视 unseen 类（seen bias）
        """
        logits = in_package['logits']        # [B, 150] 训练时只有 seen 列
        labels = in_package['batch_label']   # [B] 全局索引

        if labels.dim() > 1:
            labels = torch.argmax(labels, dim=1)

        # 全局索引 → seen 局部索引 (0~149)
        seen_labels = torch.zeros_like(labels)
        for i, cls_idx in enumerate(self.seenclass):
            seen_labels[labels == cls_idx] = i

        # CE loss（主损失）
        loss_CE = F.cross_entropy(logits, seen_labels)

        # Self-calibration loss（辅助损失）
        # 用全 200 类的 logits 计算（需要从 in_package 里取）
        logits_200 = in_package.get('logits_200', None)
        if logits_200 is not None and self.config.__dict__.get('lambda_cal', 0) > 0:
            prob_all    = F.softmax(logits_200, dim=-1)              # [B, 200]
            prob_unseen = prob_all[:, self.unseenclass]              # [B, 50]
            mass_unseen = prob_unseen.sum(dim=1)                     # [B]
            loss_cal    = -torch.log(mass_unseen.mean() + 1e-8)     # 标量
            loss = loss_CE + self.config.lambda_cal * loss_cal
        else:
            loss_cal = torch.tensor(0.0)
            loss = loss_CE

        return {'loss': loss, 'loss_CE': loss_CE, 'loss_cal': loss_cal}
