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

    核心目的:
        把 patch i 和 patch j 之间的"空间相对关系"编码成 64 维向量,
        作为 FAE 注意力里"几何偏置"的输入。
        因为这只跟 ViT patch 网格固定坐标有关 (24×24, 每张图都一样),
        所以可以预计算一次, 永远不变 → register_buffer 存为 GPU 常量。

    与图像内容的关系:
        无关。位置不是从图像内容里"算"出来的, 而是按 patch 在 token
        序列里的索引推算出的网格坐标 (索引 0=(0,0), 索引 575=(23,23))。
        CLIP ViT-L/14@336px 输出固定 24×24=576 个 patch, 这个网格是
        ViT 的结构性事实, 跟拍的是鸟还是车毫无关系。

    输出:
        embedding [576, 576, 64]
            embedding[i, j] = patch i → patch j 的相对位置编码
            自身 (i==j) 也有编码, 但语义上是"零位移"
    """
    def __init__(self, grid_size=(24, 24), dim_g=64, wave_len=1000.0):
        """
        Args:
            grid_size: (H, W) ViT patch 网格大小, 这里固定 (24, 24) 对应
                       CLIP ViT-L/14@336px 的输出
            dim_g:     位置编码维度, 必须能被 8 整除 (因为后面拆成 4 种几何
                       量 × sin/cos 两组, 每组 dim_g/8 个频率)
            wave_len:  正弦编码的波长基数, 类似 NLP Transformer 的 10000
                       这里用 1000.0 是 TransZero++ 的设定
        """
        super().__init__()
        self.grid_size = grid_size
        self.dim_g = dim_g
        self.wave_len = wave_len
        # ★ register_buffer: 注册为模型状态但不参与训练
        #   - 会跟着 .to(device) 转移到 GPU
        #   - 会被保存到 state_dict
        #   - 不会出现在 .parameters() 里, 不会被优化器更新
        self.register_buffer('geometry_embedding', self._compute_embedding())

    def _compute_embedding(self):
        """预计算 [576, 576, dim_g] 的几何编码张量, 只在 __init__ 时跑一次"""
        H, W = self.grid_size                  # (24, 24)
        seq_len = H * W                        # 576

        # ---- 第 1 步: 给每个 patch 算一个"伪 box"坐标 ----
        # 每个 patch 当作图像里 1×1 的小方块, 左上右下角:
        # 索引 0  → 网格 (0, 0) → box (0,0,1,1)
        # 索引 24 → 网格 (1, 0) → box (1,0,2,1)
        # 索引 575→ 网格 (23,23) → box (23,23,24,24)
        x = torch.arange(H).float()            # [0, 1, ..., 23]
        y = torch.arange(W).float()
        px_min = x.view(-1, 1).expand(-1, W).contiguous().view(-1)  # [576] 每个 patch 的 x_min
        py_min = y.view(1, -1).expand(H, -1).contiguous().view(-1)  # [576] 每个 patch 的 y_min
        px_max = px_min + 1
        py_max = py_min + 1

        # ---- 第 2 步: 算 box 中心 + 宽高 ----
        # 这里宽高都恒为 1 (每个 patch 都是 1×1), 但保留通用代码以便扩展
        cx = (px_min + px_max) * 0.5           # [576]
        cy = (py_min + py_max) * 0.5
        w = px_max - px_min + 1.0              # [576] 实际值都是 2.0
        h = py_max - py_min + 1.0

        # ---- 第 3 步: 算两两 patch 之间的 4 种相对几何量 ----
        # delta_x[i, j] = patch i 的 cx - patch j 的 cx, 归一化后取 log
        # 取 log 让"距离 1 vs 距离 23"的差别在编码空间被压缩, 避免远距离
        # 占据数值主导。clamp(min=1e-3) 防止 log(0)。
        delta_x = cx.unsqueeze(0) - cx.unsqueeze(1)             # [576, 576]
        delta_x = torch.clamp(torch.abs(delta_x / w.unsqueeze(0)), min=1e-3).log()
        delta_y = cy.unsqueeze(0) - cy.unsqueeze(1)
        delta_y = torch.clamp(torch.abs(delta_y / h.unsqueeze(0)), min=1e-3).log()
        # 这里 delta_w / delta_h 在固定网格上恒为 0 (所有 patch 同尺寸),
        # 写出来是为了和 TransZero++ 论文公式对齐, 也方便迁移到非均匀网格
        delta_w = torch.log(w.unsqueeze(0) / w.unsqueeze(1))
        delta_h = torch.log(h.unsqueeze(0) / h.unsqueeze(1))

        # 堆成 [576, 576, 4] 四种几何量
        pos_mat = torch.stack([delta_x, delta_y, delta_w, delta_h], dim=-1)

        # ---- 第 4 步: 正弦位置编码 (类似 NLP Transformer) ----
        # 把 4 个标量几何量扩展成 dim_g 维向量, 走 sin/cos 不同频率
        # dim_g=64 → 每种几何量编码到 dim_g/4=16 维 → 拆成 sin 8 维 + cos 8 维
        feat_range = torch.arange(self.dim_g / 8).float()                   # [8]
        dim_mat = 1.0 / (self.wave_len ** (feat_range / (self.dim_g / 8)))  # [8] 频率序列
        dim_mat = dim_mat.view(1, 1, 1, -1)                                  # [1,1,1,8]
        # 乘 100 是为了让 log 后的几何量数值范围合适, 落在 sin/cos 有信息
        # 量的区间, 避免 sin(微小值)≈0 的退化
        pos_mat = pos_mat.unsqueeze(-1) * 100.0                              # [576,576,4,1]
        mul_mat = (pos_mat * dim_mat).view(seq_len, seq_len, -1)             # [576,576,32]
        # sin 部分 + cos 部分拼起来 → 64 维
        embedding = torch.cat([mul_mat.sin(), mul_mat.cos()], dim=-1)        # [576,576,64]

        # ★ 用 float16 存: 576×576×64 = 21M 个数,
        #   float32 占 84MB, float16 减半到 42MB
        #   在 GeometryMultiHeadAttention 里用时会临时转回 float32
        return embedding.half()

    def forward(self, batch_size):
        """
        把 [576, 576, 64] 的预计算编码扩展到 batch 维度, 配合 attention 用

        Args:
            batch_size: 当前 batch 的样本数 B
        Returns:
            [B, 576, 576, 64] (float16) 所有样本共享同一份几何编码
        """
        return self.geometry_embedding.unsqueeze(0).expand(
            batch_size, -1, -1, -1)


class GeometryMultiHeadAttention(nn.Module):
    """
    带几何位置编码的多头自注意力 (FAE 的核心操作, TransZero++ 风格)
    =====================================================
    与标准多头注意力的差别:
        标准:    att(i, j) = Q_i · K_j / √d
        本模块:  att(i, j) = Q_i · K_j / √d  -  geometry_weight(i, j)
                                              ─────────────────────
                                              ★ 注意是减号 ★
        其中 geometry_weight ≥ 0 由 BoxRelationalEmbedding 提供的相对位置
        编码经一层 Linear + ReLU 学到, 每个 head 独立学一组。

    为什么是"减"而不是"加":
        TransZero++ 想做的是"解耦"位置纠缠, 不是"加强"空间感知。
        几何相关度高的 patch 对 (空间上相近) 注意力被扣减,
        逼模型用"特征确实语义相关"作为关注依据, 而不是"靠得近"
        这种 cheap shortcut。对 CUB 细粒度任务尤其重要。

    内部结构 (Post-LN 风格):
        x → Q,K,V 投影 → 多头分头 → att = QK/√d - geo_weight
            → softmax → dropout → @ V → 输出投影 → Add(x) + LN
    """
    def __init__(self, dim_com, heads, dim_g=64, dropout=0.1):
        """
        Args:
            dim_com: 公共维度 (在 VGSR 里是 512), 必须能被 heads 整除
            heads:   注意力头数 (在 VGSR 里默认 4)
            dim_g:   几何编码维度 (64), 来自 BoxRelationalEmbedding
            dropout: 注意力权重 dropout 比例
        """
        super().__init__()
        assert dim_com % heads == 0
        self.heads = heads
        self.d_k = dim_com // heads             # 每个头的维度 = 512/4 = 128

        # 标准多头注意力的 Q/K/V/O 投影
        self.fc_q = nn.Linear(dim_com, dim_com)
        self.fc_k = nn.Linear(dim_com, dim_com)
        self.fc_v = nn.Linear(dim_com, dim_com)
        self.fc_o = nn.Linear(dim_com, dim_com)

        # ★ 每个 head 独立的几何权重生成器
        #   输入: [N, N, dim_g] 几何编码
        #   输出: [N, N, 1]    每对 patch 的几何偏置标量
        #   不同 head 学不同的"位置纠缠模式" (有的 head 抑制远距离, 有的抑制近距离)
        self.WGs = nn.ModuleList([nn.Linear(dim_g, 1, bias=True)
                                  for _ in range(heads)])

        # 输出后的 LayerNorm (Post-LN 风格, 与 FAELayer 内 FFN 后的 LN 配套)
        self.ln = nn.LayerNorm(dim_com)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, geometry_emb):
        """
        Args:
            x:            [B, N, dim_com]    输入特征 (在 VGSR 里 N=576)
            geometry_emb: [B, N, N, dim_g]   预计算几何编码 (float16)
        Returns:
            [B, N, dim_com]  解耦位置纠缠后的特征 + 残差 + LN
        """
        B, N, D = x.shape

        # ---- 第 1 步: 投影 + 分头 + permute 到 [B, heads, N, d_k] ----
        q = self.fc_q(x).view(B, N, self.heads, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(x).view(B, N, self.heads, self.d_k).permute(0, 2, 1, 3)
        v = self.fc_v(x).view(B, N, self.heads, self.d_k).permute(0, 2, 1, 3)

        # ---- 第 2 步: 标准 scaled dot-product attention ----
        att = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)         # [B, h, N, N]

        # ---- 第 3 步: 计算每个 head 的几何偏置 ----
        # geometry_emb 是 float16 节省显存, Linear 层是 float32, 这里转回
        geo_flat = geometry_emb.float().reshape(-1, geometry_emb.shape[-1])    # [B*N*N, dim_g]
        # 对每个 head 用独立的 WGs[h] 生成几何偏置标量
        geo_per_head = [layer(geo_flat).view(B, N, N, 1).permute(0, 3, 1, 2)
                        for layer in self.WGs]                                  # 每个: [B, 1, N, N]
        # ReLU 保证几何权重非负 (>=0), 这样后面的减号才会"扣减"而不是"反向加强"
        geo_weights = F.relu(torch.cat(geo_per_head, dim=1))                   # [B, h, N, N]

        # ---- 第 4 步: 减去几何偏置 (核心操作!) ----
        # ★ 注意是减号: 空间相关度越大 → att 被扣得越多 → 越不容易被关注
        # 这就是 TransZero++ "解耦位置纠缠"的实现
        att = att - geo_weights

        # ---- 第 5 步: softmax + dropout + 加权求 V ----
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(B, N, D)

        # ---- 第 6 步: 输出投影 + 残差 + LN ----
        out = self.fc_o(out)
        return self.ln(x + out)


#FAE模块
class FAELayer(nn.Module):
    """
    Feature Augmentation Encoder 单层 (TransZero++)
    =====================================================
    结构: Geometry-aware Self-Attention → FFN → Add & LN

    核心目的:
        CLIP ViT 内部的 position embedding 让 patch 之间产生位置纠缠
        (相邻 patch 因位置编码相近而特征相似, 模型容易"看身边")
        FAE 在注意力得分中**反向扣除几何相关度**, 逼模型抛开空间临近
        的 shortcut, 只根据语义内容决定要不要互相关注。
        对 CUB 细粒度鸟类任务有益: 喙形/羽毛纹理这种跨区域语义判别
        必须靠纯语义关联, 不能依赖空间近邻偏置。

    数据流:
        x [B, 576, dim_com]
            │
            ↓  几何感知自注意力 (GeometryMultiHeadAttention)
            │   att = Q·K/√d - geo_weight   (注意是减号)
            │   geo_weight ≥ 0, 由相对位置编码 → 单层 Linear → ReLU 得到
            │   空间相关度高的 patch 对, 注意力被扣减
            │
            ↓  内部已做 Add(x) + LayerNorm
            │
            ↓  FFN: 512 → 1024 → 512 (扩展 → ReLU → 压缩)
            │
            ↓  Dropout
            │
            ↓  Add(残差) + LayerNorm   ← 标准 Transformer 残差
            │
            ↓
        x' [B, 576, dim_com]   解耦了位置纠缠的视觉表示
    """
    def __init__(self, dim_com, heads, dropout=0.1, dim_g=64):
        super().__init__()
        # 几何感知多头注意力:
        #   att = softmax(Q·K/√d - relu(W_g · geo_emb))
        #   把"两个 patch 空间上有多相关"作为负偏置, 抑制位置纠缠
        self.attn = GeometryMultiHeadAttention(dim_com, heads, dim_g, dropout)

        # 标准 Transformer FFN: 扩展 2 倍后再压回原维度
        # 用 ReLU 而不是 GELU, 是 TransZero++ 论文的实现选择
        self.ffn = nn.Sequential(
            nn.Linear(dim_com, dim_com * 2),  # 512 → 1024
            nn.ReLU(inplace=True),
            nn.Linear(dim_com * 2, dim_com),  # 1024 → 512
        )

        # FFN 之后的 LayerNorm (Post-LN 写法)
        self.ln = nn.LayerNorm(dim_com)

        # FFN 输出的 dropout, 作正则化, 防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, geometry_emb):
        """
        Args:
            x:            [B, 576, dim_com]    投影后的 patch 特征
            geometry_emb: [B, 576, 576, dim_g] 预计算的两两相对位置编码 (float16)

        Returns:
            x: [B, 576, dim_com] 解耦位置纠缠后的 patch 表示, 送给 v2s/s2v 共用
        """
        # ---- 第一步: 几何感知自注意力 ----
        # 注意: GeometryMultiHeadAttention 内部已经做了 Add(x) + LayerNorm
        # 所以这里直接用返回值, 不再手动加残差
        x = self.attn(x, geometry_emb)

        # ---- 第二步: FFN + 残差 + LayerNorm ----
        # 标准 Transformer 第二级残差结构
        # ffn(x) 学逐 patch 的非线性变换, dropout 提供正则化
        # x + dropout(ffn(x)) 保留原信息, LayerNorm 稳定数值
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
                 text_residual=0.2, visual_residual=0.2, use_fae=True):
        super().__init__()
        self.dim_f = dim_f
        self.dim_com = dim_com
        self.weight_s2v = weight_s2v
        self.text_residual   = text_residual    # v2s 文本残差系数 (保护 unseen 文本)
        self.visual_residual = visual_residual  # s2v 视觉残差系数 (保护 CLIP 原始视觉)
        self.use_fae = use_fae                  # ★ FAE 消融开关 (False 时跳过 FAE)

        # 输入投影 768 → dim_com (两路共享视觉投影, 文本投影独立)
        self.embed_cv = nn.Linear(dim_f, dim_com)       # 共享视觉投影
        self.embed_text = nn.Linear(dim_f, dim_com)     # 共享文本投影

        # 几何位置编码 (预计算, 不训练) - 仅在 use_fae=True 时使用
        if self.use_fae:
            self.box_emb = BoxRelationalEmbedding(grid_size=grid_size, dim_g=dim_g)
            # FAE: 共享, 两路看到一致的视觉理解 (减少参数 + 提升耦合)
            self.fae = FAELayer(dim_com, heads, dropout, dim_g=dim_g)
        else:
            self.box_emb = None
            self.fae = None

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

        # ★ Attention Pooling: 替代 mean(dim=1)，让模型学哪些 patch 重要
        self.attn_pool = nn.Linear(dim_com, 1)

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

        # ========== 共享视觉编码 ==========
        # 两路共用同一个 memory，视觉理解一致，节省参数
        vis    = self.embed_cv(patches)                          # [B, 576, dim_com]
        if self.use_fae:
            geo_emb = self.box_emb(B)                            # [B, 576, 576, dim_g]
            memory = self.fae(vis, geo_emb)                      # [B, 576, dim_com]
        else:
            # 消融 FAE：直接用线性投影后的视觉表示
            memory = vis                                          # [B, 576, dim_com]

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

        # ========== 增强后的视觉/文本表示 (cosine_only 模式用) ==========
        # v_enh: 视觉端经 s2v 池化 + 投影回 768
        v_enh_512 = F_p_s2v.mean(dim=1)                          # [B, dim_com]
        v_enh = self.proj_visual(v_enh_512)                      # [B, dim_f=768]

        # t_enh: 文本端 (每张图独立) + 投影回 768
        t_enh = self.proj_text(F_p_v2s)                          # [B, N_cls, dim_f=768]

        return {
            'local_score': local_score,
            'v_enh': v_enh,        # [B, 768]
            't_enh': t_enh,        # [B, N_cls, 768]
        }


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
        use_fae       = getattr(config, 'use_fae', True)         # ★ FAE 消融开关
        self.local_weight = getattr(config, 'local_weight', 0.3)  # 局部分数加权系数
        # ★ 计分模式
        # 'add' (默认): logits = base_logits + β × local_score
        # 'cosine_only': logits = cosine(v_enh, t_enh) × scale
        self.score_mode = getattr(config, 'score_mode', 'add')
        self.visual_residual = visual_residual
        self.text_residual   = text_residual
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
            use_fae=use_fae,
        )

        # ---- 温度系数 (可学习) ----
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # ---- CIG (Confidence-Inverse Gating) 动态门控 ----
        # 设计思想：gate 不学习 cls_token → MLP，而是 CLIP 自身置信度的函数
        #   gate = alpha * (1 - max_prob) ** tau
        #   max_prob 在全 200 类（含 unseen）上 softmax，绕开"监督盲区"
        #   CLIP 自信 → conf 高 → gate 低 → base 主导
        #   CLIP 不自信 → conf 低 → gate 高 → local_score 救援
        #
        # 仅 2 个可学标量：alpha (整体强度), tau (曲线形状)
        # 对 seen / unseen 类完全对称，不会塌缩成全局降权
        self.gate_alpha = nn.Parameter(torch.tensor(1.0))   # 缩放系数
        self.gate_tau   = nn.Parameter(torch.tensor(1.0))   # 温度

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

        # ── 双分支输出 ──
        cm_out = self.cross_tf(patches, all_text, cls_token)
        local_score = cm_out['local_score']                      # [B, 200]

        # ── 计分模式选择 ──
        # 'add' (默认): logits = base_logits + β × local_score      （CrossModal 是补丁）
        # 'cosine_only': logits = cosine(v_enh, t_enh) × scale      （CrossModal 唯一决定）
        if self.score_mode == 'cosine_only':
            v_enh_raw = cm_out['v_enh']                          # [B, 768]
            t_enh_raw = cm_out['t_enh']                          # [B, 200, 768]
            B_size = patches.size(0)

            # ===== 视觉端残差融合 =====
            # visual_residual: 用 cls 残差防止视觉端被训歪
            if cls_token is not None:
                v_enh = (1.0 - self.visual_residual) * v_enh_raw + \
                        self.visual_residual * cls_token         # [B, 768]
            else:
                v_enh = v_enh_raw

            # ===== 文本端: seen 列增强, unseen 列保留 CLIP 原始 (方案 3) =====
            # 关键: unseen 类的 t_enh 经过的 proj_text 在训练时无监督
            #       会变成随机噪声 → 直接用 all_text 绕过可训练层
            # seen 列: 经 v2s 增强 (有训练监督)
            # unseen 列: 直接 all_text (CLIP 原始, 不经任何可训练层)
            t_enh_seen = (1.0 - self.text_residual) * t_enh_raw + \
                         self.text_residual * all_text.unsqueeze(0)  # [B, 200, 768]

            # 构造 mask: unseen 位置全用 all_text
            t_enh = all_text.unsqueeze(0).expand(B_size, -1, -1).contiguous()  # [B, 200, 768]
            t_enh = t_enh.clone()                                # 避免修改 expand 的 view
            t_enh[:, self.seenclass, :] = t_enh_seen[:, self.seenclass, :]
            # unseen 50 列 = all_text (零更新, 完全冻结)

            # 余弦相似度 + 温度缩放
            v_n = F.normalize(v_enh, dim=-1).unsqueeze(1)        # [B, 1, 768]
            t_n = F.normalize(t_enh, dim=-1)                     # [B, 200, 768]
            logits_200 = (v_n * t_n).sum(dim=-1) * logit_scale   # [B, 200]
        else:
            # ── add 模式 (CIG 版: Confidence-Inverse Gating) ──
            # 动态门控: 用 CLIP 在全 200 类上的置信度反推 gate
            # 关键设计:
            #   1. base_logits.detach() 避免 gate 梯度干扰 base 分支
            #   2. softmax 在全 200 类上算, unseen 类也参与 → 绕过监督盲区
            #   3. gate 是无参标量函数, 不会塌缩成全局降权
            #   4. alpha/tau 仅 2 个可学参数, 微调强度和形状
            with torch.no_grad():
                prob = F.softmax(base_logits, dim=-1)            # [B, 200]
                conf = prob.max(dim=-1).values                   # [B]
            uncertainty = (1.0 - conf).clamp(min=0.0, max=1.0)   # [B]
            # 用 softplus 保证 alpha/tau 始终为正且平滑
            alpha = F.softplus(self.gate_alpha)                  # 标量 > 0
            tau   = F.softplus(self.gate_tau)                    # 标量 > 0
            gate  = (alpha * uncertainty.pow(tau)).unsqueeze(-1) # [B, 1]
            logits_200 = base_logits + gate * local_score

        if is_train:
            logits = logits_200[:, self.seenclass]               # [B, 150]
        else:
            logits = logits_200                                  # [B, 200]

        return {
            'logits':      logits,
            'logits_200':  logits_200,   # compute_loss 里 calibration loss 用
            'base_logits': base_logits,  # [B, 200] CLIP 余弦, 用于一致性 loss
            'local_score': local_score,  # [B, 200] 双分支输出, 用于一致性 loss
            'clip_S_pp':   logits,
            'clip_pred':   logits,
        }

    def compute_loss(self, in_package):
        """
        训练 Loss = CE loss
                  + λ_cal     × self-calibration loss
                  + λ_consist × consistency loss (KL between base / local on seen)
                  + λ_l2sp    × adapter L2-SP regularization

        - CE: 主分类损失 (仅 seen 150 类)
        - cal: 强推 unseen 概率 (实测无效, 默认关闭)
        - consist: 防止 local_score 学出和 CLIP base 矛盾的反向信号
        - l2sp: 限制 adapter 输出偏离 CLIP 原始嵌入的程度
        """
        logits = in_package['logits']        # [B, 150] 训练时只有 seen 列
        labels = in_package['batch_label']   # [B] 全局索引

        if labels.dim() > 1:
            labels = torch.argmax(labels, dim=1)

        # 全局索引 → seen 局部索引 (0~149)
        # ★ CrossEntropy 要求 target 必须是 LongTensor
        seen_labels = torch.zeros_like(labels, dtype=torch.long)
        for i, cls_idx in enumerate(self.seenclass):
            seen_labels[labels == cls_idx] = i

        # ========== CE loss (主损失) ==========
        loss_CE = F.cross_entropy(logits, seen_labels)
        loss = loss_CE

        # ========== Self-calibration loss (辅助损失, 实测无效) ==========
        logits_200 = in_package.get('logits_200', None)
        loss_cal = torch.tensor(0.0, device=logits.device)
        if logits_200 is not None and self.config.__dict__.get('lambda_cal', 0) > 0:
            prob_all    = F.softmax(logits_200, dim=-1)              # [B, 200]
            prob_unseen = prob_all[:, self.unseenclass]              # [B, 50]
            mass_unseen = prob_unseen.sum(dim=1)                     # [B]
            loss_cal    = -torch.log(mass_unseen.mean() + 1e-8)
            loss = loss + self.config.lambda_cal * loss_cal

        # ========== Consistency loss (一致性正则) ==========
        # 让 local_score 在 seen 类上的排序与 base_logits 一致
        # 防止双分支学出和 CLIP 矛盾的反向信号
        base_logits = in_package.get('base_logits', None)
        local_score = in_package.get('local_score', None)
        loss_consist = torch.tensor(0.0, device=logits.device)
        lambda_consist = self.config.__dict__.get('lambda_consist', 0)
        if (base_logits is not None and local_score is not None
                and lambda_consist > 0):
            T = self.config.__dict__.get('consist_temp', 2.0)
            # 仅在 seen 列计算, unseen 列没有可学梯度反传过来无意义
            base_seen  = base_logits[:, self.seenclass].detach()     # [B, 150]
            local_seen = local_score[:, self.seenclass]              # [B, 150]
            base_p     = F.softmax(base_seen / T, dim=-1)
            local_logp = F.log_softmax(local_seen / T, dim=-1)
            loss_consist = F.kl_div(
                local_logp, base_p, reduction='batchmean') * (T * T)
            loss = loss + lambda_consist * loss_consist

        # ========== Adapter L2-SP (防漂移) ==========
        # 让 adapter 输出 (delta) 别离 0 太远
        # adapter_ratio 已经做了一部分残差混合, 这里加直接的 L2 约束
        loss_l2sp = torch.tensor(0.0, device=logits.device)
        lambda_l2sp = self.config.__dict__.get('lambda_l2sp', 0)
        if lambda_l2sp > 0:
            delta = self.text_adapter(self.seen_text_embeds)         # [150, 768]
            loss_l2sp = (delta ** 2).mean()
            loss = loss + lambda_l2sp * loss_l2sp

        return {
            'loss': loss,
            'loss_CE': loss_CE,
            'loss_cal': loss_cal,
            'loss_consist': loss_consist,
            'loss_l2sp': loss_l2sp,
        }
