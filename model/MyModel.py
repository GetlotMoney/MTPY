"""
VGSR 主模型 (CLIP + Adapter + GPT + 双向 Transformer)
=========================================================
模块化说明：本文件内包含 VGSR 的核心建模组件与轻量辅助目标。
  1. LaSt-ViT 工具函数           — 频域池化与 top-K patch 选择
  2. Adapter                    — 文本特征轻量增强 (VDT-TransZero)
  3. CrossModalTransformer      — 视觉-语义双向 Transformer (TransZero++)
     包含子组件: BoxRelationalEmbedding, GeometryMultiHeadAttention, FAELayer
     支持 LaSt-ViT patch 子集选择后再进行几何感知交互
  4. VGSR                       — 主模型, 组合上述模块
     可选启用 LaSt-CLS 增强、LaSt patch 选择器与 AG-JEPA 辅助预测损失

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
#  工具函数: s2v 池化方法 (mean / attention / lastvit)
# ==========================================================

def _gaussian_kernel_1d(length, sigma):
    """1D 高斯核, 用于 LaSt-ViT 频域低通滤波

    ★ 严格按官方 LAST-ViT (ChengShiest/LAST-ViT/cls_pretrain/conf.py):
      - 中心化: x ∈ [-length/2+1, length/2]
      - max-归一化 (max=1), 不是 sum-归一化
      - 在频域里直接乘核 = "保留低频, 衰减高频"的低通滤波
    """
    x = torch.arange(-length // 2 + 1, length // 2 + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / torch.max(kernel)   # ★ 官方写法 /max, 不是 /sum


def lastvit_pool(F_p, k=1, sigma=None):
    """
    LaSt-ViT 频域 Top-K patch 池化 (★ 严格对齐官方 ChengShiest/LAST-ViT/cls_pretrain/conf.py)

    机制 (无可学参数):
      1. FFT 沿特征维 → 频域
      2. 高斯低通滤波 → 平滑版 x_lp
      3. 打分: diff = x_detach / |x_lp - x_detach|, 形状保持 [B, N, D]
      4. ★ 在 [B, N, D] 上 topk(dim=1, k=k), 每个 channel 独立选 top-k patch
      5. gather 后 mean(dim=1) 得到 [B, D]
      
    与之前实现的关键区别:
      ❌ 旧版: diff.mean(dim=-1) 先把 D 维平均成 [B, N], 再 topk → 全图共用 top-k
      ❌ 旧版: 加了 .abs() 取绝对值 → 偏离官方
      ✅ 新版: 不 mean(-1) 不 abs(), 在 [B, N, D] 上直接 topk → per-channel top-k
                每个 channel 关注自己最显著的 patch (part-aware)
                对 CUB 细粒度 (喙/翅膀/尾各自判别) 更合适

    论文: Vision Transformers Need More Than Registers (LAST-ViT)
    GitHub: ChengShiest/LAST-ViT (cls_pretrain/conf.py)

    Args:
        F_p:   [B, N, D] patch 特征
        k:     选 top-K 个 patch (官方默认 k=1)
        sigma: 高斯核标准差 (None=官方默认 √D)

    Returns:
        [B, D] 池化后的全局特征
    """
    B, N, D = F_p.shape

    # ★ 关键: torch.fft.* 在 CUDA 上不支持 fp16 (autocast 下会死锁)
    # 强制 FFT 部分走 fp32, 算完再转回原 dtype
    orig_dtype = F_p.dtype
    F_p_fp32 = F_p.float() if orig_dtype != torch.float32 else F_p

    # 1. 缓存原始特征 (官方写法: 变量名 x_detach 但其实没 detach, 是直接引用)
    x_detach = F_p_fp32

    # 2. FFT 沿特征维
    x_freq = torch.fft.fft(F_p_fp32, dim=-1)

    # 3. 高斯低通 (sigma 默认 √D)
    if sigma is None:
        sigma = D ** 0.5
    gs_k = _gaussian_kernel_1d(D, sigma).to(F_p_fp32.device)   # [D] fp32
    x_freq = torch.fft.fftshift(x_freq, dim=-1)
    x_freq = x_freq * gs_k                                      # 广播乘
    x_freq = torch.fft.ifftshift(x_freq, dim=-1)
    x_lp = torch.fft.ifft(x_freq, dim=-1).real                  # [B, N, D]

    # 4. ★ 严格官方公式: 不取 abs, 不 mean(-1), 保持 [B, N, D]
    diff = x_detach / (torch.abs(x_lp - x_detach) + 1e-6)       # [B, N, D]

    # 5. ★ per-channel topk: 每个 channel 独立选自己最显著的 k 个 patch
    _, indices = torch.topk(diff, k=k, dim=1, largest=True)     # [B, k, D]
    sel_p = torch.gather(x_detach, 1, indices)                  # [B, k, D]

    # 6. K 个 patch 平均 (k=1 时等价于 squeeze)
    pooled = sel_p.mean(dim=1)                                  # [B, D]

    # 7. 转回原 dtype (autocast 兼容)
    return pooled.to(orig_dtype) if orig_dtype != torch.float32 else pooled


def lastvit_select_patches(F_p, K=64, sigma=None, largest=True, formula='v2_abs_mean'):
    """
    LaSt-ViT v5 (路径 C): 用频域显著性选 top-K patch 索引

    第一性原理 (★ 2026-05-31 勘误):
        报告原文: "比值越高的 Patch 说明在通道内越平稳, 越符合真实前景对象"
        diff = x / (|x_lp - x| + ε): 比值大 → 高频残差小 → patch 平稳同质
        topk(largest=True) 选最平稳的 K 个 token (滤掉 high-norm artifact / 背景 lazy token)
        topk(largest=False) 选最不平稳的 K 个 token (高频/纹理 part-aware token)
        ★ 反向对照实验用 largest=False 验证机制
        ★ 'both' 模式: 选 K/2 个最平稳前景 + K/2 个最高频 part, 双通道融合

    Args:
        F_p:     [B, N, D] patch 特征 (CLIP 原始 patches)
        K:       选 top-K 个 patch (推荐 64 或 128). both 模式下 K 必须是偶数
        sigma:   高斯核 σ (None=√D)
        largest: True=选最平稳前景 (默认), False=选最高频 part, 'both'=K/2 平稳 + K/2 高频
        formula: 'v1_strict' / 'v2_abs_mean' (默认) / 'v3_norm'

    Returns:
        topk_indices: [B, K]  long tensor, 每个样本选的 K 个 patch 索引
        patch_score:  [B, N]  float tensor, 每个 patch 的得分 (供调试)
    """
    B, N, D = F_p.shape

    # 强制 fp32 (FFT 不支持 fp16)
    orig_dtype = F_p.dtype
    F_p_fp32 = F_p.float() if orig_dtype != torch.float32 else F_p

    # 1. FFT + 高斯低通
    x_freq = torch.fft.fft(F_p_fp32, dim=-1)
    if sigma is None:
        sigma = D ** 0.5
    gs_k = _gaussian_kernel_1d(D, sigma).to(F_p_fp32.device)
    x_freq = torch.fft.fftshift(x_freq, dim=-1)
    x_freq = x_freq * gs_k
    x_freq = torch.fft.ifftshift(x_freq, dim=-1)
    x_lp = torch.fft.ifft(x_freq, dim=-1).real

    # 2. 计算 patch 得分 (3 种公式)
    if formula == 'v1_strict':
        # 报告原文: S_i = (1/D) Σ_d  x[i,d] / (|x_lp[i,d] - x[i,d]| + ε)
        diff = F_p_fp32 / (torch.abs(x_lp - F_p_fp32) + 1e-6)   # [B, N, D]
        patch_score = diff.mean(dim=-1)                          # [B, N]
    elif formula == 'v3_norm':
        patch_score = 1.0 / (torch.norm(x_lp - F_p_fp32, dim=-1) + 1e-6)  # [B, N]
    else:
        # V2 默认 (P3.x 都用): abs() + mean
        diff = F_p_fp32 / (torch.abs(x_lp - F_p_fp32) + 1e-6)   # [B, N, D]
        patch_score = diff.abs().mean(dim=-1)                    # [B, N]

    # 3. 选 top-K patch 索引 (3 种方向)
    if isinstance(largest, str) and largest.lower() == 'both':
        # ★ 双向选择: K/2 个最平稳前景 + K/2 个最高频 part
        K_half = K // 2
        _, idx_top = torch.topk(patch_score, k=K_half, dim=1, largest=True)   # 平稳前景
        _, idx_bot = torch.topk(patch_score, k=K-K_half, dim=1, largest=False) # 高频 part
        topk_indices = torch.cat([idx_top, idx_bot], dim=1)                    # [B, K]
    else:
        _, topk_indices = torch.topk(patch_score, k=K, dim=1, largest=bool(largest))

    return topk_indices, patch_score


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
                 text_residual=0.2, visual_residual=0.2, use_fae=True,
                 pool_method='mean', lastvit_k=8, lastvit_sigma=10.0):
        super().__init__()
        self.dim_f = dim_f
        self.dim_com = dim_com
        self.weight_s2v = weight_s2v
        self.text_residual   = text_residual    # v2s 文本残差系数 (保护 unseen 文本)
        self.visual_residual = visual_residual  # s2v 视觉残差系数 (保护 CLIP 原始视觉)
        self.use_fae = use_fae                  # ★ FAE 消融开关 (False 时跳过 FAE)

        # ★ 池化方法开关 (s2v 全局池化方式)
        # 'mean'      → 576 个 patch 平均 (基线 H=72.24%)
        # 'attention' → 学一个 Linear 给每 patch 打分 (实测 H=71.83%, -0.37%)
        # 'lastvit'   → 频域 Top-K 选择 (无参数, 待验证)
        self.pool_method = pool_method
        self.lastvit_k = lastvit_k
        self.lastvit_sigma = lastvit_sigma
        # ★ H2 dmp fixed 模式下的 lambda 标量 (mlp 模式时被外部 pool_lambda_dyn 覆盖)
        # 默认 0.5 等权融合 mean+LaSt
        self.pool_lambda_fixed = 0.5

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

        # ★ Attention Pooling 模块 (仅 pool_method='attention' 时调用)
        # 学一个 Linear, 给每个 patch 打分, softmax 后加权求和
        # 实测 H=71.83% (-0.37%), 但保留作为对照实验
        if self.pool_method == 'attention':
            self.attn_pool = nn.Linear(dim_com, 1)

    def forward(self, patches, text, cls_token=None, unseen_idx=None,
                attr_text=None,
                weight_s2v_dyn=None, pool_lambda_dyn=None,
                lastvit_select_k=0, lastvit_select_sigma=0.0,
                lastvit_select_largest=True,
                lastvit_select_formula='v2_abs_mean'):
        """
        Args:
            patches:   [B, 576, 768]  CLIP patch 特征 (不含 CLS)
            text:      [N_cls, 768]   原始文本特征
            cls_token: [B, 768]       CLIP CLS token (保留接口兼容)
            attr_text: [N_attr, 768]   可选属性文本原型, MOD-001 属性路由辅助项用
            unseen_idx: [N_unseen]    可选; 若提供, t_enh 在 proj_text
                                       前对 unseen 列 detach, 防止 unseen
                                       无监督梯度流入 decoder_v2s/FAE 上层
            lastvit_select_k: int     ★ v5 路径 C: LaSt patch 选择器 K. 0=不启用
            lastvit_select_sigma: float  高斯核 σ, 0=自动用 √D
        Returns:
            local_score: [B, N_cls]   双分支产生的额外局部注意力分数
        """
        B = patches.size(0)

        # ========== ★ v5: LaSt patch 选择器 (路径 C) ==========
        # 第一性原理: LaSt 在 frozen CLIP 下不能用反向梯度调整 ViT, 改成做 patch 选择器
        # 让 FAE 仅在 K 个 part-aware patch 上做 cross-modal 交互, 而不是全 576
        # K=64 / 128 推荐起点, 比 mean pool 信息密度更高
        topk_indices_v5 = None    # ★ 保存到 FAE 子集 gather 用
        if lastvit_select_k > 0 and lastvit_select_k < patches.size(1):
            sigma_sel = lastvit_select_sigma if lastvit_select_sigma > 0 else None
            with torch.amp.autocast('cuda', enabled=False):
                topk_indices, _ = lastvit_select_patches(
                    patches.float(), K=lastvit_select_k, sigma=sigma_sel,
                    largest=lastvit_select_largest,
                    formula=lastvit_select_formula)
            # gather 出 [B, K, 768] 的 part-aware patches
            # 注意: indices 是 [B, K], 需要 expand 到最后一维
            idx_exp = topk_indices.unsqueeze(-1).expand(-1, -1, patches.size(-1))
            patches = torch.gather(patches, dim=1, index=idx_exp)        # [B, K, 768]
            topk_indices_v5 = topk_indices                                # [B, K] 留给 FAE

        # ========== 共享视觉编码 ==========
        # 两路共用同一个 memory，视觉理解一致，节省参数
        vis    = self.embed_cv(patches)                          # [B, N, dim_com]  N=576 或 K
        if self.use_fae:
            if patches.size(1) == 576:
                # 标准路径: 全图 FAE
                geo_emb = self.box_emb(B)                                # [B, 576, 576, dim_g]
                memory  = self.fae(vis, geo_emb)                         # [B, 576, dim_com]
            elif topk_indices_v5 is not None:
                # ★ v5 + FAE 子集 gather (路径 C 修复):
                #   LaSt 选出的 K 个 patch 仍然是 24×24 grid 上的合法点,
                #   它们之间两两的相对位置编码 = 全图 [576,576,dim_g] 的子集.
                #   advanced indexing 出 [B, K, K, dim_g] 给 FAE 做几何感知自注意力.
                full   = self.box_emb.geometry_embedding                  # [576, 576, dim_g] (fp16)
                K_sel  = topk_indices_v5.size(1)
                i_idx  = topk_indices_v5.unsqueeze(-1).expand(-1, -1, K_sel)  # [B, K, K]
                j_idx  = topk_indices_v5.unsqueeze(-2).expand(-1, K_sel, -1)  # [B, K, K]
                geo_emb = full[i_idx, j_idx]                              # [B, K, K, dim_g]
                memory  = self.fae(vis, geo_emb)                          # [B, K, dim_com]
            else:
                # patches 异常 (既非 576 也非 v5 选择), 不走 FAE
                memory = vis
        else:
            # 消融 FAE: 直接用线性投影后的视觉表示
            memory = vis                                                  # [B, N, dim_com]

        txt_com   = self.embed_text(text)                        # [N_cls, dim_com]
        txt_batch = txt_com.unsqueeze(0).expand(B, -1, -1)       # [B, N_cls, dim_com]

        # ========== MOD-001: 几何感知属性路由辅助分数 ==========
        # 用 FAE 后的 geometry-aware local memory 去匹配属性文本原型。
        # 对每个属性取最相关 patch 的相似度, 得到 [B, N_attr]。
        attr_route_score = None
        if attr_text is not None:
            attr_com = self.embed_text(attr_text.to(device=memory.device,
                                                    dtype=text.dtype))  # [N_attr, dim_com]
            mem_n = F.normalize(memory.float(), dim=-1)
            attr_n = F.normalize(attr_com.float(), dim=-1)
            patch_attr_sim = mem_n @ attr_n.T                           # [B, N_patch, N_attr]
            attr_route_score = patch_attr_sim.max(dim=1).values.to(memory.dtype)

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
        # ★ 四种池化方式可切换 (yaml: pool_method)
        if self.pool_method == 'class_attention':
            # ★★ P1: Class-aware Attention Pool (零新参数, 用 200 类 txt_com 当 query) ★★
            # 每个类用自己关心的 patch 加权 pool, 细粒度任务 (CUB 鸟) 精准对应 part
            #   F_p_s2v [B, 576, 512]  txt_com [N_cls, 512]
            scale = self.dim_com ** 0.5
            attn = (F_p_s2v @ txt_com.T) / scale                  # [B, 576, N_cls]
            attn_weights = F.softmax(attn, dim=1)                 # [B, 576, N_cls]  每类对每 patch
            # ★ 显存优化: matmul 等价 broadcast+sum, 避免 [B,576,N_cls,512]=15GB 中间张量
            #   sum_p attn[b,p,c]·F[b,p,d] = (attn^T @ F) [B, N_cls, 512]
            s2v_pooled_per_cls = attn_weights.transpose(1, 2) @ F_p_s2v   # [B, N_cls, 512]
            s2v_n_per_cls = F.normalize(s2v_pooled_per_cls, dim=-1)
            txt_n         = F.normalize(txt_com, dim=-1)          # [N_cls, 512]
            score_s2v     = (s2v_n_per_cls * txt_n.unsqueeze(0)).sum(dim=-1)  # [B, N_cls]
            # cosine_only 仍需要单一 s2v_pooled (取 batch 平均的类无关版本作为代表, 兼容下游 v_enh)
            s2v_pooled = F_p_s2v.mean(dim=1)
        elif self.pool_method == 'attention':
            # Attention Pooling: 学一个 Linear 给每 patch 打分 (实测 -0.37%)
            attn_logits  = self.attn_pool(F_p_s2v)             # [B, 576, 1]
            attn_weights = F.softmax(attn_logits, dim=1)
            s2v_pooled   = (F_p_s2v * attn_weights).sum(dim=1) # [B, dim_com]
            s2v_n      = F.normalize(s2v_pooled, dim=-1)
            txt_single = F.normalize(txt_com, dim=-1)
            score_s2v  = s2v_n @ txt_single.T
        elif self.pool_method == 'lastvit':
            # LaSt-ViT 频域 Top-K (无参数, 方案 16)
            # sigma=0 时让 lastvit_pool 内部用 √D 自动计算 (官方设定)
            sigma = self.lastvit_sigma if self.lastvit_sigma > 0 else None
            # ★ FFT 不支持 fp16, 强制 autocast 在此处关闭 (双保险)
            # 内部 lastvit_pool 也会转 fp32, 但显式包一层避免外层调用栈被污染
            with torch.amp.autocast('cuda', enabled=False):
                s2v_pooled = lastvit_pool(F_p_s2v.float(),
                                          k=self.lastvit_k,
                                          sigma=sigma)
            # 转回原 dtype 以保持后续计算一致 (autocast 上下文外)
            s2v_pooled = s2v_pooled.to(F_p_s2v.dtype)
            s2v_n      = F.normalize(s2v_pooled, dim=-1)
            txt_single = F.normalize(txt_com, dim=-1)
            score_s2v  = s2v_n @ txt_single.T
        elif self.pool_method == 'dmp':
            # ★★ H2: Dynamic Mean-LaSt Pooling (双专家并联 + 动态融合) ★★
            # mean pool 做稳定全局兜底, LaSt pool 提供频域选择性
            # λ_pool(x) = sigmoid(MLP_pool(cls)) 由外部传入 pool_lambda_dyn
            # z = (1-λ)·z_mean + λ·z_last
            # 优点: LaSt 单独失败 (-0.51) 但作辅助专家稳, mean 永远兜底
            #
            # 1. mean pool (稳定专家)
            z_mean = F_p_s2v.mean(dim=1)                          # [B, dim_com]
            # 2. LaSt pool (频域专家)
            sigma = self.lastvit_sigma if self.lastvit_sigma > 0 else None
            with torch.amp.autocast('cuda', enabled=False):
                z_last = lastvit_pool(F_p_s2v.float(),
                                      k=self.lastvit_k,
                                      sigma=sigma)
            z_last = z_last.to(F_p_s2v.dtype)                     # [B, dim_com]
            # 3. 动态融合 (外部传入 λ, 或 fallback 到固定 0.5)
            if pool_lambda_dyn is not None:
                lam = pool_lambda_dyn                             # [B, 1]
            else:
                # fallback: yaml pool_lambda_fixed 标量, 或默认 0.5
                lam_val = float(getattr(self, 'pool_lambda_fixed', 0.5))
                lam = torch.full((F_p_s2v.size(0), 1),
                                  lam_val, device=F_p_s2v.device,
                                  dtype=F_p_s2v.dtype)
            s2v_pooled = (1.0 - lam) * z_mean + lam * z_last      # [B, dim_com]
            s2v_n      = F.normalize(s2v_pooled, dim=-1)
            txt_single = F.normalize(txt_com, dim=-1)
            score_s2v  = s2v_n @ txt_single.T
        else:
            # 默认 mean (基线 H=72.24%)
            s2v_pooled = F_p_s2v.mean(dim=1)                   # [B, dim_com]
            s2v_n      = F.normalize(s2v_pooled, dim=-1)
            txt_single = F.normalize(txt_com, dim=-1)
            score_s2v  = s2v_n @ txt_single.T

        # ========== 加权融合 (支持外部传入动态 w) ==========
        # weight_s2v_dyn: [B, 1] 来自 cls_token-driven MLP, None 时 fallback 到 self.weight_s2v 标量
        if weight_s2v_dyn is not None:
            w = weight_s2v_dyn                                   # [B, 1]
            local_score = w * score_s2v + (1.0 - w) * score_v2s   # [B, N_cls]
        else:
            local_score = self.weight_s2v * score_s2v + \
                          (1.0 - self.weight_s2v) * score_v2s     # [B, N_cls]

        # ========== 增强后的视觉/文本表示 (cosine_only 模式用) ==========
        # v_enh: 视觉端经 s2v 池化 + 投影回 768 (复用上面池化结果)
        v_enh_512 = s2v_pooled                                   # [B, dim_com]
        v_enh = self.proj_visual(v_enh_512)                      # [B, dim_f=768]

        # t_enh: 文本端 (每张图独立) + 投影回 768
        # 注: PyTorch 自动微分确保 unseen 列即使 forward 也无梯度回流
        # (loss 只用 seen logits → t_enh[unseen] 输入梯度严格为 0, 实测验证)
        # 所以无需在此处 detach, 简单 proj_text 整体即可
        t_enh = self.proj_text(F_p_v2s)                          # [B, N_cls, 768]

        return {
            'local_score': local_score,
            'v_enh': v_enh,        # [B, 768]
            't_enh': t_enh,        # [B, N_cls, 768]
            # ★ 单独导出双分支分数, cosine_only + 辅助 CE 用
            'score_s2v': score_s2v,  # [B, N_cls]
            'score_v2s': score_v2s,  # [B, N_cls]
            'attr_route_score': attr_route_score,  # [B, N_attr] or None
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
                 unseen_text_embeds,     # [50,  768]
                 class_attr=None,        # [200, 312]
                 attr_text_embeds=None): # [312, 768]
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

        # ---- MOD-001: 几何感知属性路由所需的固定属性材料 ----
        self.use_geo_attr_routing = bool(getattr(config, 'use_geo_attr_routing', False))
        if class_attr is not None:
            self.register_buffer('class_attr', class_attr.float(), persistent=False)
        else:
            self.class_attr = None
        if attr_text_embeds is not None:
            self.register_buffer(
                'attr_text_embeds',
                F.normalize(attr_text_embeds.float(), dim=1),
                persistent=False)
        else:
            self.attr_text_embeds = None
        if self.use_geo_attr_routing and (
                self.class_attr is None or self.attr_text_embeds is None):
            raise ValueError(
                "use_geo_attr_routing=True requires class_attr and attr_text_embeds.")

        # ---- MOD-002: 拓扑感知的自适应文本属性库 ----
        self.use_text_attr_reservoir = bool(
            getattr(config, 'use_text_attr_reservoir', False))
        if self.use_text_attr_reservoir and (
                self.class_attr is None or self.attr_text_embeds is None):
            raise ValueError(
                "use_text_attr_reservoir=True requires class_attr and attr_text_embeds.")
        if self.use_text_attr_reservoir:
            reservoir_hidden = int(getattr(config, 'text_attr_reservoir_hidden', 256))
            self.text_attr_reservoir_proj = nn.Sequential(
                nn.Linear(self.dim_f, reservoir_hidden),
                nn.GELU(),
                nn.Linear(reservoir_hidden, self.dim_f),
            )
            # zero init: 打开模块的训练初始点仍等价于未加 reservoir 的文本原型。
            with torch.no_grad():
                self.text_attr_reservoir_proj[-1].weight.zero_()
                self.text_attr_reservoir_proj[-1].bias.zero_()

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
        # ★ 门控模式开关
        # 'fixed' (基线 H=72.26%): logits = base + local_weight × local_score
        # 'cig'   (动态门控):       logits = base + α×(1-conf)^τ × local_score
        self.gating = getattr(config, 'gating', 'fixed')
        # ★ 计分模式
        # 'add' (默认): logits = base_logits + β × local_score
        # 'cosine_only': logits = cosine(v_enh, t_enh) × scale
        self.score_mode = getattr(config, 'score_mode', 'add')
        self.visual_residual = visual_residual
        self.text_residual   = text_residual
        # ★ 池化方法 (s2v 全局池化方式)
        pool_method   = getattr(config, 'pool_method', 'mean')
        lastvit_k     = getattr(config, 'lastvit_k', 8)
        lastvit_sigma = getattr(config, 'lastvit_sigma', 10.0)
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
            pool_method=pool_method,
            lastvit_k=lastvit_k,
            lastvit_sigma=lastvit_sigma,
        )
        # ★ H2 dmp fixed 模式 lambda 标量 (从 yaml 读)
        self.cross_tf.pool_lambda_fixed = float(
            getattr(config, 'pool_lambda_fixed', 0.5))

        # AG-JEPA: predict masked discriminative patch features from context + class text.
        self.use_ag_jepa = bool(getattr(config, 'use_ag_jepa', False))
        self.jepa_topk = int(getattr(config, 'jepa_topk', 8))
        self.jepa_neg_margin = float(getattr(config, 'jepa_neg_margin', 0.2))
        if self.use_ag_jepa:
            jepa_hidden = int(getattr(config, 'jepa_hidden', tf_common_dim))
            self.jepa_predictor = nn.Sequential(
                nn.Linear(tf_common_dim * 2, jepa_hidden),
                nn.LayerNorm(jepa_hidden),
                nn.GELU(),
                nn.Linear(jepa_hidden, tf_common_dim),
            )

        # ---- 温度系数 (可学习) ----
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # ★ LaSt-ViT-CLS 增强 (CVPR 2026, 替代 CLIP 原 lazy CLS)
        # 论文: Vision Transformers Need More Than Registers
        # 思想: ViT 的 CLS 倾向于聚合背景 patch 作为 shortcut, LaSt-ViT 用频域
        #       per-channel topk 选 part-aware patch 替代 CLS
        # 应用位置: CLIP 原 768 维 patches → 频域池化 → 增强 CLS
        # 残差融合: cls = (1-r)·CLIP_CLS + r·LaSt_CLS, 保留 CLIP zero-shot 兜底
        self.use_lastvit_cls   = bool(getattr(config, 'use_lastvit_cls', False))
        self.lastvit_cls_k     = int(getattr(config, 'lastvit_cls_k', 1))
        self.lastvit_cls_sigma = float(getattr(config, 'lastvit_cls_sigma', 0.0))
        self.lastvit_residual  = float(getattr(config, 'lastvit_residual', 0.5))

        # ★ v5 路径 C: LaSt patch 选择器配置
        # lastvit_select_k > 0 启用, 推荐 64 或 128
        # 0 = 关闭 (FAE 看全 576 patches, 跟 baseline 一样)
        self.lastvit_select_k     = int(getattr(config, 'lastvit_select_k', 0))
        self.lastvit_select_sigma = float(getattr(config, 'lastvit_select_sigma', 0.0))
        # ★ 2026-05-31 P3.6 反向对照 + 双向选择:
        #   True/False = 选平稳前景 / 高频 part
        #   'both' = K/2 平稳 + K/2 高频, 双通道融合 (P3.9 方案 A)
        _largest_raw = getattr(config, 'lastvit_select_largest', True)
        if isinstance(_largest_raw, str) and _largest_raw.lower() == 'both':
            self.lastvit_select_largest = 'both'
        else:
            self.lastvit_select_largest = bool(_largest_raw)
        # ★ 2026-05-31 公式版本: 'v1_strict' / 'v2_abs_mean' (默认) / 'v3_norm'
        self.lastvit_select_formula = str(getattr(config, 'lastvit_select_formula', 'v2_abs_mean'))

        # ★ 2026-05-25 LaSt-CLS v3 (路径 A): 加可训练投影层
        # 第一性原理: frozen CLIP 下, lastvit_pool 输出是零参数固定特征,
        # 跟 CLIP_CLS 不在同一对齐空间. 加 lastvit_proj 让 LaSt 信号能被任务训练拉动到正确方向.
        # 结构: Linear(768→768) + LayerNorm, identity init (训练初等价 v2)
        if self.use_lastvit_cls:
            self.lastvit_proj = nn.Sequential(
                nn.Linear(self.dim_f, self.dim_f),
                nn.LayerNorm(self.dim_f),
            )
            # identity init: weight ≈ I, bias = 0 → 训练初 lastvit_proj(x) ≈ x, 等价 v2
            with torch.no_grad():
                self.lastvit_proj[0].weight.copy_(torch.eye(self.dim_f))
                self.lastvit_proj[0].bias.zero_()

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

        # ─────────────────────────────────────────────────────────
        # ★ F4/F5: 三层动态门控 (add 模式专用)
        # ─────────────────────────────────────────────────────────
        # gating_dynamic:
        #   None / 'fixed' → 不启用 (走原 fixed/cig 分支)
        #   'mlp'          → 启用外层 α(x) MLP_α(cls_token)
        # weight_s2v_mode:
        #   'fixed' → local = 0.5·s2v + 0.5·v2s (yaml weight_s2v 标量)
        #   'mlp'   → local = w(x)·s2v + (1-w(x))·v2s, w(x)=sigmoid(MLP_w(cls_token))
        self.gating_dynamic = getattr(config, 'gating_dynamic', 'fixed')
        self.weight_s2v_mode = getattr(config, 'weight_s2v_mode', 'fixed')

        gate_hidden = int(getattr(config, 'gate_hidden', 64))

        # 外层 α(x) MLP, 输出 [B, 1] 经 sigmoid 限定 [0,1]
        # 初始化最后一层偏置, 让初始 α ≈ local_weight (默认 0.3)
        # sigmoid(x) = 0.3 → x ≈ -0.847
        if self.gating_dynamic == 'mlp':
            self.alpha_net = nn.Sequential(
                nn.Linear(self.dim_f, gate_hidden),
                nn.LayerNorm(gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, 1),
            )
            init_alpha = float(getattr(config, 'gating_alpha_init', 0.3))
            init_logit = float(np.log(init_alpha / max(1e-6, 1 - init_alpha)))
            with torch.no_grad():
                self.alpha_net[-1].bias.fill_(init_logit)
                self.alpha_net[-1].weight.zero_()  # 初始时 α ≡ sigmoid(bias) 与图无关

        # 内层 w(x) MLP, 输出 [B, 1] 经 sigmoid 限定 [0,1]
        # 初始化让 w ≈ weight_s2v (默认 0.5 → bias=0)
        if self.weight_s2v_mode == 'mlp':
            self.w_net = nn.Sequential(
                nn.Linear(self.dim_f, gate_hidden),
                nn.LayerNorm(gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, 1),
            )
            init_w = float(getattr(config, 'weight_s2v_init', 0.5))
            init_w_logit = float(np.log(init_w / max(1e-6, 1 - init_w)))
            with torch.no_grad():
                self.w_net[-1].bias.fill_(init_w_logit)
                self.w_net[-1].weight.zero_()

        # ─────────────────────────────────────────────────────────
        # ★ H2: Dynamic Mean-LaSt Pooling (DMP) — Pool-level dynamic gate
        # ─────────────────────────────────────────────────────────
        # pool_dynamic:
        #   'fixed' (默认) → 用 yaml pool_lambda_fixed 标量 (cross_tf 内部读)
        #   'mlp'         → λ_pool(x) = sigmoid(MLP_pool(cls_token))
        #                   仅 pool_method='dmp' 时生效
        #
        # 物理含义: 看图自适应决定 mean (稳定全局) vs LaSt (频域选择性) 比例
        # 末层 zero init + bias=logit(0.5), 训练前 λ ≡ 0.5 等价 fixed 等权
        self.pool_dynamic = getattr(config, 'pool_dynamic', 'fixed')
        if self.pool_dynamic == 'mlp':
            self.pool_net = nn.Sequential(
                nn.Linear(self.dim_f, gate_hidden),
                nn.LayerNorm(gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, 1),
            )
            init_lam = float(getattr(config, 'pool_lambda_init', 0.5))
            init_lam_logit = float(np.log(init_lam / max(1e-6, 1 - init_lam)))
            with torch.no_grad():
                self.pool_net[-1].bias.fill_(init_lam_logit)
                self.pool_net[-1].weight.zero_()

        # ─────────────────────────────────────────────────────────
        # ★ G3: CoCoOp-inspired Conditional Text Adapter
        # ─────────────────────────────────────────────────────────
        # 思想: 不用 CoCoOp 原 prompt token (本项目无 prompt), 改用图像条件化文本残差
        #   π(x) = MLP_meta(cls_token)                # [B, 768]
        #   t̃_c(x) = Norm(t_c + ρ · π(x))             # 每图自己的 200 类原型
        #   base_logits = γ · cos(cls, t̃_c(x))
        # 仅作用于 base_logits, 不喂进 cross_tf, 改动局部.
        # ρ 控制扰动幅度, init=0.0 训练前等价 baseline (无扰动)
        self.use_conditional_text = bool(getattr(config, 'use_conditional_text', False))
        if self.use_conditional_text:
            meta_hidden = int(getattr(config, 'meta_net_hidden', 48))
            self.meta_net = nn.Sequential(
                nn.Linear(self.dim_f, meta_hidden),
                nn.LayerNorm(meta_hidden),
                nn.GELU(),
                nn.Linear(meta_hidden, self.dim_f),
            )
            # zero init: π(x)=0 训练前等价 baseline
            with torch.no_grad():
                self.meta_net[-1].weight.zero_()
                self.meta_net[-1].bias.zero_()
            self.cond_text_ratio = float(getattr(config, 'conditional_text_ratio', 0.05))

        # ─────────────────────────────────────────────────────────
        # ★ 动态 Residual & Base Blend (cosine_only 专用)
        # ─────────────────────────────────────────────────────────
        # residual_mode:
        #   'fixed'             → 用 yaml 里的 visual_residual / text_residual 标量
        #   'learnable_global'  → 1 个 vr + 1 个 tr 标量 sigmoid([0,1]) 全局学
        #   'learnable_split'   → 1 个 vr + tr_seen + tr_unseen sigmoid 分类别学
        #                          unseen 初始化偏 0.95 (sigmoid(3)≈0.95) 强保护
        #
        # cosine_base_blend_mode:
        #   'fixed'             → 用 yaml 标量 cosine_base_blend
        #   'learnable'         → 1 个 cb sigmoid([0,1]) 让模型自学 base 的混合比
        self.residual_mode = getattr(config, 'residual_mode', 'fixed')
        self.cosine_base_blend_mode = getattr(config, 'cosine_base_blend_mode', 'fixed')

        # logit 形式的可学参数 (注册后 sigmoid → [0,1])
        # 初始化对应 sigmoid 值: 0.0→0.5, 3.0→0.953, -3.0→0.047
        vr_init   = float(getattr(config, 'visual_residual_init', 0.0))
        trs_init  = float(getattr(config, 'text_residual_seen_init', 0.0))
        tru_init  = float(getattr(config, 'text_residual_unseen_init', 3.0))
        cb_init   = float(getattr(config, 'cosine_base_blend_init', -1.0))  # sigmoid(-1)=0.27

        if self.residual_mode != 'fixed':
            self.visual_residual_logit = nn.Parameter(torch.tensor(vr_init))
            if self.residual_mode == 'learnable_split':
                self.text_residual_seen_logit   = nn.Parameter(torch.tensor(trs_init))
                self.text_residual_unseen_logit = nn.Parameter(torch.tensor(tru_init))
            else:  # learnable_global
                self.text_residual_logit = nn.Parameter(torch.tensor(trs_init))

        if self.cosine_base_blend_mode == 'learnable':
            self.cosine_base_blend_logit = nn.Parameter(torch.tensor(cb_init))

    def get_adapted_seen_text(self):
        """Adapter 残差增强的 seen 类文本 [150, 768]"""
        x = self.seen_text_embeds
        adapted = self.adapter_ratio * self.text_adapter(x) + \
                  (1.0 - self.adapter_ratio) * x
        return F.normalize(adapted, dim=1)

    def _apply_text_attr_reservoir(self, all_text):
        """
        MOD-002: 用 CUB 属性文本原型构造受限 text reservoir 残差。

        class_attr 给出每个类别对应 312 个专家属性的强度；这里只取 top-K
        属性形成类别属性原型，再经过一个低秩投影产生小幅文本残差。
        现有 Pearson topology loss 会约束输出后的 all_text，防止 seen-only
        训练把 unseen 语义拓扑拉散。
        """
        if not self.use_text_attr_reservoir:
            return all_text

        device = all_text.device
        dtype = all_text.dtype
        attr_score = self.class_attr.to(device=device, dtype=torch.float32)  # [C, A]
        attr_text = self.attr_text_embeds.to(device=device, dtype=dtype)     # [A, D]

        k_attr = int(getattr(self.config, 'text_attr_reservoir_topk', 32))
        k_attr = max(1, min(k_attr, attr_score.size(1)))
        temp = float(getattr(self.config, 'text_attr_reservoir_temp', 10.0))

        top_val, top_idx = attr_score.topk(k=k_attr, dim=1)
        top_weight = F.softmax(top_val * temp, dim=1).to(dtype)
        sparse_weight = torch.zeros(attr_score.size(0), attr_score.size(1),
                                    device=device, dtype=dtype)
        sparse_weight.scatter_(1, top_idx, top_weight)

        reservoir_proto = sparse_weight @ attr_text                         # [C, D]
        reservoir_proto = F.normalize(reservoir_proto, dim=-1)
        residual = self.text_attr_reservoir_proj(reservoir_proto)
        ratio = float(getattr(self.config, 'text_attr_reservoir_ratio', 0.05))
        return F.normalize(all_text + ratio * residual.to(dtype), dim=-1)

    def _topology_pearson_loss(self, enh_text=None):
        """
        类别角度拓扑保持 Pearson loss:
        让增强后类别原型的两两余弦结构，保持 CLIP 原始文本空间的相对拓扑。

        TPR 原文使用 Pearson correlation 而不是 MSE。这里用 1-corr 作为
        非负损失；等价于最大化两个 pairwise cosine 矩阵的相关性。

        Args:
            enh_text:
                None       → 约束静态 Adapter 后的 200 类文本原型
                [B,C,D]    → 约束 cosine_only 最终参与分类的 per-image t_enh
        """
        if enh_text is None:
            adapted_seen = self.get_adapted_seen_text()                     # [150, 768]
            device = adapted_seen.device
            dtype = adapted_seen.dtype
        else:
            device = enh_text.device
            dtype = enh_text.dtype

        base_text = torch.zeros(self.nclass, self.dim_f,
                                device=device, dtype=dtype)

        base_seen = self.seen_text_embeds.to(device=device, dtype=dtype)
        base_unseen = self.unseen_text_embeds.to(device=device, dtype=dtype)

        base_text[self.seenclass] = base_seen
        base_text[self.unseenclass] = base_unseen

        if enh_text is None:
            enh_text = torch.zeros_like(base_text)
            enh_text[self.seenclass] = adapted_seen
            enh_text[self.unseenclass] = base_unseen

        base_text = F.normalize(base_text.float(), dim=-1)
        enh_text = F.normalize(enh_text.float(), dim=-1)

        base_sim = base_text @ base_text.T                                  # [C, C]
        if enh_text.dim() == 2:
            enh_sim = enh_text @ enh_text.T                                 # [C, C]
        else:
            enh_sim = torch.matmul(enh_text, enh_text.transpose(-1, -2))     # [B, C, C]

        off_diag = ~torch.eye(self.nclass, dtype=torch.bool,
                              device=device)
        base_vec = base_sim.detach()[off_diag]                              # [C*C-C]

        if enh_sim.dim() == 2:
            enh_vec = enh_sim[off_diag].unsqueeze(0)                         # [1, M]
        else:
            enh_vec = enh_sim[:, off_diag]                                  # [B, M]
        base_vec = base_vec.unsqueeze(0).expand_as(enh_vec)                  # [B, M]

        enh_centered = enh_vec - enh_vec.mean(dim=1, keepdim=True)
        base_centered = base_vec - base_vec.mean(dim=1, keepdim=True)
        numerator = (enh_centered * base_centered).sum(dim=1)
        denominator = (
            torch.sqrt((enh_centered ** 2).sum(dim=1) + 1e-8) *
            torch.sqrt((base_centered ** 2).sum(dim=1) + 1e-8)
        )
        corr = numerator / denominator
        return (1.0 - corr).mean()

    def _ag_jepa_loss(self, patches, all_text, labels):
        """
        AG-JEPA auxiliary objective.

        Select the patches most aligned with the ground-truth class text, hide them from
        the context summary, then predict their abstract visual feature from the remaining
        patches plus the class semantic prototype. A negative class text is used as a
        lightweight counterfactual term.
        """
        device = patches.device
        if (not self.use_ag_jepa) or patches is None or all_text is None:
            zero = torch.tensor(0.0, device=device if patches is not None else labels.device)
            return zero, zero

        B, N, _ = patches.shape
        k = max(1, min(int(self.jepa_topk), N - 1))
        labels = labels.to(device=device, dtype=torch.long)
        class_text = all_text[labels].to(device=device, dtype=patches.dtype)

        with torch.no_grad():
            patch_n = F.normalize(patches.float(), dim=-1)
            text_n = F.normalize(class_text.float(), dim=-1)
            patch_score = torch.einsum('bnd,bd->bn', patch_n, text_n)
            _, masked_idx = torch.topk(patch_score, k=k, dim=1, largest=True)

        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, masked_idx, True)
        keep = ~mask

        patch_z = self.cross_tf.embed_cv(patches)
        target = patch_z[mask].view(B, k, -1).mean(dim=1).detach()

        keep_f = keep.unsqueeze(-1).to(patch_z.dtype)
        context = (patch_z * keep_f).sum(dim=1) / keep_f.sum(dim=1).clamp_min(1.0)
        text_z = self.cross_tf.embed_text(class_text)

        pred = self.jepa_predictor(torch.cat([context, text_z], dim=-1))
        pos_sim = F.cosine_similarity(pred, target, dim=-1)
        loss_jepa = (1.0 - pos_sim).mean()

        seen = self.seenclass.to(device=device)
        label_pos = torch.zeros_like(labels)
        for i, cls_idx in enumerate(seen):
            label_pos[labels == cls_idx] = i
        neg_pos = (label_pos + 1) % seen.numel()
        neg_text = all_text[seen[neg_pos]].to(device=device, dtype=patches.dtype)
        neg_text_z = self.cross_tf.embed_text(neg_text)
        pred_neg = self.jepa_predictor(torch.cat([context.detach(), neg_text_z], dim=-1))
        neg_sim = F.cosine_similarity(pred_neg, target, dim=-1)
        loss_jepa_neg = F.relu(neg_sim - pos_sim.detach() + self.jepa_neg_margin).mean()

        return loss_jepa, loss_jepa_neg

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
        # ===== 消融模式分支 (clip_only / adapter_only) =====
        # clip_only:    纯 CLIP 余弦, 不经过 Adapter / FAE / CrossModalTransformer
        # adapter_only: seen 文本经 Adapter 微调, 视觉端用 CLS, 跳过 FAE / CrossModal
        # vgsr (默认):   走下方完整流程
        model_mode = getattr(self.config, 'model_mode', 'vgsr')
        if model_mode in ('clip_only', 'adapter_only'):
            # 提取 CLS token (兼容多种输入形状)
            if clip_features.dim() == 3 and clip_features.size(1) == 577:
                cls_token = clip_features[:, 0, :]
            elif clip_features.dim() == 3 and clip_features.size(1) == 1:
                cls_token = clip_features.squeeze(1)
            elif clip_features.dim() == 2:
                cls_token = clip_features
            else:
                cls_token = clip_features.mean(dim=1)  # 兜底

            logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)

            # 文本端: clip_only 用原始 CLIP, adapter_only 让 Adapter 起作用
            if model_mode == 'adapter_only':
                seen_text = self.get_adapted_seen_text()                # [150, 768]
            else:
                seen_text = F.normalize(self.seen_text_embeds, dim=1)   # [150, 768]

            # 拼接 200 类全局文本 (unseen 始终用 CLIP 原始)
            all_text = torch.zeros(self.nclass, self.dim_f,
                                   device=cls_token.device, dtype=cls_token.dtype)
            all_text[self.seenclass]   = seen_text
            all_text[self.unseenclass] = self.unseen_text_embeds

            vis_n  = F.normalize(cls_token, dim=1)                      # [B, 768]
            text_n = F.normalize(all_text, dim=1)                       # [200, 768]
            logits_200 = vis_n @ text_n.T * logit_scale                 # [B, 200]

            if is_train:
                logits = logits_200[:, self.seenclass]                  # [B, 150]
            else:
                logits = logits_200                                     # [B, 200]

            return {
                'logits':      logits,
                'logits_200':  logits_200,
                'base_logits': logits_200,                              # 占位, 保持接口兼容
                'local_score': torch.zeros_like(logits_200),            # 占位, 让 KL 一致性自动为 0
                'clip_S_pp':   logits,
                'clip_pred':   logits,
            }
        # ===== 消融分支结束, 下方为 vgsr 全功能流程 =====

        # 分离 CLS token 和 patches
        if clip_features.dim() == 3 and clip_features.size(1) == 577:
            cls_token = clip_features[:, 0, :]                   # [B, 768]
            patches   = clip_features[:, 1:, :]                  # [B, 576, 768]
        else:
            patches = self._prepare_patches(clip_features)       # [B, 576, 768]
            cls_token = None

        # ★ LaSt-ViT-CLS 增强 (CVPR 2026)
        # ──────────────────────────────────────────────────────────────
        # ★ 2026-05-25 修复 (方案 A): LaSt-CLS 仅在 base_logits 的最终 cosine 用
        # ──────────────────────────────────────────────────────────────
        # 原 v1 实现 (失败 H=71.18 vs baseline 72.27):
        #   cls_token = (1-r)·CLIP_CLS + r·LaSt_CLS   ← 直接覆盖, 后续 G3/FAE/cross_tf 都看 LaSt
        #   问题: G3 meta_net 训练逻辑基于原 cls 设计, LaSt 注入后 π(x) 方向偏移
        #         FAE 没看到 LaSt 增强, base 路径偏 part-aware / local 路径偏全局, 两者打架
        #         GZSL-S 跌 4 个点 (-1.09 H)
        #
        # 修复方案 (本版 v2): 切断 LaSt → G3 / FAE 间接污染
        #   保留 cls_token 原值, 仅在最终 base_logits 计算时用 LaSt 增强版
        #   - cls_token (原 CLIP CLS): G3 meta_net / FAE / cross_tf / consistency 全部用
        #   - cls_for_base (LaSt 增强): 仅 base_logits = cos(cls_for_base, t̃_c)
        #   保证 LaSt 只影响 "图像 → 类别" 这一处, 不污染下游
        cls_for_base = cls_token  # 默认 = cls_token, 不开 LaSt 时无副作用
        if self.use_lastvit_cls and cls_token is not None and patches.size(-1) == self.dim_f:
            sigma_cls = self.lastvit_cls_sigma if self.lastvit_cls_sigma > 0 else None
            with torch.amp.autocast('cuda', enabled=False):
                cls_lastvit = lastvit_pool(
                    patches.float(),                              # [B, 576, 768] fp32
                    k=self.lastvit_cls_k,
                    sigma=sigma_cls,                              # None → √D = √768 ≈ 27.7
                )                                                 # [B, 768]
            cls_lastvit = cls_lastvit.to(cls_token.dtype)
            # ★ v3: 投影到 CLIP-aligned 空间, 让 LaSt 信号可训练
            cls_lastvit = self.lastvit_proj(cls_lastvit)              # [B, 768]
            # 残差融合: r=0 完全用 CLIP CLS (基线), r=1 完全用 LaSt CLS (激进)
            # ★ 关键: 只赋给 cls_for_base, 不动 cls_token
            cls_for_base = (1.0 - self.lastvit_residual) * cls_token + \
                           self.lastvit_residual * cls_lastvit    # [B, 768]

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        seen_text = self.get_adapted_seen_text()                 # [150, 768]

        # 构建全 200 类文本
        all_text = torch.zeros(self.nclass, self.dim_f,
                               device=patches.device, dtype=patches.dtype)
        all_text[self.seenclass]   = seen_text
        all_text[self.unseenclass] = self.unseen_text_embeds
        text_reservoir_features = None
        if self.use_text_attr_reservoir:
            all_text = self._apply_text_attr_reservoir(all_text)
            text_reservoir_features = all_text

        # ── 基线分数: CLIP 原始余弦相似度 (完全不动, 保留零样本能力) ──
        if cls_token is not None:
            vis_n = F.normalize(cls_token, dim=1)                # [B, 768]   ★ G3 / FAE 看的是这个
        else:
            vis_n = F.normalize(patches.mean(dim=1), dim=1)

        # ★ LaSt-CLS 修复 v2: 仅 base_logits 的最终 cos 用 LaSt 增强版
        # vis_n_for_base 用于 cos(cls, t̃_c), 其它地方 (meta_net / cross_tf) 用 vis_n
        if cls_for_base is not None and cls_for_base is not cls_token:
            vis_n_for_base = F.normalize(cls_for_base, dim=1)
        else:
            vis_n_for_base = vis_n

        # ★ G3: CoCoOp Conditional Text Adapter (Step 3 稳健版)
        #   π(x) = MLP_meta(cls), normalize 限模长
        #   仅给 seen 类加 ρ·π, unseen 列永远是 CLIP 原始 (保护 GZSL-U)
        #   t̃_c(x) = Norm(t_c + ρ·π̂(x))   for c ∈ seen
        #            t_c                     for c ∈ unseen
        #   base_logits = γ · cos(cls, t̃_c(x))
        #
        # Step 3 设计动机:
        #   v1 (broadcast 200 类) GZSL-U=0% 灾难, 即使 ρ=0.005 也崩
        #   原因: π 学到方向性偏置 (seen-favoring direction), 不是幅度问题
        #   修复: π 永远不影响 unseen 文本, normalize 限模长防漂
        if (self.use_conditional_text and cls_token is not None
                and self.cond_text_ratio > 0):
            pi_x = self.meta_net(cls_token)                       # [B, 768]
            pi_x = F.normalize(pi_x, dim=-1)                      # ★ 单位向量限模长
            B_size = cls_token.size(0)
            # ★ 关键: clone 后只改 seen 列, unseen 列永远是 CLIP 原始
            all_text_cond = all_text.unsqueeze(0).expand(B_size, -1, -1).clone()  # [B, 200, 768]
            all_text_cond[:, self.seenclass, :] = (
                all_text[self.seenclass].unsqueeze(0)
                + self.cond_text_ratio * pi_x.unsqueeze(1)
            )
            text_n_cond = F.normalize(all_text_cond, dim=-1)                      # [B, 200, 768]
            # cosine: vis_n_for_base [B, 768] vs text_n_cond [B, 200, 768]
            # ★ LaSt-CLS 修复 v2: 这里用 vis_n_for_base (LaSt 增强版), 而 meta_net 上面用的是 vis_n (原 cls)
            base_logits = (vis_n_for_base.unsqueeze(1) * text_n_cond).sum(dim=-1) * logit_scale  # [B, 200]
        else:
            text_n = F.normalize(all_text, dim=1)                # [200, 768]
            # ★ LaSt-CLS 修复 v2: 不开 G3 时 base_logits 也用 vis_n_for_base
            base_logits = vis_n_for_base @ text_n.T * logit_scale         # [B, 200]

        # ── 双分支输出 ──
        # 注: cosine-only 模式下原来想给 cross_tf 传 unseen_idx 做"早期 detach"
        # 实测验证 (verify_grad.py): forward 整体过但 loss 只用 seen logits 时,
        # unseen 列输入梯度 PyTorch 自动微分会严格为 0, 不需要 detach
        # 所以保持简单调用即可
        # ★ 内层动态门控 w(x): 仅 add 模式 + weight_s2v_mode='mlp' 时启用
        weight_s2v_dyn = None
        if (self.score_mode == 'add'
                and self.weight_s2v_mode == 'mlp'
                and cls_token is not None):
            weight_s2v_dyn = torch.sigmoid(self.w_net(cls_token))   # [B, 1]
        # ★ 池化级动态门控 λ_pool(x): 仅 pool_method='dmp' + pool_dynamic='mlp' 时启用
        pool_lambda_dyn = None
        if (self.pool_dynamic == 'mlp'
                and self.cross_tf.pool_method == 'dmp'
                and cls_token is not None):
            pool_lambda_dyn = torch.sigmoid(self.pool_net(cls_token))  # [B, 1]
        attr_text = self.attr_text_embeds if self.use_geo_attr_routing else None
        cm_out = self.cross_tf(patches, all_text, cls_token,
                               attr_text=attr_text,
                               weight_s2v_dyn=weight_s2v_dyn,
                               pool_lambda_dyn=pool_lambda_dyn,
                               lastvit_select_k=getattr(self, 'lastvit_select_k', 0),
                               lastvit_select_sigma=getattr(self, 'lastvit_select_sigma', 0.0),
                               lastvit_select_largest=getattr(self, 'lastvit_select_largest', True),
                               lastvit_select_formula=getattr(self, 'lastvit_select_formula', 'v2_abs_mean'))
        local_score = cm_out['local_score']                      # [B, 200]

        # ── 计分模式选择 ──
        # 'add' (默认): logits = base_logits + β × local_score      （CrossModal 是补丁）
        # 'cosine_only': logits = cosine(v_enh, t_enh) × scale      （CrossModal 唯一决定）
        topology_text = text_reservoir_features
        if self.score_mode == 'cosine_only':
            v_enh_raw = cm_out['v_enh']                          # [B, 768]
            t_enh_raw = cm_out['t_enh']                          # [B, 200, 768]
            B_size = patches.size(0)

            # ===== 视觉端残差融合 (支持 fixed / learnable) =====
            # visual_residual: 用 cls 残差防止视觉端被训歪
            if self.residual_mode != 'fixed':
                vr = torch.sigmoid(self.visual_residual_logit)
            else:
                vr = self.visual_residual
            if cls_token is not None:
                v_enh = (1.0 - vr) * v_enh_raw + vr * cls_token       # [B, 768]
            else:
                v_enh = v_enh_raw

            # ===== 文本端: 流形共享 + 梯度截断 (cross_tf 内已做早期 detach) =====
            # 旧方案 v1/v2 (失败): unseen 50 列直接用 all_text 绕过 proj_text
            #   → seen 在 adapter 投影流形, unseen 在 CLIP 原始流形, 几何错位
            #   → cosine 相似度系统性低估 unseen 类
            #
            # 当前方案 v4 (流形共享 + 早期 detach):
            #   ★ cross_tf 内部在 proj_text 前就把 unseen 列 detach
            #   ★ 这里 t_enh_raw 已经是"unseen 列输入梯度被截断后过 proj_text 的结果"
            #   seen/unseen 共享同一个 proj_text → 几何流形一致
            #   unseen 输入梯度不流入 decoder_v2s/FAE 上层 → 不污染 seen 训练
            #   proj_text 自身权重仍受 seen 监督 (期望行为)
            #
            # 这里只做残差融合, 不再事后 detach
            # ★ 支持 fixed / learnable_global / learnable_split 三模式
            if self.residual_mode == 'learnable_split':
                tr_s = torch.sigmoid(self.text_residual_seen_logit)
                tr_u = torch.sigmoid(self.text_residual_unseen_logit)
            elif self.residual_mode == 'learnable_global':
                tr_g = torch.sigmoid(self.text_residual_logit)
                tr_s = tr_g
                tr_u = tr_g
            else:  # fixed
                tr_s = self.text_residual
                tr_u = self.text_residual

            t_enh_seen   = (1.0 - tr_s) * t_enh_raw[:, self.seenclass, :] + \
                           tr_s * all_text[self.seenclass].unsqueeze(0)
            t_enh_unseen = (1.0 - tr_u) * t_enh_raw[:, self.unseenclass, :] + \
                           tr_u * all_text[self.unseenclass].unsqueeze(0)

            # 拼回完整 [B, 200, 768]
            t_enh = torch.zeros(B_size, self.nclass, self.dim_f,
                                 device=patches.device, dtype=patches.dtype)
            t_enh[:, self.seenclass, :]   = t_enh_seen
            t_enh[:, self.unseenclass, :] = t_enh_unseen

            # 余弦相似度 + 温度缩放
            v_n = F.normalize(v_enh, dim=-1).unsqueeze(1)        # [B, 1, 768]
            t_n = F.normalize(t_enh, dim=-1)                     # [B, 200, 768]
            logits_200 = (v_n * t_n).sum(dim=-1) * logit_scale   # [B, 200]
            topology_text = t_enh

            # ── [E7] cosine_base_blend: 把 CLIP base 硬接回主路径 ──
            # 设计动机: cosine_only 把 base_logits 踢出主路径后 (E0 实测 H=37.41),
            #   distill 软约束完全救不回来。这里直接做线性混合, 让 base 的 zero-shot
            #   兜底重新进入 forward 主路径, 等价"软 add 模式":
            #     β=0.0 → 纯 cosine_only (E0/E1 行为)
            #     β=0.3 → 30% CLIP 兜底 + 70% learned cosine (推荐起点)
            #     β=1.0 → 完全等价 base_logits (退化, 但保留接口)
            #
            # ★ 支持 fixed / learnable 双模式
            if self.cosine_base_blend_mode == 'learnable':
                cb_blend = torch.sigmoid(self.cosine_base_blend_logit)
                logits_200 = (1.0 - cb_blend) * logits_200 + cb_blend * base_logits
            else:
                cb_blend = float(getattr(self.config, 'cosine_base_blend', 0.0))
                if cb_blend > 0.0:
                    logits_200 = (1.0 - cb_blend) * logits_200 + cb_blend * base_logits
        else:
            # ── add 模式 ──
            # 三种门控可切换:
            #   gating='fixed' (基线 H=72.26%):
            #       logits = base_logits + local_weight × local_score
            #
            #   gating='cig'   (Confidence-Inverse Gating, 实测 H=71.80%):
            #       gate = α × (1-conf)^τ
            #       logits = base_logits + gate × local_score
            #
            #   gating_dynamic='mlp' (★ F4/F5 新增, 优先级最高):
            #       gate = sigmoid(MLP_α(cls_token))
            #       看图自适应决定要多少局部补丁信息
            if self.gating_dynamic == 'mlp' and cls_token is not None:
                gate = torch.sigmoid(self.alpha_net(cls_token))      # [B, 1]
                logits_200 = base_logits + gate * local_score
            elif self.gating == 'cig':
                # 动态门控: 用 CLIP 在全 200 类上的置信度反推 gate
                # 关键设计:
                #   1. base_logits.detach() 避免 gate 梯度干扰 base 分支
                #   2. softmax 在全 200 类上算, unseen 类也参与 → 绕过监督盲区
                #   3. alpha/tau 仅 2 个可学参数, 微调强度和形状
                with torch.no_grad():
                    prob = F.softmax(base_logits, dim=-1)            # [B, 200]
                    conf = prob.max(dim=-1).values                   # [B]
                uncertainty = (1.0 - conf).clamp(min=0.0, max=1.0)   # [B]
                alpha = F.softplus(self.gate_alpha)                  # 标量 > 0
                tau   = F.softplus(self.gate_tau)                    # 标量 > 0
                gate  = (alpha * uncertainty.pow(tau)).unsqueeze(-1) # [B, 1]
                logits_200 = base_logits + gate * local_score
            else:
                # 'fixed' 基线版: 固定 local_weight (默认 0.3)
                logits_200 = base_logits + self.local_weight * local_score

        if is_train:
            logits = logits_200[:, self.seenclass]               # [B, 150]
        else:
            logits = logits_200                                  # [B, 200]

        # ── 可选: 把 cosine_only 中间产物透传给 compute_loss 用作约束 ──
        # raw 版本 (未残差融合) 用于锚定 loss; cls_token / all_text 作为 target
        out = {
            'logits':      logits,
            'logits_200':  logits_200,   # compute_loss 里 calibration loss 用
            'base_logits': base_logits,  # [B, 200] CLIP 余弦, 用于一致性 loss
            'local_score': local_score,  # [B, 200] 双分支输出, 用于一致性 loss
            'text_topology_features': topology_text,  # cosine_only 的最终 t_enh
            'clip_S_pp':   logits,
            'clip_pred':   logits,
            # ★ G2 MSDN 互蒸馏 + cosine_only 双分支辅助 CE 用 (add 模式也透传)
            'score_s2v':   cm_out.get('score_s2v'),  # [B, 200]
            'score_v2s':   cm_out.get('score_v2s'),  # [B, 200]
            'attr_route_score': cm_out.get('attr_route_score'),  # [B, 312] or None
            'jepa_patches': patches,
            'all_text':     all_text,
        }
        if self.score_mode == 'cosine_only':
            # 仅 cosine_only 分支才有这些; add 模式下 v_enh/t_enh 不参与最终 logits
            out['v_enh_raw']  = cm_out['v_enh']           # [B, 768]      锚到 cls_token
            out['t_enh_raw']  = cm_out['t_enh']           # [B, 200, 768] unseen 列锚到 all_text
            out['cls_token']  = cls_token                  # [B, 768] 或 None
            out['all_text']   = all_text                   # [200, 768]
        return out

    def compute_loss(self, in_package):
        """
        训练 Loss = CE loss
                  + λ_cal     × self-calibration loss
                  + λ_consist × consistency loss (KL between base / local on seen)
                  + λ_l2sp    × adapter L2-SP regularization
                  + λ_topo    × topology-preserving Pearson regularization

        - CE: 主分类损失 (仅 seen 150 类)
        - cal: 强推 unseen 概率 (实测无效, 默认关闭)
        - consist: 防止 local_score 学出和 CLIP base 矛盾的反向信号
        - l2sp: 限制 adapter 输出偏离 CLIP 原始嵌入的程度
        - topo: 保持 CLIP 类别原型的两两角度拓扑, 缓解 seen-only 训练拉散 unseen 空间
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
            prob_unseen = prob_all[:, self.unseenclass]              # [B, 50]  给未见类一点概率
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
            # ★ 动态权重: lambda_consist / (1 + γ·cons) 防 cons 量级失控
            # cons=0 → 基线; cons=9 → ~0.5×; cons=∞ → 上限 0.5 不再增长
            # 仅 consist_dynamic=True 时启用 (默认 False 兼容旧实验)
            if self.config.__dict__.get('consist_dynamic', False):
                gamma = float(self.config.__dict__.get('consist_dynamic_gamma', 0.1))
                with torch.no_grad():
                    scale = 1.0 / (1.0 + gamma * loss_consist.detach())
                loss = loss + (lambda_consist * scale) * loss_consist
            else:
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

        # ========== Topology-preserving Pearson (类别角度拓扑保持) ==========
        # 创新树候选: topology-preserving-class-angle-regularization
        # 用 Pearson 最大化增强后 pairwise cosine 矩阵与 CLIP 原始矩阵的相关性。
        # cosine_only 下优先约束最终 t_enh; 其他模式退回约束静态 Adapter 文本原型。
        loss_topo = torch.tensor(0.0, device=logits.device)
        lambda_topo = self.config.__dict__.get(
            'lambda_topo_pearson',
            self.config.__dict__.get('lambda_topo_mse', 0))
        if lambda_topo > 0:
            topo_text = in_package.get('text_topology_features', None)
            loss_topo = self._topology_pearson_loss(topo_text)
            loss = loss + lambda_topo * loss_topo

        # ========== [新增] cosine_only 锚定三件套 ==========
        # 仅 score_mode='cosine_only' 时启用 (in_package 才有 v_enh_raw/t_enh_raw/cls_token/all_text)
        # L1 v_anchor:        v_enh 不要离 CLIP CLS 太远  (修视觉端漂移)
        # L2 t_unseen_anchor: t_enh[unseen] 不要离原始 CLIP 文本太远 (修 unseen 列共享 proj_text 的间接漂移)
        # L3 distill:         logits_200 全 200 类向 base_logits 蒸馏 (软对齐, 防止 seen-only CE 把全空间拉歪)
        # ========== [AG-JEPA] masked semantic patch prediction ==========
        loss_jepa = torch.tensor(0.0, device=logits.device)
        loss_jepa_neg = torch.tensor(0.0, device=logits.device)
        lambda_jepa = self.config.__dict__.get('lambda_jepa', 0)
        lambda_jepa_neg = self.config.__dict__.get('lambda_jepa_neg', 0)
        if self.use_ag_jepa and (lambda_jepa > 0 or lambda_jepa_neg > 0):
            jepa_patches = in_package.get('jepa_patches', None)
            all_text_jepa = in_package.get('all_text', None)
            if jepa_patches is not None and all_text_jepa is not None:
                loss_jepa, loss_jepa_neg = self._ag_jepa_loss(
                    jepa_patches, all_text_jepa, labels)
                loss = loss + lambda_jepa * loss_jepa
                loss = loss + lambda_jepa_neg * loss_jepa_neg

        # ========== [MOD-001] 几何感知属性路由辅助 loss ==========
        # 对每个训练样本, 取其类别专家属性向量中 top-K 属性作为正属性集合。
        # FAE 后的局部视觉 memory 必须把概率质量路由到这些属性原型上。
        loss_geo_attr = torch.tensor(0.0, device=logits.device)
        lambda_geo_attr = self.config.__dict__.get('lambda_geo_attr_routing', 0)
        attr_route_score = in_package.get('attr_route_score', None)  # [B, 312]
        if lambda_geo_attr > 0 and attr_route_score is not None:
            target_attr = self.class_attr.to(
                device=logits.device, dtype=attr_route_score.dtype)[labels]  # [B, 312]
            k_attr = int(self.config.__dict__.get('geo_attr_route_topk', 16))
            k_attr = max(1, min(k_attr, target_attr.size(1)))
            pos_idx = target_attr.topk(k=k_attr, dim=1).indices
            temp = float(self.config.__dict__.get('geo_attr_route_temp', 14.28))
            attr_logp = F.log_softmax(attr_route_score.float() * temp, dim=-1)
            pos_logp = attr_logp.gather(1, pos_idx)
            loss_geo_attr = -torch.logsumexp(pos_logp, dim=1).mean()
            loss = loss + lambda_geo_attr * loss_geo_attr

        loss_v_anchor       = torch.tensor(0.0, device=logits.device)
        loss_t_unseen_anchor = torch.tensor(0.0, device=logits.device)
        loss_distill        = torch.tensor(0.0, device=logits.device)

        lambda_v_anchor        = self.config.__dict__.get('lambda_v_anchor', 0)
        lambda_t_unseen_anchor = self.config.__dict__.get('lambda_t_unseen_anchor', 0)
        lambda_distill         = self.config.__dict__.get('lambda_distill', 0)

        v_enh_raw = in_package.get('v_enh_raw', None)
        t_enh_raw = in_package.get('t_enh_raw', None)
        cls_token = in_package.get('cls_token', None)
        all_text  = in_package.get('all_text',  None)

        # L1: 1 - cos(v_enh_raw, cls_token.detach())
        if lambda_v_anchor > 0 and v_enh_raw is not None and cls_token is not None:
            loss_v_anchor = (1.0 - F.cosine_similarity(
                v_enh_raw, cls_token.detach(), dim=-1)).mean()
            loss = loss + lambda_v_anchor * loss_v_anchor

        # L2: 对 unseen 50 列, batch 平均后, 与 CLIP 原始文本做 1-cos
        # 注: t_enh_raw 形状 [B, 200, 768], 每张图一份, batch mean 取共识
        if lambda_t_unseen_anchor > 0 and t_enh_raw is not None and all_text is not None:
            t_enh_unseen_mean = t_enh_raw[:, self.unseenclass, :].mean(dim=0)   # [50, 768]
            target_unseen     = all_text[self.unseenclass].detach()             # [50, 768]
            loss_t_unseen_anchor = (1.0 - F.cosine_similarity(
                t_enh_unseen_mean, target_unseen, dim=-1)).mean()
            loss = loss + lambda_t_unseen_anchor * loss_t_unseen_anchor

        # L3: KL(softmax(logits_200/T) || softmax(base_logits/T)) 全 200 类
        if (lambda_distill > 0 and logits_200 is not None and base_logits is not None):
            T_d = self.config.__dict__.get('distill_temp', 4.0)
            base_p = F.softmax(base_logits.detach() / T_d, dim=-1)              # [B, 200]
            log_p  = F.log_softmax(logits_200 / T_d, dim=-1)                    # [B, 200]
            loss_distill = F.kl_div(log_p, base_p, reduction='batchmean') * (T_d * T_d)
            loss = loss + lambda_distill * loss_distill

        # ========== [新增] 双分支辅助 CE (cosine_only 用) ==========
        # 审稿人意见: cosine_only 中 CE 监督的是组合空间 cos(v_enh, t_enh),
        # 不是 score_s2v / score_v2s 自己, 失去了 add 模式里"双分类器互相补充"的设计
        # 修复: 给两路分数加辅助 CE, 让 s2v / v2s 各自也是合格的分类器
        loss_aux_s2v = torch.tensor(0.0, device=logits.device)
        loss_aux_v2s = torch.tensor(0.0, device=logits.device)
        lambda_aux_s2v = self.config.__dict__.get('lambda_aux_s2v', 0)
        lambda_aux_v2s = self.config.__dict__.get('lambda_aux_v2s', 0)

        score_s2v = in_package.get('score_s2v', None)   # [B, 200]
        score_v2s = in_package.get('score_v2s', None)   # [B, 200]

        if lambda_aux_s2v > 0 and score_s2v is not None:
            # 训练时只切 seen 列, 每路独立做 CE
            score_s2v_seen = score_s2v[:, self.seenclass]   # [B, 150]
            # 余弦本身在 [-1, 1], 加温度让 logits 量级更接近主 CE
            T_aux = self.config.__dict__.get('aux_temp', 14.28)  # 跟 logit_scale 同量级
            loss_aux_s2v = F.cross_entropy(score_s2v_seen * T_aux, seen_labels)
            loss = loss + lambda_aux_s2v * loss_aux_s2v

        if lambda_aux_v2s > 0 and score_v2s is not None:
            score_v2s_seen = score_v2s[:, self.seenclass]
            T_aux = self.config.__dict__.get('aux_temp', 14.28)
            loss_aux_v2s = F.cross_entropy(score_v2s_seen * T_aux, seen_labels)
            loss = loss + lambda_aux_v2s * loss_aux_v2s

        # ========== [G2] MSDN++ Mutual Branch Distillation ==========
        # 让 score_s2v 和 score_v2s 在 seen 类上互相蒸馏, 防一个分支塌缩
        #   p_s2v = softmax(score_s2v[:,seen] / T)
        #   p_v2s = softmax(score_v2s[:,seen] / T)
        #   L_msdn = (T²/2) [ KL(p_s2v.detach() || p_v2s) + KL(p_v2s.detach() || p_s2v) ]
        # 注意 .detach() 防梯度循环, 一边教另一边
        loss_msdn = torch.tensor(0.0, device=logits.device)
        lambda_msdn = self.config.__dict__.get('lambda_msdn', 0)
        if (lambda_msdn > 0 and score_s2v is not None and score_v2s is not None):
            T_msdn = self.config.__dict__.get('msdn_temp', 2.0)
            # 仅 seen 列 (有梯度), unseen 列没有监督蒸馏过去也学不到东西
            s2v_seen = score_s2v[:, self.seenclass] / T_msdn        # [B, 150]
            v2s_seen = score_v2s[:, self.seenclass] / T_msdn        # [B, 150]
            # 两边分别软化 + log_softmax
            p_s2v       = F.softmax(s2v_seen, dim=-1)
            p_v2s       = F.softmax(v2s_seen, dim=-1)
            log_p_s2v   = F.log_softmax(s2v_seen, dim=-1)
            log_p_v2s   = F.log_softmax(v2s_seen, dim=-1)
            # KL(p_s2v.detach() || p_v2s): s2v 教 v2s
            kl_s2v_to_v2s = F.kl_div(log_p_v2s, p_s2v.detach(), reduction='batchmean')
            # KL(p_v2s.detach() || p_s2v): v2s 教 s2v
            kl_v2s_to_s2v = F.kl_div(log_p_s2v, p_v2s.detach(), reduction='batchmean')
            loss_msdn = (T_msdn * T_msdn / 2.0) * (kl_s2v_to_v2s + kl_v2s_to_s2v)
            loss = loss + lambda_msdn * loss_msdn

        # ========== [CGC C5] Seen-Unseen Margin (反事实校准最小可行版) ==========
        # 直接对全 200 类 logits 做"max_seen - max_unseen"约束, 防止长训 unseen 漂移
        # 物理含义: 不允许 seen 类 logit 比 unseen 类 logit 高出 δ 以上 (hinge loss)
        # 与 G1 / G3 / Cons 互补: 它们都软对齐, 这个直接 hard cap
        # 推荐 λ_bias=0.05, δ=0.0 (严格 ≤0)
        loss_bias = torch.tensor(0.0, device=logits.device)
        lambda_bias = self.config.__dict__.get('lambda_bias', 0)
        if lambda_bias > 0:
            logits_200_full = in_package.get('logits_200', None)
            if logits_200_full is not None:
                delta = float(self.config.__dict__.get('bias_delta', 0.0))
                # 在全 200 类上算 max, 不切训练时的 seen mask
                max_seen   = logits_200_full[:, self.seenclass].max(dim=1).values     # [B]
                max_unseen = logits_200_full[:, self.unseenclass].max(dim=1).values   # [B]
                # hinge: seen 比 unseen 高出 > δ 时惩罚, 否则 0
                loss_bias = F.relu(max_seen - max_unseen - delta).mean()
                loss = loss + lambda_bias * loss_bias

        return {
            'loss': loss,
            'loss_CE': loss_CE,
            'loss_cal': loss_cal,
            'loss_consist': loss_consist,
            'loss_l2sp': loss_l2sp,
            'loss_topo': loss_topo,
            'loss_v_anchor': loss_v_anchor,
            'loss_t_unseen_anchor': loss_t_unseen_anchor,
            'loss_distill': loss_distill,
            'loss_aux_s2v': loss_aux_s2v,
            'loss_aux_v2s': loss_aux_v2s,
            'loss_msdn': loss_msdn,
            'loss_bias': loss_bias,
            'loss_jepa': loss_jepa,
            'loss_jepa_neg': loss_jepa_neg,
            'loss_geo_attr': loss_geo_attr,
        }
