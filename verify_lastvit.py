"""
验证 model/MyModel.py 的 lastvit_pool 与官方 LAST-ViT 实现数值一致

官方代码: C:/Users/Administrator/Desktop/项目/LAST-ViT-main/LAST-ViT-main/cls_pretrain/conf.py
         (class dense_vit.forward, 第 132-180 行)

我的代码: c:/Users/Administrator/Desktop/项目/DVSR/model/MyModel.py
         (lastvit_pool 函数, 第 39+ 行)

测试方法:
  1. 构造固定 seed 的 [B=2, N=196, D=768] 输入张量
  2. 用官方原代码跑一遍 (从 conf.py 抄过来)
  3. 用 my lastvit_pool 跑一遍
  4. 对比输出张量的最大绝对差 + cosine 相似度
"""

import torch
import sys

# ============================================================
# 1. 官方原代码 (从 conf.py 132-180 行抄出来, 一字不改)
# ============================================================
def official_gaussian_kernel_1d(kernel_size, sigma):
    kernel = torch.exp(-0.5 * (torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1).float() / sigma) ** 2)
    kernel = kernel / torch.max(kernel)
    return kernel


def official_lastvit_forward(x_with_cls):
    """
    严格抄自官方 dense_vit.forward 的核心 9 行
    输入: [B, 1+N, D] (含 CLS token)
    输出: [B, D] cls_token (经过 LaSt-ViT pooling)
    """
    cached_kernel = official_gaussian_kernel_1d(768, 768 ** 0.5).to(x_with_cls.device).unsqueeze(0).unsqueeze(0)
    
    x_detach = x_with_cls[:, 1:]                                  # [B, N, D]
    x = torch.fft.fft(x_with_cls[:, 1:], dim=-1)
    gs_k = cached_kernel.to(x.device)
    x = torch.fft.fftshift(x, dim=-1)
    x = x * (gs_k)
    x = torch.fft.ifftshift(x, dim=-1)
    x = torch.fft.ifft(x, dim=-1).real
    diff = x_detach / torch.abs(x - x_detach)                     # ★ 没 abs(diff), 没 +eps
    _, indices = torch.topk(diff, k=1, dim=1, largest=True)       # ★ 在 [B, N, D] 上 topk
    sel_p = torch.gather(x_detach, 1, indices)                    # [B, k=1, D]
    cls_token = torch.mean(sel_p, dim=1)                          # [B, D]
    return cls_token


# ============================================================
# 2. 我的实现 (从 MyModel.py 复制 lastvit_pool)
# ============================================================
def my_gaussian_kernel_1d(length, sigma):
    x = torch.arange(-length // 2 + 1, length // 2 + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / torch.max(kernel)


def my_lastvit_pool(F_p, k=1, sigma=None):
    """完全复制 model/MyModel.py 的 lastvit_pool 当前版本"""
    B, N, D = F_p.shape
    orig_dtype = F_p.dtype
    F_p_fp32 = F_p.float() if orig_dtype != torch.float32 else F_p

    x_detach = F_p_fp32
    x_freq = torch.fft.fft(F_p_fp32, dim=-1)

    if sigma is None:
        sigma = D ** 0.5
    gs_k = my_gaussian_kernel_1d(D, sigma).to(F_p_fp32.device)
    x_freq = torch.fft.fftshift(x_freq, dim=-1)
    x_freq = x_freq * gs_k
    x_freq = torch.fft.ifftshift(x_freq, dim=-1)
    x_lp = torch.fft.ifft(x_freq, dim=-1).real

    diff = x_detach / (torch.abs(x_lp - x_detach) + 1e-6)         # 我多了 +1e-6
    _, indices = torch.topk(diff, k=k, dim=1, largest=True)
    sel_p = torch.gather(x_detach, 1, indices)
    pooled = sel_p.mean(dim=1)

    return pooled.to(orig_dtype) if orig_dtype != torch.float32 else pooled


# ============================================================
# 3. 数值验证
# ============================================================
def main():
    # 固定 seed, 构造确定性输入
    torch.manual_seed(42)
    B, N, D = 2, 196, 768

    # 输入: [B, 1+N, D] (含 CLS), 模拟 ViT 输出
    x_full = torch.randn(B, 1 + N, D)              # [2, 197, 768]
    x_no_cls = x_full[:, 1:]                       # [2, 196, 768]

    # 官方版 (k=1)
    out_official = official_lastvit_forward(x_full)

    # 我的版 (k=1)
    out_mine = my_lastvit_pool(x_no_cls, k=1, sigma=None)

    # 对比
    print("=" * 60)
    print("LAST-ViT 实现一致性验证")
    print("=" * 60)
    print(f"输入 shape:        [B={B}, N={N}, D={D}]")
    print(f"官方输出 shape:    {tuple(out_official.shape)}")
    print(f"我的输出 shape:    {tuple(out_mine.shape)}")
    print()

    # 数值差
    diff_abs = (out_official - out_mine).abs()
    max_diff = diff_abs.max().item()
    mean_diff = diff_abs.mean().item()

    # cosine
    cos = torch.nn.functional.cosine_similarity(
        out_official.flatten().unsqueeze(0),
        out_mine.flatten().unsqueeze(0),
    ).item()

    print(f"最大绝对差:        {max_diff:.6e}")
    print(f"平均绝对差:        {mean_diff:.6e}")
    print(f"Cosine 相似度:     {cos:.10f}")
    print()

    # 判定
    if max_diff < 1e-5 and cos > 0.99999:
        print("✅ ✅ ✅  数值完全一致 (差异 < 1e-5, cosine > 0.99999)")
        print("        → 我的实现严格对齐官方")
        return 0
    elif max_diff < 1e-3 and cos > 0.999:
        print("⚠️  数值近似一致 (差异 < 1e-3)")
        print("    → 可能是 +1e-6 防除零造成的微小差异")
        # 进一步分析: 看是不是只有靠近 0 的 diff 受影响
        zero_mask = (diff_abs > 1e-5)
        print(f"    超过 1e-5 的元素占比: {zero_mask.float().mean().item()*100:.4f}%")
        return 0
    else:
        print("❌ ❌ ❌  数值显著不一致! 我的实现没对齐官方!")
        # 调试: 看哪一步差异最大
        return 1


if __name__ == "__main__":
    sys.exit(main())
