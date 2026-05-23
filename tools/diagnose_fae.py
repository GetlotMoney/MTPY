"""
诊断 FAE 是否真的在工作
======================
通过加载训练好的 best_model 检查：
1. WGs（几何权重生成器）是否学到非零权重
2. geo_weights 的实际数值范围
3. FAE 输出与无 FAE 输出的差异程度
4. 注意力分布是否真的被几何偏置改变
"""
import os, sys, torch
import numpy as np
import yaml
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.MyModel import VGSR

# 加载配置（强制开启 FAE）
with open('./config/VGSR_cub_gzsl.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
config = {k: v['value'] if isinstance(v, dict) and 'value' in v else v for k, v in config.items()}
config['use_fae'] = True
config = SimpleNamespace(**config)
device = config.device

# 找最新的有 FAE 的 best_model
import glob
candidates = sorted(glob.glob('train_log/CUB/best_model_CUB_*.pth'), reverse=True)
ckpt_path = None
for c in candidates:
    sz = os.path.getsize(c) / 1e6
    # 完整 FAE 模型约 80MB（无 FAE 是 31MB）
    if sz > 60:
        ckpt_path = c
        break

print(f"使用 checkpoint: {ckpt_path}")
print(f"  size: {os.path.getsize(ckpt_path)/1e6:.1f}MB")

# 构造模型
seenclass = torch.arange(150).to(device)
unseenclass = torch.arange(150, 200).to(device)
seen_text = torch.randn(150, 768).to(device)
unseen_text = torch.randn(50, 768).to(device)

model = VGSR(config, seenclass, unseenclass, seen_text, unseen_text).to(device)
state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"  missing keys: {len(missing)}")
print(f"  unexpected keys: {len(unexpected)}")
model.eval()

# ===== 检查 1: WGs (几何权重生成器) 是否学到了东西 =====
print("\n" + "=" * 60)
print("检查 1: WGs (4 个 head 的几何权重生成器) 参数")
print("=" * 60)
fae_attn = model.cross_tf.fae.attn
for h_idx, layer in enumerate(fae_attn.WGs):
    w = layer.weight.detach().cpu().numpy().flatten()
    b = layer.bias.detach().cpu().numpy().item()
    print(f"  head {h_idx}: weight ‖·‖={np.linalg.norm(w):.4f} "
          f"(mean={w.mean():.4f}, std={w.std():.4f}), bias={b:.4f}")

# ===== 检查 2: 几何编码 + WGs 实际产出的 geo_weights =====
print("\n" + "=" * 60)
print("检查 2: geo_weights 实际数值范围")
print("=" * 60)
B = 1
geo_emb = model.cross_tf.box_emb(B).to(device)         # [1, 576, 576, 64]
print(f"  geo_emb shape: {geo_emb.shape}, dtype: {geo_emb.dtype}")
print(f"  geo_emb stats: min={geo_emb.float().min().item():.3f} "
      f"max={geo_emb.float().max().item():.3f} "
      f"mean={geo_emb.float().mean().item():.3f}")

geo_flat = geo_emb.float().reshape(-1, 64)
import torch.nn.functional as F
geo_per_head = [layer(geo_flat).view(B, 576, 576, 1).permute(0, 3, 1, 2)
                for layer in fae_attn.WGs]
geo_weights = F.relu(torch.cat(geo_per_head, dim=1))      # [1, 4, 576, 576]
print(f"\n  geo_weights shape: {geo_weights.shape}")
print(f"  geo_weights stats:")
print(f"    min:  {geo_weights.min().item():.4f}")
print(f"    max:  {geo_weights.max().item():.4f}")
print(f"    mean: {geo_weights.mean().item():.4f}")
print(f"    std:  {geo_weights.std().item():.4f}")
print(f"    >0 比例: {(geo_weights > 0).float().mean().item()*100:.1f}%")

# 对每个 head 单独看
for h in range(4):
    gw_h = geo_weights[0, h]
    print(f"  head {h}: max={gw_h.max().item():.3f}, "
          f"mean={gw_h.mean().item():.3f}, "
          f"{(gw_h > 0).float().mean().item()*100:.1f}% non-zero")

# ===== 检查 3: 跑一张真图，对比 FAE 前后 vis 的差异 =====
print("\n" + "=" * 60)
print("检查 3: FAE 前后视觉特征差异")
print("=" * 60)

# 加载一张真实测试 patch
test_patch = torch.load(
    './data/cache/CUB_test_unseen_patch_features.pt',
    map_location='cpu', weights_only=True
)[0:1].to(device).float()                # [1, 576, 768]
print(f"  test patch shape: {test_patch.shape}")

with torch.no_grad():
    vis = model.cross_tf.embed_cv(test_patch)             # [1, 576, 512]
    geo_emb = model.cross_tf.box_emb(1).to(device)
    memory = model.cross_tf.fae(vis, geo_emb)             # [1, 576, 512]

diff = (memory - vis).abs()
print(f"  vis  stats: mean={vis.abs().mean().item():.4f}, std={vis.std().item():.4f}")
print(f"  mem stats: mean={memory.abs().mean().item():.4f}, std={memory.std().item():.4f}")
print(f"  |memory - vis| stats:")
print(f"    mean: {diff.mean().item():.4f}")
print(f"    max:  {diff.max().item():.4f}")
print(f"    余弦相似度 vis vs memory:")

# 比较 vis 和 memory 的余弦相似度（每个 patch）
cos = F.cosine_similarity(vis, memory, dim=-1)             # [1, 576]
print(f"    mean: {cos.mean().item():.4f}")
print(f"    min:  {cos.min().item():.4f}")
print(f"    max:  {cos.max().item():.4f}")

# ===== 检查 4: 比较"减 geo_weights 前后"的 attention 分布 =====
print("\n" + "=" * 60)
print("检查 4: FAE attention 是否真被 geo_weights 影响")
print("=" * 60)

with torch.no_grad():
    q = fae_attn.fc_q(vis).view(1, 576, 4, 128).permute(0, 2, 1, 3)
    k = fae_attn.fc_k(vis).view(1, 576, 4, 128).permute(0, 2, 1, 3)
    att_raw = torch.matmul(q, k.transpose(-2, -1)) / (128 ** 0.5)        # [1, 4, 576, 576]
    att_after = att_raw - geo_weights

    softmax_raw   = F.softmax(att_raw, dim=-1)
    softmax_after = F.softmax(att_after, dim=-1)

    diff_attn = (softmax_after - softmax_raw).abs()
    print(f"  raw attention stats:   mean={att_raw.mean().item():.3f}, std={att_raw.std().item():.3f}")
    print(f"  after - geo stats:     mean={att_after.mean().item():.3f}, std={att_after.std().item():.3f}")
    print(f"  |softmax_after - softmax_raw|:")
    print(f"    mean: {diff_attn.mean().item():.6f}")
    print(f"    max:  {diff_attn.max().item():.4f}")

    # 看 attention 主要权重
    top_raw = softmax_raw.max(dim=-1).values.mean().item()
    top_after = softmax_after.max(dim=-1).values.mean().item()
    print(f"  每个 query patch 的最大注意力权重:")
    print(f"    无 geo_weights: {top_raw:.4f}")
    print(f"    有 geo_weights: {top_after:.4f}")

print("\n" + "=" * 60)
print("结论判断")
print("=" * 60)
gw_mean = geo_weights.mean().item()
diff_mean = diff_attn.mean().item()

if gw_mean < 0.01:
    print("⚠️  WGs 学到的 geo_weights 几乎为 0 → FAE 几何解耦机制未生效")
elif diff_mean < 0.0001:
    print(f"⚠️  geo_weights 数值正常 ({gw_mean:.3f}) 但对 attention 分布影响极小 → 实际效果有限")
else:
    print(f"✓  FAE 在工作: geo_weights mean={gw_mean:.3f}, attention diff mean={diff_mean:.5f}")
