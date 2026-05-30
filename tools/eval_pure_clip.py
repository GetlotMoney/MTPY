"""
纯 CLIP Zero-Shot Baseline 评估脚本
=========================================
目的: 跑一个完全 zero-shot 的 CLIP 余弦分类作为论文 Table 的最底线对照.
- 视觉端: CLIP ViT-L/14@336px frozen, 仅取 CLS token
- 文本端: "a photo of a {class_name}[, a type of bird/animal]." 经 CLIP text encoder
- 不要 GPT 描述, 不要 Adapter, 不要 conditional prompt, 不要 FAE
- 完全不训练, 直接 GZSL 评估输出 H / U / S / ZSL

使用:
    python tools/eval_pure_clip.py --dataset CUB
    python tools/eval_pure_clip.py --dataset AWA2 --device cuda:0
    python tools/eval_pure_clip.py --dataset SUN  --device cuda:0
"""

import argparse
import os
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import clip

# 让脚本既能从根目录运行也能从 tools 目录运行
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# ★ 修复: 必须先把 PROJECT_ROOT 放到 sys.path 最前面, 并把 tools/ 自身从 sys.path 移除,
#   否则 "from tools.dataset import ..." 会被 tools/ 同级的 tools.py 抢先解析 (graphviz 报错).
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from tools.dataset import CUBDataLoader, AWA2DataLoader, SUNDataLoader   # noqa: E402
from tools.helper_func import (                                          # noqa: E402
    compute_per_class_acc,
    compute_per_class_acc_gzsl,
    map_label,
)


PROMPT_TEMPLATES = {
    'CUB':  "a photo of a {c}, a type of bird.",
    'AWA2': "a photo of a {c}.",
    'SUN':  "a photo of a {c}.",
}


def build_dataloader(dataset, device):
    """构造对应数据集的 DataLoader. 数据路径与训练脚本保持一致."""
    if dataset == 'CUB':
        return CUBDataLoader(data_path='./', device=device,
                             is_scale=False, is_unsupervised_attr=False, is_balance=True)
    if dataset == 'AWA2':
        return AWA2DataLoader(data_path='./', device=device,
                              is_scale=False, is_unsupervised_attr=False, is_balance=True)
    if dataset == 'SUN':
        return SUNDataLoader(data_path='./', device=device,
                             is_scale=False, is_unsupervised_attr=False, is_balance=True)
    raise ValueError(f"Unknown dataset: {dataset}")


def encode_class_text(class_names, template, clip_model, device):
    """用统一 prompt 模板编码所有类名 → [num_class, 768]"""
    prompts = [template.format(c=c.replace('_', ' ')) for c in class_names]
    text_inputs = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
    with torch.no_grad():
        text_embeds = clip_model.encode_text(text_inputs).float()
    return text_embeds


def load_or_extract_image_features(dataset, dataloader, clip_model, device):
    """优先用 CUB 已缓存的 test 特征 (秒出), 否则 online 提取 (用 CLIP CLS).

    返回 (seen_feat, seen_label, unseen_feat, unseen_label), feat 形状 [N, 768].
    """
    cache_dir = os.path.join(PROJECT_ROOT, 'data', 'cache')

    seen_feat_path   = os.path.join(cache_dir, f'{dataset}_test_seen_features.pt')
    seen_label_path  = os.path.join(cache_dir, f'{dataset}_test_seen_labels.pt')
    unseen_feat_path = os.path.join(cache_dir, f'{dataset}_test_unseen_features.pt')
    unseen_label_path = os.path.join(cache_dir, f'{dataset}_test_unseen_labels.pt')

    have_cache = (os.path.exists(seen_feat_path) and os.path.exists(unseen_feat_path)
                  and os.path.exists(seen_label_path) and os.path.exists(unseen_label_path))

    if have_cache:
        print(f"  [✓] Using cached test features under {cache_dir}/")
        seen_feat    = torch.load(seen_feat_path,   map_location=device, weights_only=True).float()
        seen_label   = torch.load(seen_label_path,  map_location=device, weights_only=True).long()
        unseen_feat  = torch.load(unseen_feat_path, map_location=device, weights_only=True).float()
        unseen_label = torch.load(unseen_label_path, map_location=device, weights_only=True).long()
        return seen_feat, seen_label, unseen_feat, unseen_label

    # 没缓存就 online 提取 CLIP CLS 特征
    print(f"  [·] No cache for {dataset}, extracting CLIP CLS features online...")

    def _extract(loader, name):
        feats, labels = [], []
        n = 0
        clip_model.eval()
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(device)
                # encode_image 返回 CLS [B, 768]
                f = clip_model.encode_image(imgs).float()
                feats.append(f.cpu())
                labels.append(lbls.cpu())
                n += imgs.size(0)
                if n % 500 < imgs.size(0):
                    print(f"      {name}: extracted {n} images...")
        return torch.cat(feats).to(device), torch.cat(labels).to(device).long()

    seen_feat,   seen_label   = _extract(dataloader.test_seen_loader,   'test_seen')
    unseen_feat, unseen_label = _extract(dataloader.test_unseen_loader, 'test_unseen')
    return seen_feat, seen_label, unseen_feat, unseen_label


@torch.no_grad()
def evaluate_pure_clip(dataset, device='cuda:0', logit_scale_value=100.0):
    """主流程: 加载 CLIP → 编码类文本 → 加载图像特征 → 余弦分类 → 输出指标"""
    print("=" * 64)
    print(f"  Pure CLIP Zero-Shot Baseline  |  Dataset: {dataset}")
    print(f"  CLIP: ViT-L/14@336px (frozen)")
    print(f"  Prompt: \"{PROMPT_TEMPLATES[dataset]}\"")
    print(f"  No GPT descriptions / No Adapter / No FAE / No conditional prompt")
    print("=" * 64)

    t0 = time.time()

    # 1) 加载 CLIP
    print("[1/5] Loading CLIP (ViT-L/14@336px)...")
    clip_model, _ = clip.load("ViT-L/14@336px", device=device)
    clip_model = clip_model.float()
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_model.eval()

    # 2) 加载数据集元信息
    print(f"[2/5] Loading {dataset} dataset metadata...")
    dataloader = build_dataloader(dataset, device)
    class_names    = dataloader.class_names
    seen_classes   = dataloader.seenclasses.long()
    unseen_classes = dataloader.unseenclasses.long()
    print(f"      Total classes: {len(class_names)}")
    print(f"      Seen / Unseen: {seen_classes.numel()} / {unseen_classes.numel()}")

    # 3) 编码类名文本
    print(f"[3/5] Encoding class text features with CLIP...")
    template = PROMPT_TEMPLATES[dataset]
    text_embeds = encode_class_text(class_names, template, clip_model, device)
    print(f"      text_embeds: {tuple(text_embeds.shape)}")

    # 4) 加载/提取图像 CLS 特征
    print(f"[4/5] Loading test image CLIP features...")
    seen_feat, seen_label, unseen_feat, unseen_label = load_or_extract_image_features(
        dataset, dataloader, clip_model, device)
    print(f"      test_seen   : {tuple(seen_feat.shape)}, labels in [{seen_label.min().item()}, {seen_label.max().item()}]")
    print(f"      test_unseen : {tuple(unseen_feat.shape)}, labels in [{unseen_label.min().item()}, {unseen_label.max().item()}]")

    # 5) 纯 CLIP 余弦分类
    print(f"[5/5] Computing CLIP cosine logits and per-class accuracy...")
    text_n = F.normalize(text_embeds, dim=-1)         # [num_class, 768]
    in_package = {'device': device}

    def _logits(feat):
        f_n = F.normalize(feat, dim=-1)
        return f_n @ text_n.T * logit_scale_value     # [B, num_class]

    # GZSL-S: 在全类空间预测 seen 测试样本
    logits_seen   = _logits(seen_feat)
    pred_gzsl_s   = torch.argmax(logits_seen, dim=1)
    acc_S = compute_per_class_acc_gzsl(seen_label, pred_gzsl_s, seen_classes, in_package)

    # GZSL-U: 在全类空间预测 unseen 测试样本
    logits_unseen = _logits(unseen_feat)
    pred_gzsl_u   = torch.argmax(logits_unseen, dim=1)
    acc_U = compute_per_class_acc_gzsl(unseen_label, pred_gzsl_u, unseen_classes, in_package)

    # ZSL: 仅在 unseen 类空间预测 unseen 测试样本
    pred_zsl = torch.argmax(logits_unseen[:, unseen_classes], dim=1)
    mapped   = map_label(unseen_label, unseen_classes).to(device)
    acc_ZSL  = compute_per_class_acc(mapped, pred_zsl, unseen_classes.numel())

    # Harmonic Mean
    H = 2 * acc_S * acc_U / (acc_S + acc_U) if (acc_S + acc_U) > 0 else 0.0

    elapsed = time.time() - t0

    # 输出
    print()
    print("=" * 64)
    print(f"  Pure CLIP Zero-Shot Result on {dataset}")
    print("=" * 64)
    print(f"  GZSL-U (Unseen Acc, search=all classes) : {acc_U*100:6.2f}%")
    print(f"  GZSL-S (Seen   Acc, search=all classes) : {acc_S*100:6.2f}%")
    print(f"  GZSL-H (Harmonic Mean)                  : {H*100:6.2f}%")
    print(f"  ZSL    (Unseen Acc, search=unseen only) : {acc_ZSL*100:6.2f}%")
    print(f"  Elapsed: {elapsed:.1f}s")
    print("=" * 64)
    return {'U': acc_U, 'S': acc_S, 'H': H, 'ZSL': acc_ZSL}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pure CLIP Zero-Shot Baseline')
    parser.add_argument('--dataset', type=str, default='CUB',
                        choices=['CUB', 'AWA2', 'SUN'])
    parser.add_argument('--device',  type=str, default='cuda:0')
    parser.add_argument('--logit_scale', type=float, default=100.0,
                        help='CLIP cosine 温度, 默认 100 (与训练时 logit_scale.exp() 上限一致)')
    args = parser.parse_args()

    evaluate_pure_clip(args.dataset, device=args.device,
                       logit_scale_value=args.logit_scale)
