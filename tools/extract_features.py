"""
预提取 CLIP 图像特征并缓存到文件
运行一次后，训练脚本自动读取缓存，无需每次实时提取

用法:
    F:\Anaconda\envs\dvsr_gpu\python.exe tools/extract_features.py

输出文件 (float16 存储, 节省一半内存):
    data/cache/CUB_train_patch_features.pt    → 训练集 patch 特征 [7057, 576, 768] float16 (~6.2GB)
    data/cache/CUB_train_labels.pt            → 训练集标签 [7057]
    data/cache/CUB_train_features.pt          → 训练集 CLS 特征 [7057, 768] float32 (兼容旧逻辑)

注意:
    1. 训练集提取时使用 test_transform (无数据增强)，特征固定不变
    2. 只缓存训练集 (重复使用), 测试集实时提取 (只跑一次, 不是瓶颈)
    3. 使用 float16 存储 patch 特征, 训练时在 GPU 上转回 float32
"""

import os
import sys
import torch
import clip
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import scipy.io as sio

# 直接把 tools 目录下的文件单独导入，绕过 tools/__init__.py 和 tools/tools.py
import importlib.util

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_dataset_mod = _load_module("dataset", os.path.join(_root, "tools", "dataset.py"))
_helper_mod  = _load_module("helper_func", os.path.join(_root, "tools", "helper_func.py"))

CUBDataLoader          = _dataset_mod.CUBDataLoader
ImgDataset             = _dataset_mod.ImgDataset
get_clip_spatial_features = _helper_mod.get_clip_spatial_features

DEVICE = 'cuda:0'
CACHE_DIR = './data/cache'
BATCH_SIZE = 64

os.makedirs(CACHE_DIR, exist_ok=True)

print("=" * 50)
print("  CLIP Feature Extraction for CUB")
print("=" * 50)

# ── 加载 CLIP ──
print("\n[1/3] Loading CLIP (ViT-L/14@336px)...")
clip_model, _ = clip.load("ViT-L/14@336px", device=DEVICE)
clip_model = clip_model.float().eval()
for p in clip_model.parameters():
    p.requires_grad = False
print("      Done.")

# ── 加载数据集元数据 ──
print("\n[2/3] Loading CUB metadata...")
dataloader = CUBDataLoader('.', DEVICE, is_balance=False)

# 训练集用无增强的 transform（特征才能固定复用）
test_transform = transforms.Compose([
    transforms.Resize((336, 336), interpolation=Image.BICUBIC),
    transforms.CenterCrop(336),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711)),
])

# 重新构建训练集 Dataset（用 test_transform，无增强）
import numpy as np
import scipy.io as sio

train_dataset_no_aug = ImgDataset(
    dataloader.train_dataset.image_files,
    dataloader.train_dataset.labels,
    'CUB', test_transform,
    dataloader.root_dir
)

print(f"      Train: {len(train_dataset_no_aug)} images")
print(f"      Test seen: {len(dataloader.test_seen_dataset)} images")
print(f"      Test unseen: {len(dataloader.test_unseen_dataset)} images")


def extract_features(dataset, desc, return_patches=False):
    """
    提取整个 dataset 的 CLIP 特征
    Args:
        return_patches: True 返回完整 patch [N, 576, 768], False 只返回 CLS [N, 768]
    """
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)
    all_cls = []
    all_patches = []
    all_labels = []
    total = len(loader)

    for i, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        with torch.no_grad():
            spatial = get_clip_spatial_features(clip_model, images).float()  # [B, 577, 768]
        all_cls.append(spatial[:, 0, :].cpu())                                # CLS
        if return_patches:
            # float16 存储节省一半空间, [B, 576, 768]
            all_patches.append(spatial[:, 1:, :].half().cpu())
        all_labels.append(labels.cpu())

        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"      {desc}: [{i+1}/{total}] batches done", end='\r')

    print()
    result = {
        'cls':    torch.cat(all_cls, dim=0),
        'labels': torch.cat(all_labels, dim=0),
    }
    if return_patches:
        result['patches'] = torch.cat(all_patches, dim=0)
    return result


# ── 提取特征 ──
print("\n[3/3] Extracting features...")

# 训练集: 同时提取 CLS + patch (patch 用 float16 存, 节省空间)
train_data = extract_features(train_dataset_no_aug, "Train", return_patches=True)

torch.save(train_data['cls'],
           os.path.join(CACHE_DIR, 'CUB_train_features.pt'))
torch.save(train_data['labels'],
           os.path.join(CACHE_DIR, 'CUB_train_labels.pt'))
torch.save(train_data['patches'],
           os.path.join(CACHE_DIR, 'CUB_train_patch_features.pt'))

print(f"      Train CLS:   {train_data['cls'].shape}  dtype={train_data['cls'].dtype}")
print(f"      Train Patch: {train_data['patches'].shape}  dtype={train_data['patches'].dtype}")

# 尺寸信息
patch_mb = train_data['patches'].element_size() * train_data['patches'].nelement() / (1024**2)
print(f"      Train patch size: {patch_mb:.1f} MB")

# 测试集: 只存 CLS + labels (评估只需要 CLS，不需要 patch)
# test_seen
seen_data = extract_features(dataloader.test_seen_dataset, "Test-Seen", return_patches=False)
torch.save(seen_data['cls'],    os.path.join(CACHE_DIR, 'CUB_test_seen_features.pt'))
torch.save(seen_data['labels'], os.path.join(CACHE_DIR, 'CUB_test_seen_labels.pt'))
print(f"      Test-Seen CLS: {seen_data['cls'].shape}")

# test_unseen
unseen_data = extract_features(dataloader.test_unseen_dataset, "Test-Unseen", return_patches=False)
torch.save(unseen_data['cls'],    os.path.join(CACHE_DIR, 'CUB_test_unseen_features.pt'))
torch.save(unseen_data['labels'], os.path.join(CACHE_DIR, 'CUB_test_unseen_labels.pt'))
print(f"      Test-Unseen CLS: {unseen_data['cls'].shape}")

print("\n" + "=" * 50)
print("  All features extracted and saved to data/cache/")
print("  Next time training will load from cache automatically.")
print("=" * 50)
