"""
预提取 CLIP 图像特征并缓存到文件
==========================================================
默认: 单视角 (无增强)，速度最快，所有训练样本特征固定
可选: 多视角 (NUM_VIEWS=K)，训练集每张图存 K 个增强版本，
      模拟数据增强但仍走缓存模式（训练时随机抽 1 个视角）

用法:
    F:\Anaconda\envs\dassl_clip\python.exe tools/extract_features.py [num_views]

    示例:
      python tools/extract_features.py        # 默认单视角
      python tools/extract_features.py 4      # 4 视角增强缓存

输出文件:
    单视角:
      data/cache/CUB_train_features.pt        [N, 768]      float32
      data/cache/CUB_train_patch_features.pt  [N, 576, 768] float16
      data/cache/CUB_train_labels.pt          [N]
    多视角 (K>1):
      data/cache/CUB_train_features_aug.pt        [K, N, 768]      float32
      data/cache/CUB_train_patch_features_aug.pt  [K, N, 576, 768] float16
      data/cache/CUB_train_labels.pt              [N]
      data/cache/CUB_train_views.pt               (标量) K

注意:
    1. 测试集永远只用 test_transform 提取一次 (评估必须确定性)
    2. 多视角缓存仅训练集生效
    3. K=4 时 patch 文件 ~25GB, 注意磁盘空间
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

import importlib.util


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_dataset_mod = _load_module("dataset", os.path.join(_root, "tools", "dataset.py"))
_helper_mod  = _load_module("helper_func", os.path.join(_root, "tools", "helper_func.py"))

CUBDataLoader = _dataset_mod.CUBDataLoader
ImgDataset    = _dataset_mod.ImgDataset
get_clip_spatial_features = _helper_mod.get_clip_spatial_features

DEVICE = 'cuda:0'
CACHE_DIR = './data/cache'
BATCH_SIZE = 64

# ★ 多视角参数: 命令行第 1 个参数, 默认 1 (无增强)
NUM_VIEWS = int(sys.argv[1]) if len(sys.argv) > 1 else 1

os.makedirs(CACHE_DIR, exist_ok=True)

print("=" * 60)
print(f"  CLIP Feature Extraction for CUB  (NUM_VIEWS = {NUM_VIEWS})")
print("=" * 60)

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

# 测试集 transform (确定性, 永远用)
test_transform = transforms.Compose([
    transforms.Resize((336, 336), interpolation=Image.BICUBIC),
    transforms.CenterCrop(336),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711)),
])

# 训练集增强 transform (多视角时用)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(336, scale=(0.6, 1.0),
                                 interpolation=Image.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711)),
])

print(f"      Train images : {len(dataloader.train_dataset)}")
print(f"      Test seen    : {len(dataloader.test_seen_dataset)}")
print(f"      Test unseen  : {len(dataloader.test_unseen_dataset)}")


def extract_features(dataset, desc, return_patches=False):
    """提取整个 dataset 的 CLIP 特征 (走 1 次)"""
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)
    all_cls = []
    all_patches = []
    all_labels = []
    total = len(loader)

    for i, (images, labels) in enumerate(loader):
        images = images.to(DEVICE)
        with torch.no_grad():
            spatial = get_clip_spatial_features(clip_model, images).float()
        all_cls.append(spatial[:, 0, :].cpu())
        if return_patches:
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


print("\n[3/3] Extracting features...")

# ============================================================
#   训练集: 单视角 / 多视角
# ============================================================
if NUM_VIEWS <= 1:
    # ── 单视角 (无增强), 与之前一致 ──
    print("\n  [Train] Single-view extraction (no augmentation)")
    train_dataset_no_aug = ImgDataset(
        dataloader.train_dataset.image_files,
        dataloader.train_dataset.labels,
        'CUB', test_transform, dataloader.root_dir,
    )
    train_data = extract_features(train_dataset_no_aug, "Train", return_patches=True)

    torch.save(train_data['cls'],
               os.path.join(CACHE_DIR, 'CUB_train_features.pt'))
    torch.save(train_data['labels'],
               os.path.join(CACHE_DIR, 'CUB_train_labels.pt'))
    torch.save(train_data['patches'],
               os.path.join(CACHE_DIR, 'CUB_train_patch_features.pt'))

    print(f"      Train CLS:   {train_data['cls'].shape}  dtype={train_data['cls'].dtype}")
    print(f"      Train Patch: {train_data['patches'].shape}  dtype={train_data['patches'].dtype}")
    patch_mb = train_data['patches'].element_size() * train_data['patches'].nelement() / (1024**2)
    print(f"      Train patch size: {patch_mb:.1f} MB")
else:
    # ── 多视角 (K 个增强版本) ──
    print(f"\n  [Train] Multi-view extraction (K = {NUM_VIEWS} views)")
    print(f"  ★ 警告: 磁盘占用约 {NUM_VIEWS * 6.2:.1f} GB")

    all_cls_views    = []
    all_patches_views = []
    train_labels_first = None

    for k in range(NUM_VIEWS):
        # 每个 view 用不同的随机种子产生不同增强
        torch.manual_seed(1000 + k)
        np.random.seed(1000 + k)

        train_dataset_aug = ImgDataset(
            dataloader.train_dataset.image_files,
            dataloader.train_dataset.labels,
            'CUB', train_transform, dataloader.root_dir,
        )
        print(f"\n  ── View [{k+1}/{NUM_VIEWS}] ──")
        view_data = extract_features(train_dataset_aug, f"Train-V{k+1}",
                                     return_patches=True)
        all_cls_views.append(view_data['cls'])           # [N, 768]
        all_patches_views.append(view_data['patches'])   # [N, 576, 768] f16
        if train_labels_first is None:
            train_labels_first = view_data['labels']

    # 堆叠成 [K, N, ...]
    cls_aug    = torch.stack(all_cls_views, dim=0)        # [K, N, 768]
    patch_aug  = torch.stack(all_patches_views, dim=0)    # [K, N, 576, 768]

    torch.save(cls_aug,
               os.path.join(CACHE_DIR, 'CUB_train_features_aug.pt'))
    torch.save(patch_aug,
               os.path.join(CACHE_DIR, 'CUB_train_patch_features_aug.pt'))
    torch.save(train_labels_first,
               os.path.join(CACHE_DIR, 'CUB_train_labels.pt'))
    torch.save(torch.tensor(NUM_VIEWS),
               os.path.join(CACHE_DIR, 'CUB_train_views.pt'))

    print(f"\n      Train CLS (aug):   {cls_aug.shape}  dtype={cls_aug.dtype}")
    print(f"      Train Patch (aug): {patch_aug.shape}  dtype={patch_aug.dtype}")
    patch_gb = patch_aug.element_size() * patch_aug.nelement() / (1024**3)
    print(f"      Train patch size: {patch_gb:.2f} GB")

# ============================================================
#   测试集 (永远单视角 + test_transform)
# ============================================================
print("\n  [Test] Single-view extraction (deterministic)")
seen_data = extract_features(dataloader.test_seen_dataset, "Test-Seen", return_patches=True)
torch.save(seen_data['cls'],     os.path.join(CACHE_DIR, 'CUB_test_seen_features.pt'))
torch.save(seen_data['labels'],  os.path.join(CACHE_DIR, 'CUB_test_seen_labels.pt'))
torch.save(seen_data['patches'], os.path.join(CACHE_DIR, 'CUB_test_seen_patch_features.pt'))
print(f"      Test-Seen CLS:   {seen_data['cls'].shape}")
print(f"      Test-Seen Patch: {seen_data['patches'].shape}")

unseen_data = extract_features(dataloader.test_unseen_dataset, "Test-Unseen", return_patches=True)
torch.save(unseen_data['cls'],     os.path.join(CACHE_DIR, 'CUB_test_unseen_features.pt'))
torch.save(unseen_data['labels'],  os.path.join(CACHE_DIR, 'CUB_test_unseen_labels.pt'))
torch.save(unseen_data['patches'], os.path.join(CACHE_DIR, 'CUB_test_unseen_patch_features.pt'))
print(f"      Test-Unseen CLS:   {unseen_data['cls'].shape}")
print(f"      Test-Unseen Patch: {unseen_data['patches'].shape}")

print("\n" + "=" * 60)
print("  All features extracted and saved to data/cache/")
if NUM_VIEWS > 1:
    print(f"  Multi-view cache (K={NUM_VIEWS}) created.")
    print("  Set config.use_aug_cache: True in YAML to enable in training.")
print("=" * 60)
