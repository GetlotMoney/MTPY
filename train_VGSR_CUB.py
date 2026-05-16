import os
import torch
import torch.optim as optim
import numpy as np
import yaml
import clip
from types import SimpleNamespace
from datetime import datetime

from model.MyModel import VGSR
from tools.dataset import CUBDataLoader
from tools.helper_func import eval_zs_gzsl, get_clip_spatial_features

CACHE_DIR = './data/cache'
CACHE_TRAIN_FEAT   = os.path.join(CACHE_DIR, 'CUB_train_features.pt')
CACHE_TRAIN_LABEL  = os.path.join(CACHE_DIR, 'CUB_train_labels.pt')
CACHE_TRAIN_PATCH  = os.path.join(CACHE_DIR, 'CUB_train_patch_features.pt')  # [N, 576, 768] float16

# ==========================================
#   日志
# ==========================================
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
LOG_FILE = f"./train_log/CUB/training_log_CUB_{current_time}.txt"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


def print_log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding='utf-8') as f:
        f.write(msg + "\n")


# ==========================================
#   加载配置
# ==========================================
with open('./config/VGSR_cub_gzsl.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
config = {k: v['value'] if isinstance(v, dict) and 'value' in v else v for k, v in config.items()}
config = SimpleNamespace(**config)

if not hasattr(config, 'device'):
    config.device = 'cuda:0'

print_log("=" * 60)
print_log("  CLIP + Adapter + GPT  |  CUB GZSL Training")
print_log("=" * 60)
print_log(f"  Log file  : {LOG_FILE}")
print_log(f"  Device    : {config.device}")
print_log(f"  Epochs    : {config.epochs}")
print_log(f"  Batch size: {config.batch_size}")
print_log(f"  Adapter r : {getattr(config, 'adapter_ratio', 0.2)}")
print_log("=" * 60)

# ==========================================
#   加载 CLIP (冻结)
# ==========================================
print_log("\n[1/5] Loading CLIP (ViT-L/14@336px)...")
clip_model, _ = clip.load("ViT-L/14@336px", device=config.device)
clip_model = clip_model.float()
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False
print_log("      CLIP loaded & frozen.")

# ==========================================
#   加载数据
# ==========================================
print_log("\n[2/5] Loading CUB Dataset...")
dataloader = CUBDataLoader('.', config.device, is_balance=False)
print_log(f"      Train images : {dataloader.ntrain_clip}")
print_log(f"      Seen classes : {len(dataloader.seenclasses)}  (used for training)")
print_log(f"      Unseen classes: {len(dataloader.unseenclasses)}  (zero-shot target)")
print_log(f"      Test seen    : {len(dataloader.test_seen_loader.dataset)} images")
print_log(f"      Test unseen  : {len(dataloader.test_unseen_loader.dataset)} images")

# 固定随机种子
seed = config.random_seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# ==========================================
#   加载/提取训练集图像特征缓存
# ==========================================
# 优先级: patch 缓存 (完整 576) > CLS 缓存 (仅 CLS) > 实时提取
HAS_PATCH_CACHE = os.path.exists(CACHE_TRAIN_PATCH) and os.path.exists(CACHE_TRAIN_LABEL)
HAS_CLS_CACHE   = os.path.exists(CACHE_TRAIN_FEAT)  and os.path.exists(CACHE_TRAIN_LABEL)

train_patches  = None   # [N, 576, 768] float16 存 CPU (节省GPU显存)
train_features = None   # [N, 768]      float32 (仅向后兼容)
train_labels   = None

if HAS_PATCH_CACHE:
    print_log("\n[★] Loading train PATCH + CLS features from cache (fast mode)...")
    train_patches = torch.load(CACHE_TRAIN_PATCH, map_location='cpu')    # [N, 576, 768] float16
    train_labels  = torch.load(CACHE_TRAIN_LABEL, map_location=config.device)
    # 同时加载 CLS token (必须, 作为 CrossModalTransformer 的 cls_base 残差基础)
    train_cls = torch.load(CACHE_TRAIN_FEAT, map_location='cpu') if HAS_CLS_CACHE else None
    if train_cls is None:
        print_log("      WARNING: CLS cache missing, fallback to patch.mean()")
    else:
        print_log(f"      train_cls:     {train_cls.shape}  dtype={train_cls.dtype}")
    print_log(f"      train_patches: {train_patches.shape}  dtype={train_patches.dtype}  device=cpu")
    print_log(f"      train_labels:  {train_labels.shape}")
    USE_CACHE = 'patch'
elif HAS_CLS_CACHE:
    print_log("\n[★] Loading train CLS features from cache (legacy mode)...")
    train_features = torch.load(CACHE_TRAIN_FEAT,  map_location=config.device)
    train_labels   = torch.load(CACHE_TRAIN_LABEL, map_location=config.device)
    train_cls = None
    print_log(f"      train_features: {train_features.shape}  train_labels: {train_labels.shape}")
    USE_CACHE = 'cls'
else:
    print_log("\n[!] Cache not found. Will extract features on-the-fly (slow).")
    print_log("    Run: python tools/extract_features.py  to generate cache.")
    train_cls = None
    USE_CACHE = None

# ==========================================
#   生成类名文本特征 (CLIP 模板 prompt)
# ==========================================
print_log("\n[3/5] Encoding class name text features...")
class_names = [c.split('.')[-1].replace("_", " ") for c in dataloader.class_names]
prompts = [f"a photo of a {c}, a type of bird." for c in class_names]
text_inputs = torch.cat([clip.tokenize(p) for p in prompts]).to(config.device)
with torch.no_grad():
    class_text_embeds = clip_model.encode_text(text_inputs).float()  # [200, 768]
print_log(f"      class_text_embeds: {class_text_embeds.shape}")

# ==========================================
#   加载 GPT-4 描述并编码
# ==========================================
gpt_text_embeds = None
gpt4_data_path = os.path.join('.', 'data', 'gpt4_data', 'cub.pt')

if os.path.exists(gpt4_data_path):
    print_log("\n[4/5] Loading & encoding GPT-4 descriptions...")
    gpt4_sentences = torch.load(gpt4_data_path, map_location='cpu', weights_only=False)
    n_cls = len(gpt4_sentences)
    n_desc = len(list(gpt4_sentences.values())[0])
    print_log(f"      {n_cls} classes × {n_desc} descriptions/class")

    gpt_text_embeds_list = []
    hit, miss = 0, 0
    for cls_name in dataloader.class_names:
        gpt_key = '.'.join(cls_name.split('.')[1:]).lower()
        if gpt_key in gpt4_sentences:
            sentences = gpt4_sentences[gpt_key]
            tokens = torch.cat([clip.tokenize(s) for s in sentences]).to(config.device)
            with torch.no_grad():
                feats = clip_model.encode_text(tokens).float()
            gpt_text_embeds_list.append(feats.mean(dim=0))
            hit += 1
        else:
            idx = dataloader.class_names.index(cls_name)
            gpt_text_embeds_list.append(class_text_embeds[idx])
            miss += 1

    gpt_text_embeds = torch.stack(gpt_text_embeds_list)  # [200, 768]
    print_log(f"      GPT hit: {hit} classes | fallback: {miss} classes")
    print_log(f"      gpt_text_embeds: {gpt_text_embeds.shape}")
else:
    print_log(f"\n[4/5] WARNING: GPT-4 data not found at {gpt4_data_path}, using class name only")

# ==========================================
#   初始化模型
# ==========================================
print_log("\n[5/5] Initializing model...")

# 只取 seen 类的文本特征传给模型训练
# unseen 类的原始 CLIP 文本特征用于评估时拼接
seen_gpt_embeds = gpt_text_embeds[dataloader.seenclasses] \
    if gpt_text_embeds is not None else class_text_embeds[dataloader.seenclasses]
# unseen 类也用 GPT 7句平均（语义更丰富），不再用类名模板
# GPT 描述是纯文本语义，不含视觉监督信号，不算信息泄露
unseen_clip_embeds = gpt_text_embeds[dataloader.unseenclasses] \
    if gpt_text_embeds is not None else class_text_embeds[dataloader.unseenclasses]

model = VGSR(
    config,
    dataloader.seenclasses,
    dataloader.unseenclasses,
    seen_text_embeds=seen_gpt_embeds,       # [150, 768] seen 类 GPT 文本
    unseen_text_embeds=unseen_clip_embeds,  # [50, 768]  unseen 类原始 CLIP 文本
).to(config.device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print_log(f"      Total params    : {total_params:,}")
print_log(f"      Trainable params: {trainable_params:,}")

# ==========================================
#   优化器 & 调度器
# ==========================================
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

# ==========================================
#   训练循环
# ==========================================
iters_per_epoch = dataloader.ntrain_clip // config.batch_size
total_iters = iters_per_epoch * config.epochs

best_H = 0.0
best_metrics = {'U': 0, 'S': 0, 'H': 0, 'ZS': 0, 'epoch': 0}

# 最佳模型保存路径
BEST_MODEL_DIR = './train_log/CUB'
BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, f'best_model_CUB_{current_time}.pth')

print_log("\n" + "=" * 60)
print_log(f"  Start Training: {config.epochs} epochs × {iters_per_epoch} iters/epoch = {total_iters} iters")
print_log("=" * 60)

for epoch in range(1, config.epochs + 1):

    # ---------- 训练阶段 ----------
    model.train()
    epoch_loss = 0.0
    epoch_iters = 0

    print_log(f"\n{'─'*60}")
    print_log(f"  Epoch [{epoch}/{config.epochs}]  Training...")
    print_log(f"{'─'*60}")

    for step in range(iters_per_epoch):
        optimizer.zero_grad()

        if USE_CACHE == 'patch':
            # ── 随机采样：每步随机取 batch ──
            idx = torch.randperm(len(train_patches))[:config.batch_size]
            batch_label = train_labels[idx]
            patch_batch = train_patches[idx].to(config.device).float()  # [B, 576, 768]
            if train_cls is not None:
                cls_batch = train_cls[idx].to(config.device).float().unsqueeze(1)  # [B, 1, 768]
                clip_features = torch.cat([cls_batch, patch_batch], dim=1)  # [B, 577, 768]
            else:
                cls_batch = patch_batch.mean(dim=1, keepdim=True)
                clip_features = torch.cat([cls_batch, patch_batch], dim=1)
        elif USE_CACHE == 'cls':
            idx = torch.randperm(len(train_features))[:config.batch_size]
            batch_label = train_labels[idx]
            clip_features = train_features[idx].unsqueeze(1)  # [B, 1, 768]
        else:
            # ── 实时模式：读图 → CLIP 提取特征 ──
            batch_label, batch_images, batch_att = dataloader.next_batch(config.batch_size)
            clip_features = get_clip_spatial_features(clip_model, batch_images).float()

        # Forward (训练模式：只用 150 seen 类 logits)
        out_package = model(clip_features, is_train=True)
        in_package = out_package.copy()
        in_package['batch_label'] = batch_label
        loss_pack = model.compute_loss(in_package)  # logits_200 已在 out_package 里
        loss = loss_pack['loss']

        # Backward
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_iters += 1

        # 每 20 步打印一次进度
        if (step + 1) % 20 == 0 or (step + 1) == iters_per_epoch:
            avg_loss = epoch_loss / epoch_iters
            print_log(f"  Step [{step+1:3d}/{iters_per_epoch}] | "
                      f"Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f}")

    # 更新学习率
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    avg_epoch_loss = epoch_loss / epoch_iters

    print_log(f"\n  >> Epoch [{epoch}/{config.epochs}] Train Summary")
    print_log(f"     Avg Loss : {avg_epoch_loss:.4f}")
    print_log(f"     LR       : {current_lr:.6f}")
    # ---------- 测试阶段 ----------
    print_log(f"\n  >> Epoch [{epoch}/{config.epochs}] Evaluating GZSL...")
    gzsl_bias = getattr(config, 'gzsl_bias', 0.0)
    acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(
        dataloader, clip_model, model, config.device,
        bias_unseen=gzsl_bias)

    # 更新最佳结果
    if H > best_H:
        best_H = H
        best_metrics = {
            'U': acc_novel,
            'S': acc_seen,
            'H': H,
            'ZS': acc_zs,
            'epoch': epoch
        }
        # 保存最佳模型权重
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print_log(f"  [★] Best model saved → {BEST_MODEL_PATH}")

    # 打印当前 epoch 结果
    print_log(f"\n  ┌─ Epoch [{epoch}/{config.epochs}] Results ─────────────────────")
    print_log(f"  │  GZSL-U (Unseen Acc) : {acc_novel*100:.2f}%")
    print_log(f"  │  GZSL-S (Seen Acc)   : {acc_seen*100:.2f}%")
    print_log(f"  │  GZSL-H (Harmonic)   : {H*100:.2f}%  {'★ NEW BEST' if H == best_H else ''}")
    print_log(f"  │  ZSL    (ZS Acc)     : {acc_zs*100:.2f}%")
    print_log(f"  └──────────────────────────────────────────────────────")

# ==========================================
#   最终汇总
# ==========================================
print_log("\n" + "=" * 60)
print_log("  Training Finished!")
print_log("=" * 60)
print_log(f"  Best Results @ Epoch {best_metrics['epoch']}")
print_log(f"  ┌─────────────────────────────────────")
print_log(f"  │  GZSL-U : {best_metrics['U']*100:.2f}%")
print_log(f"  │  GZSL-S : {best_metrics['S']*100:.2f}%")
print_log(f"  │  GZSL-H : {best_metrics['H']*100:.2f}%")
print_log(f"  │  ZSL    : {best_metrics['ZS']*100:.2f}%")
print_log(f"  └─────────────────────────────────────")
print_log(f"\n  Log saved to: {LOG_FILE}")
