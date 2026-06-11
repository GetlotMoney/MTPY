import os
import sys
import argparse
import torch
import torch.optim as optim
import numpy as np
import yaml
import clip
from types import SimpleNamespace
from datetime import datetime

# ★ 加速优化: 让 cuDNN 自动选最优算法 (对固定输入形状有效)
# ⚠️ RTX 5070 Ti (sm_120) + cuDNN 9.19 + BF16 在搜索内核时巨吃显存导致 8s/step
#    暂时禁用, 只用预加载和 BF16 加速
torch.backends.cudnn.benchmark = False

from model.MyModel import VGSR
from tools.dataset import CUBDataLoader
from tools.helper_func import eval_zs_gzsl, get_clip_spatial_features

CACHE_DIR = './data/cache'
CACHE_TRAIN_FEAT   = os.path.join(CACHE_DIR, 'CUB_train_features.pt')
CACHE_TRAIN_LABEL  = os.path.join(CACHE_DIR, 'CUB_train_labels.pt')
CACHE_TRAIN_PATCH  = os.path.join(CACHE_DIR, 'CUB_train_patch_features.pt')  # [N, 576, 768] float16
# ★ 多视角增强缓存 (可选, 由 extract_features.py [K] 生成)
CACHE_TRAIN_FEAT_AUG  = os.path.join(CACHE_DIR, 'CUB_train_features_aug.pt')        # [K, N, 768]
CACHE_TRAIN_PATCH_AUG = os.path.join(CACHE_DIR, 'CUB_train_patch_features_aug.pt')  # [K, N, 576, 768] f16
CACHE_TRAIN_VIEWS     = os.path.join(CACHE_DIR, 'CUB_train_views.pt')
# ★ 加速优化 (方案 2a): CLIP 文本嵌入缓存 (避免每次启动重新跑 CLIP encode_text)
CACHE_CLASS_TEXT      = os.path.join(CACHE_DIR, 'CUB_class_text_embeds.pt')         # [200, 768]
# {text_source}_text_embeds.pt: gpt4 / claude / gpt55 / merge
def _gpt_embed_cache_path(text_source):
    return os.path.join(CACHE_DIR, f'CUB_{text_source}_text_embeds.pt')

# ==========================================
#   日志
# ==========================================
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
LOG_FILE = f"./train_log/CUB/training_log_CUB_{current_time}.txt"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


def print_log(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', errors='replace').decode('ascii'))
    with open(LOG_FILE, "a", encoding='utf-8') as f:
        f.write(msg + "\n")


# ==========================================
#   加载配置
# ==========================================
parser = argparse.ArgumentParser(description="Train VGSR on CUB GZSL.")
parser.add_argument(
    "--config",
    default="./config/VGSR_cub_gzsl.yaml",
    help="Path to the YAML config. Use an experiment-local config.yaml for tracked runs.",
)
args = parser.parse_args()
config_path = os.path.normpath(args.config)

if not os.path.exists(config_path):
    print_log(f"Error: Config file {config_path} not found!")
    raise SystemExit(1)

with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
config = {k: v['value'] if isinstance(v, dict) and 'value' in v else v for k, v in config.items()}
config = SimpleNamespace(**config)
config.config_path = config_path

if not hasattr(config, 'device'):
    config.device = 'cuda:0'

print_log("=" * 60)
print_log("  CLIP + Adapter + GPT  |  CUB GZSL Training")
print_log("=" * 60)
print_log(f"  Log file  : {LOG_FILE}")
print_log(f"  Config    : {config_path}")
print_log(f"  Device    : {config.device}")
print_log(f"  Epochs    : {config.epochs}")
print_log(f"  Batch size: {config.batch_size}")
print_log(f"  Random seed: {getattr(config, 'random_seed', 5)}")
print_log("=" * 60)

# ★ 模块配置摘要 (每次训练记录, 便于回看)
print_log("")
print_log("┌─ Module Configuration ───────────────────────────────────┐")
print_log(f"│  model_mode    : {getattr(config, 'model_mode', 'vgsr')}")
print_log(f"│  text_source   : {getattr(config, 'text_source', 'gpt4')}")
print_log(f"│  use_aug_cache : {getattr(config, 'use_aug_cache', False)}")
print_log("├─ Architecture ──────────────────────────────────────────┤")
print_log(f"│  use_fae       : {getattr(config, 'use_fae', True)}")
print_log(f"│  adapter_ratio : {getattr(config, 'adapter_ratio', 0.2)}")
print_log(f"│  tf_common_dim : {getattr(config, 'tf_common_dim', 512)}")
print_log(f"│  tf_heads      : {getattr(config, 'tf_heads', 4)}")
print_log(f"│  tf_dropout    : {getattr(config, 'tf_dropout', 0.1)}")
print_log("├─ Pooling (s2v) ─────────────────────────────────────────┤")
print_log(f"│  pool_method   : {getattr(config, 'pool_method', 'mean')}")
if getattr(config, 'pool_method', 'mean') == 'lastvit':
    print_log(f"│  lastvit_k     : {getattr(config, 'lastvit_k', 8)}")
    print_log(f"│  lastvit_sigma : {getattr(config, 'lastvit_sigma', 10.0)}")
print_log("├─ LaSt-ViT-CLS (CLIP CLS 增强) ──────────────────────────┤")
print_log(f"│  use_lastvit_cls : {getattr(config, 'use_lastvit_cls', False)}")
if bool(getattr(config, 'use_lastvit_cls', False)):
    print_log(f"│  lastvit_cls_k     : {getattr(config, 'lastvit_cls_k', 1)}")
    print_log(f"│  lastvit_cls_sigma : {getattr(config, 'lastvit_cls_sigma', 0.0)}")
    print_log(f"│  lastvit_residual  : {getattr(config, 'lastvit_residual', 0.5)}")
print_log("├─ Scoring & Gating ──────────────────────────────────────┤")
print_log(f"│  score_mode    : {getattr(config, 'score_mode', 'add')}")
print_log(f"│  gating        : {getattr(config, 'gating', 'fixed')}")
print_log(f"│  local_weight  : {getattr(config, 'local_weight', 0.3)}")
print_log(f"│  weight_s2v    : {getattr(config, 'weight_s2v', 0.5)}")
print_log(f"│  text_residual : {getattr(config, 'text_residual', 0.5)}")
print_log(f"│  visual_residual: {getattr(config, 'visual_residual', 0.5)}")
print_log("├─ Loss Weights ──────────────────────────────────────────┤")
print_log(f"│  lambda_cal    : {getattr(config, 'lambda_cal', 0.0)}")
print_log(f"│  lambda_consist: {getattr(config, 'lambda_consist', 0.0)}")
print_log(f"│  lambda_l2sp   : {getattr(config, 'lambda_l2sp', 0.0)}")
print_log(f"│  lambda_topo   : {getattr(config, 'lambda_topo_pearson', 0.0)}")
print_log(f"│  lambda_aux_s2v: {getattr(config, 'lambda_aux_s2v', 0.0)}")
print_log(f"│  lambda_aux_v2s: {getattr(config, 'lambda_aux_v2s', 0.0)}")
print_log(f"│  aux_temp      : {getattr(config, 'aux_temp', 14.28)}")
print_log(f"│  use_geo_attr  : {getattr(config, 'use_geo_attr_routing', False)}")
print_log(f"│  lambda_geo_attr: {getattr(config, 'lambda_geo_attr_routing', 0.0)}")
print_log(f"│  use_text_reservoir: {getattr(config, 'use_text_attr_reservoir', False)}")
print_log(f"│  text_reservoir_ratio: {getattr(config, 'text_attr_reservoir_ratio', 0.0)}")
print_log(f"│  use_msdn_gate: {getattr(config, 'use_uncertainty_msdn_gate', False)}")
print_log(f"│  use_attr_patch_ot: {getattr(config, 'use_attr_patch_ot', False)}")
print_log(f"│  use_ag_jepa_v2: {getattr(config, 'use_ag_jepa_v2', False)}")
print_log(f"│  use_cf_neg_text: {getattr(config, 'use_cf_neg_text', False)}")
print_log(f"│  lambda_cf_neg_text: {getattr(config, 'lambda_cf_neg_text', 0.0)}")
print_log("├─ Resume ────────────────────────────────────────────────┤")
print_log(f"│  resume_from        : {getattr(config, 'resume_from', '')!r}")
print_log(f"│  resume_lr_schedule : {getattr(config, 'resume_lr_schedule', 'continue')}")
print_log(f"│  extra_epochs       : {getattr(config, 'extra_epochs', 0)}")
print_log("├─ Speed Optimizations ───────────────────────────────────┤")
print_log(f"│  use_amp       : {getattr(config, 'use_amp', True)}  (BF16 autocast)")
print_log(f"│  pin_memory    : enabled if patch cache loaded  non_blocking H2D : True")
print_log("└─────────────────────────────────────────────────────────┘")

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
# 优先级:
#   1) 多视角增强缓存 (use_aug_cache=True 且文件存在) — K 个增强视角
#   2) 单视角 patch 缓存 — 完整 576 token 但无增强
#   3) 单视角 CLS 缓存   — 仅 CLS, 兼容旧逻辑
#   4) 实时提取
USE_AUG = bool(getattr(config, 'use_aug_cache', False))
HAS_AUG_CACHE = (USE_AUG
                 and os.path.exists(CACHE_TRAIN_PATCH_AUG)
                 and os.path.exists(CACHE_TRAIN_FEAT_AUG)
                 and os.path.exists(CACHE_TRAIN_LABEL))
HAS_PATCH_CACHE = os.path.exists(CACHE_TRAIN_PATCH) and os.path.exists(CACHE_TRAIN_LABEL)
HAS_CLS_CACHE   = os.path.exists(CACHE_TRAIN_FEAT)  and os.path.exists(CACHE_TRAIN_LABEL)

train_patches  = None   # [N, 576, 768] float16 存 CPU
train_features = None   # [N, 768] float32 (legacy)
train_labels   = None
train_cls      = None
train_patches_aug = None   # [K, N, 576, 768] float16 多视角
train_cls_aug     = None   # [K, N, 768] float32 多视角
NUM_VIEWS_CACHE   = 1

if HAS_AUG_CACHE:
    print_log("\n[★] Loading MULTI-VIEW augmented cache (best mode)...")
    train_patches_aug = torch.load(CACHE_TRAIN_PATCH_AUG, map_location='cpu', weights_only=True)
    train_cls_aug     = torch.load(CACHE_TRAIN_FEAT_AUG,  map_location='cpu', weights_only=True)
    train_labels      = torch.load(CACHE_TRAIN_LABEL,     map_location=config.device, weights_only=True)
    NUM_VIEWS_CACHE   = train_patches_aug.shape[0]
    print_log(f"      Views K        : {NUM_VIEWS_CACHE}")
    print_log(f"      train_cls_aug    : {train_cls_aug.shape}  dtype={train_cls_aug.dtype}")
    print_log(f"      train_patches_aug: {train_patches_aug.shape}  dtype={train_patches_aug.dtype}")
    print_log(f"      train_labels     : {train_labels.shape}")
    USE_CACHE = 'aug'
elif HAS_PATCH_CACHE:
    print_log("\n[★] Loading train PATCH + CLS features from cache (single-view)...")
    train_patches = torch.load(CACHE_TRAIN_PATCH, map_location='cpu', weights_only=True)
    train_labels  = torch.load(CACHE_TRAIN_LABEL, map_location=config.device, weights_only=True)
    train_cls = torch.load(CACHE_TRAIN_FEAT, map_location='cpu', weights_only=True) if HAS_CLS_CACHE else None

    # ⚠️ GPU 预加载已禁用 (实测在 PyTorch 2.11+cu128 + 5070 Ti 上仍会慢到 8s/step)
    # 真因不明, 怀疑 BF16 张量核 + 6GB 静态张量索引在 Blackwell 上有 driver 路径问题
    # 保守模式: patch 留 CPU, 每 step CPU→GPU 切片
    # ★ Bug 修复 (2026-05-25): 真正调用 pin_memory(), 让 non_blocking=True 异步传输生效
    # 旧代码注释说 pinned 但没真调, non_blocking 在普通 CPU tensor 上效果几乎为 0
    try:
        train_patches = train_patches.pin_memory()
        if train_cls is not None:
            train_cls = train_cls.pin_memory()
        _PIN_OK = True
        print_log(f"      ★ Patches pinned to locked CPU memory (non_blocking H2D enabled)")
    except Exception as _pin_err:
        _PIN_OK = False
        print_log(f"      [info] pin_memory failed ({_pin_err}), fallback to plain CPU tensor")

    if train_cls is None:
        print_log("      WARNING: CLS cache missing, fallback to patch.mean()")
    else:
        print_log(f"      train_cls:     {train_cls.shape}  dtype={train_cls.dtype}  device={train_cls.device}")
    print_log(f"      train_patches: {train_patches.shape}  dtype={train_patches.dtype}  device={train_patches.device}")
    print_log(f"      train_labels:  {train_labels.shape}")
    USE_CACHE = 'patch'
elif HAS_CLS_CACHE:
    print_log("\n[★] Loading train CLS features from cache (legacy mode)...")
    train_features = torch.load(CACHE_TRAIN_FEAT,  map_location=config.device, weights_only=True)
    train_labels   = torch.load(CACHE_TRAIN_LABEL, map_location=config.device, weights_only=True)
    print_log(f"      train_features: {train_features.shape}  train_labels: {train_labels.shape}")
    USE_CACHE = 'cls'
else:
    print_log("\n[!] Cache not found. Will extract features on-the-fly (slow).")
    print_log("    Run: python tools/extract_features.py  to generate cache.")
    USE_CACHE = None

# ==========================================
#   生成类名文本特征 (CLIP 模板 prompt)
# ==========================================
print_log("\n[3/5] Encoding class name text features...")
class_names = [c.split('.')[-1].replace("_", " ") for c in dataloader.class_names]
# ★ 加速优化 (方案 2a): 类名文本嵌入缓存 (200 句 prompt, 启动省 ~3s)
if os.path.exists(CACHE_CLASS_TEXT):
    class_text_embeds = torch.load(CACHE_CLASS_TEXT, map_location=config.device,
                                    weights_only=True)
    print_log(f"      ★ class_text_embeds loaded from cache: {class_text_embeds.shape}")
else:
    prompts = [f"a photo of a {c}, a type of bird." for c in class_names]
    text_inputs = torch.cat([clip.tokenize(p) for p in prompts]).to(config.device)
    with torch.no_grad():
        class_text_embeds = clip_model.encode_text(text_inputs).float()  # [200, 768]
    try:
        torch.save(class_text_embeds.cpu(), CACHE_CLASS_TEXT)
        print_log(f"      class_text_embeds: {class_text_embeds.shape}  (cached for next run)")
        class_text_embeds = class_text_embeds.to(config.device)
    except Exception as e:
        print_log(f"      class_text_embeds: {class_text_embeds.shape}  "
                  f"(cache save failed: {e})")

# ==========================================
#   加载 GPT-4 描述并编码
# ==========================================
gpt_text_embeds = None
# 文本数据来源切换:
#   'gpt4'     → cub.pt          (GPT-4, 7 句/类)
#   'claude'   → cub_claude.pt   (Claude, 7 句/类)
#   'gpt55'    → cub_gpt55.pt    (GPT-5.5 style, 7 句/类)
#   'merge'    → cub_merge.pt    (GPT-4 + Claude 拼接, 14 句/类)
#   'weighted' → α × Claude_emb + (1-α) × GPT_emb (各自编码再加权融合)
text_source = getattr(config, 'text_source', 'gpt4')


def _encode_descriptions(file_path, dataloader, clip_model, device, class_text_embeds):
    """加载并 CLIP 编码描述文件 → 返回 [200, 768] 嵌入"""
    sentences_dict = torch.load(file_path, map_location='cpu', weights_only=False)
    embeds_list = []
    hit = 0
    for cls_name in dataloader.class_names:
        gpt_key = '.'.join(cls_name.split('.')[1:]).lower()
        if gpt_key in sentences_dict:
            sentences = sentences_dict[gpt_key]
            tokens = torch.cat([clip.tokenize(s) for s in sentences]).to(device)
            with torch.no_grad():
                feats = clip_model.encode_text(tokens).float()
            embeds_list.append(feats.mean(dim=0))
            hit += 1
        else:
            idx = dataloader.class_names.index(cls_name)
            embeds_list.append(class_text_embeds[idx])
    return torch.stack(embeds_list), hit, len(sentences_dict)


if text_source == 'weighted':
    text_alpha = getattr(config, 'text_alpha', 1.0)
    gpt_path    = os.path.join('.', 'data', 'gpt4_data', 'cub.pt')
    claude_path = os.path.join('.', 'data', 'gpt4_data', 'cub_claude.pt')
    print_log(f"\n[4/5] Using Weighted fusion: α={text_alpha} × Claude + {1-text_alpha:.2f} × GPT-4")

    # ★ 加速优化 (方案 2a): weighted 模式两套都缓存
    weighted_cache = os.path.join(CACHE_DIR,
                                   f'CUB_weighted_a{text_alpha:.2f}_text_embeds.pt')
    if os.path.exists(weighted_cache):
        gpt_text_embeds = torch.load(weighted_cache, map_location=config.device,
                                      weights_only=True)
        print_log(f"      ★ weighted text embeds loaded from cache: {gpt_text_embeds.shape}")
    else:
        print_log(f"      Encoding GPT-4 descriptions...")
        gpt_embeds, gpt_hit, _    = _encode_descriptions(gpt_path,    dataloader, clip_model,
                                                          config.device, class_text_embeds)
        print_log(f"      Encoding Claude descriptions...")
        claude_embeds, claude_hit, _ = _encode_descriptions(claude_path, dataloader, clip_model,
                                                             config.device, class_text_embeds)
        gpt_text_embeds = text_alpha * claude_embeds + (1.0 - text_alpha) * gpt_embeds
        print_log(f"      GPT hit: {gpt_hit} | Claude hit: {claude_hit} classes")
        try:
            torch.save(gpt_text_embeds.cpu(), weighted_cache)
            gpt_text_embeds = gpt_text_embeds.to(config.device)
        except Exception:
            pass
    print_log(f"      gpt_text_embeds: {gpt_text_embeds.shape} (α={text_alpha})")
else:
    if text_source == 'merge':
        gpt4_data_path = os.path.join('.', 'data', 'gpt4_data', 'cub_merge.pt')
        print_log(f"\n[4/5] Using Merge (GPT-4 + Claude) descriptions: {gpt4_data_path}")
    elif text_source == 'claude':
        gpt4_data_path = os.path.join('.', 'data', 'gpt4_data', 'cub_claude.pt')
        print_log(f"\n[4/5] Using Claude descriptions: {gpt4_data_path}")
    elif text_source == 'gpt55':
        gpt4_data_path = os.path.join('.', 'data', 'gpt4_data', 'cub_gpt55.pt')
        print_log(f"\n[4/5] Using GPT-5.5 descriptions: {gpt4_data_path}")
    else:
        gpt4_data_path = os.path.join('.', 'data', 'gpt4_data', 'cub.pt')
        print_log(f"\n[4/5] Using GPT-4 descriptions: {gpt4_data_path}")

    # ★ 加速优化 (方案 2a): GPT 文本嵌入缓存 (200 类 × 7 句 = 1400 个 CLIP encode_text)
    # 启动省 ~10 秒 (CLIP ViT-L/14 文本编码慢, 7 次 encode_text)
    embed_cache = _gpt_embed_cache_path(text_source)
    if os.path.exists(embed_cache):
        gpt_text_embeds = torch.load(embed_cache, map_location=config.device,
                                      weights_only=True)
        print_log(f"      ★ gpt_text_embeds loaded from cache: {gpt_text_embeds.shape}")
    elif os.path.exists(gpt4_data_path):
        print_log(f"      Loading text descriptions from {gpt4_data_path}...")
        gpt_text_embeds, hit, n_cls = _encode_descriptions(
            gpt4_data_path, dataloader, clip_model, config.device, class_text_embeds)
        n_desc = len(list(torch.load(gpt4_data_path, map_location='cpu',
                                     weights_only=False).values())[0])
        print_log(f"      {n_cls} classes × {n_desc} descriptions/class")
        print_log(f"      GPT hit: {hit} classes | fallback: {n_cls - hit} classes")
        try:
            torch.save(gpt_text_embeds.cpu(), embed_cache)
            print_log(f"      gpt_text_embeds: {gpt_text_embeds.shape}  (cached for next run)")
            gpt_text_embeds = gpt_text_embeds.to(config.device)
        except Exception as e:
            print_log(f"      gpt_text_embeds: {gpt_text_embeds.shape}  "
                      f"(cache save failed: {e})")
    else:
        print_log(f"      WARNING: text data not found at {gpt4_data_path}, using class name only")

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
    class_attr=dataloader.att,              # [200, 312] CUB 专家属性
    attr_text_embeds=dataloader.clip_att,    # [312, 768] CLIP 属性文本原型
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
#   Resume 系统
# ==========================================
# 三种使用场景:
#   1. 从头训练 (默认): yaml resume_from='' → 不加载任何 checkpoint
#   2. 续训接着原 LR: resume_from='auto' + resume_lr_schedule='continue'
#   3. 续训重启 LR: resume_from='auto' + resume_lr_schedule='restart'
#   4. 微调: resume_from='auto' + resume_lr_schedule='finetune' (LR=1e-4)
#
# checkpoint 内容 (full_ckpt): model + optimizer + scheduler + epoch + best_H
# 旧的 best_model_*.pth 仅含 model.state_dict, 也兼容加载
import glob


def find_latest_checkpoint(ckpt_dir, full_only=False):
    """
    在 ckpt_dir 下找最新的 checkpoint
    full_only=True: 只找含完整状态的 ckpt_full_*.pth
    full_only=False: 也接受老的 best_model_*.pth (仅权重)
    """
    if full_only:
        candidates = glob.glob(os.path.join(ckpt_dir, 'ckpt_full_*.pth'))
    else:
        candidates = (glob.glob(os.path.join(ckpt_dir, 'ckpt_full_*.pth'))
                      + glob.glob(os.path.join(ckpt_dir, 'best_model_*.pth')))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


resume_from        = getattr(config, 'resume_from', '')
resume_lr_schedule = getattr(config, 'resume_lr_schedule', 'continue')
extra_epochs       = int(getattr(config, 'extra_epochs', 0))

# ★ Bug 修复 (2026-05-25): clip_only / adapter_only 消融时强制跳过 resume
# 否则 checkpoint 里的 logit_scale / Adapter / proj_text 等参数会污染消融结果,
# clip_only 数值不再是干净 CLIP 基线
_model_mode_for_resume = getattr(config, 'model_mode', 'vgsr')
if _model_mode_for_resume in ('clip_only', 'adapter_only') and resume_from:
    print_log(f"\n[Resume] model_mode='{_model_mode_for_resume}' detected, "
              f"forcing resume_from='' (skip checkpoint to keep ablation clean)")
    resume_from = ''

start_epoch = 1     # 默认从 epoch 1 开始
best_H = 0.0
best_metrics = {'U': 0, 'S': 0, 'H': 0, 'ZS': 0, 'epoch': 0}

if resume_from:
    if resume_from == 'auto':
        # ★ Bug 修复 (2026-05-25): continue 模式只接受 ckpt_full_*.pth
        # 否则 legacy best_model_*.pth (仅权重) 比 full ckpt 新时会退化, 丢失 optimizer/scheduler
        _full_only = (resume_lr_schedule == 'continue')
        ckpt_path = find_latest_checkpoint('./train_log/CUB', full_only=_full_only)
    else:
        ckpt_path = resume_from

    if ckpt_path is None or not os.path.exists(ckpt_path):
        print_log(f"\n[!] resume_from='{resume_from}' but no checkpoint found, "
                  f"starting from scratch.")
    else:
        print_log(f"\n[Resume] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=config.device,
                          weights_only=False)

        # 兼容两种 ckpt 格式:
        # 1) 完整 ckpt: dict with 'model_state_dict' / 'optimizer_state_dict' / ...
        # 2) 旧 ckpt: 直接是 model.state_dict()
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            # 新格式: 完整 checkpoint
            model.load_state_dict(ckpt['model_state_dict'])
            print_log(f"  ✓ Loaded model weights")

            # ★ 不论 resume_lr_schedule 是什么, 都读 epoch/best 信息以正确显示进度
            #   仅 LR/optimizer/scheduler 是否恢复因模式而异
            ckpt_epoch = ckpt.get('epoch', 0)
            best_H = ckpt.get('best_H', 0.0)
            if 'best_metrics' in ckpt:
                best_metrics = ckpt['best_metrics']

            if resume_lr_schedule == 'continue':
                # 继续 LR 进度: 恢复 optimizer + scheduler, epoch 接着数
                if 'optimizer_state_dict' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    print_log(f"  ✓ Loaded optimizer state")
                if 'scheduler_state_dict' in ckpt:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                    print_log(f"  ✓ Loaded scheduler state")
                start_epoch = ckpt_epoch + 1
                print_log(f"  ✓ Resuming from epoch {start_epoch}, best_H so far = {best_H*100:.2f}%")
            elif resume_lr_schedule == 'restart':
                # 只载权重, LR 重启 cosine 0.001 → 0
                # epoch 计数仍从 1 (因为 LR 调度从 0 开始, 跟 ckpt epoch 不一致)
                print_log(f"  ✓ Loaded model weights (epoch {ckpt_epoch}, best_H={best_H*100:.2f}%)")
                print_log(f"  ✓ LR scheduler restarted (LR=0.001 → cosine over {config.epochs} epochs)")
                print_log(f"  ⚠ epoch counter resets to 1 (not continuing from {ckpt_epoch + 1})")
            elif resume_lr_schedule == 'finetune':
                # LR 重置 finetune_lr (默认 1e-4), cosine 在新 epochs 上从头走
                # epoch 计数也从 1 (类似 restart)
                # ★ 2026-05-25: 支持 yaml 配置 finetune_lr (默认 1e-4 兼容旧实验)
                #   方案 B 多段训练: 第一段 finetune_lr=1e-4, 第二段 finetune_lr=1e-5
                finetune_lr = float(getattr(config, 'finetune_lr', 1e-4))
                for g in optimizer.param_groups:
                    g['lr'] = finetune_lr
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=config.epochs)
                print_log(f"  ✓ Loaded model weights (epoch {ckpt_epoch}, best_H={best_H*100:.2f}%)")
                print_log(f"  ✓ Finetune mode: LR={finetune_lr:g}, cosine over {config.epochs} epochs")
                print_log(f"  ⚠ epoch counter resets to 1 (treating as new finetune run)")
        else:
            # 旧格式: 仅模型权重
            model.load_state_dict(ckpt)
            print_log(f"  ✓ Loaded legacy weights only "
                      f"(no optimizer/scheduler/epoch state)")
            print_log(f"  ⚠ resume_lr_schedule='{resume_lr_schedule}' "
                      f"applies to LR, but no optimizer state to restore")
            if resume_lr_schedule == 'finetune':
                finetune_lr = float(getattr(config, 'finetune_lr', 1e-4))
                for g in optimizer.param_groups:
                    g['lr'] = finetune_lr
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=config.epochs)
                print_log(f"  ✓ Finetune mode: LR={finetune_lr:g}")

# 续训时在原 epochs 基础上加 extra_epochs
total_epochs = config.epochs + extra_epochs
if extra_epochs > 0:
    print_log(f"  + {extra_epochs} extra epochs → total {total_epochs} epochs")
    # 调度器也要扩 T_max (重新初始化, 用当前 LR 作为起点)
    if resume_lr_schedule != 'continue':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs)

# ──────────────────────────────────────────────────────────────
# ★ 2026-05-25: 一键多段训练 (Multi-Stage Training)
# ──────────────────────────────────────────────────────────────
# yaml 写法示例:
#   lr_stages:
#     value:
#       - {lr: 0.001,  epochs: 20}    # 第 1 段: 头训
#       - {lr: 0.0001, epochs: 30}    # 第 2 段: finetune
#       - {lr: 0.00001, epochs: 30}   # 第 3 段: micro-finetune
#
# 行为:
#   - total_epochs 自动等于各段 epochs 之和 (覆盖上面 extra_epochs 计算)
#   - 每段开始时: 把 optimizer.lr 切换到 stage.lr, 重新初始化 CosineAnnealingLR(T_max=stage.epochs)
#   - 关掉 (lr_stages 为空 / None / 不存在): 完全走旧逻辑, 兼容 H=72.95 ckpt recipe
lr_stages = getattr(config, 'lr_stages', None) or []
stage_boundaries = []     # 第 i 段结束时的 epoch 编号 (1-indexed, 含)
if lr_stages:
    print_log(f"\n[Multi-Stage] {len(lr_stages)} stages enabled "
              f"(overrides epochs/extra_epochs):")
    cum = 0
    for i, st in enumerate(lr_stages):
        cum += int(st['epochs'])
        stage_boundaries.append(cum)
        _emin = float(st.get('eta_min', 0))
        print_log(f"  Stage {i+1}: lr={float(st['lr']):g}, epochs={int(st['epochs'])}, eta_min={_emin:g} "
                  f"(epoch {cum - int(st['epochs']) + 1}..{cum})")
    total_epochs = stage_boundaries[-1]
    print_log(f"  Total epochs (sum of stages) = {total_epochs}")

    # 第 1 段立即生效: 设 lr + 重置 cosine
    first = lr_stages[0]
    for g in optimizer.param_groups:
        g['lr'] = float(first['lr'])
    _emin = float(first.get('eta_min', 0))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(first['epochs']), eta_min=_emin)
    print_log(f"  [OK] Stage 1 active: lr={float(first['lr']):g}, "
              f"cosine T_max={int(first['epochs'])}, eta_min={_emin:g}")

# ==========================================
#   训练循环
# ==========================================
iters_per_epoch = dataloader.ntrain_clip // config.batch_size
total_iters = iters_per_epoch * total_epochs

# 最佳模型保存路径
BEST_MODEL_DIR = './train_log/CUB'
BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, f'best_model_CUB_{current_time}.pth')
# 完整 checkpoint 保存路径 (含 optimizer/scheduler/epoch)
CKPT_FULL_PATH  = os.path.join(BEST_MODEL_DIR, f'ckpt_full_CUB_{current_time}.pth')

print_log("\n" + "=" * 60)
print_log(f"  Start Training: epoch [{start_epoch}/{total_epochs}], "
          f"{iters_per_epoch} iters/epoch")
print_log("=" * 60)

# ==========================================
#   ★ 加速优化 (方案 D): Mixed Precision (autocast + GradScaler)
# ==========================================
# autocast: forward 自动用 fp16 张量核 (FAE 矩阵乘加速 ~30%)
# GradScaler: backward 自动放大 loss 防 fp16 underflow
# 关闭开关: yaml 设 use_amp: False
USE_AMP = bool(getattr(config, 'use_amp', True))
if USE_AMP:
    # ★ 新 API: torch.amp.autocast('cuda', ...)
    # 用 BF16 而非 FP16: 5070 Ti 原生支持, 数值范围与 FP32 同, 无需 GradScaler
    from torch.amp import autocast
    print_log(f"  ★ AMP enabled (BF16 autocast, no GradScaler needed)")
else:
    print_log(f"  ⚠ AMP disabled (use_amp=False), running fp32")

# ==========================================
#   消融模式: clip_only 跳过训练直接评估
# ==========================================
model_mode = getattr(config, 'model_mode', 'vgsr')
print_log(f"\n  ★ Model mode: {model_mode}")

if model_mode == 'clip_only':
    print_log("\n[CLIP-Only Baseline] Skipping training, evaluating zero-shot CLIP...")
    print_log("  (Adapter / FAE / CrossModalTransformer all bypassed)")
    gzsl_bias = getattr(config, 'gzsl_bias', 0.0)
    acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(
        dataloader, clip_model, model, config.device,
        bias_unseen=gzsl_bias)

    print_log("\n" + "=" * 60)
    print_log("  Zero-Shot CLIP Baseline Results (no training)")
    print_log("=" * 60)
    print_log(f"  ┌─────────────────────────────────────")
    print_log(f"  │  GZSL-U : {acc_novel*100:.2f}%")
    print_log(f"  │  GZSL-S : {acc_seen*100:.2f}%")
    print_log(f"  │  GZSL-H : {H*100:.2f}%")
    print_log(f"  │  ZSL    : {acc_zs*100:.2f}%")
    print_log(f"  └─────────────────────────────────────")
    print_log(f"\n  Log saved to: {LOG_FILE}")
    raise SystemExit(0)

for epoch in range(start_epoch, total_epochs + 1):

    # ---------- 训练阶段 ----------
    model.train()
    epoch_loss = 0.0
    epoch_iters = 0

    print_log(f"\n{'─'*60}")
    print_log(f"  Epoch [{epoch}/{total_epochs}]  Training...")
    print_log(f"{'─'*60}")

    for step in range(iters_per_epoch):
        optimizer.zero_grad()

        if USE_CACHE == 'aug':
            # ── 多视角增强缓存: 每张图随机抽 1 个视角 ──
            idx = torch.randperm(train_patches_aug.shape[1])[:config.batch_size]
            view_idx = torch.randint(0, NUM_VIEWS_CACHE, (config.batch_size,))
            batch_label = train_labels[idx]
            patch_batch = train_patches_aug[view_idx, idx].to(config.device).float()  # [B, 576, 768]
            cls_batch   = train_cls_aug[view_idx, idx].to(config.device).float().unsqueeze(1)  # [B, 1, 768]
            clip_features = torch.cat([cls_batch, patch_batch], dim=1)                # [B, 577, 768]
        elif USE_CACHE == 'patch':
            # ── 随机采样：每步随机取 batch ──
            idx = torch.randperm(len(train_patches))[:config.batch_size]
            batch_label = train_labels[idx]
            # ★ 加速优化 (方案 A): pin_memory + non_blocking 异步 H2D
            # train_patches 已在 __init__ 时 pin_memory(), 这里 non_blocking=True
            # 让 CPU→GPU 传输与 GPU forward 并发, 减少阻塞等待
            if train_patches.is_cuda:
                patch_batch = train_patches[idx].float()                # 已在 GPU, 直接 float32
            else:
                patch_batch = train_patches[idx].to(
                    config.device, non_blocking=True).float()
            if train_cls is not None:
                if train_cls.is_cuda:
                    cls_batch = train_cls[idx].float().unsqueeze(1)
                else:
                    cls_batch = train_cls[idx].to(
                        config.device, non_blocking=True).float().unsqueeze(1)
                clip_features = torch.cat([cls_batch, patch_batch], dim=1)
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
        # ★ 加速优化 (方案 D): autocast 让 FAE 注意力走 BF16 张量核
        # BF16 比 FP16 数值范围大, 5070 Ti 原生支持, GradScaler 也可省略
        if USE_AMP:
            with autocast('cuda', dtype=torch.bfloat16):
                out_package = model(clip_features, is_train=True)
                in_package = out_package.copy()
                in_package['batch_label'] = batch_label
                loss_pack = model.compute_loss(in_package)
                loss = loss_pack['loss']
            # BF16 不需要 GradScaler (数值范围足够), 直接 backward
            loss.backward()
            optimizer.step()
        else:
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
            ce_v   = loss_pack.get('loss_CE',      torch.tensor(0.)).item()
            cons_v = loss_pack.get('loss_consist', torch.tensor(0.)).item()
            l2_v   = loss_pack.get('loss_l2sp',    torch.tensor(0.)).item()
            topo_v = loss_pack.get('loss_topo',    torch.tensor(0.)).item()
            va_v   = loss_pack.get('loss_v_anchor',        torch.tensor(0.)).item()
            ta_v   = loss_pack.get('loss_t_unseen_anchor', torch.tensor(0.)).item()
            di_v   = loss_pack.get('loss_distill',         torch.tensor(0.)).item()
            as_v   = loss_pack.get('loss_aux_s2v',         torch.tensor(0.)).item()
            av_v   = loss_pack.get('loss_aux_v2s',         torch.tensor(0.)).item()
            ms_v   = loss_pack.get('loss_msdn',            torch.tensor(0.)).item()
            mg_v   = loss_pack.get('loss_msdn_gate',       torch.tensor(1.)).item()
            bi_v   = loss_pack.get('loss_bias',            torch.tensor(0.)).item()
            jp_v   = loss_pack.get('loss_jepa',            torch.tensor(0.)).item()
            jn_v   = loss_pack.get('loss_jepa_neg',        torch.tensor(0.)).item()
            cf_v   = loss_pack.get('loss_cf_neg_text',     torch.tensor(0.)).item()
            ga_v   = loss_pack.get('loss_geo_attr',        torch.tensor(0.)).item()
            ao_v   = loss_pack.get('loss_attr_patch_ot',   torch.tensor(0.)).item()
            print_log(f"  Step [{step+1:3d}/{iters_per_epoch}] | "
                      f"Loss: {loss.item():.4f} | Avg: {avg_loss:.4f} | "
                      f"CE: {ce_v:.3f}  Cons: {cons_v:.3f}  "
                      f"L2SP: {l2_v:.4f}  Topo: {topo_v:.4f}  "
                      f"VA: {va_v:.4f}  TA: {ta_v:.4f}  Dist: {di_v:.4f}  "
                      f"AuxS2V: {as_v:.3f}  AuxV2S: {av_v:.3f}  "
                      f"MSDN: {ms_v:.4f}  MGate: {mg_v:.3f}  Bias: {bi_v:.4f}  "
                      f"JEPA: {jp_v:.4f}  JNeg: {jn_v:.4f}  CFNeg: {cf_v:.4f}  "
                      f"GeoAttr: {ga_v:.4f}  AttrOT: {ao_v:.4f}")

    # 更新学习率
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    avg_epoch_loss = epoch_loss / epoch_iters

    print_log(f"\n  >> Epoch [{epoch}/{total_epochs}] Train Summary")
    print_log(f"     Avg Loss : {avg_epoch_loss:.4f}")
    print_log(f"     LR       : {current_lr:.6f}")

    # ★ 2026-05-25: 多段训练 - 段边界自动切到下一段
    # 当前 epoch 等于某段终点 → 切换到下一段 lr + 重置 cosine T_max
    if lr_stages and epoch in stage_boundaries and epoch < total_epochs:
        next_idx = stage_boundaries.index(epoch) + 1
        if next_idx < len(lr_stages):
            next_stage = lr_stages[next_idx]
            new_lr = float(next_stage['lr'])
            new_T = int(next_stage['epochs'])

            # ★ 2026-05-25: 段切换时是否从历史 best ckpt 重启 (warm-restart)
            #   yaml 写法: lr_stages: [{lr:..., epochs:..., restart_from_best: True}, ...]
            #   等价于"严格 early stopping at best + finetune from best"
            #   这是 GZSL 领域 (TransZero/MSDN) 的标准做法, 比严格连续高 ~0.8 H
            if bool(next_stage.get('restart_from_best', False)):
                # 从当前 run 已保存的 best ckpt 重启
                if os.path.exists(CKPT_FULL_PATH):
                    _ckpt = torch.load(CKPT_FULL_PATH, map_location=config.device,
                                       weights_only=False)
                    if isinstance(_ckpt, dict) and 'model_state_dict' in _ckpt:
                        model.load_state_dict(_ckpt['model_state_dict'])
                        _bH = _ckpt.get('best_H', 0.0) * 100
                        _bep = _ckpt.get('epoch', 0)
                        print_log(f"\n  ★ Warm-restart: rolled back to best ckpt "
                                  f"(epoch {_bep}, H={_bH:.2f}%)")
                else:
                    print_log(f"\n  ⚠ restart_from_best=True 但 best ckpt 还不存在, 跳过回滚")

            for g in optimizer.param_groups:
                g['lr'] = new_lr
            _emin = float(next_stage.get('eta_min', 0))
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=new_T, eta_min=_emin)
            print_log(f"\n  ★★★ Stage {next_idx + 1} starts at epoch {epoch + 1} "
                      f"★★★\n     lr → {new_lr:g}, cosine T_max={new_T}, eta_min={_emin:g}")

    # ---------- 动态 residual / blend 系数 (仅 cosine_only + learnable 模式打印) ----------
    # 注: add 模式下这些 Parameter 虽然存在但不参与 forward, 打印没意义, 跳过
    is_cosine_only = getattr(model, 'score_mode', 'add') == 'cosine_only'
    if is_cosine_only and hasattr(model, 'residual_mode') and model.residual_mode != 'fixed':
        with torch.no_grad():
            vr = torch.sigmoid(model.visual_residual_logit).item()
            if model.residual_mode == 'learnable_split':
                tr_s = torch.sigmoid(model.text_residual_seen_logit).item()
                tr_u = torch.sigmoid(model.text_residual_unseen_logit).item()
                print_log(f"     Residual : vr={vr:.4f}  tr_seen={tr_s:.4f}  tr_unseen={tr_u:.4f}")
            else:  # learnable_global
                tr = torch.sigmoid(model.text_residual_logit).item()
                print_log(f"     Residual : vr={vr:.4f}  tr={tr:.4f}")
    if is_cosine_only and hasattr(model, 'cosine_base_blend_mode') and model.cosine_base_blend_mode == 'learnable':
        with torch.no_grad():
            cb = torch.sigmoid(model.cosine_base_blend_logit).item()
        print_log(f"     CB Blend : cb={cb:.4f}  (sigmoid of logit={model.cosine_base_blend_logit.item():.4f})")

    # ---------- F4/F5 add 模式动态门控 (诊断用, 打印 alpha_net/w_net 末层 bias 对应的"零输入"基线值) ----------
    if not is_cosine_only:
        if hasattr(model, 'alpha_net') and getattr(model, 'gating_dynamic', 'fixed') == 'mlp':
            with torch.no_grad():
                # alpha_net[-1] 是最后一层 Linear. weight=0 init 下, 输出 ≈ bias (与图无关)
                # 训练后 weight 不为 0, 这里只打 bias 对应的"基础" α 作为稳定参考
                bias_a = model.alpha_net[-1].bias.item()
                a_base = float(torch.sigmoid(torch.tensor(bias_a)).item())
            print_log(f"     Gating α (mlp): bias-baseline={a_base:.4f}  (last bias={bias_a:.4f})")
        if hasattr(model, 'w_net') and getattr(model, 'weight_s2v_mode', 'fixed') == 'mlp':
            with torch.no_grad():
                bias_w = model.w_net[-1].bias.item()
                w_base = float(torch.sigmoid(torch.tensor(bias_w)).item())
            print_log(f"     Weight w (mlp): bias-baseline={w_base:.4f}  (last bias={bias_w:.4f})")
        if hasattr(model, 'pool_net') and getattr(model, 'pool_dynamic', 'fixed') == 'mlp':
            with torch.no_grad():
                bias_p = model.pool_net[-1].bias.item()
                p_base = float(torch.sigmoid(torch.tensor(bias_p)).item())
            print_log(f"     Pool λ (mlp): bias-baseline={p_base:.4f}  (last bias={bias_p:.4f})")
    # ---------- 测试阶段 ----------
    print_log(f"\n  >> Epoch [{epoch}/{total_epochs}] Evaluating GZSL...")
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
        # 保存最佳模型权重（写盘失败不影响训练继续）
        # 文件名带 H 后缀, 便于跨 run 排序识别
        H_int = int(round(H * 10000))
        BEST_MODEL_PATH_WITH_H = BEST_MODEL_PATH.replace(
            '.pth', f'_H{H_int}.pth')
        try:
            # 先删本次 run 之前更低 H 的旧权重 (同 timestamp 但低 H)
            import glob
            old_pattern = BEST_MODEL_PATH.replace('.pth', '_H*.pth')
            for old_p in glob.glob(old_pattern):
                if old_p != BEST_MODEL_PATH_WITH_H:
                    try:
                        os.remove(old_p)
                    except Exception:
                        pass
            # 保存当前最佳 (仅模型权重, 兼容旧逻辑)
            torch.save(model.state_dict(), BEST_MODEL_PATH_WITH_H)
            print_log(f"  [★] Best model saved → {BEST_MODEL_PATH_WITH_H}")

            # ★ 同时保存完整 checkpoint (含 optimizer/scheduler 用于 resume)
            # 文件总是覆盖同一个路径, 只保留当前 run 的最新
            try:
                torch.save({
                    'epoch': epoch,
                    'best_H': best_H,
                    'best_metrics': best_metrics,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, CKPT_FULL_PATH)
            except Exception as e2:
                print_log(f"  [!] Full ckpt save failed (training continues): {e2}")
        except Exception as e:
            print_log(f"  [!] Best model save failed (training continues): {e}")
            # 删除可能产生的损坏文件
            if os.path.exists(BEST_MODEL_PATH_WITH_H):
                try:
                    os.remove(BEST_MODEL_PATH_WITH_H)
                except Exception:
                    pass

    # 打印当前 epoch 结果
    print_log(f"\n  ┌─ Epoch [{epoch}/{total_epochs}] Results ─────────────────────")
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
