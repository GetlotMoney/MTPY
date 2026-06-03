import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import yaml
import clip
from types import SimpleNamespace
from datetime import datetime

# [引用] VGSR 模型
from model.VGSR import VGSR
# [引用] AWA2 数据集加载器
from tools.dataset import AWA2DataLoader
# [引用] 统一评估函数
from tools.helper_func import eval_zs_gzsl, get_clip_spatial_features

# ==========================================
#   日志记录辅助功能
# ==========================================
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
LOG_FILE = f"./train_log/AWA2/training_log_AWA2_{current_time}.txt"

if not os.path.exists(os.path.dirname(LOG_FILE)):
    os.makedirs(os.path.dirname(LOG_FILE))


def print_log(message):
    print(message)
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")


# ==========================================
#   主训练逻辑
# ==========================================

# 加载 Config
parser = argparse.ArgumentParser(description="Train VGSR on AWA2 GZSL.")
parser.add_argument(
    "--config",
    default="./config/VGSR_awa2_gzsl.yaml",
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
    config.device = 'cuda:1'

# AWA2 使用 Cosine 度量，无需强制改 Loss，沿用配置文件即可
print_log(f"Training Start... Log will be saved to: {LOG_FILE}")
print_log("=" * 50)
print_log(f"Dataset: AWA2 (Animals with Attributes 2)")
print_log(f"Config file: {config_path}")
print_log(f"Training config: {config}")
print_log("=" * 50)

# 1. 加载 CLIP
print_log("Loading CLIP (ViT-L/14@336px)...")
clip_model, _ = clip.load("ViT-L/14@336px", device=config.device)
clip_model = clip_model.float()
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

# 2. 加载 DataLoader
print_log("Loading AWA2 Dataset...")
dataloader = AWA2DataLoader('.', config.device, is_balance=False)

# 3. 固定随机种子
seed = config.random_seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# ============================================================
# [新增关键步骤] 生成 AWA2 类名文本特征 (用于 Residual Fusion)
# ============================================================
print_log("Encoding Class Names for AWA2 (Residual Fusion)...")

# 1. 获取 AWA2 类名并清洗
# AWA2 类名通常格式为 "killer+whale"，需要把 "+" 替换为空格
if hasattr(dataloader, 'class_names'):
    class_names = [c.replace("+", " ") for c in dataloader.class_names]
else:
    # 必须报错，否则 Residual Fusion 无法工作
    raise ValueError("AWA2DataLoader must have 'class_names' attribute! Please update tools/dataset.py")

# 2. 构建 Prompt
# AWA2 推荐 Prompt: "a photo of a {c}." (或者 "a photo of a {c}, a type of animal.")
# 既然你的代码里用的是短的，就保持原样
prompts = [f"a photo of a {c}." for c in class_names]

# 3. 提取特征
text_inputs = torch.cat([clip.tokenize(p) for p in prompts]).to(config.device)
with torch.no_grad():
    class_text_embeds = clip_model.encode_text(text_inputs)
    class_text_embeds = class_text_embeds.float()  # [50, 768]
# ============================================================

# 4. 初始化 VGSR 模型
print_log("Initializing VGSR Model...")
# [修改] 传入 class_text_embeds
model = VGSR(
    config,
    dataloader.att,
    dataloader.clip_att,
    dataloader.seenclasses,
    dataloader.unseenclasses,
    class_text_embeds=class_text_embeds  # <--- 新增参数
).to(config.device)

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0005)

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

# 计算迭代次数
niters = dataloader.ntrain_clip * config.epochs // config.batch_size
report_interval = niters // config.epochs
best_performance = [0, 0, 0, 0]  # U, S, H, ZS
best_performance_zsl = 0

print_log(f"Start VGSR Training for AWA2 ({niters} iters)...")

for i in range(niters):
    model.train()
    optimizer.zero_grad()

    try:
        batch_label, batch_images, batch_att = dataloader.next_batch(config.batch_size)
    except Exception as e:
        print_log(f"Error in dataloader: {e}")
        continue

    # 提取特征
    clip_features = get_clip_spatial_features(clip_model, batch_images).float().to(config.device)

    # Model Forward
    out_package = model(clip_features)

    # Loss Calculation
    in_package = out_package.copy()
    in_package['batch_label'] = batch_label

    loss_pack = model.compute_loss(in_package)
    loss = loss_pack['loss']

    loss_CE = loss_pack.get('loss_CE', torch.tensor(0.0))
    loss_cal = loss_pack.get('loss_cal', torch.tensor(0.0))
    loss_reg = loss_pack.get('loss_reg', torch.tensor(0.0))
    loss_con = loss_pack.get('loss_con', torch.tensor(0.0))

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # 更新学习率
    if (i + 1) % report_interval == 0:
        scheduler.step()

    # Evaluation
    if i % report_interval == 0 and i > 0:
        print_log(f"Iter {i}/{niters}: Evaluating...")

        acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(dataloader, clip_model, model, config.device)

        if H > best_performance[2]:
            best_performance = [acc_novel, acc_seen, H, acc_zs]
            # save_path = f'save_model/VGSR_AWA2_Best.pth'
            # if not os.path.exists('save_model'): os.makedirs('save_model')
            # torch.save(model.state_dict(), save_path)

        if acc_zs > best_performance_zsl:
            best_performance_zsl = acc_zs

        current_lr = optimizer.param_groups[0]['lr']

        # === [新增] 监控 Alpha 和 Dynamic Weight ===
        # 获取 Alpha
        if hasattr(model, 'module'):
            fusion_module = model.module.adaptive_fusion
        else:
            fusion_module = model.adaptive_fusion
        current_alpha = fusion_module.alpha.item() if hasattr(fusion_module.alpha, 'item') else fusion_module.alpha

        # 获取 Avg Dynamic Weight
        avg_dyn_weight = out_package['dynamic_weight'].mean().item() if 'dynamic_weight' in out_package else 0.0
        # ==========================================

        print_log('-' * 60)
        print_log(f"Epoch {int(i // report_interval)} | LR: {current_lr:.6g}")
        print_log(f"Loss Total: {loss.item():.5f}")

        # [核心] 打印融合状态
        print_log(f"Fusion State: Alpha = {current_alpha:.4f} | Avg Correction Weight = {avg_dyn_weight:.4f}")

        print_log(
            f"Details : CE={loss_CE.item():.4f} | Cal={loss_cal.item():.4f} | Reg={loss_reg.item():.4f} | "
            f"Con={loss_con.item():.4f}")
        print_log(f"Current : U={acc_novel:.5f}%  S={acc_seen:.5f}%  H={H:.5f}%  |  ZS={acc_zs:.5f}%")
        print_log(
            f"Best (GZSL): U={best_performance[0]:.5f}% S={best_performance[1]:.5f}% H={best_performance[2]:.5f}%")
        print_log(f"Best (ZSL) : Acc={best_performance_zsl:.5f}%")
        print_log('-' * 60)

print_log(f"Training Finished. Best H: {best_performance[2]:.2f}%. Log saved to {LOG_FILE}")
