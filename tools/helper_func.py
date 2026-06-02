import torch
import torch.nn.functional as F
import os

# 测试集特征缓存路径
_CACHE_DIR = './data/cache'
_CACHE_TEST_SEEN_FEAT     = os.path.join(_CACHE_DIR, 'CUB_test_seen_features.pt')
_CACHE_TEST_SEEN_LABEL    = os.path.join(_CACHE_DIR, 'CUB_test_seen_labels.pt')
_CACHE_TEST_SEEN_PATCH    = os.path.join(_CACHE_DIR, 'CUB_test_seen_patch_features.pt')
_CACHE_TEST_UNSEEN_FEAT   = os.path.join(_CACHE_DIR, 'CUB_test_unseen_features.pt')
_CACHE_TEST_UNSEEN_LABEL  = os.path.join(_CACHE_DIR, 'CUB_test_unseen_labels.pt')
_CACHE_TEST_UNSEEN_PATCH  = os.path.join(_CACHE_DIR, 'CUB_test_unseen_patch_features.pt')

# 全局缓存（只加载一次）
_test_cache = {}

def _load_test_cache(device):
    """加载测试集特征缓存到内存（只执行一次）

    优先加载 patch 缓存（让 FAE 能用到真实空间信息）；
    若缺失则回退为 CLS-only（FAE 看到 576 个相同 token，等价于关闭空间信息）。
    """
    global _test_cache
    if _test_cache:
        return True
    has_seen   = os.path.exists(_CACHE_TEST_SEEN_FEAT)   and os.path.exists(_CACHE_TEST_SEEN_LABEL)
    has_unseen = os.path.exists(_CACHE_TEST_UNSEEN_FEAT) and os.path.exists(_CACHE_TEST_UNSEEN_LABEL)
    if not (has_seen and has_unseen):
        return False

    _test_cache['seen_feat']    = torch.load(_CACHE_TEST_SEEN_FEAT,   map_location='cpu', weights_only=True)
    _test_cache['seen_label']   = torch.load(_CACHE_TEST_SEEN_LABEL,  map_location=device, weights_only=True)
    _test_cache['unseen_feat']  = torch.load(_CACHE_TEST_UNSEEN_FEAT, map_location='cpu', weights_only=True)
    _test_cache['unseen_label'] = torch.load(_CACHE_TEST_UNSEEN_LABEL,map_location=device, weights_only=True)

    # patch 缓存（可选）
    has_seen_patch   = os.path.exists(_CACHE_TEST_SEEN_PATCH)
    has_unseen_patch = os.path.exists(_CACHE_TEST_UNSEEN_PATCH)

    if has_seen_patch and has_unseen_patch:
        # patch 用 float16 存盘, 放在 CPU, 按 batch 切片再上 GPU 转 float32
        # ⚠️ 测试 patch GPU 预加载已禁用 (与训练 patch 一起占 10GB+, 留给激活的余量太少, 8s/step)
        # 训练 patch 6.2GB 优先 (每 step 命中, 收益最大), 测试 patch 走 CPU H2D 切片
        seen_patch   = torch.load(_CACHE_TEST_SEEN_PATCH,   map_location='cpu', weights_only=True)
        unseen_patch = torch.load(_CACHE_TEST_UNSEEN_PATCH, map_location='cpu', weights_only=True)
        _test_cache['seen_patch']   = seen_patch
        _test_cache['unseen_patch'] = unseen_patch
        _test_cache['has_patch']    = True
        print(f"[cache] Test patch features loaded (CPU, eval H2D per-batch): "
              f"seen={tuple(_test_cache['seen_patch'].shape)} "
              f"unseen={tuple(_test_cache['unseen_patch'].shape)}")
    else:
        _test_cache['has_patch'] = False
        print("[cache] Test patch features NOT found, falling back to CLS-only "
              "(FAE will see 576 identical tokens — spatial signal is lost).")

    return True

def get_clip_spatial_features(clip_model, images):
    with torch.no_grad():
        # 卷积层切块 卷积核大小14x14 步长14 在336x336的图上滑动，切出576块
        x = clip_model.visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        # 构造CLS Token，图像块特征拼接
        class_embedding = clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                                      dtype=x.dtype, device=x.device)
        # 将CLS Token拼接到图像块的前面 “火车头”
        x = torch.cat([class_embedding, x], dim=1)
        # 位置编码，定位每块在哪儿
        x = x + clip_model.visual.positional_embedding.to(x.dtype)
        x = clip_model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        # 经过Transformer 全局交互，上下文感知
        x = clip_model.visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = clip_model.visual.ln_post(x)
        # 投影到768维空间，可以直接跟我的语义去交互
        if clip_model.visual.proj is not None:
            x = x @ clip_model.visual.proj
    # 返回的x包括全局特征和局部特征   全局：x[:, 0, :]   局部：x[:, 1:, :]
    # global_feat = x[:, 0, :] 形状[64, 768]  local_feat = x[:, 1:, :] 形状[64, 576, 768]
    return x

# acc_seen = val_gzsl(test_seen_feature, test_seen_label, seenclasses, in_package,bias=bias_seen)
# 核心：在线特征提取与预测
def extract_and_predict(loader, clip_model, model, device, target_classes=None, bias=0, is_zsl=False):
    model.eval()
    clip_model.eval()  # 确保 CLIP 也在 eval 模式
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            # 1. 实时通过 CLIP 提取特征 全局特征
            # encode_image 输出 [B, 768] (如果是 ViT-L/14)
            # features = clip_model.encode_image(batch_images).float()

            # === 修改点 ===
            features = get_clip_spatial_features(clip_model, batch_images).float()

            # 2. 将特征输入你的 DVIE 模型
            out_package = model(features)
            output = out_package['clip_S_pp']
            # 3. 处理预测 (ZSL vs GZSL)
            if is_zsl:
                # ZSL: 仅在 Unseen 类中比较
                output_t = output.clone()
                # 这里的逻辑是: 把非 Unseen 的类分数设为极小，或者只取 Unseen 的列
                # 原代码通常是给 Target Classes 加分，或者Mask掉其他类
                # 这里采用 Mask 逻辑：只看 Unseen 列
                # 注意：target_classes 必须是 Unseen Classes 的索引
                pred = torch.argmax(output_t.data[:, target_classes], 1)
                # 注意：这里 pred 返回的是 0~49 的相对索引，需要映射回全局索引用于对比？
                # 通常 acc_zs 是在 50 类内部算的，所以 label 也需要映射
            else:
                # GZSL: Seen + Unseen
                if target_classes is not None:
                    output[:, target_classes] = output[:, target_classes] + bias
                pred = torch.argmax(output.data, 1)

            predicted_labels.append(pred.cpu())
            true_labels.append(batch_labels.cpu())
    return torch.cat(true_labels), torch.cat(predicted_labels)


def _estimate_prior_unseen_bias(model, device, batch_size=64):
    """Estimate a single unseen logit shift from the cached test distribution."""
    cfg = getattr(model, 'config', None)
    mode = getattr(cfg, 'prior_correction', 'none') if cfg is not None else 'none'
    if mode in (False, None, 'none', 'off', ''):
        return 0.0
    if not _test_cache:
        return 0.0

    temp = float(getattr(cfg, 'prior_temp', 1.0))
    max_bias = float(getattr(cfg, 'prior_max_bias', 3.0))
    target = str(getattr(cfg, 'prior_target', 'balanced'))

    prior_sum = None
    total = 0
    feat_sets = [
        (_test_cache['seen_feat'], _test_cache.get('seen_patch')),
        (_test_cache['unseen_feat'], _test_cache.get('unseen_patch')),
    ]

    with torch.no_grad():
        for feat, patch_cache in feat_sets:
            N = feat.size(0)
            for i in range(0, N, batch_size):
                batch_feat = feat[i:i + batch_size].to(device)
                cls = batch_feat.unsqueeze(1)
                if patch_cache is not None:
                    if patch_cache.is_cuda:
                        patches = patch_cache[i:i + batch_size].float()
                    else:
                        patches = patch_cache[i:i + batch_size].to(
                            device, non_blocking=True).float()
                else:
                    patches = cls.expand(-1, 576, -1).contiguous()
                logits = model(torch.cat([cls, patches], dim=1),
                               is_train=False)['clip_S_pp']
                prob = F.softmax(logits / max(temp, 1e-6), dim=-1)
                prior_sum = prob.sum(dim=0) if prior_sum is None else prior_sum + prob.sum(dim=0)
                total += prob.size(0)

    if prior_sum is None or total == 0:
        return 0.0

    prior = prior_sum / float(total)
    seen_mass = prior[model.seenclass].sum().clamp_min(1e-8)
    unseen_mass = prior[model.unseenclass].sum().clamp_min(1e-8)

    if target == 'class_frequency':
        target_seen = model.seenclass.numel() / float(model.nclass)
        target_unseen = model.unseenclass.numel() / float(model.nclass)
    else:
        target_seen = 0.5
        target_unseen = 0.5

    seen_shift = torch.log(torch.tensor(target_seen, device=device) / seen_mass)
    unseen_shift = torch.log(torch.tensor(target_unseen, device=device) / unseen_mass)
    bias = (unseen_shift - seen_shift).clamp(min=-max_bias, max=max_bias)
    bias_value = float(bias.item())
    print(f"  >> [PriorCorrection] seen_mass={seen_mass.item():.4f} "
          f"unseen_mass={unseen_mass.item():.4f} unseen_bias={bias_value:+.3f}",
          flush=True)
    return bias_value


def eval_zs_gzsl(dataloader, clip_model, model, device, bias_seen=0, bias_unseen=0):
    model.eval()

    seenclasses   = dataloader.seenclasses
    unseenclasses = dataloader.unseenclasses.long()
    in_package = {'model': model, 'device': device}

    # 优先用缓存（快），没有缓存用实时提取（慢）
    use_cache = _load_test_cache(device)

    effective_bias_unseen = bias_unseen

    with torch.no_grad():
        if use_cache:
            seen_patch_cpu   = _test_cache.get('seen_patch')   if _test_cache.get('has_patch') else None
            unseen_patch_cpu = _test_cache.get('unseen_patch') if _test_cache.get('has_patch') else None

            # ★ 加速优化: eval 时临时把测试 patch 上 GPU
            # 显存预算严格：系统 2~4 GB + 训练 patch (CPU 模式时 0) + 模型激活 ~2 GB
            # 在 RTX 5070 Ti (16 GB) 上, eval 时 free 显存 ~10 GB, 4 GB 测试 patch 安全
            # 用 2.0 倍余量阈值, 不够就走 CPU 老路
            seen_patch   = None
            unseen_patch = None
            try:
                if seen_patch_cpu is not None and not seen_patch_cpu.is_cuda:
                    free_gb = torch.cuda.mem_get_info(device)[0] / 1e9
                    test_size_gb = (seen_patch_cpu.numel() * seen_patch_cpu.element_size()
                                     + unseen_patch_cpu.numel() * unseen_patch_cpu.element_size()) / 1e9
                    if free_gb > test_size_gb * 2.0:
                        seen_patch   = seen_patch_cpu.to(device, non_blocking=True)
                        unseen_patch = unseen_patch_cpu.to(device, non_blocking=True)
                        # 一次性传完, 后续 batch 直接索引零 H2D
                    else:
                        seen_patch   = seen_patch_cpu
                        unseen_patch = unseen_patch_cpu
                else:
                    seen_patch   = seen_patch_cpu
                    unseen_patch = unseen_patch_cpu
            except Exception:
                seen_patch   = seen_patch_cpu
                unseen_patch = unseen_patch_cpu

            effective_bias_unseen = bias_unseen + _estimate_prior_unseen_bias(
                model, device)

            acc_seen  = _eval_from_cache(
                _test_cache['seen_feat'], _test_cache['seen_label'],
                model, seenclasses, in_package, device,
                bias_unseen=effective_bias_unseen, patches_cache=seen_patch)
            acc_novel, acc_zs = _eval_unseen_from_cache(
                _test_cache['unseen_feat'], _test_cache['unseen_label'],
                model, unseenclasses, in_package, device,
                bias_unseen=effective_bias_unseen, patches_cache=unseen_patch)

            # ★ 释放临时 GPU 张量, 不污染训练 step 显存
            if seen_patch is not None and seen_patch.is_cuda and not seen_patch_cpu.is_cuda:
                del seen_patch, unseen_patch
                torch.cuda.empty_cache()
        else:
            acc_seen  = val_gzsl_online(dataloader.test_seen_loader, clip_model, model,
                                        seenclasses, in_package, bias=bias_seen)
            acc_novel, acc_zs = val_zs_gzsl_online(dataloader.test_unseen_loader, clip_model,
                                                    model, unseenclasses, in_package, bias=bias_unseen)

    H = (2 * acc_seen * acc_novel) / (acc_seen + acc_novel) if (acc_seen + acc_novel) > 0 else 0
    return acc_seen, acc_novel, H, acc_zs


def _eval_from_cache(feat, labels, model, target_classes, in_package, device, batch_size=64,
                     bias_unseen=0.0, patches_cache=None):
    """用缓存特征评估 GZSL seen 准确率（完整模型，包含 local_score）

    batch_size=64: FAE 注意力 O(B²), 跟训练一致避免显存峰值翻 4 倍
    patches_cache: 可选 [N, 576, 768]（float16 on CPU）。若提供则用真实空间 patch；
                   否则用 CLS 复制 576 份（FAE 看到全相同 token，空间信号丢失）。
    """
    all_pred = []
    N = feat.size(0)
    model.eval()
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch_feat = feat[i:i+batch_size].to(device)         # [B, 768]
            cls = batch_feat.unsqueeze(1)                        # [B, 1, 768]
            if patches_cache is not None:
                # 真实 spatial patch（float16 → float32 on GPU）
                # 智能搬运: 已在 GPU 直接索引, 否则上 GPU (★ non_blocking 异步)
                if patches_cache.is_cuda:
                    patches = patches_cache[i:i+batch_size].float()
                else:
                    patches = patches_cache[i:i+batch_size].to(
                        device, non_blocking=True).float()
            else:
                # 兼容旧缓存：CLS 复制
                patches = cls.expand(-1, 576, -1).contiguous()   # [B, 576, 768]
            clip_input = torch.cat([cls, patches], dim=1)        # [B, 577, 768]
            out = model(clip_input, is_train=False)
            logits = out['clip_S_pp'].clone()                    # [B, 200]
            if bias_unseen != 0:
                logits[:, model.unseenclass] = logits[:, model.unseenclass] + bias_unseen
            pred = torch.argmax(logits, dim=1)
            all_pred.append(pred)
    predicted = torch.cat(all_pred)
    return compute_per_class_acc_gzsl(labels, predicted, target_classes, in_package)


def _eval_unseen_from_cache(feat, labels, model, unseen_classes, in_package, device, batch_size=64,
                            bias_unseen=0.0, patches_cache=None):
    """用缓存特征评估 GZSL unseen + ZSL 准确率（完整模型）

    batch_size=64: 跟训练一致, 避免 FAE O(B²) 显存峰值
    patches_cache: 可选 [N, 576, 768]（float16 on CPU）。
    """
    all_pred_gzsl = []
    all_pred_zsl  = []
    # ★ G3 诊断: 累计 logits 均值/最大值, 用于排查 seen/unseen logit bias
    diag_seen_max  = []
    diag_unseen_max = []
    diag_seen_mean = []
    diag_unseen_mean = []
    seen_classes_idx = model.seenclass if hasattr(model, 'seenclass') else None
    N = feat.size(0)
    model.eval()
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch_feat = feat[i:i+batch_size].to(device)
            cls = batch_feat.unsqueeze(1)
            if patches_cache is not None:
                if patches_cache.is_cuda:
                    patches = patches_cache[i:i+batch_size].float()
                else:
                    patches = patches_cache[i:i+batch_size].to(
                        device, non_blocking=True).float()
            else:
                patches = cls.expand(-1, 576, -1).contiguous()
            clip_input = torch.cat([cls, patches], dim=1)
            out = model(clip_input, is_train=False)
            logits = out['clip_S_pp'].clone()                    # [B, 200]
            # ★ G3 诊断: 收集 logits 在 seen / unseen 列上的统计
            if seen_classes_idx is not None:
                with torch.no_grad():
                    seen_l   = logits[:, seen_classes_idx]
                    unseen_l = logits[:, unseen_classes]
                    diag_seen_max.append(seen_l.max(dim=1).values.detach())
                    diag_unseen_max.append(unseen_l.max(dim=1).values.detach())
                    diag_seen_mean.append(seen_l.mean(dim=1).detach())
                    diag_unseen_mean.append(unseen_l.mean(dim=1).detach())
            # 给 unseen 类加偏置
            if bias_unseen != 0:
                logits[:, unseen_classes] = logits[:, unseen_classes] + bias_unseen
            pred_gzsl = torch.argmax(logits, dim=1)
            pred_zsl  = torch.argmax(logits[:, unseen_classes], dim=1)
            all_pred_gzsl.append(pred_gzsl)
            all_pred_zsl.append(pred_zsl)
    pred_gzsl = torch.cat(all_pred_gzsl)
    pred_zsl  = torch.cat(all_pred_zsl)
    acc_gzsl = compute_per_class_acc_gzsl(labels, pred_gzsl, unseen_classes, in_package)
    mapped_labels = map_label(labels, unseen_classes)
    acc_zsl  = compute_per_class_acc(mapped_labels, pred_zsl, unseen_classes.size(0))

    # ★ G3 诊断打印 (在 unseen 测试集上, 显眼标记便于排查)
    if seen_classes_idx is not None and len(diag_seen_max) > 0:
        s_max  = torch.cat(diag_seen_max).mean().item()
        u_max  = torch.cat(diag_unseen_max).mean().item()
        s_mean = torch.cat(diag_seen_mean).mean().item()
        u_mean = torch.cat(diag_unseen_mean).mean().item()
        print(f"  >> [G3-DIAG] unseen-set: max_seen={s_max:.3f}  max_unseen={u_max:.3f}  "
              f"Δmax={s_max-u_max:+.3f} | "
              f"mean_seen={s_mean:.3f}  mean_unseen={u_mean:.3f}  "
              f"Δmean={s_mean-u_mean:+.3f}", flush=True)

    return acc_gzsl, acc_zsl


def map_label(label, classes):
    # 创建一个与 label 形状相同的张量 mapped_label，初始值全部设为 -1
    mapped_label = torch.LongTensor(label.size()).fill_(-1)
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    # 将 label 对应到 unseen_classes 的新索引
    return mapped_label


def val_gzsl_online(loader, clip_model, model, target_classes, in_package, bias=0):
    device = in_package['device']
    true_labels, predicted_labels = extract_and_predict(loader, clip_model, model, device, target_classes, bias, is_zsl=False)

    # 将 Tensor 转到 device 上计算准确率
    true_labels = true_labels.to(device)
    predicted_labels = predicted_labels.to(device)
    return compute_per_class_acc_gzsl(true_labels, predicted_labels, target_classes, in_package)


def val_zs_gzsl_online(loader, clip_model, model, unseen_classes, in_package, bias=0):
    device = in_package['device']
    clip_model.eval()
    model.eval()

    pred_gzsl_list = []
    pred_zs_t_list = []
    true_labels_list = []

    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            # 提取特征 全局特征
            # features = clip_model.encode_image(batch_images).float()
            # 此处修改，提取空间特征图
            features = get_clip_spatial_features(clip_model, batch_images).float()
            # 模型推理
            out_package = model(features)
            output = out_package['clip_S_pp']

            # --- ZSL (T) ---
            # 只在 Unseen 类别中找最大值
            pred_zs_t = torch.argmax(output.data[:, unseen_classes], 1)

            # --- GZSL ---
            output[:, unseen_classes] = output[:, unseen_classes] + bias
            pred_gzsl = torch.argmax(output.data, 1)

            pred_gzsl_list.append(pred_gzsl.cpu())
            pred_zs_t_list.append(pred_zs_t.cpu())
            true_labels_list.append(batch_labels.cpu())

    true_labels = torch.cat(true_labels_list).to(device)
    predicted_label_gzsl = torch.cat(pred_gzsl_list).to(device)
    predicted_label_zs_t = torch.cat(pred_zs_t_list).to(device)

    # GZSL Unseen Accuracy
    acc_gzsl = compute_per_class_acc_gzsl(true_labels, predicted_label_gzsl, unseen_classes, in_package)

    # ZSL Accuracy (需要映射 Label)
    mapped_true_labels = map_label(true_labels, unseen_classes)
    acc_zs_t = compute_per_class_acc(mapped_true_labels, predicted_label_zs_t, unseen_classes.size(0))

    return acc_gzsl, acc_zs_t


def compute_per_class_acc(test_label, predicted_label, nclass):
    test_label = test_label.to(predicted_label.device)
    acc_per_class = torch.FloatTensor(nclass).fill_(0)
    for i in range(nclass):
        idx = (test_label == i)  # 选择当前类别 i 的样本
        if idx.sum() > 0:
            acc_per_class[i] = torch.sum(test_label[idx] == predicted_label[idx]).float() / torch.sum(idx).float()
    return acc_per_class.mean().item()


def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes, in_package):
    device = in_package['device']
    per_class_accuracies = torch.zeros(target_classes.size()[0]).float().to(device).detach()  # 用于存储每个类别的准确率
    # 确保预测类别在正确的计算设备上
    predicted_label = predicted_label.to(device)

    for i in range(target_classes.size()[0]):  # 遍历所有目标类别
        is_class = test_label == target_classes[i]  # 选出该类别的样本索引
        if is_class.sum() > 0:
            per_class_accuracies[i] = torch.div((predicted_label[is_class] == test_label[is_class]).sum().float(), is_class.sum().float())
    return per_class_accuracies.mean().item()
