# tools/helper_func.py 完整代码讲解

> CLIP 特征提取 + GZSL 评估协议的核心函数文件。
> 训练脚本里 `eval_zs_gzsl` 调用的就是这里的函数。

---

## 目录

- [0. 这个文件在干什么](#0-这个文件在干什么)
- [1. get_clip_spatial_features：从 CLIP 提 576 patch 特征](#1-get_clip_spatial_features从-clip-提-576-patch-特征)
- [2. extract_and_predict：提取特征 + 预测（废弃/复用）](#2-extract_and_predict提取特征--预测废弃复用)
- [3. eval_zs_gzsl：主评估入口](#3-eval_zs_gzsl主评估入口)
- [4. map_label：标签重映射](#4-map_label标签重映射)
- [5. val_gzsl_online：评估 seen 类精度](#5-val_gzsl_online评估-seen-类精度)
- [6. val_zs_gzsl_online：同时评估 GZSL 和 ZSL](#6-val_zs_gzsl_online同时评估-gzsl-和-zsl)
- [7. compute_per_class_acc：普通的 per-class 精度](#7-compute_per_class_acc普通的-per-class-精度)
- [8. compute_per_class_acc_gzsl：GZSL 下的 per-class 精度](#8-compute_per_class_acc_gzslgzsl-下的-per-class-精度)
- [9. 评估指标完整流程图](#9-评估指标完整流程图)

---

## 0. 这个文件在干什么

**两大功能：**

1. **CLIP 特征提取**：`get_clip_spatial_features` 实时从 CLIP 的视觉编码器中提取 576 个 patch 特征（训练/评估时用）
2. **GZSL 评估**：一系列函数算四个核心指标
   - **S**（seen 类精度，GZSL 搜索空间 = 全 200 类）
   - **U**（unseen 类精度，GZSL 搜索空间 = 全 200 类）
   - **H**（调和平均 = `2SU/(S+U)`）
   - **ZS**（unseen 类精度，但搜索空间只有 50 个 unseen 类）

**为什么要 per-class 精度，不用普通 accuracy？**
因为 GZSL 测试集每类样本数不平衡，直接算 `正确数/总数` 会被大类主导。**per-class 精度**是先每个类分别算精度，再对类求平均，更公平。

---

## 1. get_clip_spatial_features：从 CLIP 提 576 patch 特征

这个函数**手动展开** CLIP 的视觉编码器，按步骤得到 patch 级特征。

```python
def get_clip_spatial_features(clip_model, images):
    with torch.no_grad():
        # 卷积层切块 卷积核大小14x14 步长14 在336x336的图上滑动，切出576块
        x = clip_model.visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
```

**逐步讲解：**

### 1.1 函数签名
```python
def get_clip_spatial_features(clip_model, images):
```
- `clip_model`：加载好的 CLIP 模型（整个对象）
- `images`：`[B, 3, 336, 336]` 的图像张量（已经过预处理和归一化）

### 1.2 关闭梯度
```python
with torch.no_grad():
```
- CLIP 冻结，不需要梯度，关掉可以节省一半显存

### 1.3 Conv1：切块
```python
x = clip_model.visual.conv1(images)
```

CLIP ViT-L/14@336 的 `visual.conv1` 是一个：
- 卷积核大小 `14 × 14`
- stride（步长）`14`
- 输入通道 3（RGB）
- 输出通道 1024（CLIP ViT-L 内部维度）

**效果：把图像切成 patch 并嵌入。**

**形状变化：**
```
images: [B, 3, 336, 336]
conv1:  [B, 1024, 24, 24]   (336/14 = 24)
```

- `24 × 24 = 576` 就是 patch 数量
- 每个 patch 是 1024 维向量

### 1.4 Reshape：展平空间维
```python
x = x.reshape(x.shape[0], x.shape[1], -1)
```

```
[B, 1024, 24, 24] → [B, 1024, 576]
```

- `-1`：自动计算这个维度大小，`24*24=576`

### 1.5 Permute：交换维度
```python
x = x.permute(0, 2, 1)
```

```
[B, 1024, 576] → [B, 576, 1024]
```

现在 shape 是 `[batch, 序列长度, 特征维度]`，符合 Transformer 的标准输入格式。

### 1.6 拼接 CLS token

```python
        class_embedding = clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                                      dtype=x.dtype, device=x.device)
        x = torch.cat([class_embedding, x], dim=1)
```

**什么是 CLS token？**

Transformer 通常在序列最前面加一个"特殊 token"，叫 CLS（class token）。它不是真实 patch，而是一个可学习的向量，用来收集全局信息。

**代码拆解：**

```python
clip_model.visual.class_embedding
```
- 这是 CLIP 里预训练好的 CLS 向量，形状 `[1024]`

```python
torch.zeros(x.shape[0], 1, x.shape[-1], ...)
```
- 形状 `[B, 1, 1024]`，全零张量
- 作用：构造一个可以广播的"承载器"

```python
class_embedding = clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(...)
```
- `[1024]` + `[B, 1, 1024]` → 广播 → `[B, 1, 1024]`
- 每个样本都得到一个相同的 CLS 向量

```python
x = torch.cat([class_embedding, x], dim=1)
```
- `[B, 1, 1024]` + `[B, 576, 1024]` → `[B, 577, 1024]`
- CLS 拼在最前面，所以索引 `[:, 0, :]` 是 CLS，`[:, 1:, :]` 是 576 个 patch

### 1.7 加位置编码

```python
        x = x + clip_model.visual.positional_embedding.to(x.dtype)
```

- `positional_embedding`: `[577, 1024]`（577 = 1 CLS + 576 patch）
- 每个位置加一个不同的向量，告诉模型"我是第几个 patch"
- `x` 是 `[B, 577, 1024]`，广播相加

**为什么要位置编码？**  
Transformer 的自注意力本身不知道"哪个 patch 在上哪个在下"，加位置编码后才有空间感。

### 1.8 LayerNorm + Permute

```python
        x = clip_model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)
```

- `ln_pre`：LayerNorm，对最后一维归一化
- `.permute(1, 0, 2)`：`[B, 577, 1024]` → `[577, B, 1024]`

**为什么要换维度？**  
PyTorch 旧版 Transformer 默认用 `[seq_len, batch, dim]` 格式（`batch_first=False`）。CLIP 用的是这种格式。

### 1.9 Transformer 主体

```python
        x = clip_model.visual.transformer(x)
        x = x.permute(1, 0, 2)
```

- `transformer`：ViT 的 24 层 Transformer Encoder（对 ViT-L）
- 每层：自注意力 + FFN + 残差 + LN
- 输出形状不变：`[577, B, 1024]`
- 再 permute 回来：`[B, 577, 1024]`

**这一步最耗时**，CLIP 的 ViT-L 有 24 层，每层 16 个头。

### 1.10 最终归一化 + 投影

```python
        x = clip_model.visual.ln_post(x)
        if clip_model.visual.proj is not None:
            x = x @ clip_model.visual.proj
```

- `ln_post`：最后一次 LayerNorm
- `visual.proj`: `[1024, 768]` 的投影矩阵（把 ViT 内部 1024 维投到 CLIP 共享的 768 维空间）
- `x @ proj`：矩阵乘，`[B, 577, 1024] @ [1024, 768]` → `[B, 577, 768]`

**为什么要投影到 768？**  
CLIP 的视觉和文本编码器最终要在**同一个 768 维空间**里比较相似度。图像和文本走到最后都要 768 维。

### 1.11 返回

```python
    return x
```

返回 `[B, 577, 768]`，包含 CLS 和 576 个 patch。

**使用方式：**
- 全局特征：`x[:, 0, :]` → `[B, 768]`（CLS token）
- 局部特征：`x[:, 1:, :]` → `[B, 576, 768]`（576 个 patch）

你的 VGSR 模型用的是后者。

---

## 2. extract_and_predict：提取特征 + 预测（废弃/复用）

```python
def extract_and_predict(loader, clip_model, model, device, target_classes=None, bias=0, is_zsl=False):
    model.eval()
    clip_model.eval()
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            features = get_clip_spatial_features(clip_model, batch_images).float()

            out_package = model(features)
            output = out_package['clip_S_pp']
            if is_zsl:
                output_t = output.clone()
                pred = torch.argmax(output_t.data[:, target_classes], 1)
            else:
                if target_classes is not None:
                    output[:, target_classes] = output[:, target_classes] + bias
                pred = torch.argmax(output.data, 1)

            predicted_labels.append(pred.cpu())
            true_labels.append(batch_labels.cpu())
    return torch.cat(true_labels), torch.cat(predicted_labels)
```

**讲解：**

这个函数是**通用的预测流程**。目前主代码里**没直接用它**（`val_gzsl_online` 只用了它，`val_zs_gzsl_online` 是独立版本），但逻辑和下面两个函数类似，理解它就理解其他。

**逐步讲解：**

```python
model.eval()
clip_model.eval()
```
- 切换到评估模式（关 Dropout、BatchNorm 统计量固定）

```python
predicted_labels = []
true_labels = []
```
- 收集每个 batch 的结果

```python
with torch.no_grad():
    for batch_images, batch_labels in loader:
```
- `loader` 是 PyTorch DataLoader，迭代得到 `(图像, 标签)` batch
- `no_grad`：评估时不需要梯度

```python
        batch_images = batch_images.to(device)
        features = get_clip_spatial_features(clip_model, batch_images).float()
```
- 搬 GPU + 提取 CLIP 特征 `[B, 577, 768]`

```python
        out_package = model(features)
        output = out_package['clip_S_pp']
```
- 调用 VGSR（默认 `is_train=False`）
- 输出 `[B, 200]` 全类 logits

```python
        if is_zsl:
            output_t = output.clone()
            pred = torch.argmax(output_t.data[:, target_classes], 1)
```

**ZSL 模式：**
- `output[:, target_classes]`：只取 unseen 50 列 → `[B, 50]`
- `torch.argmax(..., 1)`：在维度 1（类别维度）上找最大值索引
- `pred` 的范围是 `[0, 50)`，**局部索引**

```python
        else:
            if target_classes is not None:
                output[:, target_classes] = output[:, target_classes] + bias
            pred = torch.argmax(output.data, 1)
```

**GZSL 模式：**
- 如果给了 `target_classes`，给这些列的分数加 `bias`
  - 这是**校准技术（Calibrated Stacking）**：给 unseen 类加正 bias，或给 seen 类加负 bias，平衡两边的偏好
  - 本代码默认 `bias=0`，不做校准
- `argmax` 在全 200 类中找最大

```python
        predicted_labels.append(pred.cpu())
        true_labels.append(batch_labels.cpu())
```
- 收集结果，搬回 CPU（节省 GPU 显存）

```python
    return torch.cat(true_labels), torch.cat(predicted_labels)
```
- 拼接所有 batch 的结果，返回两个长张量

**`.data` 是什么？**  
是访问原始张量数据（不带梯度）。现代 PyTorch 里用 `tensor.detach()` 更推荐，但这里 `no_grad` 已经关梯度了，效果一样。

---

## 3. eval_zs_gzsl：主评估入口

```python
def eval_zs_gzsl(dataloader, clip_model, model, device, bias_seen=0, bias_unseen=0):
    model.eval()

    seenclasses = dataloader.seenclasses
    unseenclasses = dataloader.unseenclasses.long()

    in_package = {'model': model, 'device': device}

    with torch.no_grad():
        acc_seen = val_gzsl_online(dataloader.test_seen_loader, clip_model, model, seenclasses, in_package, bias=bias_seen)
        acc_novel, acc_zs = val_zs_gzsl_online(dataloader.test_unseen_loader, clip_model, model, unseenclasses, in_package, bias=bias_unseen)

    if (acc_seen + acc_novel) > 0:
        H = (2 * acc_seen * acc_novel) / (acc_seen + acc_novel)
    else:
        H = 0
    return acc_seen, acc_novel, H, acc_zs
```

**讲解：**

这是训练脚本调用的入口函数。

**参数：**
- `dataloader`：CUBDataLoader 对象
- `clip_model`：CLIP 模型
- `model`：VGSR 模型
- `device`：GPU
- `bias_seen`, `bias_unseen`：校准 bias（默认 0）

**流程：**

```python
seenclasses = dataloader.seenclasses
unseenclasses = dataloader.unseenclasses.long()
```
- 取出 seen 和 unseen 的全局类别索引
- `.long()`：转成 int64（索引张量要求）

```python
in_package = {'model': model, 'device': device}
```
- 打包成字典，方便传递

```python
acc_seen = val_gzsl_online(
    dataloader.test_seen_loader, clip_model, model, seenclasses, in_package, bias=bias_seen)
```

**评估 S（GZSL seen 类精度）：**
- 用 test_seen_loader（测试集中属于 seen 类的图片）
- `target_classes = seenclasses` 只是用来标记算哪些类的精度
- 注意：预测时是在**全 200 类**中竞争，不是 seen 类内部

```python
acc_novel, acc_zs = val_zs_gzsl_online(
    dataloader.test_unseen_loader, clip_model, model, unseenclasses, in_package, bias=bias_unseen)
```

**同时评估 U 和 ZS：**
- 用 test_unseen_loader（测试集中属于 unseen 类的图片）
- `val_zs_gzsl_online` 一次性算两个：
  - `acc_novel`（U）：在全 200 类中预测，只看 unseen 类对不对
  - `acc_zs`（ZS）：限制在 50 unseen 类中预测

```python
if (acc_seen + acc_novel) > 0:
    H = (2 * acc_seen * acc_novel) / (acc_seen + acc_novel)
else:
    H = 0
```

**调和平均（Harmonic mean）：**
- `H = 2 * S * U / (S + U)`
- 只有 S 和 U 都高，H 才会高
- 如果一边是 0（另一边再高也没用），H = 0
- 判断 `(acc_seen + acc_novel) > 0` 是防止除零

```python
return acc_seen, acc_novel, H, acc_zs
```

返回 4 个指标。

---

## 4. map_label：标签重映射

```python
def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size()).fill_(-1)
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label
```

**讲解：**

把**全局类别索引**映射到**局部索引**。

**例子：**
- `unseenclasses = [5, 12, 23, 34, ...]`（50 个全局索引）
- 某张图的全局标签 `label = 23`
- 映射后 `mapped_label = 2`（因为 23 在 unseenclasses 里排第 2 位）

**代码拆解：**

```python
mapped_label = torch.LongTensor(label.size()).fill_(-1)
```
- 创建一个和 `label` 形状相同的 LongTensor
- 填充 `-1`（表示"未映射"的默认值）

```python
for i in range(classes.size(0)):
    mapped_label[label == classes[i]] = i
```

- 遍历每个类别（比如 `i=0, 1, 2, ..., 49`）
- `label == classes[i]`：找出所有 `label` 等于 `classes[i]` 的位置
- 把这些位置的 `mapped_label` 设为 `i`

**为什么要映射？**  
算 ZSL 精度时，预测 `pred` 是 `[0, 50)` 的局部索引，但真实 label 是全局 `[0, 200)` 的索引。必须统一到同一个编号系统。

---

## 5. val_gzsl_online：评估 seen 类精度

```python
def val_gzsl_online(loader, clip_model, model, target_classes, in_package, bias=0):
    device = in_package['device']
    true_labels, predicted_labels = extract_and_predict(loader, clip_model, model, device, target_classes, bias, is_zsl=False)

    true_labels = true_labels.to(device)
    predicted_labels = predicted_labels.to(device)
    return compute_per_class_acc_gzsl(true_labels, predicted_labels, target_classes, in_package)
```

**讲解：**

简单的 wrapper：
1. 调 `extract_and_predict` 跑一遍测试集，得到 `(真实标签, 预测标签)`
2. 调 `compute_per_class_acc_gzsl` 算 per-class 精度

**注意：**
- 这里 `target_classes = seenclasses` 被传进去只用来做**校准 bias**（默认 0，没实际作用）
- 预测时 `argmax` 是在全 200 类中竞争
- 算精度时只看 seen 类样本的精度（test_seen_loader 里的图片都是 seen 类）

---

## 6. val_zs_gzsl_online：同时评估 GZSL 和 ZSL

这个函数**同时**跑 GZSL 和 ZSL 两个指标，避免重复提取特征。

```python
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
            features = get_clip_spatial_features(clip_model, batch_images).float()
            out_package = model(features)
            output = out_package['clip_S_pp']

            # --- ZSL (T) ---
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

    acc_gzsl = compute_per_class_acc_gzsl(true_labels, predicted_label_gzsl, unseen_classes, in_package)

    mapped_true_labels = map_label(true_labels, unseen_classes)
    acc_zs_t = compute_per_class_acc(mapped_true_labels, predicted_label_zs_t, unseen_classes.size(0))

    return acc_gzsl, acc_zs_t
```

**逐步讲解：**

### 6.1 提取特征 + 推理

```python
features = get_clip_spatial_features(clip_model, batch_images).float()
out_package = model(features)
output = out_package['clip_S_pp']
```

- 输出 `output`: `[B, 200]` 全类 logits

### 6.2 ZSL 预测

```python
pred_zs_t = torch.argmax(output.data[:, unseen_classes], 1)
```

- `output[:, unseen_classes]`: 只取 50 个 unseen 列 → `[B, 50]`
- `argmax(1)`: 在 50 类中找最大 → **局部索引 0~49**

### 6.3 GZSL 预测

```python
output[:, unseen_classes] = output[:, unseen_classes] + bias
pred_gzsl = torch.argmax(output.data, 1)
```

- 给 unseen 列加 bias（默认 0）
- 在全 200 类中 argmax → **全局索引 0~199**

**注意**：这里 `output` 被**原地修改**了，但因为 ZSL 已经算完了，所以没影响。

### 6.4 收集结果

```python
pred_gzsl_list.append(pred_gzsl.cpu())
pred_zs_t_list.append(pred_zs_t.cpu())
true_labels_list.append(batch_labels.cpu())
```

### 6.5 拼接

```python
true_labels = torch.cat(true_labels_list).to(device)
predicted_label_gzsl = torch.cat(pred_gzsl_list).to(device)
predicted_label_zs_t = torch.cat(pred_zs_t_list).to(device)
```

`torch.cat` 把所有 batch 的结果拼成一个长张量。

### 6.6 计算 GZSL-U

```python
acc_gzsl = compute_per_class_acc_gzsl(true_labels, predicted_label_gzsl, unseen_classes, in_package)
```

- `true_labels`：unseen 类图片的**全局**标签
- `predicted_label_gzsl`：**全局**预测（0~199）
- `unseen_classes`：要算哪些类的精度

算的是：**这些 unseen 类图片在全 200 类中被正确识别的比例**。

### 6.7 计算 ZS

```python
mapped_true_labels = map_label(true_labels, unseen_classes)
acc_zs_t = compute_per_class_acc(mapped_true_labels, predicted_label_zs_t, unseen_classes.size(0))
```

- 把真实标签从全局映射到局部（`[0, 200) → [0, 50)`）
- `predicted_label_zs_t` 本身就是局部
- 调 `compute_per_class_acc`（普通版本，不需要 in_package）

---

## 7. compute_per_class_acc：普通的 per-class 精度

```python
def compute_per_class_acc(test_label, predicted_label, nclass):
    test_label = test_label.to(predicted_label.device)
    acc_per_class = torch.FloatTensor(nclass).fill_(0)
    for i in range(nclass):
        idx = (test_label == i)
        if idx.sum() > 0:
            acc_per_class[i] = torch.sum(test_label[idx] == predicted_label[idx]).float() / torch.sum(idx).float()
    return acc_per_class.mean().item()
```

**讲解：**

计算 **per-class accuracy**（每类精度后求平均，不是整体精度）。

**逐行：**

```python
test_label = test_label.to(predicted_label.device)
```
- 确保两个张量在同一个设备上

```python
acc_per_class = torch.FloatTensor(nclass).fill_(0)
```
- 创建长度为 `nclass` 的零张量

```python
for i in range(nclass):
    idx = (test_label == i)
```
- `idx` 是一个布尔张量，长度等于 test_label，值为 True 的位置表示真实类别是 `i`

```python
    if idx.sum() > 0:
        acc_per_class[i] = torch.sum(test_label[idx] == predicted_label[idx]).float() / torch.sum(idx).float()
```

- `idx.sum() > 0`：如果这个类有样本
- `test_label[idx]`：该类的真实标签（全是 `i`）
- `predicted_label[idx]`：该类样本对应的预测
- `test_label[idx] == predicted_label[idx]`：布尔数组，正确的位置是 True
- `.sum()`：正确数
- `/ idx.sum()`：除以该类总样本数 → 精度

```python
return acc_per_class.mean().item()
```

- 对所有类的精度求平均
- `.item()`：转成 Python float

**per-class 精度 vs 普通精度：**

假设 unseen 有 50 类，类 A 有 100 张图（全对），类 B 有 10 张图（全错）。
- 普通精度：`100 / 110 ≈ 90.9%`
- per-class 精度：`(100% + 0%) / 2 = 50%`

后者更公平，不被大类主导。

---

## 8. compute_per_class_acc_gzsl：GZSL 下的 per-class 精度

```python
def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes, in_package):
    device = in_package['device']
    per_class_accuracies = torch.zeros(target_classes.size()[0]).float().to(device).detach()
    predicted_label = predicted_label.to(device)

    for i in range(target_classes.size()[0]):
        is_class = test_label == target_classes[i]
        if is_class.sum() > 0:
            per_class_accuracies[i] = torch.div((predicted_label[is_class] == test_label[is_class]).sum().float(), is_class.sum().float())
    return per_class_accuracies.mean().item()
```

**讲解：**

和 `compute_per_class_acc` 几乎一样，但是：
- `target_classes` 是要算精度的类（全局索引），不是范围 `0~nclass-1`
- 用于 GZSL：真实标签是全局的，预测也是全局的

**逐行：**

```python
per_class_accuracies = torch.zeros(target_classes.size()[0]).float().to(device).detach()
```
- `target_classes.size()[0]`：类别数量
- `.detach()`：脱离计算图（虽然已经在 no_grad 下了，双保险）

```python
for i in range(target_classes.size()[0]):
    is_class = test_label == target_classes[i]
```
- `target_classes[i]`：第 i 个目标类的全局索引
- `is_class`：布尔张量，标记该类样本

```python
if is_class.sum() > 0:
    per_class_accuracies[i] = torch.div(
        (predicted_label[is_class] == test_label[is_class]).sum().float(),
        is_class.sum().float()
    )
```

- `torch.div(a, b)` ≡ `a / b`
- 正确数 / 总数 = 精度

```python
return per_class_accuracies.mean().item()
```

---

## 9. 评估指标完整流程图

```
训练循环结束后：
    ↓
eval_zs_gzsl(dataloader, clip_model, model, device)
    ↓
    ├─→ val_gzsl_online(test_seen_loader, ..., seenclasses)
    │       ├─→ extract_and_predict(..., is_zsl=False)
    │       │       for batch in loader:
    │       │           features = get_clip_spatial_features(clip, images)  # [B, 577, 768]
    │       │           output = model(features)['clip_S_pp']                # [B, 200]
    │       │           pred = argmax(output, 1)                              # 全局索引
    │       └─→ compute_per_class_acc_gzsl(true, pred, seenclasses)
    │               → acc_seen (S) ≈ 73%
    │
    ├─→ val_zs_gzsl_online(test_unseen_loader, ..., unseenclasses)
    │       for batch in loader:
    │           features = get_clip_spatial_features(clip, images)
    │           output = model(features)['clip_S_pp']                         # [B, 200]
    │           pred_zs  = argmax(output[:, unseenclasses], 1)                # 局部 [0, 50)
    │           pred_gzsl = argmax(output, 1)                                 # 全局 [0, 200)
    │       ├─→ compute_per_class_acc_gzsl(true_global, pred_gzsl, unseenclasses)
    │       │       → acc_novel (U) ≈ 67%
    │       └─→ mapped_true = map_label(true_global, unseenclasses)           # 全局→局部
    │           compute_per_class_acc(mapped_true, pred_zs, 50)
    │               → acc_zs (ZS) ≈ 78%
    │
    └─→ H = 2 * S * U / (S + U) ≈ 70%

返回: (S, U, H, ZS)
```

---

## 结语

这个文件的核心有两点：

1. **`get_clip_spatial_features`** 把 CLIP 的视觉编码器"拆开跑"，取出完整的 577 个 token（1 个 CLS + 576 个 patch）。
   - 为什么不用 `clip_model.encode_image()`？因为它只返回 CLS token，丢了 patch 信息。

2. **四个指标的计算协议**：
   - `S`：seen 测试图 在全 200 类中 argmax，只算 seen 类的 per-class 精度
   - `U`：unseen 测试图 在全 200 类中 argmax，只算 unseen 类的 per-class 精度
   - `H`：`2SU/(S+U)` 调和平均
   - `ZS`：unseen 测试图 只在 50 unseen 类中 argmax 的 per-class 精度

**关键细节：**
- 用 **per-class** 精度而不是整体精度（样本不平衡公平性考虑）
- `map_label` 做全局↔局部索引转换
- 评估时 CLIP 和模型都要 `eval()` + `no_grad()`
- `val_zs_gzsl_online` 同时算两个指标避免重复推理

下一份文档讲 `dataset.py`。
