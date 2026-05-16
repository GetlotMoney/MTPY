# DVSR 项目完整指南

## 一、项目文件结构总览

```
DVSR/
├── model/
│   └── VGSR.py                 ← 【核心】模型定义 (Adapter + 主模型)
├── tools/
│   ├── dataset.py              ← 数据加载器 (读图片、划分训练/测试集)
│   ├── helper_func.py          ← CLIP特征提取 + GZSL评估函数
│   ├── preprocessing.py        ← 预处理工具 (当前未使用)
│   └── global_setting.py       ← 全局路径设置
├── config/
│   └── VGSR_cub_gzsl.yaml     ← 超参数配置
├── data/
│   ├── gpt4_data/cub.pt        ← GPT-4 类别描述数据
│   ├── clip_att/               ← CLIP属性特征 (当前未使用)
│   ├── CUB/images/ → 符号链接  ← CUB图片
│   └── xlsa17/data/ → 符号链接 ← 数据划分文件
├── train_VGSR_CUB.py           ← 【入口】CUB训练脚本
├── train_log/                  ← 训练日志输出
└── xlsa17/                     ← 原始xlsa17数据
```

---

## 二、完整流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           准备阶段 (只执行一次)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ① config/VGSR_cub_gzsl.yaml → 读取超参数                              │
│                                                                         │
│  ② clip.load("ViT-L/14@336px") → 加载CLIP模型 (冻结，永不训练)         │
│                                                                         │
│  ③ CUBDataLoader (tools/dataset.py):                                    │
│     - 读 xlsa17/att_splits.mat → 获取 trainval_loc, test_seen/unseen   │
│     - 构建 train_dataset (7057张, 只有150个seen类)                      │
│     - 构建 test_seen_dataset (1764张)                                   │
│     - 构建 test_unseen_dataset (2967张, 50个unseen类)                   │
│                                                                         │
│  ④ CLIP编码类名 → class_text_embeds [200, 768]                          │
│     "a photo of a Laysan Albatross, a type of bird." × 200类            │
│                                                                         │
│  ⑤ 加载GPT-4描述 (data/gpt4_data/cub.pt):                              │
│     每类7条描述 → CLIP编码 → 取平均 → gpt_text_embeds [200, 768]        │
│                                                                         │
│  ⑥ 初始化 VGSR 模型 (model/VGSR.py):                                   │
│     - 存储文本特征 (冻结)                                               │
│     - 创建 Adapter (唯一可训练, 约30万参数)                             │
│     - 创建 logit_scale (1个可训练参数)                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      训练循环 (每个iteration)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ① dataloader.next_batch(64)                                            │
│     → batch_images [64, 3, 336, 336]  (64张seen类图片)                  │
│     → batch_labels [64]               (对应的类别索引)                  │
│                                                                         │
│  ② get_clip_spatial_features(clip_model, batch_images)                  │
│     图片 → CLIP视觉编码器(冻结) → [64, 577, 768]                       │
│     (577 = 1个CLS token + 576个patch token)                            │
│                                                                         │
│  ③ model.forward(clip_features):                                        │
│     ┌─────────────────────────────────────────────────────────┐         │
│     │  image_feat = clip_features[:, 0, :]     # [64, 768]   │         │
│     │  image_feat = normalize(image_feat)                     │         │
│     │                                                         │         │
│     │  text = gpt_text_embeds                  # [200, 768]   │         │
│     │  adapted = 0.2*Adapter(text) + 0.8*text  # 残差增强     │         │
│     │  adapted = normalize(adapted)                           │         │
│     │                                                         │         │
│     │  logits = image_feat @ adapted.T * scale # [64, 200]    │         │
│     └─────────────────────────────────────────────────────────┘         │
│                                                                         │
│  ④ compute_loss:                                                        │
│     loss = CrossEntropy(logits, labels)                                 │
│                                                                         │
│  ⑤ loss.backward() → 梯度只流向 Adapter 和 logit_scale                 │
│     optimizer.step() → 更新 Adapter 权重                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      评估 (每个epoch结束)                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  eval_zs_gzsl(dataloader, clip_model, model, device):                   │
│                                                                         │
│  ① test_seen_loader → CLIP提取特征 → model.forward → argmax            │
│     在200类中预测 → 与真实label对比 → per-class acc → S                 │
│                                                                         │
│  ② test_unseen_loader → CLIP提取特征 → model.forward → argmax          │
│     GZSL: 在200类中预测 → per-class acc → U                            │
│     ZSL:  只在50个unseen类中预测 → per-class acc → ZS                  │
│                                                                         │
│  ③ H = 2*S*U / (S+U)                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 三、各文件详细说明

### 3.1 `model/VGSR.py` — 模型定义

**你要加模块，改的就是这个文件。**

```python
# === Adapter 类 (第11-29行) ===
# 输入: [N, 768]  输出: [N, 768]
# 结构: 768 → 192 → 768 (bottleneck)
# 作用: 对文本特征做轻量级变换

# === VGSR 主模型 (第35行起) ===
# __init__: 存储文本特征、创建Adapter、创建logit_scale
# forward:  image_feat × adapted_text → logits
# compute_loss: CrossEntropy
```

### 3.2 `train_VGSR_CUB.py` — 训练入口

```python
# 第1-11行:   导入
# 第30-35行:  加载config
# 第48-53行:  加载CLIP (冻结)
# 第58-63行:  加载数据
# 第72-78行:  编码类名文本特征
# 第83-109行: 加载GPT-4描述并编码
# 第114-120行: 初始化模型
# 第127-128行: 优化器
# 第133-160行: 训练循环
```

### 3.3 `tools/helper_func.py` — 特征提取 + 评估

```python
# get_clip_spatial_features(): 
#   输入: images [B, 3, 336, 336]
#   输出: [B, 577, 768] (CLS + 576 patches)
#   作用: 从CLIP视觉编码器中提取空间特征

# eval_zs_gzsl():
#   输入: dataloader, clip_model, model, device
#   输出: acc_seen, acc_novel, H, acc_zs
#   作用: 标准GZSL评估

# compute_per_class_acc_gzsl():
#   对每个类单独算准确率，再取平均 (避免类别不平衡影响)
```

### 3.4 `tools/dataset.py` — 数据加载

```python
# CUBDataLoader:
#   - 从 xlsa17 读取数据划分 (trainval_loc / test_seen_loc / test_unseen_loc)
#   - 构建 PyTorch Dataset 和 DataLoader
#   - next_batch(): 从训练集随机采样一个batch
#   - test_seen_loader / test_unseen_loader: 用于评估
```

### 3.5 `config/VGSR_cub_gzsl.yaml` — 配置

```yaml
# 当前实际使用的参数:
num_class: 200          # 类别总数
dim_f_clip: 768         # CLIP特征维度
batch_size: 64          # 批大小
epochs: 10              # 训练轮数
adapter_ratio: 0.2      # Adapter残差比例
is_bias: False          # GZSL bias开关
# 其余参数是旧模板残留，当前模型未使用
```

---

## 四、如何添加新模块

### 场景 A：在视觉端加一个增强模块

比如你想对 CLIP 的视觉特征做一个变换再去匹配文本。

**Step 1**: 在 `model/VGSR.py` 中定义新模块

```python
class VisualAdapter(nn.Module):
    """视觉端增强模块"""
    def __init__(self, dim=768, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.fc(x)
```

**Step 2**: 在 `VGSR.__init__` 中实例化

```python
# 在 self.text_adapter = Adapter(...) 下面加:
self.visual_adapter = VisualAdapter(self.dim_f, reduction=4)
self.visual_adapter_ratio = 0.2
```

**Step 3**: 在 `VGSR.forward()` 中接入

```python
# 原来:
image_feat = F.normalize(image_feat, dim=1)

# 改成:
image_feat_adapted = self.visual_adapter_ratio * self.visual_adapter(image_feat) + \
                     (1 - self.visual_adapter_ratio) * image_feat
image_feat = F.normalize(image_feat_adapted, dim=1)
```

完事。其他文件不用动。

---

### 场景 B：双分支 (视觉增强 + 语义增强) 联合优化

你想同时对视觉和文本做增强，然后联合训练。

**Step 1**: 在 `model/VGSR.py` 中定义两个模块

```python
class VisualAdapter(nn.Module):
    def __init__(self, dim=768, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.fc(x)

# Adapter (文本端) 已经有了，不用重复定义
```

**Step 2**: 在 `VGSR.__init__` 中实例化

```python
# 文本端 (已有)
self.text_adapter = Adapter(self.dim_f, reduction=4)
self.adapter_ratio = 0.2

# 视觉端 (新增)
self.visual_adapter = VisualAdapter(self.dim_f, reduction=4)
self.visual_ratio = 0.2
```

**Step 3**: 在 `VGSR.forward()` 中改成双分支

```python
def forward(self, clip_features):
    # 提取全局特征
    if clip_features.dim() == 3:
        image_feat = clip_features[:, 0, :]
    else:
        image_feat = clip_features

    # ====== 视觉分支增强 ======
    image_feat = self.visual_ratio * self.visual_adapter(image_feat) + \
                 (1 - self.visual_ratio) * image_feat
    image_feat = F.normalize(image_feat, dim=1)

    # ====== 文本分支增强 ======
    text_feats_base = self.gpt_text_embeds if self.use_gpt else self.class_text_embeds
    adapted_text = self.adapter_ratio * self.text_adapter(text_feats_base) + \
                   (1 - self.adapter_ratio) * text_feats_base
    adapted_text = F.normalize(adapted_text, dim=1)

    # ====== 匹配 ======
    logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
    logits = torch.matmul(image_feat, adapted_text.T) * logit_scale

    if self.is_bias:
        logits = logits + self.mask_bias * self.bias

    return {'logits': logits, 'clip_S_pp': logits, 'clip_pred': logits}
```

**训练脚本和评估函数完全不用改**，因为接口没变（输入还是 clip_features，输出还是 logits）。

---

### 场景 C：加一个额外的 Loss

比如你想加 calibration loss 防止模型偏向 seen 类。

**在 `VGSR.compute_loss()` 中加：**

```python
def compute_loss(self, in_package):
    logits = in_package['logits']
    labels = in_package['batch_label']
    if labels.dim() > 1:
        labels = torch.argmax(labels, dim=1)

    # 主损失
    loss_CE = F.cross_entropy(logits, labels)

    # Calibration loss (新增)
    probs = F.softmax(logits, dim=1)
    mass_unseen = probs[:, self.unseenclass.long()].sum(dim=1)
    loss_cal = -torch.log(mass_unseen.mean() + 1e-8)

    # 总损失
    loss = loss_CE + 0.05 * loss_cal

    return {'loss': loss, 'loss_CE': loss_CE, 'loss_cal': loss_cal}
```

---

## 五、添加模块的通用规则

| 你想做什么 | 改哪里 | 不用改什么 |
|-----------|--------|-----------|
| 加视觉增强模块 | `model/VGSR.py` 的 `__init__` + `forward` | 训练脚本、评估、数据加载 |
| 加文本增强模块 | `model/VGSR.py` 的 `__init__` + `forward` | 同上 |
| 加新的 Loss | `model/VGSR.py` 的 `compute_loss` | 同上 |
| 改超参数 | `config/VGSR_cub_gzsl.yaml` | 同上 |
| 改数据增强 | `tools/dataset.py` 的 transform | 模型、评估 |
| 改评估方式 | `tools/helper_func.py` | 模型、训练 |

**核心原则：模型的输入永远是 `clip_features`，输出永远是包含 `'clip_S_pp'` key 的 dict。只要保持这个接口，训练和评估代码就不用动。**

---

## 六、数据流中每个张量的形状

| 变量 | 形状 | 含义 |
|------|------|------|
| batch_images | [64, 3, 336, 336] | 64张RGB图片 |
| clip_features | [64, 577, 768] | CLIP空间特征 (CLS+576patches) |
| image_feat | [64, 768] | 取CLS token后的全局特征 |
| text_feats_base | [200, 768] | 200个类的GPT文本特征 |
| adapted_text | [200, 768] | Adapter增强后的文本特征 |
| logits | [64, 200] | 64张图对200类的得分 |
| batch_labels | [64] | 真实类别索引 (0~199) |
| loss | 标量 | CrossEntropy损失值 |
