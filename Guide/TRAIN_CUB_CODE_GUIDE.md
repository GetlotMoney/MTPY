# train_VGSR_CUB.py 完整代码讲解

> 训练主脚本，串起"配置 → 模型 → 数据 → 训练循环 → 评估"全流程。
> 和 `VGSR.py` 一样逐段讲解，适合小白阅读。

---

## 目录

- [0. 训练脚本在做什么](#0-训练脚本在做什么)
- [1. 导入和常量](#1-导入和常量)
- [2. 日志系统](#2-日志系统)
- [3. 加载配置文件](#3-加载配置文件)
- [4. 加载 CLIP 模型](#4-加载-clip-模型)
- [5. 加载数据集](#5-加载数据集)
- [6. 加载特征缓存](#6-加载特征缓存)
- [7. 编码类名文本](#7-编码类名文本)
- [8. 加载 GPT-4 描述](#8-加载-gpt-4-描述)
- [9. 初始化模型](#9-初始化模型)
- [10. 优化器和调度器](#10-优化器和调度器)
- [11. 训练循环](#11-训练循环)
- [12. 最终汇总](#12-最终汇总)

---

## 0. 训练脚本在做什么

这个脚本的工作流程可以概括成一张图：

```
[读配置] → [加载CLIP] → [加载数据] → [加载缓存特征]
     ↓
[编码类名文本 200×768] → [加载GPT-4描述 200×768]
     ↓
[初始化 VGSR 模型] → [构建优化器]
     ↓
for epoch in 1..N:
    for step in 1..iters_per_epoch:
        [取 batch 特征] → [前向] → [算 loss] → [反向] → [更新参数]
    [评估 GZSL: 算 U/S/H/ZS 四个指标]
    [记录最佳结果]
     ↓
[打印最终最佳指标]
```

---

## 1. 导入和常量

```python
import os
import torch
import torch.optim as optim
import numpy as np
import yaml
import clip
from types import SimpleNamespace
from datetime import datetime

from model.VGSR import VGSR
from tools.dataset import CUBDataLoader
from tools.helper_func import eval_zs_gzsl, get_clip_spatial_features

CACHE_DIR = './data/cache'
CACHE_TRAIN_FEAT   = os.path.join(CACHE_DIR, 'CUB_train_features.pt')
CACHE_TRAIN_LABEL  = os.path.join(CACHE_DIR, 'CUB_train_labels.pt')
CACHE_TRAIN_PATCH  = os.path.join(CACHE_DIR, 'CUB_train_patch_features.pt')  # [N, 576, 768] float16
```

**逐行讲解：**

| 代码 | 含义 |
|------|------|
| `import os` | 操作系统相关，拼路径、判断文件是否存在 |
| `import torch` | PyTorch 核心 |
| `import torch.optim as optim` | 优化器模块（Adam、SGD 等） |
| `import numpy as np` | 数值计算 |
| `import yaml` | 读 YAML 配置文件 |
| `import clip` | OpenAI 的 CLIP 库 |
| `from types import SimpleNamespace` | 把字典转成对象（可以用 `config.xxx` 访问而不是 `config['xxx']`） |
| `from datetime import datetime` | 生成时间戳用于日志文件名 |

**自己项目内的 import：**
- `VGSR`：主模型（上一份文档讲过）
- `CUBDataLoader`：CUB 数据加载器
- `eval_zs_gzsl`：GZSL 评估函数
- `get_clip_spatial_features`：从 CLIP 提取 spatial 特征的函数

**三个缓存文件路径：**
- `CACHE_TRAIN_FEAT`：`CUB_train_features.pt`，旧版缓存，只存 CLS token，形状 `[N, 768]`
- `CACHE_TRAIN_LABEL`：`CUB_train_labels.pt`，训练集标签，形状 `[N]`
- `CACHE_TRAIN_PATCH`：`CUB_train_patch_features.pt`，**新版缓存**，存完整 576 个 patch，形状 `[N, 576, 768]`，用 `float16` 存储节省硬盘和内存

**为什么要缓存？**  
因为 CLIP 的 ViT-L 很大，每张图跑一次大概要 50~100 毫秒。训练 5000+ 张图 × 50 个 epoch 要跑几十万次 CLIP，极慢。所以把 CLIP 的输出**一次性存下来**，训练时直接读缓存，快 10 倍以上。

---

## 2. 日志系统

```python
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
LOG_FILE = f"./train_log/CUB/training_log_CUB_{current_time}.txt"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


def print_log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding='utf-8') as f:
        f.write(msg + "\n")
```

**逐行讲解：**

- `datetime.now().strftime('%Y-%m-%d_%H-%M-%S')`：生成当前时间的字符串，比如 `2026-05-10_22-16-54`
- `LOG_FILE`：日志文件路径，带时间戳（每次运行都是新文件，不会覆盖）
- `os.makedirs(..., exist_ok=True)`：创建目录，`exist_ok=True` 表示已存在不报错

**`print_log` 函数：**
- 既在控制台打印，又写到日志文件
- `"a"` 是 append 模式，追加到文件末尾
- `encoding='utf-8'`：Windows 下防止中文乱码

**为什么要双写？**  
控制台打印可以实时看到进度，日志文件可以训练完后慢慢翻查。

---

## 3. 加载配置文件

```python
with open('./config/VGSR_cub_gzsl.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
config = {k: v['value'] if isinstance(v, dict) and 'value' in v else v for k, v in config.items()}
config = SimpleNamespace(**config)

if not hasattr(config, 'device'):
    config.device = 'cuda:0'
```

**逐行讲解：**

```python
with open(..., 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
```
- `with ... as f`：上下文管理，自动关闭文件
- `yaml.safe_load(f)`：把 YAML 文件解析成 Python 字典

**YAML 配置示例**（`VGSR_cub_gzsl.yaml` 里可能长这样）：
```yaml
batch_size:
  value: 64
epochs:
  value: 50
```
解析后是：
```python
{'batch_size': {'value': 64}, 'epochs': {'value': 50}}
```

```python
config = {k: v['value'] if isinstance(v, dict) and 'value' in v else v for k, v in config.items()}
```

**字典推导式**，语法可能有点绕，拆开看：
```python
# 等价于：
new_config = {}
for k, v in config.items():
    if isinstance(v, dict) and 'value' in v:
        new_config[k] = v['value']      # 取出 value 字段
    else:
        new_config[k] = v
config = new_config
```

结果：`{'batch_size': 64, 'epochs': 50}`

```python
config = SimpleNamespace(**config)
```

**`SimpleNamespace(**d)`**：把字典 `d` 解包成关键字参数，变成一个对象。
- 之前：`config['batch_size']`
- 现在：`config.batch_size`（更简洁）

`**config` 叫"字典解包"，相当于 `SimpleNamespace(batch_size=64, epochs=50)`。

```python
if not hasattr(config, 'device'):
    config.device = 'cuda:0'
```

**`hasattr(obj, 'x')`**：检查对象是否有属性 `x`。没有 device 字段就默认设 `cuda:0`。

---

下面打印配置信息：
```python
print_log("=" * 60)
print_log("  CLIP + Adapter + GPT  |  CUB GZSL Training")
print_log("=" * 60)
print_log(f"  Log file  : {LOG_FILE}")
print_log(f"  Device    : {config.device}")
print_log(f"  Epochs    : {config.epochs}")
print_log(f"  Batch size: {config.batch_size}")
print_log(f"  Adapter r : {getattr(config, 'adapter_ratio', 0.2)}")
print_log("=" * 60)
```

- `"=" * 60` → 60 个等号字符串，做分隔线
- `getattr(config, 'adapter_ratio', 0.2)`：取 `config.adapter_ratio`，取不到就用默认 `0.2`

---

## 4. 加载 CLIP 模型

```python
print_log("\n[1/5] Loading CLIP (ViT-L/14@336px)...")
clip_model, _ = clip.load("ViT-L/14@336px", device=config.device)
clip_model = clip_model.float()
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False
print_log("      CLIP loaded & frozen.")
```

**逐行讲解：**

```python
clip_model, _ = clip.load("ViT-L/14@336px", device=config.device)
```
- `clip.load` 返回两个东西：模型和预处理函数
- `_` 表示"不要这个"（Python 惯例），因为这里预处理我们在 `dataset.py` 里自己定义
- `"ViT-L/14@336px"`：CLIP 最大的视觉模型，336px 输入分辨率，patch 大小 14

```python
clip_model = clip_model.float()
```
- CLIP 默认是 `float16`（半精度），转成 `float32` 避免数值不稳定

```python
clip_model.eval()
```
- 切换到**评估模式**：关掉 Dropout、BatchNorm 用固定统计量
- 我们只用 CLIP 提特征，不训练它

```python
for p in clip_model.parameters():
    p.requires_grad = False
```
- 遍历所有参数，关闭梯度
- **冻结**：这些参数不会被优化器更新

**为什么要冻结？**
- CLIP 是在 4 亿图文对上预训练的，已经足够好
- 如果让它被少量 CUB 数据"微调"，会破坏原有泛化能力（叫**灾难性遗忘**）

---

## 5. 加载数据集

```python
print_log("\n[2/5] Loading CUB Dataset...")
dataloader = CUBDataLoader('.', config.device, is_balance=False)
print_log(f"      Train images : {dataloader.ntrain_clip}")
print_log(f"      Seen classes : {len(dataloader.seenclasses)}  (used for training)")
print_log(f"      Unseen classes: {len(dataloader.unseenclasses)}  (zero-shot target)")
print_log(f"      Test seen    : {len(dataloader.test_seen_loader.dataset)} images")
print_log(f"      Test unseen  : {len(dataloader.test_unseen_loader.dataset)} images")
```

**讲解：**

```python
dataloader = CUBDataLoader('.', config.device, is_balance=False)
```
- `'.'`：数据根路径（当前目录）
- `config.device`：GPU 设备
- `is_balance=False`：**不**做类别均衡采样（纯随机）

**是否类别均衡？**
- `is_balance=True`：每个 batch 尽量每类都有
- `is_balance=False`：完全随机，可能某些类出现多次，某些类不出现
- 这里选 `False`，因为用的是缓存特征，随机采样更简单

**打印了什么？**
- `ntrain_clip`：训练图像总数（约 7000）
- `seenclasses`：seen 类别索引列表（长度 150）
- `unseenclasses`：unseen 类别索引列表（长度 50）
- `test_seen_loader.dataset`：测试集的 seen 类图片（用于算 `S`）
- `test_unseen_loader.dataset`：测试集的 unseen 类图片（用于算 `U` 和 `ZS`）

```python
seed = config.random_seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
```

**固定随机种子：**
- `torch.manual_seed`：PyTorch CPU 随机数
- `torch.cuda.manual_seed_all`：所有 GPU 的随机数
- `np.random.seed`：NumPy 的随机数

**为什么要固定？**  
复现实验。同一个种子每次运行结果一样。

---

## 6. 加载特征缓存

这段是优化训练速度的关键。

```python
HAS_PATCH_CACHE = os.path.exists(CACHE_TRAIN_PATCH) and os.path.exists(CACHE_TRAIN_LABEL)
HAS_CLS_CACHE   = os.path.exists(CACHE_TRAIN_FEAT)  and os.path.exists(CACHE_TRAIN_LABEL)

train_patches  = None
train_features = None
train_labels   = None
```

**讲解：**
- `os.path.exists(path)`：判断文件是否存在
- `HAS_PATCH_CACHE`：有没有新版 patch 缓存
- `HAS_CLS_CACHE`：有没有旧版 CLS 缓存
- 下面三个变量先置 `None`，等下根据情况赋值

```python
if HAS_PATCH_CACHE:
    print_log("\n[★] Loading train PATCH features from cache (fast mode, float16 on CPU)...")
    train_patches = torch.load(CACHE_TRAIN_PATCH, map_location='cpu')    # 留在 CPU
    train_labels  = torch.load(CACHE_TRAIN_LABEL, map_location=config.device)
    print_log(f"      train_patches: {train_patches.shape}  dtype={train_patches.dtype}  device=cpu")
    print_log(f"      train_labels:  {train_labels.shape}")
    USE_CACHE = 'patch'
```

**讲解：**

```python
train_patches = torch.load(CACHE_TRAIN_PATCH, map_location='cpu')
```
- `torch.load`：从磁盘加载 PyTorch 张量
- `map_location='cpu'`：加载到 CPU 内存

**为什么放 CPU 不放 GPU？**  
- `[N=7000, 576, 768] float16` ≈ `7000 × 576 × 768 × 2 字节` ≈ **6 GB**
- 如果放 GPU，显存不够（尤其 12G 卡还要装模型）
- 放 CPU 内存（通常 32G+），训练时再按 batch 搬 GPU

```python
train_labels = torch.load(CACHE_TRAIN_LABEL, map_location=config.device)
```
- 标签只有 `[N]` 很小，直接放 GPU

```python
elif HAS_CLS_CACHE:
    print_log("\n[★] Loading train CLS features from cache (legacy mode)...")
    train_features = torch.load(CACHE_TRAIN_FEAT,  map_location=config.device)
    train_labels   = torch.load(CACHE_TRAIN_LABEL, map_location=config.device)
    print_log(f"      train_features: {train_features.shape}  train_labels: {train_labels.shape}")
    USE_CACHE = 'cls'
```

**旧版 CLS 缓存：**
- `[N, 768]` 只有 CLS token，没有 patch
- 数据量小（几十 MB），直接放 GPU
- 模型训练时会把 CLS 广播成 576 个 patch 用（效果会差一些）

```python
else:
    print_log("\n[!] Cache not found. Will extract features on-the-fly (slow).")
    print_log("    Run: python tools/extract_features.py  to generate cache.")
    USE_CACHE = None
```

**没缓存：**
- 训练时实时提取，**超慢**
- 提示用户运行 `tools/extract_features.py` 先生成缓存

**三种模式对比：**

| 模式 | 训练速度 | 效果 |
|------|----------|------|
| `patch` 缓存 | 最快 | 最好（完整 576 patch） |
| `cls` 缓存 | 中等 | 次之（只有全局特征） |
| 实时 | 最慢 | 最好但慢 10 倍 |

---

## 7. 编码类名文本

```python
print_log("\n[3/5] Encoding class name text features...")
class_names = [c.split('.')[-1].replace("_", " ") for c in dataloader.class_names]
prompts = [f"a photo of a {c}, a type of bird." for c in class_names]
text_inputs = torch.cat([clip.tokenize(p) for p in prompts]).to(config.device)
with torch.no_grad():
    class_text_embeds = clip_model.encode_text(text_inputs).float()  # [200, 768]
print_log(f"      class_text_embeds: {class_text_embeds.shape}")
```

**逐行讲解：**

```python
class_names = [c.split('.')[-1].replace("_", " ") for c in dataloader.class_names]
```

CUB 的类名原始格式是 `"001.Black_footed_Albatross"`，处理成自然语言：
- `.split('.')` → `['001', 'Black_footed_Albatross']`
- `[-1]` → `'Black_footed_Albatross'`（最后一段）
- `.replace("_", " ")` → `'Black footed Albatross'`

```python
prompts = [f"a photo of a {c}, a type of bird." for c in class_names]
```

构造提示词模板（prompt template），比如：
- `"a photo of a Black footed Albatross, a type of bird."`
- `"a photo of a Cardinal, a type of bird."`
- ...

**为什么要这种句式？**  
CLIP 在预训练时见过大量 `"a photo of ..."` 的句子，这种句式效果最好。加 `"a type of bird"` 是 CUB 的特殊提示。

```python
text_inputs = torch.cat([clip.tokenize(p) for p in prompts]).to(config.device)
```

**Tokenize（分词）：**
- `clip.tokenize(p)`：把字符串编码成整数序列 `[1, 77]`（CLIP 固定长度 77）
- 对 200 个类的 prompt 分别 tokenize，得到 200 个 `[1, 77]`
- `torch.cat` 沿默认维度（0 维）拼接 → `[200, 77]`
- `.to(config.device)`：搬到 GPU

```python
with torch.no_grad():
    class_text_embeds = clip_model.encode_text(text_inputs).float()
```

- `torch.no_grad()`：关梯度，节省显存（反正 CLIP 不训练）
- `encode_text`：CLIP 的文本编码器，`[200, 77]` → `[200, 768]`
- `.float()`：转 float32

**结果：** `class_text_embeds` 是 `[200, 768]`，每个类一个文本向量。

---

## 8. 加载 GPT-4 描述

```python
gpt_text_embeds = None
gpt4_data_path = os.path.join('.', 'data', 'gpt4_data', 'cub.pt')

if os.path.exists(gpt4_data_path):
    print_log("\n[4/5] Loading & encoding GPT-4 descriptions...")
    gpt4_sentences = torch.load(gpt4_data_path, map_location='cpu', weights_only=False)
    n_cls = len(gpt4_sentences)
    n_desc = len(list(gpt4_sentences.values())[0])
    print_log(f"      {n_cls} classes × {n_desc} descriptions/class")
```

**讲解：**

`cub.pt` 是一个字典，大概长这样：
```python
{
    'black footed albatross': [
        '一段描述',
        '另一段描述',
        ... 共 7 段
    ],
    'cardinal': [...],
    ...
}
```

- `len(gpt4_sentences)`：类别数（200）
- `list(gpt4_sentences.values())[0]`：取第一个类的描述列表
- `len(...)` = 7（每类 7 段描述）

```python
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
```

**逐步讲解：**

```python
gpt_key = '.'.join(cls_name.split('.')[1:]).lower()
```
- `cls_name`: `"001.Black_footed_Albatross"`
- `.split('.')` → `['001', 'Black_footed_Albatross']`
- `[1:]` → `['Black_footed_Albatross']`（去掉编号）
- `'.'.join(...)` → `'Black_footed_Albatross'`
- `.lower()` → `'black_footed_albatross'`

（注意：代码里有个小 bug，这里应该再 `.replace('_', ' ')` 才和 GPT 字典的 key 完全一致，但这里依赖 GPT 字典的 key 格式。）

```python
if gpt_key in gpt4_sentences:
    sentences = gpt4_sentences[gpt_key]         # 7 段描述
    tokens = torch.cat([clip.tokenize(s) for s in sentences]).to(config.device)  # [7, 77]
    with torch.no_grad():
        feats = clip_model.encode_text(tokens).float()  # [7, 768]
    gpt_text_embeds_list.append(feats.mean(dim=0))      # [768] 对 7 段描述求平均
    hit += 1
```

**关键操作：`feats.mean(dim=0)`** 把一个类的 7 段描述嵌入**平均**成一个向量，让表示更鲁棒。

```python
else:
    idx = dataloader.class_names.index(cls_name)
    gpt_text_embeds_list.append(class_text_embeds[idx])  # 回退到类名 prompt
    miss += 1
```

如果 GPT 字典里找不到这个类（字典不全），就用前面的 `class_text_embeds` 作为兜底。

```python
    gpt_text_embeds = torch.stack(gpt_text_embeds_list)  # [200, 768]
    print_log(f"      GPT hit: {hit} classes | fallback: {miss} classes")
    print_log(f"      gpt_text_embeds: {gpt_text_embeds.shape}")
```

- `torch.stack` 沿新维度堆叠，200 个 `[768]` → `[200, 768]`

```python
else:
    print_log(f"\n[4/5] WARNING: GPT-4 data not found at {gpt4_data_path}, using class name only")
```

如果没 GPT 数据，退化成只用类名 prompt。

---

## 9. 初始化模型

```python
print_log("\n[5/5] Initializing model...")

seen_gpt_embeds = gpt_text_embeds[dataloader.seenclasses] \
    if gpt_text_embeds is not None else class_text_embeds[dataloader.seenclasses]
unseen_clip_embeds = class_text_embeds[dataloader.unseenclasses]

model = VGSR(
    config,
    dataloader.seenclasses,
    dataloader.unseenclasses,
    seen_text_embeds=seen_gpt_embeds,
    unseen_text_embeds=unseen_clip_embeds,
).to(config.device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print_log(f"      Total params    : {total_params:,}")
print_log(f"      Trainable params: {trainable_params:,}")
```

**逐行讲解：**

```python
seen_gpt_embeds = gpt_text_embeds[dataloader.seenclasses] \
    if gpt_text_embeds is not None else class_text_embeds[dataloader.seenclasses]
```

**三元表达式**（条件表达式）：
- 如果有 GPT 数据，用 GPT 描述嵌入
- 否则用类名 prompt 嵌入

反斜杠 `\` 是 Python 的换行续行符。

```python
unseen_clip_embeds = class_text_embeds[dataloader.unseenclasses]
```

**Unseen 类总是用原始 CLIP 类名嵌入**，不用 GPT 描述。

**为什么？**
- GPT 描述可能带偏（训练时模型会适应 GPT 的风格）
- unseen 类要保持"CLIP 原始语义"作为零样本基准
- 这是论文里的一个经验设计

```python
model = VGSR(
    config,
    dataloader.seenclasses,
    dataloader.unseenclasses,
    seen_text_embeds=seen_gpt_embeds,
    unseen_text_embeds=unseen_clip_embeds,
).to(config.device)
```

实例化 VGSR 并搬到 GPU。

```python
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

- `p.numel()`：参数张量的元素总数
- `total_params`：所有参数
- `trainable_params`：只算 `requires_grad=True` 的

**`{total_params:,}`** 里的 `:,` 是格式化符号，每 3 位加逗号（123,456,789）。

---

## 10. 优化器和调度器

```python
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
```

**讲解：**

**Adam 优化器：**
- `lr=0.001`：学习率（步长）
- `weight_decay=1e-4`：权重衰减（正则化项）
  - 相当于在 loss 里加 `0.0001 * ||w||²`
  - 作用：防止参数过大，减轻过拟合

**CosineAnnealingLR 调度器：**
- 学习率按**余弦曲线**从初始值衰减到 0
- `T_max=config.epochs`：总周期等于训练 epoch 数
- 前期快、后期慢，让训练更稳定

**学习率变化示意：**
```
lr
0.001 ─╮
        ╲
         ╲  (余弦曲线)
          ╲
           ╲_____
0      时间         epochs
```

---

## 11. 训练循环

### 11.1 准备

```python
iters_per_epoch = dataloader.ntrain_clip // config.batch_size
total_iters = iters_per_epoch * config.epochs

best_H = 0.0
best_metrics = {'U': 0, 'S': 0, 'H': 0, 'ZS': 0, 'epoch': 0}
```

- `iters_per_epoch`：一个 epoch 多少个 step（步数），例如 `7000 / 64 ≈ 109`
- `//`：整数除法，结果向下取整
- `best_H`、`best_metrics`：记录训练过程中最佳结果

### 11.2 外层循环：epoch

```python
for epoch in range(1, config.epochs + 1):
    model.train()
    epoch_loss = 0.0
    epoch_iters = 0

    print_log(f"\n{'─'*60}")
    print_log(f"  Epoch [{epoch}/{config.epochs}]  Training...")
    print_log(f"{'─'*60}")
```

- `model.train()`：切换到**训练模式**（开 Dropout、BatchNorm 用 mini-batch 统计）
- 每个 epoch 重置累计 loss 和步数

### 11.3 内层循环：step

```python
    for step in range(iters_per_epoch):
        optimizer.zero_grad()
```

- `optimizer.zero_grad()`：清空上一步的梯度
- **必须写**，否则梯度会累加

### 11.4 取 batch 特征（三种模式）

```python
        if USE_CACHE == 'patch':
            idx = torch.randperm(len(train_patches))[:config.batch_size]
            batch_label = train_labels[idx]
            clip_features = train_patches[idx].to(config.device).float()
```

**Patch 缓存模式：**
- `torch.randperm(N)`：生成 `[0, N)` 的随机排列
- `[:batch_size]`：取前 `batch_size` 个 → 随机采样
- `train_patches[idx]`：从 CPU 切 batch，`[batch_size, 576, 768]`
- `.to(device).float()`：搬 GPU 并转 float32

```python
        elif USE_CACHE == 'cls':
            idx = torch.randperm(len(train_features))[:config.batch_size]
            batch_label = train_labels[idx]
            clip_features = train_features[idx].unsqueeze(1)  # [B, 1, 768]
```

**CLS 缓存模式：**
- `[N, 768]` → 取 batch → `[B, 768]`
- `.unsqueeze(1)` → `[B, 1, 768]`（加一个维度）
- VGSR 的 `_prepare_patches` 会把它广播成 `[B, 576, 768]`

```python
        else:
            batch_label, batch_images, batch_att = dataloader.next_batch(config.batch_size)
            clip_features = get_clip_spatial_features(clip_model, batch_images).float()
```

**实时模式：**
- `dataloader.next_batch`：读图、加载、预处理
- `get_clip_spatial_features`：实时用 CLIP 提特征（慢）

### 11.5 前向 + 反向

```python
        out_package = model(clip_features, is_train=True)
        in_package = out_package.copy()
        in_package['batch_label'] = batch_label
        loss_pack = model.compute_loss(in_package)
        loss = loss_pack['loss']

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_iters += 1
```

**逐行讲解：**

```python
out_package = model(clip_features, is_train=True)
```
- 调用 `model.__call__` → 实际调 `forward`
- `is_train=True`：只返回 seen 150 类 logits
- `out_package` = `{'logits': [B, 150], 'clip_S_pp': ..., 'clip_pred': ...}`

```python
in_package = out_package.copy()
in_package['batch_label'] = batch_label
```
- 复制字典（避免改动原字典）
- 加上标签，准备喂给 `compute_loss`

```python
loss_pack = model.compute_loss(in_package)
loss = loss_pack['loss']
```
- 调用 VGSR 的 `compute_loss` 方法
- 得到 `{'loss': ..., 'loss_CE': ...}`
- 取 `loss`（一个标量张量）

```python
loss.backward()
```
- **反向传播**：计算所有参数对 loss 的梯度
- 梯度存在每个参数的 `.grad` 属性里

```python
optimizer.step()
```
- **参数更新**：用 Adam 规则根据 `.grad` 更新参数
- `w = w - lr * 某种梯度变形`

```python
epoch_loss += loss.item()
epoch_iters += 1
```
- `loss.item()`：把标量张量转成 Python 的 float
- 累计方便后面算平均

### 11.6 打印进度

```python
        if (step + 1) % 20 == 0 or (step + 1) == iters_per_epoch:
            avg_loss = epoch_loss / epoch_iters
            print_log(f"  Step [{step+1:3d}/{iters_per_epoch}] | "
                      f"Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f}")
```

- 每 20 步或最后一步打印
- `{step+1:3d}`：整数占 3 位
- `{loss.item():.4f}`：小数保留 4 位

### 11.7 更新学习率

```python
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    avg_epoch_loss = epoch_loss / epoch_iters
```

- `scheduler.step()`：每个 epoch 结束调一次，学习率按余弦曲线衰减
- `optimizer.param_groups[0]['lr']`：取当前学习率（用于打印）

### 11.8 评估

```python
    print_log(f"\n  >> Epoch [{epoch}/{config.epochs}] Evaluating GZSL...")
    acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(
        dataloader, clip_model, model, config.device)
```

- 调用 `eval_zs_gzsl`（后面在 `helper_func.py` 文档里详解）
- 返回 4 个指标：
  - `acc_seen`: GZSL 下 seen 类的精度（S）
  - `acc_novel`: GZSL 下 unseen 类的精度（U）
  - `H`: 调和平均（Harmonic mean）
  - `acc_zs`: 仅在 unseen 类中竞争的精度（ZS，即 ZSL 传统协议）

**调和平均 H 的公式：**
```
H = 2 × S × U / (S + U)
```
只有 S 和 U 都很高，H 才会高。

### 11.9 记录最佳

```python
    if H > best_H:
        best_H = H
        best_metrics = {
            'U': acc_novel,
            'S': acc_seen,
            'H': H,
            'ZS': acc_zs,
            'epoch': epoch
        }
```

- 如果这个 epoch 的 H 比历史最佳高，更新记录

```python
    print_log(f"\n  ┌─ Epoch [{epoch}/{config.epochs}] Results ─────────────────────")
    print_log(f"  │  GZSL-U (Unseen Acc) : {acc_novel*100:.2f}%")
    print_log(f"  │  GZSL-S (Seen Acc)   : {acc_seen*100:.2f}%")
    print_log(f"  │  GZSL-H (Harmonic)   : {H*100:.2f}%  {'★ NEW BEST' if H == best_H else ''}")
    print_log(f"  │  ZSL    (ZS Acc)     : {acc_zs*100:.2f}%")
    print_log(f"  └──────────────────────────────────────────────────────")
```

- 乘 100 转成百分比
- `'★ NEW BEST' if H == best_H else ''`：三元表达式，本轮是最佳就打星

---

## 12. 最终汇总

```python
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
```

打印整个训练的最佳结果和日志文件路径。

---

## 结语

这个训练脚本是个典型的"**配置驱动**"的训练框架：
- 所有超参数在 YAML 配置文件里
- 脚本本身只管"加载 → 训练 → 评估"的流程
- 模型、数据、评估都解耦到其他模块

**核心要点：**
1. CLIP 冻结，只训练 VGSR 内的 Adapter + Transformer
2. 用特征缓存加速训练（6GB patch 缓存在 CPU）
3. 训练时只监督 seen 150 类的 CE loss
4. 每个 epoch 评估一次 GZSL 四指标

下一份文档会讲 `helper_func.py`（评估函数）和 `dataset.py`（数据加载）。
