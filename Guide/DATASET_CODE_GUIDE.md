# tools/dataset.py 完整代码讲解

> 数据加载器文件。定义了 `ImgDataset` 和 3 个数据集专用的 `DataLoader`（CUB/SUN/AWA2）。
> 三个 DataLoader 结构几乎一模一样，所以只详讲 `CUBDataLoader`，另外两个对照看即可。

---

## 目录

- [0. 这个文件在干什么](#0-这个文件在干什么)
- [1. 导入](#1-导入)
- [2. ImgDataset：图片数据集](#2-imgdataset图片数据集)
- [3. CUBDataLoader：CUB 数据加载器](#3-cubdataloadercub-数据加载器)
- [4. read_mat_metadata：读取 .mat 元数据](#4-read_mat_metadata读取-mat-元数据)
- [5. get_idx_classes：类别索引分组](#5-get_idx_classes类别索引分组)
- [6. next_batch：采样一个 batch](#6-next_batch采样一个-batch)
- [7. SUNDataLoader 和 AWA2DataLoader](#7-sundataloader-和-awa2dataloader)
- [8. 数据流总览](#8-数据流总览)

---

## 0. 这个文件在干什么

**核心任务：** 把磁盘上的图片文件和元数据（标签、类别名、属性）包装成训练和评估用的数据加载器。

**提供三层抽象：**

| 层级 | 类名 | 作用 |
|------|------|------|
| 底层 | `ImgDataset(Dataset)` | 单张图片的读取逻辑 |
| 中层 | `DataLoader`（PyTorch） | 批量加载测试集 |
| 顶层 | `CUBDataLoader` / `SUNDataLoader` / `AWA2DataLoader` | 统一封装：元数据 + 训练集采样 + 测试集 DataLoader |

**数据来源（xlsa17 标准协议）：**
- `res101.mat`：包含所有图片路径和对应标签
- `att_splits.mat`：包含训练/测试划分、类别属性、类别名称

---

## 1. 导入

```python
import os, sys, torch, pickle
import numpy as np
import scipy.io as sio
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
```

**逐行讲解：**

| 代码 | 含义 |
|------|------|
| `os, sys` | 路径操作、系统相关（sys 实际没用到） |
| `torch` | PyTorch |
| `pickle` | Python 的序列化工具，读 `.pkl` 文件 |
| `numpy as np` | 数值计算 |
| `scipy.io as sio` | 读取 MATLAB 的 `.mat` 文件 |
| `PIL.Image` | 图像读取（PIL = Python Imaging Library） |
| `torchvision.transforms` | 图像预处理工具（裁剪、归一化等） |
| `Dataset, DataLoader` | PyTorch 数据加载的两个基类 |

**什么是 `.mat` 文件？**  
MATLAB 的数据格式。xlsa17 benchmark（零样本学习的标准数据集）的元数据都用 .mat 存储。`scipy.io.loadmat` 可以读它。

---

## 2. ImgDataset：图片数据集

这是一个继承 `torch.utils.data.Dataset` 的类。PyTorch 的自定义数据集需要实现 3 个方法：
- `__init__`：初始化
- `__len__`：返回数据集大小
- `__getitem__`：按索引取一个样本

### 2.1 类定义和构造函数

```python
class ImgDataset(Dataset):
    def __init__(self, image_files, labels, dataset_name, transform=None, root_dir=None):
        self.image_files = image_files
        self.labels = labels
        self.dataset_name = dataset_name
        self.transform = transform
        self.root_dir = root_dir
```

**参数讲解：**

| 参数 | 类型 | 含义 |
|------|------|------|
| `image_files` | np.ndarray | 图片路径数组，来自 `res101.mat` |
| `labels` | np.ndarray | 标签数组（从 0 开始） |
| `dataset_name` | str | `'CUB'` / `'SUN'` / `'AWA2'`（决定路径处理逻辑） |
| `transform` | torchvision.Compose | 图像预处理管线 |
| `root_dir` | str | 数据集根目录，比如 `./data/CUB` |

### 2.2 `__len__`

```python
    def __len__(self):
        return len(self.image_files)
```

**讲解：** 返回数据集总样本数。PyTorch DataLoader 靠这个决定循环次数。

### 2.3 `__getitem__`：核心

```python
    def __getitem__(self, idx):
        # 1. 获取原始路径字符串
        img_path_raw = self.image_files[idx]
        if isinstance(img_path_raw, np.ndarray):
            img_path_raw = img_path_raw[0]
```

**讲解：**

- `idx`：PyTorch 会用 `0 ~ len(self)-1` 的索引调用这个方法
- `self.image_files[idx]`：取出路径

**为什么要 `if isinstance(img_path_raw, np.ndarray)`？**

`.mat` 文件读出来的字符串常常被套了一层 numpy 数组，比如 `array(['/path/to/img.jpg'], dtype='<U50')`。需要 `[0]` 取出真正的字符串。

### 2.4 路径清洗

```python
        parts = img_path_raw.split('/')
        rel_path = img_path_raw

        if self.dataset_name == 'CUB':
            if 'images' in parts:
                idx_start = parts.index('images')
                rel_path = '/'.join(parts[idx_start:])
            else:
                if len(parts) > 6:
                    rel_path = '/'.join(parts[6:])
        elif self.dataset_name == 'AWA2':
            if 'JPEGImages' in parts:
                idx_start = parts.index('JPEGImages')
                rel_path = '/'.join(parts[idx_start:])
            else:
                if len(parts) > 5:
                    rel_path = '/'.join(parts[5:])
        elif self.dataset_name == 'SUN':
            if 'images' in parts:
                idx_start = parts.index('images')
                rel_path = '/'.join(parts[idx_start:])
            else:
                if len(parts) > 7:
                    rel_path = '/'.join(parts[7:])
```

**讲解：**

`.mat` 里的路径通常是**原作者电脑上的绝对路径**，比如：
```
/home/xianyan/data/CUB/CUB_200_2011/images/001.Black_footed_Albatross/xxx.jpg
```

但你电脑上路径不一样，必须**截取出相对部分**再拼接你本地的 root。

**处理逻辑（分两段）：**

1. **优先用关键字定位**：
   ```python
   if 'images' in parts:
       idx_start = parts.index('images')
       rel_path = '/'.join(parts[idx_start:])
   ```
   - `parts = ['home', 'xianyan', 'data', 'CUB', 'CUB_200_2011', 'images', '001....', 'xxx.jpg']`
   - `parts.index('images')` 找到 `'images'` 的位置（索引 5）
   - `parts[5:]` = `['images', '001.....', 'xxx.jpg']`
   - `'/'.join(...)` = `'images/001..../xxx.jpg'`（相对路径）

2. **兜底：按固定位置截取**：
   ```python
   if len(parts) > 6:
       rel_path = '/'.join(parts[6:])
   ```
   - 如果路径中没 `'images'` 关键字，退化成"扔掉前 6 段"
   - 这是 `preprocessing.py` 里的老逻辑

**不同数据集的关键字：**
- CUB: `images`
- AWA2: `JPEGImages`
- SUN: `images`

### 2.5 拼接完整路径 + 加载图片

```python
        full_path = os.path.join(self.root_dir, rel_path)

        try:
            image = Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load {full_path}, using black image.")
            image = Image.new('RGB', (336, 336))

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label
```

**讲解：**

```python
full_path = os.path.join(self.root_dir, rel_path)
```
- 用 `os.path.join` 拼接路径（自动处理斜杠）
- 比如 `./data/CUB` + `images/001..../xxx.jpg` → `./data/CUB/images/001..../xxx.jpg`

```python
try:
    image = Image.open(full_path).convert('RGB')
except Exception as e:
    print(f"Warning: Could not load {full_path}, using black image.")
    image = Image.new('RGB', (336, 336))
```

**容错加载：**
- `Image.open`：打开图片
- `.convert('RGB')`：转成 RGB（防止灰度图或 RGBA）
- 如果加载失败（文件损坏或路径错），用**黑色图**代替，避免训练中断
- 注意：应该关注这个 warning，否则数据有问题而不自知

```python
if self.transform:
    image = self.transform(image)
```
- 应用预处理（后面详说）

```python
label = self.labels[idx]
return image, label
```
- 返回 `(图像张量, 标签)` 元组
- PyTorch DataLoader 会自动把多个样本 stack 成 batch

---

## 3. CUBDataLoader：CUB 数据加载器

### 3.1 构造函数（上半部分）

```python
class CUBDataLoader():
    def __init__(self, data_path, device, is_scale=False, is_unsupervised_attr=False, is_balance=True):
        self.data_path = data_path
        self.device = device
        self.dataset = 'CUB'
        self.root_dir = os.path.join(self.data_path, 'data', self.dataset)
        self.is_balance = is_balance
        self.is_unsupervised_attr = is_unsupervised_attr

        print(f"Loading metadata from {self.root_dir}...")
```

**参数讲解：**

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `data_path` | `'.'` | 项目根路径 |
| `device` | `'cuda:0'` | GPU 设备 |
| `is_scale` | False | 是否缩放特征（未用到） |
| `is_unsupervised_attr` | False | 无监督属性标志（未用到） |
| `is_balance` | True | 是否做类别均衡采样 |

**`self.root_dir`**：`./data/CUB`，存图片的根目录。

注意 `class CUBDataLoader():` 后面的空括号是 Python 2 的写法，在 Python 3 里等价于 `class CUBDataLoader:`。

### 3.2 定义图像变换

```python
        # 1. 定义训练集增强 (Data Augmentation)
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(336, scale=(0.6, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        # 2. 定义测试集预处理
        self.test_transform = transforms.Compose([
            transforms.Resize((336, 336), interpolation=Image.BICUBIC),
            transforms.CenterCrop(336),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
```

**讲解：**

`transforms.Compose([...])`：把多个变换串起来，按顺序执行。

**训练集变换（`train_transform`）：**

```python
transforms.RandomResizedCrop(336, scale=(0.6, 1.0), interpolation=Image.BICUBIC)
```
- 随机裁剪一块（面积占原图 60%~100%），然后 resize 到 `336×336`
- **数据增强**：每次裁的位置不同，让模型看到更多视角
- `BICUBIC`：双三次插值（画质最好）

```python
transforms.RandomHorizontalFlip()
```
- 50% 概率水平翻转（左右镜像）
- 也是数据增强

```python
transforms.ToTensor()
```
- PIL Image → PyTorch Tensor
- 同时：`[0, 255]` 整数 → `[0, 1]` 浮点
- 维度：`[H, W, C]` → `[C, H, W]`

```python
transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073),     # mean (R, G, B)
    (0.26862954, 0.26130258, 0.27577711)     # std  (R, G, B)
)
```
- 用 CLIP 预训练时的统计量归一化
- `x = (x - mean) / std`
- **必须用这套数字**，因为 CLIP 就是在这个分布上训练的

**测试集变换（`test_transform`）：**

```python
transforms.Resize((336, 336), interpolation=Image.BICUBIC)
transforms.CenterCrop(336)
```
- Resize 到 `336×336` 后中心裁 `336×336`
- 第二步有点冗余（Resize 已经是 336 了），但保险

**为什么测试集不做增强？**
- 测试时要稳定的结果，不能随机
- 用 `Resize + CenterCrop` 固定处理

### 3.3 加载元数据 + 构建 Dataset

```python
        self.read_mat_metadata()

        self.train_dataset = ImgDataset(self.image_files[self.trainval_loc],
                                        self.labels[self.trainval_loc],
                                        'CUB', self.train_transform, self.root_dir)
        self.test_seen_dataset = ImgDataset(self.image_files[self.test_seen_loc],
                                            self.labels[self.test_seen_loc],
                                            'CUB', self.test_transform, self.root_dir)
        self.test_unseen_dataset = ImgDataset(self.image_files[self.test_unseen_loc],
                                              self.labels[self.test_unseen_loc],
                                              'CUB', self.test_transform, self.root_dir)
```

**讲解：**

`self.read_mat_metadata()`：读 `.mat` 文件（下面详解）。

构建 3 个 Dataset：
- `train_dataset`：训练集图片（用 `train_transform` 含增强）
- `test_seen_dataset`：测试集中 seen 类的图片（`test_transform` 无增强）
- `test_unseen_dataset`：测试集中 unseen 类的图片（`test_transform` 无增强）

**`self.image_files[self.trainval_loc]`**：
- `self.image_files` 是所有图片路径的数组
- `self.trainval_loc` 是训练+验证集的**索引数组**（从 `.mat` 里来）
- NumPy 高级索引：取出这些位置的路径

### 3.4 统计和辅助

```python
        self.ntrain_clip = len(self.train_dataset)
        self.ntrain_class = len(self.seenclasses)
        self.ntest_class = len(self.unseenclasses)
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        self.get_idx_classes()

        self.test_seen_loader = DataLoader(self.test_seen_dataset, batch_size=50, shuffle=False, num_workers=0)
        self.test_unseen_loader = DataLoader(self.test_unseen_dataset, batch_size=50, shuffle=False, num_workers=0)
```

**讲解：**

```python
self.ntrain_clip = len(self.train_dataset)    # 训练图片数（约 7000）
self.ntrain_class = len(self.seenclasses)     # seen 类数（150）
self.ntest_class = len(self.unseenclasses)    # unseen 类数（50）
self.allclasses = torch.arange(0, 200).long() # 全类别索引 [0, 1, ..., 199]
```

```python
self.get_idx_classes()
```
- 按类别分组训练集索引（下面会讲）

```python
self.test_seen_loader = DataLoader(self.test_seen_dataset, batch_size=50, shuffle=False, num_workers=0)
self.test_unseen_loader = DataLoader(self.test_unseen_dataset, batch_size=50, shuffle=False, num_workers=0)
```

**构建测试集的 DataLoader：**
- `batch_size=50`：每批 50 张
- `shuffle=False`：测试时不打乱
- `num_workers=0`：**Windows 上必须是 0**，多进程会卡死

**为什么 Windows 不能用多进程？**  
PyTorch 的 `num_workers>0` 依赖 `fork` 系统调用，Windows 没有 fork。虽然能用 spawn 模拟，但慢且容易出 bug。

**为什么训练集没 DataLoader？**  
因为训练集用自定义的 `next_batch()` 来采样（可以类别均衡），下面详讲。

---

## 4. read_mat_metadata：读取 .mat 元数据

```python
    def read_mat_metadata(self):
        mat_path = os.path.join(self.data_path, 'data/xlsa17/data', self.dataset, 'res101.mat')
        split_path = os.path.join(self.data_path, 'data/xlsa17/data', self.dataset, 'att_splits.mat')
```

**讲解：**

两个 .mat 文件：
- `res101.mat`：图片路径 + 所有图片的标签
- `att_splits.mat`：数据集划分 + 属性 + 类名

### 4.1 读图片路径和标签

```python
        res101 = sio.loadmat(mat_path)
        self.image_files = np.squeeze(res101['image_files'])
        self.labels = res101['labels'].astype(int).squeeze() - 1
```

**讲解：**

```python
res101 = sio.loadmat(mat_path)
```
- 返回一个字典：`{'image_files': ..., 'labels': ...}`

```python
self.image_files = np.squeeze(res101['image_files'])
```
- `np.squeeze`：去掉所有长度为 1 的维度
- 比如 `[1, 11788, 1]` → `[11788]`

```python
self.labels = res101['labels'].astype(int).squeeze() - 1
```
- `.astype(int)`：float → int
- `.squeeze()`：去掉长度 1 的维度
- `- 1`：MATLAB 标签从 1 开始，Python 从 0 开始，所以减 1

### 4.2 读数据集划分

```python
        att_splits = sio.loadmat(split_path)
        self.trainval_loc = att_splits['trainval_loc'].squeeze() - 1
        self.test_seen_loc = att_splits['test_seen_loc'].squeeze() - 1
        self.test_unseen_loc = att_splits['test_unseen_loc'].squeeze() - 1
```

**讲解：**

三个划分（都是**索引数组**，指向 `image_files`）：
- `trainval_loc`：训练+验证集
- `test_seen_loc`：测试集中来自 seen 类的图片
- `test_unseen_loc`：测试集中来自 unseen 类的图片

**都要减 1**（MATLAB 从 1 开始）。

### 4.3 取出 seen/unseen 类别

```python
        self.seenclasses = torch.from_numpy(np.unique(self.labels[self.trainval_loc])).to(self.device)
        self.unseenclasses = torch.from_numpy(np.unique(self.labels[self.test_unseen_loc])).to(self.device)
```

**讲解：**

- `self.labels[self.trainval_loc]`：训练集所有样本的标签
- `np.unique(...)`：去重，得到 seen 类的全局索引
- `torch.from_numpy`：NumPy → PyTorch 张量
- `.to(self.device)`：搬 GPU

**结果：**
- `seenclasses`：长度 150 的张量（CUB）
- `unseenclasses`：长度 50 的张量

### 4.4 读属性

```python
        att = att_splits['att'].T
        self.att = torch.from_numpy(att).float().to(self.device)
```

**讲解：**

- `att_splits['att']`：`.mat` 里的属性矩阵
- `.T`：转置（MATLAB 存的是列主序，转成行主序）
- `.float()`：float32
- 形状：`[200, 312]`（CUB 有 312 维属性）

**什么是"属性"？**  
CUB 每个类都有 312 个人工标注的属性（比如"has_wing_color_red"），评分 0~100。这在传统 GZSL 方法里是关键输入，但在你的 VGSR 模型里用的是 CLIP 文本+GPT 描述，属性没用到。

### 4.5 读类名

```python
        if 'allclasses_names' in att_splits:
            self.class_names = [str(c[0]) for c in att_splits['allclasses_names'].squeeze()]
            print(f"Successfully loaded {len(self.class_names)} class names from att_splits.mat")
        else:
            raise ValueError(f"'allclasses_names' key not found in {split_path}!")
```

**讲解：**

```python
att_splits['allclasses_names'].squeeze()
```
- `.mat` 里类名是嵌套的数组，比如 `[[array(['001.Black_footed_Albatross'])], [array([...])], ...]`
- `.squeeze()` 展平成一维

```python
self.class_names = [str(c[0]) for c in ...]
```
- 列表推导式
- 每个 `c` 还是个数组，`c[0]` 取字符串
- `str(...)` 确保是 Python 字符串

**结果：**
```python
self.class_names = [
    '001.Black_footed_Albatross',
    '002.Laysan_Albatross',
    ...
]
```

### 4.6 读 CLIP 属性向量

```python
        clip_att_path = os.path.join(self.data_path, 'data/clip_att', f'{self.dataset}_attribute.pkl')
        if os.path.exists(clip_att_path):
            with open(clip_att_path, 'rb') as f:
                self.clip_att = pickle.load(f)
            if not isinstance(self.clip_att, torch.Tensor):
                self.clip_att = torch.from_numpy(self.clip_att)
            self.clip_att = self.clip_att.float().to(self.device)
        else:
            print(f"ERROR: CLIP attribute file not found at {clip_att_path}")
            raise FileNotFoundError(f"Missing {clip_att_path}")
```

**讲解：**

- 读取 `data/clip_att/CUB_attribute.pkl`
- 这是**预先用 CLIP 文本编码器算好的属性向量**（"has red wing" 之类的描述 → CLIP 文本向量）
- 当前 VGSR 模型**没用到**它（但保留读取，可能未来会用）

**`'rb'`** 是"read binary"（二进制读）。pkl 是二进制格式。

**`pickle.load(f)`** 从文件反序列化 Python 对象。

---

## 5. get_idx_classes：类别索引分组

```python
    def get_idx_classes(self):
        self.idxs_list = []
        train_labels = self.labels[self.trainval_loc]
        for i in range(self.ntrain_class):
            label_c = self.seenclasses[i].item()
            idx_c = np.where(train_labels == label_c)[0]
            self.idxs_list.append(idx_c)
```

**讲解：**

构建**按类别分组的索引列表**，用于类别均衡采样。

**逐步：**

```python
train_labels = self.labels[self.trainval_loc]
```
- 训练集所有样本的标签（长度 ≈ 7000）

```python
for i in range(self.ntrain_class):    # i = 0, 1, ..., 149
    label_c = self.seenclasses[i].item()
    idx_c = np.where(train_labels == label_c)[0]
    self.idxs_list.append(idx_c)
```

- `self.seenclasses[i].item()`：第 i 个 seen 类的**全局索引**（比如 5）
- `np.where(train_labels == label_c)`：找出训练集里所有标签等于 5 的位置
- 返回 `(indices,)` 元组，`[0]` 取出数组
- 这些索引是**训练集内部的索引**（`trainval_loc` 里的相对位置）

**结果：**
```python
self.idxs_list = [
    array([3, 17, 29, ...]),    # 类 0 的所有样本在训练集中的位置
    array([4, 12, 88, ...]),    # 类 1 的所有样本
    ...
    array([...]),               # 类 149 的所有样本
]
```

**干啥用？** 下面 `next_batch` 做均衡采样时用。

---

## 6. next_batch：采样一个 batch

```python
    def next_batch(self, batch_size):
        if self.is_balance:
            batch_idxs_local = []
            n_samples_class = max(batch_size // self.ntrain_class, 1)
            sampled_classes = np.random.choice(np.arange(self.ntrain_class),
                                               min(self.ntrain_class, batch_size), replace=False)
            for i_c in sampled_classes:
                idxs = self.idxs_list[i_c]
                sampled_idx = np.random.choice(idxs, n_samples_class)
                batch_idxs_local.append(sampled_idx)
            batch_idxs_local = np.concatenate(batch_idxs_local)
        else:
            batch_idxs_local = np.random.choice(self.ntrain_clip, batch_size, replace=False)

        batch_images = []
        batch_labels = []
        for idx in batch_idxs_local:
            img, label = self.train_dataset[idx]
            batch_images.append(img)
            batch_labels.append(label)

        batch_images = torch.stack(batch_images).to(self.device)
        batch_labels = torch.tensor(batch_labels).long().to(self.device)
        batch_att = self.att[batch_labels]
        return batch_labels, batch_images, batch_att
```

### 6.1 均衡采样分支

```python
if self.is_balance:
    batch_idxs_local = []
    n_samples_class = max(batch_size // self.ntrain_class, 1)
```

- `n_samples_class`：每类取几个样本
- 例如 `batch_size=64`，`ntrain_class=150`：`64 // 150 = 0`，`max(0, 1) = 1`
- 所以每类 1 个，但总数变成 150 不等于 64（所以下面才有 `min(batch_size, ntrain_class)`）

```python
    sampled_classes = np.random.choice(
        np.arange(self.ntrain_class),
        min(self.ntrain_class, batch_size),
        replace=False
    )
```

- `np.random.choice(a, size, replace=False)`：从 `a` 中**不放回**抽 `size` 个
- 抽出 `min(150, 64) = 64` 个不重复的类

```python
    for i_c in sampled_classes:
        idxs = self.idxs_list[i_c]
        sampled_idx = np.random.choice(idxs, n_samples_class)
        batch_idxs_local.append(sampled_idx)
```

- 每个抽到的类，再从中随机取 `n_samples_class` 张图
- 默认 `replace=True`（可重复），万一某类样本少也能采出

```python
    batch_idxs_local = np.concatenate(batch_idxs_local)
```

- 拼接：`[array([3]), array([17]), ...]` → `array([3, 17, ...])`

**均衡采样的效果：** batch 里每类都出现过（至少每类 1 张）。

### 6.2 随机采样分支

```python
else:
    batch_idxs_local = np.random.choice(self.ntrain_clip, batch_size, replace=False)
```

- 不放回随机抽 `batch_size` 个索引
- 更快，但类别可能不均衡

### 6.3 按索引加载图片

```python
batch_images = []
batch_labels = []
for idx in batch_idxs_local:
    img, label = self.train_dataset[idx]
    batch_images.append(img)
    batch_labels.append(label)
```

- 调用 `ImgDataset.__getitem__`（实际用 `dataset[idx]` 语法）
- 每张图都走一次 transform（包括随机增强）

### 6.4 组装 batch

```python
batch_images = torch.stack(batch_images).to(self.device)
batch_labels = torch.tensor(batch_labels).long().to(self.device)
batch_att = self.att[batch_labels]
return batch_labels, batch_images, batch_att
```

**讲解：**

```python
torch.stack(batch_images)
```
- 把列表里的 N 个 `[3, 336, 336]` 张量堆叠成 `[N, 3, 336, 336]`

```python
batch_att = self.att[batch_labels]
```
- 根据标签索引取对应的属性向量
- `self.att`: `[200, 312]`，`batch_labels`: `[N]` → `batch_att`: `[N, 312]`

**返回 3 个东西：**
- `batch_labels`：`[N]` 全局标签
- `batch_images`：`[N, 3, 336, 336]` 预处理后的图像
- `batch_att`：`[N, 312]` 属性向量（VGSR 当前不用）

---

## 7. SUNDataLoader 和 AWA2DataLoader

这两个和 `CUBDataLoader` **结构完全一样**，只有细节不同：

### 7.1 主要差异

| 差异点 | CUB | SUN | AWA2 |
|--------|-----|-----|------|
| 数据集名 | `'CUB'` | `'SUN'` | `'AWA2'` |
| 图片目录关键字 | `images` | `images` | `JPEGImages` |
| 类别数 | 200 | 717 | 50 |
| seen/unseen | 150/50 | 645/72 | 40/10 |
| 属性维度 | 312 | 102 | 85 |
| 测试 batch_size | 50 | 64 | 128 |
| train_transform 做增强 | 是 | 否（和测试一样） | 否 |

### 7.2 SUN 的关键差异

SUN 的 `transform` 和 `test_transform` 是**一样的**（都没做随机增强）：

```python
self.transform = transforms.Compose([
    transforms.Resize((336, 336), interpolation=Image.BICUBIC),
    transforms.CenterCrop(336),
    transforms.ToTensor(),
    transforms.Normalize(...)
])
```

可能是因为 SUN 的类别很多（717 类），每类样本少，做增强反而不稳定。

### 7.3 AWA2 的关键差异

- 图片根目录用 `JPEGImages`（AWA2 官方数据集的约定）
- 测试 batch_size 大（128），因为图片少（50 类 × 每类几百张 ≈ 37K 张，但 unseen 测试约 7K 张）

### 7.4 其他的都是 copy-paste

`read_mat_metadata`、`get_idx_classes`、`next_batch` 和 CUB 版本几乎一模一样。唯一的区别就是读不同数据集的 .mat 文件。

---

## 8. 数据流总览

```
磁盘文件:
    ├─ data/xlsa17/data/CUB/res101.mat        (图片路径 + 标签)
    ├─ data/xlsa17/data/CUB/att_splits.mat    (划分 + 属性 + 类名)
    ├─ data/clip_att/CUB_attribute.pkl        (CLIP 属性向量, 未用)
    └─ data/CUB/images/001.Black_footed_Albatross/*.jpg   (图片)
           ↓
CUBDataLoader(...) 初始化时:
    1. read_mat_metadata()
        ├─ self.image_files: [11788] 所有图片路径
        ├─ self.labels:      [11788] 所有标签
        ├─ self.trainval_loc, test_seen_loc, test_unseen_loc: 划分索引
        ├─ self.seenclasses:   [150] seen 类全局索引
        ├─ self.unseenclasses: [50]  unseen 类全局索引
        ├─ self.att:         [200, 312] 属性矩阵
        └─ self.class_names: ['001.Black_footed_Albatross', ...] 200 个类名
    
    2. 构建 3 个 ImgDataset (train / test_seen / test_unseen)
    3. 构建 2 个 DataLoader (test_seen_loader / test_unseen_loader)
    4. get_idx_classes() 构建 idxs_list

↓
训练时:
    每个 step 调 next_batch(batch_size):
        (均衡/随机采样) → 按索引调 ImgDataset.__getitem__ →
        ├─ 路径清洗 (去掉原作者的绝对路径部分)
        ├─ PIL 加载图片
        └─ transform (训练: 随机裁剪+翻转+归一化 / 测试: 固定 resize+归一化)
        ↓ 
        返回 (batch_labels, batch_images, batch_att)

↓
评估时:
    for batch in test_seen_loader 或 test_unseen_loader:
        - DataLoader 自动按 batch_size 打包
        - 走 test_transform (无增强)
        - 返回 (images, labels)
```

---

## 结语

`dataset.py` 的核心是：

1. **`ImgDataset`**：把 `.mat` 里的路径转成本地路径，加载图片并应用 transform。
   - 关键：处理原作者的绝对路径 → 本地相对路径

2. **`CUBDataLoader`**（及兄弟类）：统一封装
   - 元数据（标签、类名、属性）
   - 训练集采样（`next_batch`，支持类别均衡）
   - 测试集 DataLoader（PyTorch 标准方式）

**重要约定：**
- Windows 下 `num_workers=0`（多进程会卡死）
- 训练集用自定义 `next_batch`（方便均衡采样）
- 测试集用 PyTorch DataLoader（固定遍历）
- 归一化常数用 CLIP 的统计量（必须）

**三个 DataLoader 差异小：**
- 数据集名、图片目录关键字、类别数、测试 batch_size

理解了 CUBDataLoader，SUN 和 AWA2 照葫芦画瓢即可。
