# MyModel.py 原子级教程（纯小白版）

> 本教程把 `model/MyModel.py` 从头到尾**逐行**讲清楚，**每一句 Python 语法都解释**，遇到 PyTorch 概念就解释 PyTorch。读完这份文档应该能独立看懂整个文件。

---

## 目录

- [0. 你需要知道的预备知识](#0-你需要知道的预备知识)
- [1. 文件开头：注释 + import](#1-文件开头注释--import)
- [2. 模块一：Adapter](#2-模块一adapter)
- [3. 模块二之一：BoxRelationalEmbedding](#3-模块二之一boxrelationalembedding)
- [4. 模块二之二：GeometryMultiHeadAttention](#4-模块二之二geometrymultiheadattention)
- [5. 模块二之三：FAELayer](#5-模块二之三faelayer)
- [6. 模块二之四：CrossModalTransformer](#6-模块二之四crossmodaltransformer)
- [7. 模块三：VGSR 主模型](#7-模块三vgsr-主模型)

---

## 0. 你需要知道的预备知识

### 0.1 Python 基础

| 写法 | 含义 |
|---|---|
| `x = 5` | 把 5 这个值绑定到名字 `x` |
| `def f(a, b):` | 定义一个叫 f 的函数, 接收两个参数 a 和 b |
| `class A:` | 定义一个叫 A 的类 |
| `class A(B):` | 定义类 A，**继承自** B（A 拥有 B 的所有功能再加自己的） |
| `self.x = 5` | 把 5 存进当前对象的 x 字段 |
| `[a for a in lst]` | 列表推导式：从 lst 里每个 a 出发，生成新列表 |
| `if a > 0:` | 条件判断 |
| `return v` | 函数返回 v |
| `# 注释` | 这一行 # 后面都是注释，不执行 |
| `"""docstring"""` | 三引号字符串，常用于类/函数开头做文档 |

### 0.2 PyTorch 基础（pytroch 用 `torch` 导入）

| 概念 | 含义 |
|---|---|
| `torch.Tensor` | 多维数组（类似 numpy 数组），但能放在 GPU 上、能算梯度 |
| `tensor.shape` | 张量的形状，比如 `[64, 768]` 表示 64 行 768 列 |
| `tensor.dim()` | 张量的维数，`[64, 768]` 是 2 维 |
| `tensor.size(i)` | 第 i 维的大小，`tensor.size(0)` 就是 64 |
| `tensor.float()` | 把元素类型转成 float32 |
| `tensor.half()` | 转成 float16（占一半显存） |
| `tensor.view(...)` | 改变形状但不复制数据（要求内存连续） |
| `tensor.reshape(...)` | 改变形状（必要时复制） |
| `tensor.permute(0,2,1)` | 重新排列维度顺序 |
| `tensor.unsqueeze(0)` | 在第 0 维插入一个长度为 1 的新维度 |
| `tensor.expand(...)` | 把长度为 1 的维度"广播"成指定大小（不复制内存） |
| `nn.Module` | PyTorch 所有神经网络模块的基类 |
| `nn.Linear(in, out)` | 全连接层 y = x · W + b |
| `nn.ReLU()` | 激活函数 max(0, x) |
| `nn.LayerNorm(d)` | 在最后一维做归一化 |
| `nn.Dropout(p)` | 训练时以概率 p 随机置零 |
| `nn.Sequential(a, b, c)` | 把 a, b, c 串起来按顺序执行 |
| `nn.Parameter(t)` | 把张量 t 注册为"可学习参数"（会被优化器更新） |
| `self.register_buffer('name', t)` | 把张量 t 注册为"模型状态但不学习"（会跟模型上 GPU、保存权重，但不被优化） |
| `F.cross_entropy` | 交叉熵损失（自带 softmax） |
| `F.softmax(x, dim=-1)` | 把最后一维变成概率分布（和为 1） |
| `F.normalize(x, dim=-1)` | 把最后一维变成单位向量（长度 1） |

### 0.3 形状记号约定

整个文件里我会写：
- `[B, 576, 768]` 表示一个 3 维张量，三个维度大小分别是 B（batch 数）、576、768
- B = batch_size（一批有多少张图，比如 64）
- 768 是 CLIP 的特征维度
- 576 = 24×24，是 CLIP 把 336×336 图像切成的 patch 数量

### 0.4 这个文件做的事（一句话）

把图像（已经被 CLIP 编码成 patch 向量）和文本（类别名）通过 Adapter + 双向 Transformer 融合，输出每个类别的得分（logits），用于零样本图像分类。

---

## 1. 文件开头：注释 + import

```python
"""
VGSR 主模型 (CLIP + Adapter + GPT + 双向 Transformer)
=========================================================
模块化说明：本文件内包含三个独立模块...
"""
```

### 逐行讲解

**`"""`** 是三引号字符串，三引号之间所有内容都是字符串。Python 里**写在文件开头**或**类/函数开头第一句**的字符串叫 **docstring**（文档字符串），是给读代码的人看的说明，**不影响程序运行**。

这段 docstring 在告诉读者：
- 这个文件实现了 VGSR 模型
- 文件里有 3 个模块（Adapter / CrossModalTransformer / VGSR）
- 模型的接口契约：`forward(clip_features, is_train=False)` 返回字典且必含 `'clip_S_pp'`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
```

### 逐行讲解

| 行 | 含义 |
|---|---|
| `import torch` | 导入 PyTorch 主模块。后面就能用 `torch.xxx` 调用功能 |
| `import torch.nn as nn` | 导入"神经网络层"子模块，并起别名 `nn`。后面用 `nn.Linear` 这种短写法 |
| `import torch.nn.functional as F` | 导入"函数式接口"子模块，别名 `F`。后面用 `F.relu` 这种 |
| `import numpy as np` | 导入 numpy 数组库，别名 `np`。这个文件主要是为了后面用 `np.log` |

**`nn` vs `F` 的区别**：
- `nn.ReLU()` 是一个**类**，要先实例化（`relu = nn.ReLU()`）再调用（`relu(x)`），用于 `nn.Sequential` 里
- `F.relu(x)` 是一个**函数**，直接调用一次就行，用于 `forward` 里临时算

---

## 2. 模块一：Adapter

### 2.1 类定义

```python
class Adapter(nn.Module):
```

### 逐行讲解

- **`class Adapter`**：定义一个名字叫 Adapter 的类（一个"类型"）
- **`(nn.Module)`**：括号里写父类。这个类继承自 `nn.Module`，意思是：**"我是一个神经网络模块"**，这样就拥有了 PyTorch 帮我们准备好的全部能力（参数管理、转 GPU、保存权重等）
- **`:`**：冒号表示后面是这个类的内容（缩进的部分）

### 2.2 docstring（类说明）

```python
    """
    Bottleneck 结构的轻量 Adapter
    结构: Linear(d → d/r) → ReLU → Linear(d/r → d)
    作用: 对 seen 类文本特征做任务相关的语义微调
    """
```

类内第一句字符串就是这个类的 docstring。说明了：
- **结构**：是一个"瓶颈"形状的小网络。先把维度从 d 压缩到 d/r（变小），过 ReLU 激活，再扩回 d。
- **r 是缩减比**（reduction ratio）。比如 d=768, r=4，那中间维度就是 192。
- **作用**：对文本特征做"轻量增强"，让冻结的 CLIP 文本特征更适合下游任务。

### 2.3 `__init__` 方法

```python
    def __init__(self, c_in, reduction=4):
        super().__init__()
```

#### 逐行讲解

- **`def __init__`**：定义构造方法。**Python 创建对象时会自动调用这个方法**。比如写 `a = Adapter(768)`，就等于调用 `Adapter.__init__(a, 768)`。
- **`self`**：第一个参数永远是 `self`，代表"刚创建出来的这个对象自身"。
- **`c_in`**：输入维度（比如 768）
- **`reduction=4`**：默认参数。如果调用时不传，默认是 4。
- **`super().__init__()`**：**调用父类的构造方法**。`super()` = "我的父类"（这里就是 `nn.Module`）。**继承 `nn.Module` 必须写这句**，否则 PyTorch 没法初始化它的内部状态。

```python
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
        )
```

#### 逐行讲解

- **`self.fc = ...`**：把右边的对象**绑定**到当前对象的 `fc` 字段。后面 `forward` 里就能用 `self.fc(x)`。
- **`nn.Sequential(a, b, c)`**：把多个层串起来。输入会依次经过 a → b → c。
- **`nn.Linear(c_in, c_in // reduction, bias=False)`**：一个全连接层
  - 输入维度 `c_in`（比如 768）
  - 输出维度 `c_in // reduction` （`//` 是**整除**，768 // 4 = 192）
  - `bias=False` 表示不要偏置项（只有矩阵乘法 y = xW，没有 +b）
- **`nn.ReLU(inplace=True)`**：ReLU 激活函数。`inplace=True` 表示直接在原张量上改（省一点显存）。
- **`nn.Linear(c_in // reduction, c_in, bias=False)`**：再一个全连接，把 192 维扩回 768 维。

**整体效果**：768 → 192 → ReLU → 768，参数量 = 768×192×2 ≈ 295K。

### 2.4 `forward` 方法

```python
    def forward(self, x):
        return self.fc(x)
```

#### 逐行讲解

- **`def forward(self, x)`**：定义前向传播方法。**`forward` 是 PyTorch 约定的方法名**。
- **`return self.fc(x)`**：把 x 喂进 self.fc（那个 Sequential），返回结果。

**为什么用 `self.fc(x)` 而不是 `self.fc.forward(x)`？**
PyTorch 给 `nn.Module` 重载了"括号调用"，写 `self.fc(x)` 等价于 `self.fc.__call__(x)`，里面会做一些 hook 处理然后调用 `forward`。**永远写括号调用，不要直接调 `forward`**。

### 2.5 整体怎么用

```python
adapter = Adapter(768, reduction=4)   # 创建一个 Adapter
text_feat = torch.randn(150, 768)     # 假装是 150 个类的文本特征
output = adapter(text_feat)           # 输出形状 [150, 768]
```


---

## 3. 模块二之一：BoxRelationalEmbedding

这个类**预计算**一张 `[576, 576, 64]` 的"位置关系表"，给后面的注意力用。

### 3.1 类定义

```python
class BoxRelationalEmbedding(nn.Module):
```

继承自 `nn.Module`，含义同前。这次它没有可学习参数，但用了 PyTorch 的"buffer"机制管理一个常量张量。

### 3.2 `__init__`

```python
    def __init__(self, grid_size=(24, 24), dim_g=64, wave_len=1000.0):
        super().__init__()
        self.grid_size = grid_size
        self.dim_g = dim_g
        self.wave_len = wave_len
```

#### 逐行讲解

- **`grid_size=(24, 24)`**：默认参数是一个**元组**（用 `(...)` 包起来），表示 patch 网格大小是 24 行 × 24 列。
- **`dim_g=64`**：位置编码的维度，默认 64。
- **`wave_len=1000.0`**：正弦编码的"波长基数"，控制频率分布。
- **`super().__init__()`**：调父类构造，必写。
- **`self.grid_size = grid_size`**：把传进来的参数存到对象里，后面可能要用。

```python
        self.register_buffer('geometry_embedding', self._compute_embedding())
```

#### 逐行讲解

- **`self.register_buffer('name', tensor)`**：把张量**注册为 buffer**。
- buffer 和 Parameter 的区别：

| | `nn.Parameter` | `register_buffer` |
|---|---|---|
| 会被优化器更新吗 | ✅ 是 | ❌ 否（保持不变） |
| 会跟 `.to(cuda)` 上 GPU 吗 | ✅ 是 | ✅ 是 |
| 会被 `torch.save` 保存吗 | ✅ 是 | ✅ 是 |
| 用在哪 | 可学习权重 | 固定常量（PE、统计量等） |

- **`'geometry_embedding'`**：这个 buffer 的名字（字符串）。后面就能用 `self.geometry_embedding` 访问它。
- **`self._compute_embedding()`**：调用下面这个私有方法生成实际张量。**前面的下划线** `_` 表示"内部用，不希望外部调"（Python 约定，不强制）。

### 3.3 `_compute_embedding` 方法（核心计算）

```python
    def _compute_embedding(self):
        H, W = self.grid_size                  # (24, 24)
        seq_len = H * W                        # 576
```

#### 逐行讲解

- **`H, W = self.grid_size`**：**元组解包**。把 `(24, 24)` 拆成 `H = 24, W = 24`。
- **`seq_len = H * W`**：算出一共多少个 patch，24×24=576。

```python
        x = torch.arange(H).float()            # [0, 1, ..., 23]
        y = torch.arange(W).float()
```

- **`torch.arange(H)`**：生成 `[0, 1, 2, ..., H-1]` 这种序列。`H=24` 时是 `[0, 1, ..., 23]`。返回的是整数张量。
- **`.float()`**：转成 float32（后面要做除法、对数运算，必须用浮点数）。

```python
        px_min = x.view(-1, 1).expand(-1, W).contiguous().view(-1)  # [576] 每个 patch 的 x_min
        py_min = y.view(1, -1).expand(H, -1).contiguous().view(-1)  # [576] 每个 patch 的 y_min
```

#### 这两行有点绕，逐步拆解

`x` 现在是 `[0, 1, 2, ..., 23]`，形状 `[24]`。

**第一步 `x.view(-1, 1)`**：
- `view` 重新整形。`-1` 表示"自动算"。
- 把 `[24]` 变成 `[24, 1]`，看起来像一列：
  ```
  [[0],
   [1],
   ...,
   [23]]
  ```

**第二步 `.expand(-1, W)`**：
- 把第二维从 1 广播到 W=24。`-1` 表示"保持"。
- 形状变成 `[24, 24]`：
  ```
  [[0,  0,  0,  ..., 0 ],
   [1,  1,  1,  ..., 1 ],
   ...,
   [23, 23, 23, ..., 23]]
  ```
- 注意 expand **不复制内存**，只是"假装"扩展了。

**第三步 `.contiguous()`**：
- expand 出来的张量内存不连续，调用 `view` 之前需要 `contiguous()` 让内存重新排好。

**第四步 `.view(-1)`**：
- 拉平成 1 维，形状 `[576]`。
- 顺序是按行读：`[0,0,...,0, 1,1,...,1, ..., 23,23,...,23]`
- 含义：**第 i 个元素是 patch 索引 i 对应的 x 坐标**。
  - 索引 0 的 patch 在 (行=0, 列=0) → x=0
  - 索引 24 的 patch 在 (行=1, 列=0) → x=1
  - 索引 575 的 patch 在 (行=23, 列=23) → x=23

**`py_min` 类似**，但因为 `y.view(1, -1)` 是把 y 当成"一行"来扩展，所以最终顺序是：
`[0, 1, 2, ..., 23, 0, 1, 2, ..., 23, ..., 0, 1, ..., 23]`（每行从 0 到 23 重复 24 次）。

```python
        px_max = px_min + 1
        py_max = py_min + 1
```

每个 patch 当成 1×1 的方块，左上角 `(px_min, py_min)`，右下角就是 `(px_min+1, py_min+1)`。

```python
        cx = (px_min + px_max) * 0.5           # [576]
        cy = (py_min + py_max) * 0.5
        w = px_max - px_min + 1.0              # [576] 实际值都是 2.0
        h = py_max - py_min + 1.0
```

- **cx, cy**：每个 patch 的中心坐标
- **w, h**：宽高（这里因为每个 patch 都是 1×1 + 1 = 2，恒为 2）
- 写得通用是为了将来兼容非均匀网格

```python
        delta_x = cx.unsqueeze(0) - cx.unsqueeze(1)             # [576, 576]
        delta_x = torch.clamp(torch.abs(delta_x / w.unsqueeze(0)), min=1e-3).log()
```

#### 这两行非常关键，详细解释

**目标**：算出 `delta_x[i, j] = patch i 和 patch j 在 x 方向的相对距离`，最终得到 `[576, 576]` 矩阵。

**`cx.unsqueeze(0)`**：在最前面插入一个新维度，形状从 `[576]` 变成 `[1, 576]`。
**`cx.unsqueeze(1)`**：在第 1 位插入新维度，形状从 `[576]` 变成 `[576, 1]`。

**`cx.unsqueeze(0) - cx.unsqueeze(1)`**：
- 形状 `[1, 576] - [576, 1]`
- PyTorch **广播机制**自动把它们扩成 `[576, 576]`
- 结果 `[i, j] = cx[j] - cx[i]`（注意第 0 维是 i 那行，第 1 维是 j 那列；`unsqueeze(0)` 是横向广播 j，`unsqueeze(1)` 是纵向广播 i）

**`torch.abs(...)`**：取绝对值（不在乎方向，只看距离）

**`/ w.unsqueeze(0)`**：除以宽度做归一化（因为 w 都是 2，等于全部除以 2）

**`torch.clamp(..., min=1e-3)`**：把太小的值钳到 0.001，防止下一步 log 出现 `log(0) = -inf`

**`.log()`**：取自然对数，把"距离 1 vs 距离 23"的差距压缩进 log 空间，避免远距离数值过大主导编码

```python
        delta_y = cy.unsqueeze(0) - cy.unsqueeze(1)
        delta_y = torch.clamp(torch.abs(delta_y / h.unsqueeze(0)), min=1e-3).log()
        delta_w = torch.log(w.unsqueeze(0) / w.unsqueeze(1))
        delta_h = torch.log(h.unsqueeze(0) / h.unsqueeze(1))
```

- `delta_y` 是 y 方向相对距离，做法同 delta_x
- `delta_w` 和 `delta_h` 是宽高比的对数。因为 w/h 都恒为 2，所以这里实际值都是 `log(1) = 0`，但代码留着是为了和论文公式对齐 + 兼容非均匀网格

```python
        pos_mat = torch.stack([delta_x, delta_y, delta_w, delta_h], dim=-1)
```

- **`torch.stack`**：把多个相同形状的张量沿着新的维度拼起来
- 4 个 `[576, 576]` 沿最后维度堆 → `[576, 576, 4]`
- 第 i 个位置的最后一维 4 个数是这对 patch 的 (Δx, Δy, Δw, Δh)

```python
        feat_range = torch.arange(self.dim_g / 8).float()                   # [8]
        dim_mat = 1.0 / (self.wave_len ** (feat_range / (self.dim_g / 8)))  # [8] 频率序列
        dim_mat = dim_mat.view(1, 1, 1, -1)                                  # [1,1,1,8]
```

#### 这是正弦位置编码的频率部分

- **`self.dim_g / 8 = 64 / 8 = 8`**：要生成 8 个不同频率
- **`torch.arange(8).float()`** = `[0, 1, 2, ..., 7]`
- **`self.wave_len ** (feat_range / 8)`** = `1000^(0/8), 1000^(1/8), ..., 1000^(7/8)`，是从 1 到约 1000 的几何级数
- **`1.0 / (...)`**：取倒数，得到从 1 到 1/1000 递减的频率
- **`.view(1, 1, 1, -1)`**：把形状 `[8]` 变成 `[1,1,1,8]`，这样后面广播方便

```python
        pos_mat = pos_mat.unsqueeze(-1) * 100.0                              # [576,576,4,1]
        mul_mat = (pos_mat * dim_mat).view(seq_len, seq_len, -1)             # [576,576,32]
```

- **`pos_mat.unsqueeze(-1)`**：在最后插入一个新维度，形状 `[576,576,4]` → `[576,576,4,1]`
- **`* 100.0`**：把 log 后的值放大 100 倍，让 sin/cos 有信息（避免 sin(很小的数)≈0）
- **`pos_mat * dim_mat`**：广播相乘，`[576,576,4,1] × [1,1,1,8]` → `[576,576,4,8]`
  - 含义：4 个几何量 × 8 种频率 = 32 个数
- **`.view(seq_len, seq_len, -1)`**：合并最后两维，形状变 `[576, 576, 32]`

```python
        embedding = torch.cat([mul_mat.sin(), mul_mat.cos()], dim=-1)        # [576,576,64]
```

- **`.sin()`** 和 **`.cos()`**：逐元素取正弦/余弦
- **`torch.cat([a, b], dim=-1)`**：在最后维度拼接两个张量
- 32 + 32 = 64，最终形状 `[576, 576, 64]`，正好是 dim_g

```python
        return embedding.half()
```

- **`.half()`**：转成 float16，省一半显存（576×576×64 个 float32 是 84MB，float16 只要 42MB）

### 3.4 `forward` 方法

```python
    def forward(self, batch_size):
        return self.geometry_embedding.unsqueeze(0).expand(
            batch_size, -1, -1, -1)
```

#### 逐行讲解

- 输入是一个**整数** `batch_size`（不是张量）
- **`self.geometry_embedding`**：那个预计算的 `[576, 576, 64]` buffer
- **`.unsqueeze(0)`**：在最前插入 batch 维，形状变 `[1, 576, 576, 64]`
- **`.expand(batch_size, -1, -1, -1)`**：把第 0 维"广播"到 batch_size，其他维度 `-1` 保持不变
- 最终形状 `[B, 576, 576, 64]`，**所有 batch 共享同一份位置编码**（不消耗额外内存）


---

## 4. 模块二之二：GeometryMultiHeadAttention

带几何位置编码的多头自注意力。**FAE 的真正运算核心**。

### 4.1 类定义和 `__init__`

```python
class GeometryMultiHeadAttention(nn.Module):
    def __init__(self, dim_com, heads, dim_g=64, dropout=0.1):
        super().__init__()
        assert dim_com % heads == 0
        self.heads = heads
        self.d_k = dim_com // heads             # 每个头的维度 = 512/4 = 128
```

#### 逐行讲解

- **`assert dim_com % heads == 0`**：断言。`%` 是取余。要求 dim_com 必须能被 heads 整除（不然分头时会余出来）。如果不满足程序直接报错。
- **`self.d_k = dim_com // heads`**：每个注意力头的维度。512 // 4 = 128。

```python
        self.fc_q = nn.Linear(dim_com, dim_com)
        self.fc_k = nn.Linear(dim_com, dim_com)
        self.fc_v = nn.Linear(dim_com, dim_com)
        self.fc_o = nn.Linear(dim_com, dim_com)
```

四个全连接层。
- `fc_q`：把输入投影成 Q（query）
- `fc_k`：投影成 K（key）
- `fc_v`：投影成 V（value）
- `fc_o`：注意力输出后再投影一次（output projection）

这是标准多头注意力的"四件套"。

```python
        self.WGs = nn.ModuleList([nn.Linear(dim_g, 1, bias=True)
                                  for _ in range(heads)])
```

#### 详细拆解

- **`[... for _ in range(heads)]`**：列表推导式。`for _ in range(heads)` 表示循环 heads 次。`_` 是变量名，**约定俗成表示"我不在乎这个变量的值"**（这里就是循环 4 次，每次都不需要循环变量）。
- 每次循环创建一个 `nn.Linear(dim_g, 1, bias=True)`，意思是：输入 64 维，输出 1 个标量。
- 整个列表推导式生成 `[Linear1, Linear2, Linear3, Linear4]`（4 个 Linear）。
- **`nn.ModuleList([...])`**：把这个 Python 列表包装成 PyTorch 的 ModuleList。**为什么不直接用普通列表？** 因为普通列表里的 nn.Module 不会被 PyTorch 识别为子模块（参数、to GPU 都不会管它们）。ModuleList 才会被正确管理。

**这 4 个 Linear 各自学一组"几何偏置生成器"**，每个 head 一份。

```python
        self.ln = nn.LayerNorm(dim_com)
        self.dropout = nn.Dropout(dropout)
```

- **`nn.LayerNorm(dim_com)`**：在最后一维（dim_com 那维）做归一化
- **`nn.Dropout(dropout)`**：以 `dropout` 概率随机置零，训练时正则化

### 4.2 `forward` 方法

```python
    def forward(self, x, geometry_emb):
        B, N, D = x.shape
```

#### 逐行讲解

- **`x.shape`**：返回张量形状（一个 torch.Size 对象，类似元组）
- **`B, N, D = x.shape`**：元组解包。x 形状是 `[B, N, D]`，所以 B = batch, N = patch 数 (576), D = 维度 (512)。

```python
        q = self.fc_q(x).view(B, N, self.heads, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(x).view(B, N, self.heads, self.d_k).permute(0, 2, 1, 3)
        v = self.fc_v(x).view(B, N, self.heads, self.d_k).permute(0, 2, 1, 3)
```

#### 第一行详细拆解

1. **`self.fc_q(x)`**：调用 fc_q 这个 Linear，把 `[B, N, D]` 变成 `[B, N, D]`（Linear 不改最后一维之外的形状）
2. **`.view(B, N, self.heads, self.d_k)`**：把最后那 D 维拆成 `(heads, d_k)`。形状变 `[B, N, heads, d_k]` = `[B, 576, 4, 128]`
3. **`.permute(0, 2, 1, 3)`**：交换维度顺序。新顺序是 (原第 0, 原第 2, 原第 1, 原第 3)，形状变 `[B, heads, N, d_k]` = `[B, 4, 576, 128]`

**为什么要 permute**：注意力运算 `Q @ K.T` 是在每个 head 内部做的，把 heads 提到前面方便批量运算。

k 和 v 同理。

```python
        att = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)         # [B, h, N, N]
```

#### 这是标准注意力打分公式 `att = Q · K^T / √d`

- **`k.transpose(-2, -1)`**：交换最后两维。`[B, h, N, d_k]` → `[B, h, d_k, N]`
- **`torch.matmul(q, ...)`**：矩阵乘法。`[B, h, N, d_k] @ [B, h, d_k, N]` → `[B, h, N, N]`
- **`/ (self.d_k ** 0.5)`**：除以 √d_k。`**` 是幂运算，`d_k ** 0.5` = √d_k。这是 scaled dot-product 的"scaled"。

```python
        geo_flat = geometry_emb.float().reshape(-1, geometry_emb.shape[-1])    # [B*N*N, dim_g]
```

- **`geometry_emb.float()`**：从 float16 转回 float32（Linear 层要 float32）
- **`.reshape(-1, geometry_emb.shape[-1])`**：拉平前 3 维。`[B, N, N, dim_g]` → `[B*N*N, dim_g]`
  - 形状参数里 `-1` 表示"自动算"
- 这样接下来就能一次过四个 Linear

```python
        geo_per_head = [layer(geo_flat).view(B, N, N, 1).permute(0, 3, 1, 2)
                        for layer in self.WGs]
```

#### 这是列表推导式 + 复杂操作

对每个 head 的 Linear 层 `layer`（共 4 个）做：
1. `layer(geo_flat)`：把 `[B*N*N, 64]` 投影成 `[B*N*N, 1]`
2. `.view(B, N, N, 1)`：恢复成 `[B, N, N, 1]`
3. `.permute(0, 3, 1, 2)`：把 1 那维提到前面，`[B, 1, N, N]`

最终 `geo_per_head` 是 4 个 `[B, 1, N, N]` 的列表。

```python
        geo_weights = F.relu(torch.cat(geo_per_head, dim=1))                   # [B, h, N, N]
```

- **`torch.cat(..., dim=1)`**：在第 1 维拼接。4 个 `[B, 1, N, N]` 拼起来变 `[B, 4, N, N]`，刚好对应 4 个 head
- **`F.relu(...)`**：max(0, x)，**保证几何权重非负**。这一步至关重要——不然下面减号就不一定是"扣减"

```python
        att = att - geo_weights
```

**核心操作**：标准注意力分数减去几何偏置。空间相关度高的位置对，注意力被扣得越多，最终softmax 出来的注意力越小。

```python
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(B, N, D)
```

- **`F.softmax(att, dim=-1)`**：在最后一维做 softmax，每行加起来 = 1，变成"注意力分布"
- **`self.dropout(att)`**：随机置零部分注意力（正则化）
- **`torch.matmul(att, v)`**：用注意力加权求 V。形状 `[B, h, N, N] @ [B, h, N, d_k]` → `[B, h, N, d_k]`
- **`.permute(0, 2, 1, 3)`**：把 heads 维放回去。`[B, h, N, d_k]` → `[B, N, h, d_k]`
- **`.contiguous().view(B, N, D)`**：把 heads 和 d_k 合并回 D。形状 `[B, N, D]`

```python
        out = self.fc_o(out)
        return self.ln(x + out)
```

- **`self.fc_o(out)`**：输出投影
- **`x + out`**：**残差连接**（输入直接加到输出上）。这是 Transformer 的标准做法
- **`self.ln(...)`**：LayerNorm，稳定数值

返回形状 `[B, N, D]`。


---

## 5. 模块二之三：FAELayer

把 `GeometryMultiHeadAttention` 包装成一层 Transformer Block。

### 5.1 类定义和 `__init__`

```python
class FAELayer(nn.Module):
    def __init__(self, dim_com, heads, dropout=0.1, dim_g=64):
        super().__init__()
        self.attn = GeometryMultiHeadAttention(dim_com, heads, dim_g, dropout)
```

- **`self.attn`**：实例化一个 `GeometryMultiHeadAttention`，作为本层的注意力子模块

```python
        self.ffn = nn.Sequential(
            nn.Linear(dim_com, dim_com * 2),  # 512 → 1024
            nn.ReLU(inplace=True),
            nn.Linear(dim_com * 2, dim_com),  # 1024 → 512
        )
```

- **`self.ffn`**：前馈网络（FFN），标准 Transformer block 第二阶段
- 结构：512 → 1024 → ReLU → 512（先扩 2 倍再压回去）

```python
        self.ln = nn.LayerNorm(dim_com)
        self.dropout = nn.Dropout(dropout)
```

LayerNorm 和 Dropout，含义同前。

### 5.2 `forward` 方法

```python
    def forward(self, x, geometry_emb):
        x = self.attn(x, geometry_emb)
        x = self.ln(x + self.dropout(self.ffn(x)))
        return x
```

#### 逐行讲解

**`x = self.attn(x, geometry_emb)`**：
- 几何感知自注意力。**注意：`GeometryMultiHeadAttention` 内部已经做了 `x + out` 残差和 LayerNorm**，所以这里直接覆盖 x 即可，不要再手动加残差。

**`x = self.ln(x + self.dropout(self.ffn(x)))`**：
分解成：
1. `self.ffn(x)`：FFN 处理
2. `self.dropout(...)`：随机置零
3. `x + ...`：残差连接（保留输入）
4. `self.ln(...)`：LayerNorm

这是标准 Transformer block 的"FFN + Add & Norm"结构。

**返回 x**：形状仍然 `[B, N, D]`。

---

## 6. 模块二之四：CrossModalTransformer

双向视觉-语义交互模块，**FAE 是它的子模块**。

### 6.1 类定义和 `__init__`

```python
class CrossModalTransformer(nn.Module):
    def __init__(self, dim_f=768, dim_com=512, heads=4, dropout=0.1,
                 weight_s2v=0.5, grid_size=(24, 24), dim_g=64,
                 text_residual=0.2, visual_residual=0.2):
        super().__init__()
```

接收一堆超参数：
- `dim_f=768`：CLIP 输出的特征维度
- `dim_com=512`：内部公共维度（投影到这个维度做 attention）
- `heads=4`：多头数
- `weight_s2v=0.5`：两个分支的融合权重

```python
        self.dim_f = dim_f
        self.dim_com = dim_com
        self.weight_s2v = weight_s2v
        self.text_residual   = text_residual
        self.visual_residual = visual_residual
```

把超参数存进对象。

```python
        self.embed_cv = nn.Linear(dim_f, dim_com)       # 共享视觉投影
        self.embed_text = nn.Linear(dim_f, dim_com)     # 共享文本投影
```

两个投影层：把 768 维降到 512 维。视觉用 `embed_cv`，文本用 `embed_text`。

```python
        self.box_emb = BoxRelationalEmbedding(grid_size=grid_size, dim_g=dim_g)
        self.fae = FAELayer(dim_com, heads, dropout, dim_g=dim_g)
```

实例化 BoxRelationalEmbedding 和 FAELayer。

```python
        self.decoder_v2s = nn.TransformerDecoderLayer(
            d_model=dim_com, nhead=heads, dim_feedforward=dim_com * 2,
            dropout=dropout, batch_first=True)
        self.decoder_s2v = nn.TransformerDecoderLayer(
            d_model=dim_com, nhead=heads, dim_feedforward=dim_com * 2,
            dropout=dropout, batch_first=True)
```

#### `nn.TransformerDecoderLayer`

PyTorch 内置的 Transformer Decoder Block，结构是：
1. self-attention（query 自注意力）
2. cross-attention（query 和 memory 做注意力）
3. FFN

参数说明：
- `d_model`：内部维度
- `nhead`：多头数
- `dim_feedforward`：FFN 中间维度
- `batch_first=True`：输入形状是 `[B, N, D]`（不是默认的 `[N, B, D]`）

**两个 decoder**：
- `decoder_v2s`：v2s 分支用，文本 query 视觉
- `decoder_s2v`：s2v 分支用，视觉 query 文本

```python
        self.proj_visual = nn.Linear(dim_com, dim_f)
        self.proj_text   = nn.Linear(dim_com, dim_f)
```

两个投影回 768 维（实际本版本没用到，留下兼容接口）。

### 6.2 `forward` 方法

```python
    def forward(self, patches, text, cls_token=None):
        B = patches.size(0)
        geo_emb = self.box_emb(B)                                # [B, 576, 576, dim_g]
```

- **`patches.size(0)`**：取第 0 维大小，就是 B
- **`self.box_emb(B)`**：调用 BoxRelationalEmbedding 的 forward，返回 `[B, 576, 576, 64]`

```python
        vis    = self.embed_cv(patches)                          # [B, 576, dim_com]
        memory = self.fae(vis, geo_emb)                          # [B, 576, dim_com]
```

- **`self.embed_cv(patches)`**：把 patch 从 768 投影到 512
- **`self.fae(vis, geo_emb)`**：用 FAE 做几何感知自注意力，**输出叫 `memory`**（在 decoder 里 memory 是 KV 来源的标准命名）

```python
        txt_com   = self.embed_text(text)                        # [N_cls, dim_com]
        txt_batch = txt_com.unsqueeze(0).expand(B, -1, -1)       # [B, N_cls, dim_com]
```

- **`self.embed_text(text)`**：文本投影到 512 维
- **`.unsqueeze(0).expand(B, -1, -1)`**：加 batch 维并广播。每个 batch 都用同一份文本

```python
        F_p_v2s = self.decoder_v2s(tgt=txt_batch,
                                    memory=memory)               # [B, N_cls, dim_com]
```

#### v2s 分支

- **`tgt=txt_batch`**：query 来源（target sequence）
- **`memory=memory`**：KV 来源（要被关注的内容）
- 含义：**每个类的文本去 576 个 patch 里找相关的视觉区域**
- 输出 `F_p_v2s[b, k]` = 第 b 张图、第 k 类文本"看完视觉"后的表示

```python
        F_p_s2v = self.decoder_s2v(tgt=memory,
                                    memory=txt_batch)            # [B, 576, dim_com]
```

#### s2v 分支（方向相反）

- query 是视觉，KV 是文本
- 含义：**每个 patch 去文本里找相关的语义**
- 输出 `F_p_s2v[b, p]` = 第 b 张图、第 p 个 patch"看完文本"后的表示

```python
        v2s_n   = F.normalize(F_p_v2s, dim=-1)                  # [B, N_cls, dim_com]
        txt_n   = F.normalize(txt_batch, dim=-1)                 # [B, N_cls, dim_com]
        score_v2s = (v2s_n * txt_n).sum(dim=-1)                  # [B, N_cls]
```

#### 计算 v2s 分数

- **`F.normalize(x, dim=-1)`**：把最后一维变成单位向量（除以自己的 L2 范数）
- **`(v2s_n * txt_n).sum(dim=-1)`**：逐元素乘后求和 = 余弦相似度
- 得到 `[B, N_cls]`：每张图对每个类的 v2s 分数

```python
        s2v_pooled = F_p_s2v.mean(dim=1)                         # [B, dim_com] 576→全局
        s2v_n      = F.normalize(s2v_pooled, dim=-1)             # [B, dim_com]
        txt_single = F.normalize(txt_com, dim=-1)                # [N_cls, dim_com]
        score_s2v  = s2v_n @ txt_single.T                        # [B, N_cls]
```

#### 计算 s2v 分数

- **`F_p_s2v.mean(dim=1)`**：在第 1 维（576 个 patch）求平均，得到全局视觉表示 `[B, dim_com]`
- **`s2v_n @ txt_single.T`**：矩阵乘法。`@` 是 Python 的矩阵乘法运算符。`[B, dim_com] @ [dim_com, N_cls]` → `[B, N_cls]`
- 也是余弦相似度

```python
        local_score = self.weight_s2v * score_s2v + \
                      (1.0 - self.weight_s2v) * score_v2s        # [B, N_cls]
        return local_score
```

- 加权融合两路分数。`\` 是 Python 的换行符（这一行没结束的意思）
- 返回 `[B, N_cls]`：每张图对每个类的"局部分数"


---

## 7. 模块三：VGSR 主模型

把上面所有模块组装起来。

### 7.1 类定义和 `__init__` 头部

```python
class VGSR(nn.Module):
    def __init__(self, config, seenclass, unseenclass,
                 seen_text_embeds,       # [150, 768]
                 unseen_text_embeds):    # [50,  768]
        super().__init__()
        self.config = config
        self.nclass = config.num_class   # 200
        self.dim_f = config.dim_f_clip   # 768
        self.seenclass = seenclass       # [150] 全局索引
        self.unseenclass = unseenclass   # [50]  全局索引
```

#### 逐行讲解

- 接收 5 个参数：
  - `config`：超参数容器（一个 SimpleNamespace 对象，可以用点号访问字段）
  - `seenclass`：训练集 150 个类的全局索引
  - `unseenclass`：测试用的 50 个未见类索引
  - `seen_text_embeds`：seen 类的文本特征 `[150, 768]`
  - `unseen_text_embeds`：unseen 类的文本特征 `[50, 768]`

- **`config.num_class`**：从 config 取出 num_class 字段（200）

### 7.2 注册文本特征为不学习的参数

```python
        self.seen_text_embeds   = nn.Parameter(
            F.normalize(seen_text_embeds,   dim=1), requires_grad=False)
        self.unseen_text_embeds = nn.Parameter(
            F.normalize(unseen_text_embeds, dim=1), requires_grad=False)
```

#### 详细拆解

- **`F.normalize(seen_text_embeds, dim=1)`**：在第 1 维（每行）做 L2 归一化。每行变成单位向量
- **`nn.Parameter(..., requires_grad=False)`**：注册成 Parameter，但**不参与梯度更新**
  - 为什么不用 `register_buffer`？两者都不学习，但 Parameter 在 print(model) 时会显示出来更清楚
  - `requires_grad=False` 是关键开关，等价于"我是只读的"

### 7.3 Adapter 子模块

```python
        self.adapter_ratio = getattr(config, 'adapter_ratio', 0.2)
        self.text_adapter  = Adapter(self.dim_f, reduction=4)
```

- **`getattr(config, 'adapter_ratio', 0.2)`**：从 config 拿 adapter_ratio，**没有就用默认 0.2**。这是 Python 内置函数，安全访问对象属性
- **`Adapter(768, reduction=4)`**：实例化前面那个 Adapter 类

### 7.4 CrossModalTransformer 子模块

```python
        tf_common_dim = getattr(config, 'tf_common_dim', 512)
        tf_heads      = getattr(config, 'tf_heads', 4)
        tf_dropout    = getattr(config, 'tf_dropout', 0.1)
        weight_s2v    = getattr(config, 'weight_s2v', 0.5)
        text_residual   = getattr(config, 'text_residual', 0.2)
        visual_residual = getattr(config, 'visual_residual', 0.2)
        self.local_weight = getattr(config, 'local_weight', 0.3)
        self.cross_tf = CrossModalTransformer(
            dim_f=self.dim_f,
            dim_com=tf_common_dim,
            heads=tf_heads,
            dropout=tf_dropout,
            weight_s2v=weight_s2v,
            grid_size=(24, 24),
            dim_g=64,
            text_residual=text_residual,
            visual_residual=visual_residual,
        )
```

读一堆超参数，实例化 CrossModalTransformer。`local_weight` 是后面用来加权 local_score 的系数。

### 7.5 可学习的温度系数

```python
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
```

#### 详细解释

- **`torch.ones([])`**：生成一个**0 维标量**张量，值为 1.0。注意 `[]` 表示空形状（标量）
- **`np.log(1 / 0.07)` = `log(14.28...)` ≈ 2.659**：CLIP 论文里的初始值
- **`nn.Parameter(...)`**：注册成可学习参数，**会被优化器更新**

为什么要存 log？因为后面会取 exp 用，存 log 可以用线性变化代替指数变化，训练更稳定。

### 7.6 `get_adapted_seen_text` 方法

```python
    def get_adapted_seen_text(self):
        x = self.seen_text_embeds
        adapted = self.adapter_ratio * self.text_adapter(x) + \
                  (1.0 - self.adapter_ratio) * x
        return F.normalize(adapted, dim=1)
```

#### 逐行讲解

- 取出 seen 类原始文本 `[150, 768]`
- **`self.text_adapter(x)`**：经过 Adapter 增强
- **残差融合**：`adapted = ratio × Adapter(x) + (1-ratio) × x`
  - ratio=0.2 时，原始信号占 80%，Adapter 增强占 20%
  - 这样保留了 CLIP 原始能力，Adapter 只做轻微微调
- **`F.normalize(adapted, dim=1)`**：再做一次单位归一化

### 7.7 `_prepare_patches` 方法（兼容多种输入格式）

```python
    def _prepare_patches(self, clip_features):
        if clip_features.dim() == 3:
            if clip_features.size(1) == 577:       # 完整 spatial
                return clip_features[:, 1:, :]
            elif clip_features.size(1) == 576:     # 纯 patch
                return clip_features
            elif clip_features.size(1) == 1:       # CLS 缓存兼容
                return clip_features.expand(-1, 576, -1)
        return clip_features.unsqueeze(1).expand(-1, 576, -1)
```

#### 逐行讲解

- **`clip_features.dim()`**：维度数。3 表示 `[B, N, D]`，2 表示 `[B, D]`
- **`clip_features.size(1)`**：第 1 维大小

支持 4 种输入格式：
1. `[B, 577, 768]`：CLIP 完整输出（CLS + 576 patches）。`[:, 1:, :]` 表示**所有 batch、跳过第 0 个 token、所有 D 维**
2. `[B, 576, 768]`：纯 patch，直接返回
3. `[B, 1, 768]`：只有 CLS 的缓存。**`.expand(-1, 576, -1)`** 把第 1 维从 1 广播到 576
4. `[B, 768]`：直接 CLS 向量。`.unsqueeze(1)` 加一维变 `[B, 1, 768]`，再 expand

### 7.8 `forward` 方法（核心）

```python
    def forward(self, clip_features, is_train=False):
        if clip_features.dim() == 3 and clip_features.size(1) == 577:
            cls_token = clip_features[:, 0, :]                   # [B, 768]
            patches   = clip_features[:, 1:, :]                  # [B, 576, 768]
        else:
            patches = self._prepare_patches(clip_features)
            cls_token = None
```

#### 逐行讲解

- **`is_train=False`**：默认参数，调用时不传就是 False
- 如果是完整的 `[B, 577, 768]`：
  - **`clip_features[:, 0, :]`**：取出每张图的第 0 个 token = CLS（全局图像表示）
  - **`clip_features[:, 1:, :]`**：取出剩下 576 个 patch
- 否则用 `_prepare_patches` 做转换

```python
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        seen_text = self.get_adapted_seen_text()                 # [150, 768]
```

- **`self.logit_scale.exp()`**：取指数。把存在 log 空间的温度系数恢复成实际值
- **`torch.clamp(..., max=100.0)`**：钳到 100 以内，防止训练初期数值爆炸
- **`seen_text`**：调用前面那个方法，得到 Adapter 增强后的 seen 类文本

```python
        all_text = torch.zeros(self.nclass, self.dim_f,
                               device=patches.device, dtype=patches.dtype)
        all_text[self.seenclass]   = seen_text
        all_text[self.unseenclass] = self.unseen_text_embeds
```

#### 详细拆解

- **`torch.zeros(200, 768)`**：创建全零的 `[200, 768]` 张量
- **`device=patches.device, dtype=patches.dtype`**：保证设备和数据类型和 patches 一致
- **`all_text[self.seenclass] = seen_text`**：**索引赋值**
  - `self.seenclass` 是 150 个全局类索引（比如 `[0, 2, 5, 7, ...]`）
  - 把 seen_text 的 150 行写到 all_text 对应的位置
- **`all_text[self.unseenclass] = self.unseen_text_embeds`**：同理把 unseen 类文本写进去
- 最终 all_text 包含全 200 类的文本特征，按全局类别索引排列

```python
        if cls_token is not None:
            vis_n = F.normalize(cls_token, dim=1)                # [B, 768]
        else:
            vis_n = F.normalize(patches.mean(dim=1), dim=1)
        text_n = F.normalize(all_text, dim=1)                    # [200, 768]
        base_logits = vis_n @ text_n.T * logit_scale             # [B, 200]
```

#### 这是 CLIP 的"基线分数"

- 视觉端：用 CLS token（如果有），否则用 patch 平均
- **`F.normalize(..., dim=1)`**：每行变单位向量
- **`vis_n @ text_n.T`**：矩阵乘法 `[B, 768] @ [768, 200]` = `[B, 200]`
  - 这是 cosine similarity（因为已经归一化）
- **`* logit_scale`**：乘温度系数放大

这一步就是 CLIP 原版的零样本分类。

```python
        local_score = self.cross_tf(patches, all_text, cls_token) # [B, 200]
```

调用 CrossModalTransformer 算"局部分数"。

```python
        logits_200 = base_logits + self.local_weight * local_score
```

**最终 logits = CLIP 基线分数 + β × 局部分数**。`local_weight` 控制局部分数的权重。

```python
        if is_train:
            logits = logits_200[:, self.seenclass]               # [B, 150]
        else:
            logits = logits_200                                  # [B, 200]
```

- 训练时只取 seen 列（150 个），用于 CE loss
- 评估时全 200 列都要

```python
        return {
            'logits':     logits,
            'logits_200': logits_200,
            'clip_S_pp':  logits,
            'clip_pred':  logits,
        }
```

返回字典。`'clip_S_pp'` 是接口契约要求的键名（评估代码里会读这个键）。

### 7.9 `compute_loss` 方法

```python
    def compute_loss(self, in_package):
        logits = in_package['logits']        # [B, 150]
        labels = in_package['batch_label']   # [B] 全局索引
```

接收一个字典 `in_package`，里面要有 `logits` 和 `batch_label`。

```python
        if labels.dim() > 1:
            labels = torch.argmax(labels, dim=1)
```

如果标签是 one-hot 形式（2 维），用 argmax 转成整数索引。

```python
        seen_labels = torch.zeros_like(labels, dtype=torch.long)
        for i, cls_idx in enumerate(self.seenclass):
            seen_labels[labels == cls_idx] = i
```

#### 详细拆解

- **`torch.zeros_like(labels, dtype=torch.long)`**：创建和 labels 相同形状的零张量，类型 long（CrossEntropy 要求）
- **`enumerate(self.seenclass)`**：遍历 self.seenclass，同时给出**索引 i** 和**值 cls_idx**
- 例如 self.seenclass = [0, 2, 5, ...]，循环时：
  - 第 1 次：i=0, cls_idx=0
  - 第 2 次：i=1, cls_idx=2
  - 第 3 次：i=2, cls_idx=5
- **`labels == cls_idx`**：返回 bool 张量，对应位置 True/False
- **`seen_labels[labels == cls_idx] = i`**：**布尔索引**赋值，把所有"标签等于 cls_idx"的位置写成 i

**作用**：把全局索引（0~199 中的 150 个 seen 索引）转成 seen 局部索引（0~149 连续）。

例：全局索引 5 对应 seen 局部索引 2。

```python
        loss_CE = F.cross_entropy(logits, seen_labels)
```

交叉熵损失。`logits` 形状 `[B, 150]`，`seen_labels` 形状 `[B]`，每个元素是 0~149 的整数。

`F.cross_entropy` 内部做了：
1. softmax(logits)
2. -log(P[正确类])
3. 求平均

```python
        logits_200 = in_package.get('logits_200', None)
        if logits_200 is not None and self.config.__dict__.get('lambda_cal', 0) > 0:
            prob_all    = F.softmax(logits_200, dim=-1)              # [B, 200]
            prob_unseen = prob_all[:, self.unseenclass]              # [B, 50]
            mass_unseen = prob_unseen.sum(dim=1)                     # [B]
            loss_cal    = -torch.log(mass_unseen.mean() + 1e-8)
            loss = loss_CE + self.config.lambda_cal * loss_cal
        else:
            loss_cal = torch.tensor(0.0)
            loss = loss_CE
```

#### Self-calibration loss（辅助，可选）

- **`in_package.get('logits_200', None)`**：字典安全访问，没有就返回 None
- **`self.config.__dict__.get('lambda_cal', 0) > 0`**：检查 config 里 lambda_cal 是否大于 0（即是否启用此 loss）

如果启用：
1. 算全 200 类概率
2. 取 unseen 列概率，求和得到"unseen 总概率质量"
3. `loss_cal = -log(平均 unseen 质量)`：unseen 概率越高，loss 越小
4. 加权加进总 loss

这个 loss 作用是**强制模型不要把所有概率都给 seen 类**，防止"seen bias"。

```python
        return {'loss': loss, 'loss_CE': loss_CE, 'loss_cal': loss_cal}
```

返回字典，主 loss 在 `'loss'` 键里，外面会调 `loss.backward()`。

---

## 总结：整个文件做的事

```
图像 → CLIP → patches [B, 576, 768] + cls_token [B, 768]
                    │                    │
                    ↓                    ↓
                CrossModalTransformer   余弦相似度
                    │   (FAE + 双向 attention)
                    │   注意 FAE 是减号几何偏置
                    ↓                    ↓
                local_score [B, 200]  base_logits [B, 200]
                    └────────┬───────────┘
                             ↓
                  logits_200 = base + β × local
                             ↓
              训练: 取 seen 150 列 → CE loss
              评估: 全 200 类 argmax
```

读懂这个文件，等于读懂了：
- PyTorch nn.Module 的写法
- 多头注意力的实现
- Transformer encoder/decoder 的用法
- 几何位置编码的算法
- 残差连接 + LayerNorm 的标准写法
- 零样本分类的损失设计
