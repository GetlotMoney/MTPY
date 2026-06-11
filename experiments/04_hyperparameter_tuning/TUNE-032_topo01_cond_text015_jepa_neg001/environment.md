# 运行环境

## 硬件

| 项目 | 值 |
|------|---|
| Host | PC-20260227YXOR |
| GPU | NVIDIA RTX 4090 |
| CUDA | 12.1 |
| CPU | Intel i9-13900K |
| RAM | 64GB |

## 软件

| 项目 | 值 |
|------|---|
| 操作系统 | Windows 11 22H2 |
| Python | 3.10.13 |
| conda 环境 | dvsr_gpu |
| PyTorch | 2.1.2+cu121 |
| clip | OpenAI 1.0.1 |

## 运行命令

```powershell
conda activate dvsr_gpu
cd C:\Users\Administrator\Desktop\项目\DVSR
python train_VGSR_CUB.py --config experiments/04_hyperparameter_tuning/TUNE-032_topo01_cond_text015_jepa_neg001/config.yaml
```

## 数据完整性

| 数据 | 路径 | 状态 |
|------|------|------|
| xlsa17 | data/xlsa17/ | ✅ 已存在 |
| CUB images | data/CUB/images/ | ✅ 已存在 |
| GPT-4 描述 | data/gpt4_data/cub.pt | ✅ 已存在 |
| CLIP 缓存 | data/cache/ | ✅ 已存在 |
