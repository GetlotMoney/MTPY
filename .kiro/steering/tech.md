# Tech Stack

## Environment

- **OS**: Windows (important: `num_workers=0` in all DataLoaders — multiprocessing breaks on Windows)
- **Conda env**: `dassl_clip` (`F:\Anaconda\envs\dassl_clip`)
- **Python**: 3.x
- **CUDA**: Required; default device `cuda:0` (AWA2 uses `cuda:1`)

## Core Libraries

| Library | Role |
|---------|------|
| `torch` / `torchvision` | Model, training loop, data transforms |
| `clip` (OpenAI) | ViT-L/14@336px visual + text encoder (always frozen) |
| `numpy` | Array ops, data loading from `.mat` files |
| `scipy.io` | Reading `.mat` metadata files (xlsa17 splits) |
| `yaml` | Config loading (`SimpleNamespace` wrapping) |
| `PIL` (Pillow) | Image loading and transforms |

## CLIP Model

- Always loaded as `clip.load("ViT-L/14@336px", device=config.device)`
- Always cast to `.float()` after loading
- Always frozen: `p.requires_grad = False` for all parameters
- Feature dim: **768**
- Spatial output: `[B, 577, 768]` — index `[:, 0, :]` for CLS (global) token

## Config System

Configs are YAML files in `config/`. Values are nested dicts with a `value` key:
```yaml
batch_size:
  value: 64
```
Loaded via:
```python
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
config = {k: v['value'] if isinstance(v, dict) and 'value' in v else v for k, v in config.items()}
config = SimpleNamespace(**config)
```
Always check `hasattr(config, 'key')` before accessing optional params.

## Image Normalization

All datasets use CLIP's normalization constants:
```python
mean = (0.48145466, 0.4578275, 0.40821073)
std  = (0.26862954, 0.26130258, 0.27577711)
```
All images resized to **336×336** (matching ViT-L/14@336px).

## Training

- **Optimizer**: Adam
  - CUB: `lr=0.001`, `weight_decay=1e-4`, CosineAnnealingLR scheduler
  - AWA2/SUN: `lr=1e-5` / `2e-5`, `weight_decay=5e-4`, StepLR scheduler
- **Gradient clipping**: `clip_grad_norm_(model.parameters(), max_norm=1.0)` (AWA2, SUN)
- **Random seed**: Set via `config.random_seed` on `torch`, `torch.cuda`, and `numpy`

## Feature Caching (CUB only)

Pre-extracted CLIP features stored in `data/cache/`:
- `CUB_train_features.pt` — `[N, 768]`
- `CUB_train_labels.pt` — `[N]`

Generate with:
```bash
python tools/extract_features.py
```
When cache exists, training skips live CLIP inference (much faster).

## Common Commands

```bash
# Activate environment
conda activate dassl_clip

# Train on CUB
cd C:\Users\Administrator\Desktop\DVSR
python train_VGSR_CUB.py

# Train on AWA2
python train_VGSR_AWA2.py

# Train on SUN
python train_VGSR_SUN.py

# Pre-extract and cache CUB training features
python tools/extract_features.py
```

## Logging

Each training run writes a timestamped log to `train_log/{DATASET}/training_log_{DATASET}_{timestamp}.txt`. The `print_log()` helper writes to both stdout and the log file simultaneously.
