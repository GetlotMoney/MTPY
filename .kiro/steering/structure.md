# Project Structure

```
DVSR/
├── model/
│   └── VGSR.py                  # Core model: Adapter class + VGSR main model
├── tools/
│   ├── dataset.py               # DataLoaders: CUBDataLoader, AWA2DataLoader, SUNDataLoader
│   ├── helper_func.py           # CLIP feature extraction + GZSL evaluation functions
│   ├── extract_features.py      # One-time script to pre-cache CUB training features
│   ├── preprocessing.py         # Legacy path utilities (mostly unused)
│   ├── global_setting.py        # Global NFS_path constant (legacy, set to './')
│   └── tools.py                 # Misc utilities
├── config/
│   ├── VGSR_cub_gzsl.yaml       # Hyperparams for CUB training
│   ├── VGSR_awa2_gzsl.yaml      # Hyperparams for AWA2 training
│   └── VGSR_sun_gzsl.yaml       # Hyperparams for SUN training
├── data/
│   ├── cache/                   # Pre-extracted CUB CLIP features (.pt files)
│   ├── clip_att/                # CLIP attribute vectors per dataset (.pkl)
│   ├── gpt4_data/               # GPT-4 class descriptions (cub.pt, etc.)
│   ├── CUB/images/              # Symlink → CUB image files
│   └── xlsa17/data/             # Symlink → xlsa17 split .mat files
├── train_log/
│   ├── CUB/                     # Timestamped training logs for CUB
│   ├── AWA2/                    # Timestamped training logs for AWA2
│   └── SUN/                     # Timestamped training logs for SUN
├── xlsa17/                      # Original xlsa17 benchmark data (APY, AWA1, AWA2, CUB, SUN)
├── train_VGSR_CUB.py            # Training entry point for CUB
├── train_VGSR_AWA2.py           # Training entry point for AWA2
├── train_VGSR_SUN.py            # Training entry point for SUN
├── PROJECT_GUIDE.md             # Detailed architecture and extension guide (Chinese)
└── CHANGELOG_VDT_TransZero.md   # Change log and architecture summary (Chinese)
```

## Key Architectural Boundaries

**Only `model/VGSR.py` should be modified when:**
- Adding new modules (visual adapter, cross-attention, etc.)
- Changing the forward pass
- Adding new loss terms to `compute_loss()`

**`tools/dataset.py` should be modified when:**
- Adding a new dataset
- Changing data augmentation or image transforms
- Modifying the sampling strategy in `next_batch()`

**`tools/helper_func.py` should be modified when:**
- Changing the evaluation protocol
- Modifying how CLIP spatial features are extracted

**Config YAML files should be modified when:**
- Tuning hyperparameters (lr, batch_size, epochs, adapter_ratio, etc.)

Training scripts (`train_VGSR_*.py`) rarely need changes — they are thin wrappers that wire config → dataloader → model → optimizer → eval loop.

## Model Interface Contract

The model's public interface must be preserved:
- **Input**: `clip_features` — shape `[B, 577, 768]` (spatial) or `[B, 768]` (global)
- **Output**: dict with key `'clip_S_pp'` containing logits `[B, num_class]`
- `is_train=True` → logits over seen classes only `[B, n_seen]`
- `is_train=False` (default) → logits over all classes `[B, num_class]`

Breaking this contract requires updating `eval_zs_gzsl()` in `tools/helper_func.py`.

## Data Flow: Tensor Shapes

| Stage | Variable | Shape |
|-------|----------|-------|
| Raw images | `batch_images` | `[B, 3, 336, 336]` |
| CLIP spatial features | `clip_features` | `[B, 577, 768]` |
| Global image feature | `image_feat` | `[B, 768]` |
| Seen class text embeds | `seen_text_embeds` | `[n_seen, 768]` |
| Unseen class text embeds | `unseen_text_embeds` | `[n_unseen, 768]` |
| Full class text embeds | `all_text` | `[num_class, 768]` |
| Output logits | `logits` | `[B, num_class]` or `[B, n_seen]` |

## Data Files

- **`data/xlsa17/data/{DATASET}/res101.mat`** — image file paths + labels
- **`data/xlsa17/data/{DATASET}/att_splits.mat`** — train/test splits, class attributes, class names
- **`data/gpt4_data/cub.pt`** — dict mapping class name → list of 7 GPT-4 description strings
- **`data/clip_att/{DATASET}_attribute.pkl`** — precomputed CLIP attribute vectors (loaded but not used in current VGSR model)
- **`data/cache/CUB_train_features.pt`** — cached CLIP CLS features for CUB training set

## Naming Conventions

- Classes follow standard Python conventions (PascalCase for classes, snake_case for functions/variables)
- Dataset-specific loaders are named `{DATASET}DataLoader` (e.g., `CUBDataLoader`)
- Config files follow `VGSR_{dataset}_{task}.yaml` pattern
- Training logs follow `training_log_{DATASET}_{YYYY-MM-DD_HH-MM-SS}.txt` pattern
- Comments and documentation are written in **Chinese** throughout the codebase
