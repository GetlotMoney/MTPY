# Product: DVSR / VGSR

DVSR is a research codebase implementing **Generalized Zero-Shot Learning (GZSL)** using CLIP + Adapter + GPT-4 descriptions, based on the VDT-TransZero methodology.

## What it does

Given an image, the model classifies it into one of N classes — including **unseen classes** that were never shown during training. This is achieved by:

1. Encoding images with a frozen CLIP ViT-L/14@336px visual encoder
2. Encoding class names / GPT-4 descriptions with a frozen CLIP text encoder
3. Training a lightweight **Adapter** (bottleneck MLP, ~295K params) that refines seen-class text features
4. At inference, combining adapted seen-class features with raw CLIP unseen-class features for full-space classification

## Supported Datasets

| Dataset | Classes | Task |
|---------|---------|------|
| CUB-200-2011 | 200 bird species (150 seen / 50 unseen) | Fine-grained bird classification |
| AWA2 | 50 animal classes | Coarse-grained animal classification |
| SUN | Scene categories | Scene recognition |

## Evaluation Protocol

Uses the **xlsa17 Proposed Split** standard. Core metrics:
- **H** (Harmonic mean of S and U) — primary metric
- **S** — per-class accuracy on seen classes (GZSL search space = all classes)
- **U** — per-class accuracy on unseen classes (GZSL search space = all classes)
- **ZS** — per-class accuracy on unseen classes only (ZSL, restricted search space)
