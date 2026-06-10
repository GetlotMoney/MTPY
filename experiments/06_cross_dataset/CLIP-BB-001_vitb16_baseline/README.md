# CLIP-BB-001 ViT-B/16 baseline

类型: CLIP-BB (backbone 对比)
状态: completed (failed to converge)

## 实验目的

对比不同 CLIP backbone 对 VGSR 框架的影响。本实验用 ViT-B/16 直接应用 TUNE-024 最优配置。

## 配置对比

| 参数 | ViT-L/14@336 (baseline) | ViT-B/16 (本实验) |
|---|---:|---:|
| dim | 768 | 512 |
| patches | 576 (24x24) | 196 (14x14) |
| CUB ZSL | 81%+ | ~16% |
| 参数量 | 428M | 150M |

## 结果

| 数据集 | seed | U | S | H | ZS | 最佳 epoch |
|---|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 9.32 | 5.54 | 6.95 | 15.29 | 1 |

结论: ViT-B/16 backbone 下 VGSR 无法收敛。底层 backbone 太弱 (ZSL 仅 16%)，训练后 seen 和 unseen 都极低，模型过拟合 seen 但泛化无效。**证明 VGSR 需要至少 ViT-L 级别 backbone。**

## 日志

- logs/seed-5/console-output.txt
