# TUNE-034 训练策略消融: 单段 50ep

类型: TUNE (策略消融)
范式: TUNE-LITE
状态: done

## Baseline

- 代码起点: main (TUNE-024 配置派生)
- 数据集: CUB GZSL
- seed: 5
- 参数: topo=0.10, cond_text=0.010, jepa_neg=0.010

## 策略

```yaml
lr_stages:
  - {lr: 0.001, epochs: 50, eta_min: 1e-5}
```

单段余弦退火，50ep，无段切换。

## 结果

| 数据集 | seed | U | S | H | ZS | 最佳 epoch |
|---|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 73.10 | 71.16 | 72.12 | 81.48 | 9 |

## 结论

单段 50ep H=72.12，远低于 20+20+10 (73.85)。Stage1 过长导致 S 崩盘。
