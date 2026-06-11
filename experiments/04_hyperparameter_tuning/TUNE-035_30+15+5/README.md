# TUNE-035 训练策略消融: 30+15+5

类型: TUNE (策略消融)
范式: TUNE-LITE
状态: done

## Baseline

- 参数: topo=0.10, cond_text=0.010, jepa_neg=0.010, seed=5

## 策略

```yaml
lr_stages:
  - {lr: 0.001,  epochs: 30, eta_min: 1e-5}
  - {lr: 0.0001, epochs: 15, eta_min: 1e-6}
  - {lr: 0.00001, epochs: 5, eta_min: 1e-7}
```

## 结果

| 数据集 | U | S | H | ZS | Best |
|---|---:|---:|---:|---:|---:|
| CUB GZSL | 72.58 | 71.31 | 71.94 | 81.62 | ep9 |

## 结论

H=71.94，Stage1=30ep 太长，S 崩盘。
