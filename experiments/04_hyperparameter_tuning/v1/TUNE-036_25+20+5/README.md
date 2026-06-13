# TUNE-036 训练策略消融: 25+20+5

类型: TUNE (策略消融)
状态: done

## 策略

```yaml
lr_stages:
  - {lr: 0.001,  epochs: 25, eta_min: 1e-5}
  - {lr: 0.0001, epochs: 20, eta_min: 1e-6}
  - {lr: 0.00001, epochs: 5, eta_min: 1e-7}
```

## 结果

| 数据集 | U | S | H | ZS | Best |
|---|---:|---:|---:|---:|---:|
| CUB GZSL | 74.34 | 71.52 | 72.90 | 81.62 | ep30 |

## 结论

H=72.90，比 20+20+10 低 0.95。Stage1=25ep 已经开始 S 漂移。
