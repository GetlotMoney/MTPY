# TUNE-037 训练策略消融: 两段 30+20

类型: TUNE (策略消融)
状态: done

## 策略

```yaml
lr_stages:
  - {lr: 0.001,  epochs: 30, eta_min: 1e-5}
  - {lr: 0.0001, epochs: 20, eta_min: 1e-6}
```

## 结果

| 数据集 | U | S | H | ZS | Best |
|---|---:|---:|---:|---:|---:|
| CUB GZSL | 71.62 | 72.01 | 71.81 | 81.44 | ep9 |

## 结论

H=71.81。两段同样 Stage1 过长崩 S。去掉 Stage3 无帮助。
