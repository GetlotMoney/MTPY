# TUNE-024 组合调参: topo=0.10 + cond_text=0.01 + jepa_neg=0.01

类型: TUNE
状态: done (SGDR 20+20+10)

## 策略

```yaml
lr_stages: 20+20+10 (SGDR, eta_min > 0)
```

## 结果

| 数据集 | seed | U | S | H | ZS | Best |
|---|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 71.46 | 76.40 | 73.85 | 81.61 | ep41 |

## 历史

- 旧作弊版 (restart_from_best): H=74.09，作废
- SGDR 90ep: H=72.29，Stage1过长崩S
- **SGDR 20+20+10: H=73.85，当前干净基线**
