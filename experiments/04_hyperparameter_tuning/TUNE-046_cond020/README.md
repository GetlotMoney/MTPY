# TUNE-046 cond_text=0.020

类型: TUNE
状态: done

## 参数变化
- 基于 TUNE-024 配置
- conditional_text_ratio: 0.010 → 0.020

## 结果

| 数据集 | seed | U | S | H | ZS | Best |
|---|---:|---:|---:|---:|---:|---:|
| CUB GZSL | 5 | 70.08 | 77.29 | 73.51 | 81.55 | ep20 |

vs TUNE-024 baseline (H=73.85): **-0.34** (soft_negative)

## 结论

- cond_text 0.020 比 baseline 0.010 低 0.34
- cond_text 最优值确认：**0.008**（TUNE-043 H=73.93）
- Phase 2 cond_text 轴向搜索完成

## 最终结论

cond_text 轴向验证结果排序：
| 值 | H | 排名 |
|----|---|------|
| 0.008 | 73.93 | 🥇 最佳 |
| 0.010 | 73.85 | baseline |
| 0.015 | 73.69 | 第3 |
| 0.012 | 73.61 | 第4 |
| 0.020 | 73.51 | 第5 |
| 0.005 | 73.50 | 第6 |
