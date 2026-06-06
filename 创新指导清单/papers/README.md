# 论文 PDF 存放规则

这个目录用于保存创新树引用的论文 PDF。

命名规则：

- 优先使用原论文标题作为文件名。
- 文件名可以保留英文、空格和冒号的安全替代字符。
- 不要使用 `paper1.pdf`、`new.pdf`、`tmp.pdf` 这类无法追溯的名字。

示例：

```text
TPR Topology-Preserving Reservoirs for Generalized Zero-Shot Learning.pdf
Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model.pdf
Mutually Causal Semantic Distillation Network for Zero-Shot Learning.pdf
```

实验规则：

- 有论文来源的创意，写代码前优先读取本地 PDF。
- 本地没有 PDF 时，Codex 先尝试查找论文。
- 找不到、下载不了、打不开或无法读取时，Codex 必须汇报。
- 为了保证实验流程不断，可以继续做保守复现，但实验 README 和 review packet 必须明确标记：`未参考论文原文`。
