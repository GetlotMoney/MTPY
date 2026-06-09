# Codex 自审与风险分级审查

## 审查门

实验运行前必须先选择范式。不同范式使用不同审查门：

| 范式 | 适用实验 | 运行前必须通过 |
|---|---|---|
| `TUNE-LITE` | 只改配置副本的调参实验；不改代码的复核实验 | Codex 自审: `ACCEPTED` |
| `STANDARD` | 消融、跨数据集、只改已有开关或配置的最终复核 | Codex 自审: `ACCEPTED`；Claude 单轮: `ACCEPTED` |
| `STRICT` | 新模块、组合模块、改 forward/loss/数据流/评估逻辑的实验 | Codex 自审: `ACCEPTED`；Claude 第 1/2/3 轮全部 `ACCEPTED` |

任意必需审查返回 `REJECTED`，都不能训练。非必需审查可以跳过，但必须在 README 或 review packet 中写明当前范式。

## Codex 自审

重点检查：

- diff 是否只服务当前实验。
- 新模块关闭后是否退回 baseline。
- 配置键默认值是否安全。
- 是否无意改变 seed、学习率、epoch、评估 bias 或训练日程。
- 实验 README 是否绑定创意树节点。
- review packet 是否包含来源复核、模块关系、冲突分析和运行计划。
- 日志命名、结果记录和创新树反馈是否有位置。

`TUNE-LITE` 额外检查：

- 是否只改实验目录里的 `config.yaml`。
- 是否只改一个主变量。
- 是否记录 old value 和 new value。
- 是否保持 seed、数据集、评估逻辑和 baseline 代码不变。
- 是否声明结果只是候选调参结果，不是最终多 seed 结论。

## Claude 单轮

`STANDARD` 范式使用 Claude 单轮。单轮必须同时检查：

- 配置/代码改动是否只服务当前实验。
- baseline 是否可回退。
- seed、训练日程、评估 bias 和数据源是否可比。
- 运行命令、日志路径、结果记录位置是否完整。

单轮输出必须包含：

- 决策：`ACCEPTED` 或 `REJECTED`。
- 关键发现。
- 是否允许开跑。

## Claude 固定三轮

| 轮次 | 范围 | 必须判断 |
|---|---|---|
| 第 1 轮 | 代码/配置正确性 | 改动最小性、开关默认关闭、baseline 可回退、静默失败风险。 |
| 第 2 轮 | 实验设计和可复现性 | 指标有效性、seed/config 可比性、数据泄漏、来源复核、冲突分析。 |
| 第 3 轮 | 最终运行计划 | 命令、环境、日志、产物、回滚安全性、是否允许开跑。 |

每一轮都必须记录：

- 轮次编号。
- 关注范围。
- 决策：`ACCEPTED` 或 `REJECTED`。
- 发现或理由。

## Claude 通道

只有 `STANDARD` 和 `STRICT` 需要 Claude。`TUNE-LITE` 默认不调用 Claude，除非用户明确要求或 Codex 自审发现风险。

优先级：

```text
claude_code_worker MCP
  ↓
claude -p CLI
  ↓
环境阻塞
```

CLI 模式必须在当前项目根目录执行，让 Claude 读取当前实验的 `review-packet.md`，写回：

```text
experiments/<group>/<EXP-ID>_<slug>/claude-review.md
```

用户主要查看该文件，而不是 MCP 原始日志。

## 审查修复上限

最多 10 个审查-修复轮次。超过后：

- 不运行实验。
- 标记 `blocked`。
- 原因写为 `exceeded max review rounds`。
- 记录已修复问题、剩余阻塞点和下一步建议。

