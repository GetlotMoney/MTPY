# Codex 自审与 Claude 三轮审查

## 审查门

实验运行前必须通过：

```text
Codex 自审: ACCEPTED
Claude 第 1 轮: ACCEPTED
Claude 第 2 轮: ACCEPTED
Claude 第 3 轮: ACCEPTED
```

任意一轮 `REJECTED`，都不能训练。

## Codex 自审

重点检查：

- diff 是否只服务当前实验。
- 新模块关闭后是否退回 baseline。
- 配置键默认值是否安全。
- 是否无意改变 seed、学习率、epoch、评估 bias 或训练日程。
- 实验 README 是否绑定创意树节点。
- review packet 是否包含来源复核、模块关系、冲突分析和运行计划。
- 日志命名、结果记录和创新树反馈是否有位置。

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

