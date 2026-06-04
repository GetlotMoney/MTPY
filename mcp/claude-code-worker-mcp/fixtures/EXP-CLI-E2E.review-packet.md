# EXP-CLI-E2E MCP CLI 端到端协议测试

## 测试目标

这是一个最小端到端协议测试，不审查 MCP 源码质量，不审查 DVSR 实验设计。

本测试只验证以下链路是否跑通：

1. MCP `start` 默认使用 `cli` 模式。
2. MCP 能启动本机 `claude -p`。
3. Claude CLI 能从 stdin 收到本审查包。
4. Claude CLI 能输出固定三轮决策格式。
5. MCP 能从 stream-json 中提取文本。
6. MCP 能写入 `output_path`。
7. MCP 能把三轮一致的 `Overall Decision: ACCEPTED` 解析为 accepted。

## 判定规则

如果你能读到这个审查包，并且没有被要求修改文件、运行 Bash、运行训练或访问密钥，那么本协议测试应全部通过。

请不要读取项目文件，不要调用工具，不要审查当前 Git diff。这个包已经包含完整测试目标。

## 期望输出

请只输出简短 Markdown，并在靠前位置包含：

```text
Pass 1 Decision: ACCEPTED
Pass 2 Decision: ACCEPTED
Pass 3 Decision: ACCEPTED
Overall Decision: ACCEPTED
```

可以附加一句说明：MCP CLI 端到端协议测试已收到 stdin 审查包。
