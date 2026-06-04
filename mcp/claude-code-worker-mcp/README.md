# Claude Code 审查工作器 MCP

这是给 Codex Desktop 使用的 Node.js MCP 服务，用来把 `experiment-agent-workflow` 里的 Claude 审查步骤自动化。

默认模式是 `cli`：Codex 调用 MCP，MCP 在同一个项目目录里启动本机 `claude -p`，把审查包通过 stdin 交给 Claude CLI，Claude 完成固定三轮审查后退出，MCP 解析 `Overall Decision: ACCEPTED|REJECTED` 并把结果写回实验目录。

该默认模式使用你本机 Claude CLI 的 OAuth 登录状态，不需要 DeepSeek 或 Anthropic API key，也不依赖 Claude Code 桌面端窗口。它和 Codex 共享同一个项目工作目录，但不是同一个聊天上下文。

## 模式

- `cli`：默认模式。调用本机 `claude -p`，在 `cwd` 指定的项目目录执行审查。默认只开放 `Read` 工具，并禁止 `Bash`、`Edit`、`Write`、`MultiEdit`、`NotebookEdit`。
- `manual`：备用模式。MCP 只写入队列文件，由当前 Claude Code 会话人工读取 inbox、写回 outbox。
- `external`：后端 API 模式。通过 Anthropic 兼容接口启动独立 worker，需要 `DEEPSEEK_API_KEY` 或 `ANTHROPIC_API_KEY`。

## 功能

- `start`：启动审查任务。默认直接走 `cli`；也可显式指定 `manual` 或 `external`。
- `get`：读取任务状态、决策、变更文件、检查项和可选 diff。
- `tail`：读取 Claude CLI stream-json 日志尾部，方便看审查进度。
- `wait`：等待任务结束，直到 accepted、rejected、failed、timeout 或 canceled。
- `cancel`：取消运行中的任务。
- `doctor`：检查 Node、Git、Claude CLI、默认模式和密钥状态。
- `setup`：输出 Codex Desktop 的 MCP 配置片段。

## 安装

```powershell
cd C:\Users\Administrator\Desktop\项目\DVSR\mcp\claude-code-worker-mcp
npm install
npm run build
npm run smoke
npm run smoke:cli
```

`npm run smoke` 是快速检查，不会真实调用 Claude 审查；`npm run smoke:cli` 会使用 `fixtures/EXP-CLI-E2E.review-packet.md` 真的启动 Claude CLI，验证 stdin、后台轮询、stream-json 提取、output 写入和三轮决策解析。

## Codex Desktop 配置

构建完成后，把下面配置加入 `%USERPROFILE%\.codex\config.toml`：

```toml
[mcp_servers.claude_code_worker]
type = "stdio"
command = "node"
args = ["C:\\Users\\Administrator\\Desktop\\项目\\DVSR\\mcp\\claude-code-worker-mcp\\dist\\index.js"]
cwd = "C:\\Users\\Administrator\\Desktop\\项目\\DVSR"
startup_timeout_sec = 40
tool_timeout_sec = 1800
```

加入 MCP 服务后，重启 Codex Desktop。重启后可以让 Codex 调用 `claude_code_worker.doctor` 检查。

## CLI 模式流程

Codex 调用 `start` 时不传 `mode`，默认就是 `cli`：

```json
{
  "experiment_id": "EXP-000",
  "round": 1,
  "max_rounds": 3,
  "packet_path": "experiments/EXP-000.review-packet.md",
  "output_path": "experiments/EXP-000.claude-review.md",
  "cwd": "C:\\Users\\Administrator\\Desktop\\项目\\DVSR",
  "allowed_dirs": ["C:\\Users\\Administrator\\Desktop\\项目\\DVSR"]
}
```

MCP 会执行等价于下面的命令：

```powershell
claude -p --output-format stream-json --permission-mode plan --verbose --bare --add-dir <cwd> --tools=Read --disallowedTools=Bash,Edit,Write,MultiEdit,NotebookEdit
```

审查提示词和审查包会从 stdin 输入，不再把长 prompt 塞进命令行参数。Claude CLI 结束后，MCP 会把 Markdown 结果写到 `output_path`。

审查结果必须包含这四行：

```markdown
Pass 1 Decision: ACCEPTED
Pass 2 Decision: ACCEPTED
Pass 3 Decision: ACCEPTED
Overall Decision: ACCEPTED
```

只要任意一轮是 `REJECTED`，`Overall Decision` 就必须是 `REJECTED`。MCP 会机器校验 `Pass 1/2/3 Decision` 和 `Overall Decision` 是否同时存在且一致；如果缺失或不一致，状态会变成 `failed`。旧格式 `Decision: ACCEPTED|REJECTED` 只作为没有三轮字段时的兼容 fallback。

## 手动备用模式

只有当 CLI 不可用，或者你想让当前 Claude Code 桌面会话人工审查时，才使用 `manual`：

```json
{
  "mode": "manual",
  "experiment_id": "EXP-000",
  "round": 1,
  "max_rounds": 3,
  "packet_path": "experiments/EXP-000.review-packet.md",
  "output_path": "experiments/EXP-000.claude-review.md",
  "cwd": "C:\\Users\\Administrator\\Desktop\\项目\\DVSR",
  "allowed_dirs": ["C:\\Users\\Administrator\\Desktop\\项目\\DVSR"]
}
```

MCP 会写入：

```text
experiments/.agent-queue/inbox/<job_id>.request.json
experiments/.agent-queue/inbox/<job_id>.review-packet.md
experiments/.agent-queue/state/<job_id>.json
experiments/.agent-queue/logs/<job_id>.log
```

Claude Code 会话需要读取 inbox，并把同样格式的三轮审查结果写到 `output_path`。随后 Codex 调用 `get` 或 `wait` 读取结果。

## External 模式

`external` 保留给 API 后端：

```json
{
  "mode": "external",
  "experiment_id": "EXP-000",
  "round": 1,
  "packet_path": "experiments/EXP-000.review-packet.md",
  "output_path": "experiments/EXP-000.claude-review.md",
  "cwd": "C:\\Users\\Administrator\\Desktop\\项目\\DVSR",
  "allowed_dirs": ["C:\\Users\\Administrator\\Desktop\\项目\\DVSR"]
}
```

只有该模式需要：

```powershell
$env:ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"
$env:DEEPSEEK_API_KEY = "sk-..."
```

## 安全默认值

- 默认模式是 `cli`。
- 默认最大审查-修复轮数是 3。
- 默认模式不需要 API key。
- 默认不使用 `bypassPermissions`。
- 默认权限模式是 `plan`。
- CLI 默认只允许 Claude 读取文件，不允许修改文件或运行 Bash。
- 路径必须位于 `allowed_dirs` 内。
- 审查轮数超过 `max_rounds` 时返回 `blocked`，原因为 `exceeded max review rounds`。

## 使用范式

正式实验时，Codex 的流程是：

```text
检查 Git 状态
准备实验 checkpoint
按 backlog 选择实验
Codex 修改代码或配置
Codex 自审
MCP cli 模式调用 Claude CLI 做固定三轮审查
若 REJECTED，Codex 修复后进入下一轮，最多 3 轮
若 ACCEPTED，Codex 按实验 README 运行轻量或正式实验
复制规范日志到实验目录 logs/
更新实验 README、EXPERIMENT_REGISTRY.md、backlog.md
提交本地 Git checkpoint
```

## 参考

- MCP TypeScript SDK：https://github.com/modelcontextprotocol/typescript-sdk
- Claude Code CLI 参考：https://docs.claude.com/en/docs/claude-code/cli-reference
- Codex 配置参考：https://developers.openai.com/codex/config-reference
- DeepSeek Anthropic API 兼容说明：https://api-docs.deepseek.com/guides/anthropic_api
