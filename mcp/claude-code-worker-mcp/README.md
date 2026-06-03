# Claude Code Worker MCP

Node.js MCP server for Codex Desktop that coordinates Claude review work through a project-local file queue.

Default mode is `manual`: Codex asks this MCP to write a review request into the project, and the currently running Claude Code session reviews it and writes the result back. This default mode does not start another model process and does not require a DeepSeek or Anthropic API key.

An optional `external` mode is still available for future use. In that mode the MCP starts `claude -p` as a separate worker process, and that mode does require a usable Claude/DeepSeek backend key.

## What It Does

- `start`: queue a review request in manual mode, or start `claude -p` in external mode.
- `get`: return `server_version`, job status, changed files, checks, decision, and optional diff.
- `tail`: read the queue log or review output tail.
- `wait`: poll until the review result appears or the timeout expires.
- `cancel`: mark a pending/running manual request as canceled, or kill a running external worker.
- `doctor`: check Node, Git, Claude CLI availability, queue paths, and key status.
- `setup`: print the Codex Desktop MCP config snippet.

## Install

```powershell
cd C:\Users\Administrator\Desktop\项目\DVSR\mcp\claude-code-worker-mcp
npm install
npm run build
npm run smoke
```

## Codex Desktop Config

Add this to `%USERPROFILE%\.codex\config.toml` after building:

```toml
[mcp_servers.claude_code_worker]
type = "stdio"
command = "node"
args = ["C:\\Users\\Administrator\\Desktop\\项目\\DVSR\\mcp\\claude-code-worker-mcp\\dist\\index.js"]
cwd = "C:\\Users\\Administrator\\Desktop\\项目\\DVSR"
startup_timeout_sec = 40
tool_timeout_sec = 1800
```

Restart Codex Desktop after adding the MCP server.

## Manual Mode Flow

Codex calls `start`:

```json
{
  "experiment_id": "EXP-000",
  "round": 1,
  "max_rounds": 3,
  "mode": "manual",
  "packet_path": "experiments/EXP-000.review-packet.md",
  "output_path": "experiments/EXP-000.claude-review.md",
  "cwd": "C:\\Users\\Administrator\\Desktop\\项目\\DVSR",
  "allowed_dirs": ["C:\\Users\\Administrator\\Desktop\\项目\\DVSR"]
}
```

The MCP writes:

```text
experiments/.agent-queue/inbox/<job_id>.request.json
experiments/.agent-queue/inbox/<job_id>.review-packet.md
experiments/.agent-queue/state/<job_id>.json
experiments/.agent-queue/logs/<job_id>.log
```

The current Claude Code session reviews the request and writes Markdown to `output_path`:

```markdown
Decision: ACCEPTED

Findings:
- No blocking issues found.
```

or:

```markdown
Decision: REJECTED

Findings:
- The experiment changes more than one variable.
```

Codex then calls `get` or `wait`. The MCP parses `Decision: ACCEPTED` or `Decision: REJECTED` from `output_path` and updates the job state.

## External Mode

External mode is optional:

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

Only this mode needs:

```powershell
$env:ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"
$env:DEEPSEEK_API_KEY = "sk-..."
```

## Safety Defaults

- Default mode is `manual`.
- Default maximum review-repair rounds is 3.
- No API key is required for default mode.
- No `bypassPermissions` by default.
- External mode uses `--permission-mode plan`.
- External mode disallows Edit / Write / MultiEdit.
- Paths must stay inside `allowed_dirs`.
- Review rounds greater than `max_rounds` return `blocked` with reason `exceeded max review rounds`.

## References

- MCP TypeScript SDK: https://github.com/modelcontextprotocol/typescript-sdk
- Claude Code CLI reference: https://docs.claude.com/en/docs/claude-code/cli-reference
- Codex config reference: https://developers.openai.com/codex/config-reference
- DeepSeek Anthropic API compatibility: https://api-docs.deepseek.com/guides/anthropic_api
