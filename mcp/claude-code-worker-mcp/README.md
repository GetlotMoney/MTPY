# Claude Code Worker MCP

Node.js MCP server for Codex Desktop that runs Claude Code as an asynchronous worker. It is intended for the Claude review gate in `experiment-agent-workflow`, while letting Claude Code use DeepSeek's Anthropic-compatible API backend to reduce Codex main-thread token use.

## What It Does

- Starts Claude Code workers asynchronously with `start`.
- Returns `server_version`, job status, changed files, checks, and optional diff with `get`.
- Streams worker logs through `tail`.
- Waits for long reasoning without killing it through `wait`.
- Cancels a stuck worker with `cancel`.
- Provides `setup` and `doctor` helpers.
- Defaults to `--permission-mode plan`; it does not enable `bypassPermissions` by default.
- Restricts file paths through `allowed_dirs` and project-scoped paths.

## Install

```powershell
cd C:\Users\Administrator\Desktop\项目\DVSR\mcp\claude-code-worker-mcp
npm install
npm run build
npm run smoke
```

## DeepSeek Backend

DeepSeek documents an Anthropic-compatible endpoint:

```powershell
$env:ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"
$env:DEEPSEEK_API_KEY = "sk-..."
```

The MCP maps `DEEPSEEK_API_KEY` to `ANTHROPIC_API_KEY` for the spawned Claude Code worker unless `ANTHROPIC_API_KEY` is already set.

Default model:

```text
deepseek-v4-pro
```

Override with:

```powershell
$env:CLAUDE_WORKER_MODEL = "deepseek-v4-pro"
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

[mcp_servers.claude_code_worker.env]
ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"
CLAUDE_WORKER_MODEL = "deepseek-v4-pro"
CLAUDE_WORKER_API_KEY_ENV = "DEEPSEEK_API_KEY"
```

Do not commit real API keys. Set `DEEPSEEK_API_KEY` in your user environment or shell instead.

Restart Codex Desktop after adding the MCP server.

## Tool Examples

Start a dry-run worker:

```json
{
  "experiment_id": "EXP-000",
  "round": 1,
  "packet_path": "experiments/EXP-000.review-packet.md",
  "output_path": "experiments/EXP-000.claude-review.md",
  "cwd": "C:\\Users\\Administrator\\Desktop\\项目\\DVSR",
  "allowed_dirs": ["C:\\Users\\Administrator\\Desktop\\项目\\DVSR"],
  "dry_run": true
}
```

Start a real review:

```json
{
  "experiment_id": "EXP-000",
  "round": 1,
  "max_rounds": 10,
  "packet_path": "experiments/EXP-000.review-packet.md",
  "output_path": "experiments/EXP-000.claude-review.md",
  "cwd": "C:\\Users\\Administrator\\Desktop\\项目\\DVSR",
  "allowed_dirs": ["C:\\Users\\Administrator\\Desktop\\项目\\DVSR"],
  "timeout_seconds": 900
}
```

Then poll:

```json
{
  "job_id": "EXP-000.round-1.xxxxxxxx",
  "cwd": "C:\\Users\\Administrator\\Desktop\\项目\\DVSR",
  "include_diff": true
}
```

## Status Files

Worker state and logs are written under:

```text
experiments/.agent-state/
```

Review output is written to the `output_path` you pass, usually:

```text
experiments/EXP-XXX.claude-review.md
```

## Safety Defaults

- No `bypassPermissions` by default.
- Default Claude Code permission mode is `plan`.
- Edit tools are disallowed.
- Paths must stay inside `allowed_dirs`.
- Review rounds greater than `max_rounds` return `blocked` with reason `exceeded max review rounds`.

## References

- MCP TypeScript SDK: https://github.com/modelcontextprotocol/typescript-sdk
- Claude Code CLI reference: https://docs.claude.com/en/docs/claude-code/cli-reference
- Codex config reference: https://developers.openai.com/codex/config-reference
- DeepSeek Anthropic API compatibility: https://api-docs.deepseek.com/guides/anthropic_api
