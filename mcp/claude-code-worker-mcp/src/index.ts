#!/usr/bin/env node

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { randomUUID } from "node:crypto";
import { promises as fs } from "node:fs";
import path from "node:path";
import { z } from "zod";

const SERVER_VERSION = "0.1.0";
const DEFAULT_BASE_URL = "https://api.deepseek.com/anthropic";
const DEFAULT_MODEL = "deepseek-v4-pro";
const DEFAULT_MAX_ROUNDS = 10;
const DEFAULT_TIMEOUT_SECONDS = 900;
const DEFAULT_TAIL_BYTES = 64 * 1024;

type JobStatus = "pending" | "running" | "accepted" | "rejected" | "failed" | "timeout" | "canceled" | "blocked";

type JobState = {
  server_version: string;
  job_id: string;
  experiment_id: string;
  round: number;
  max_rounds: number;
  status: JobStatus;
  cwd: string;
  packet_path: string;
  output_path: string;
  log_path: string;
  state_path: string;
  pid: number | null;
  model: string;
  permission_mode: string;
  started_at: string;
  updated_at: string;
  completed_at: string | null;
  exit_code: number | null;
  decision: "ACCEPTED" | "REJECTED" | null;
  reason: string | null;
  changed_files: string[];
  checks: Record<string, unknown>;
};

type JobRecord = {
  state: JobState;
  child?: ChildProcessWithoutNullStreams;
  stdout: string;
  stderr: string;
};

const jobs = new Map<string, JobRecord>();

const server = new McpServer({
  name: "claude-code-worker-mcp",
  version: SERVER_VERSION
});

function textResponse(data: Record<string, unknown>) {
  return {
    content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }],
    structuredContent: data
  };
}

function nowIso() {
  return new Date().toISOString();
}

function normalizeCwd(cwd?: string) {
  return path.resolve(cwd || process.cwd());
}

function asArray(value: string[] | undefined, fallback: string[]) {
  return value && value.length > 0 ? value : fallback;
}

function resolveScopedPath(cwd: string, inputPath: string, allowedDirs: string[]) {
  const resolved = path.resolve(cwd, inputPath);
  const allowed = allowedDirs.map((dir) => path.resolve(cwd, dir));
  const ok = allowed.some((dir) => resolved === dir || resolved.startsWith(dir + path.sep));
  if (!ok) {
    throw new Error(`Path is outside allowed_dirs: ${inputPath}`);
  }
  return resolved;
}

async function ensureParent(filePath: string) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
}

async function appendFileSafe(filePath: string, text: string) {
  await ensureParent(filePath);
  await fs.appendFile(filePath, text, "utf8");
}

async function writeJson(filePath: string, data: unknown) {
  await ensureParent(filePath);
  await fs.writeFile(filePath, JSON.stringify(data, null, 2), "utf8");
}

async function readJson<T>(filePath: string): Promise<T | null> {
  try {
    return JSON.parse(await fs.readFile(filePath, "utf8")) as T;
  } catch {
    return null;
  }
}

async function runCommand(command: string, args: string[], cwd: string, timeoutMs = 20_000) {
  return new Promise<{ exitCode: number | null; stdout: string; stderr: string }>((resolve) => {
    const child = spawn(command, args, { cwd, shell: process.platform === "win32" });
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill();
    }, timeoutMs);
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("close", (exitCode) => {
      clearTimeout(timer);
      resolve({ exitCode, stdout, stderr });
    });
  });
}

async function getChangedFiles(cwd: string) {
  const result = await runCommand("git", ["status", "--short"], cwd);
  if (result.exitCode !== 0) return [];
  return result.stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

async function getDiff(cwd: string, includeDiff: boolean) {
  if (!includeDiff) return "";
  const result = await runCommand("git", ["diff", "--stat"], cwd);
  return result.exitCode === 0 ? result.stdout.trim() : result.stderr.trim();
}

function parseDecision(text: string): "ACCEPTED" | "REJECTED" | null {
  const match = text.match(/^\s*Decision\s*:\s*(ACCEPTED|REJECTED)\b/im) || text.match(/\b(ACCEPTED|REJECTED)\b/);
  return match ? (match[1] as "ACCEPTED" | "REJECTED") : null;
}

function extractClaudeText(stdout: string) {
  const parts: string[] = [];
  for (const line of stdout.split(/\r?\n/)) {
    if (!line.trim()) continue;
    try {
      const event = JSON.parse(line);
      if (typeof event.result === "string") parts.push(event.result);
      if (typeof event.message?.content === "string") parts.push(event.message.content);
      if (Array.isArray(event.message?.content)) {
        for (const item of event.message.content) {
          if (typeof item?.text === "string") parts.push(item.text);
        }
      }
      if (typeof event.delta?.text === "string") parts.push(event.delta.text);
      if (typeof event.content_block?.text === "string") parts.push(event.content_block.text);
      if (typeof event.content_block_delta?.delta?.text === "string") parts.push(event.content_block_delta.delta.text);
    } catch {
      if (!line.startsWith("{")) parts.push(line);
    }
  }
  return parts.join("\n").trim() || stdout.trim();
}

function buildReviewPrompt(args: {
  experiment_id: string;
  round: number;
  output_path: string;
  changed_files: string[];
}) {
  return [
    "You are the Claude Code worker for an experiment-agent-workflow review gate.",
    "You must review only. Do not edit files. Do not run long experiments.",
    "Return a concise Markdown review that begins with exactly one of:",
    "Decision: ACCEPTED",
    "Decision: REJECTED",
    "",
    "Reject for correctness bugs, data leakage, invalid metrics, reproducibility gaps, unrelated changes, unsafe permissions, or missing evidence.",
    "Accept only when the implementation is safe to run for this experiment.",
    "",
    `Experiment: ${args.experiment_id}`,
    `Review round: ${args.round}`,
    `The MCP server will write your final Markdown to: ${args.output_path}`,
    "",
    "Changed files visible before review:",
    args.changed_files.length > 0 ? args.changed_files.join("\n") : "(none reported)",
    "",
    "The review packet is provided on stdin. Base your review on it."
  ].join("\n");
}

async function finishJob(job: JobRecord, status: JobStatus, reason: string | null, exitCode: number | null) {
  const text = extractClaudeText(job.stdout);
  const decision = parseDecision(text);
  const finalStatus: JobStatus =
    status === "failed" || status === "timeout" || status === "canceled"
      ? status
      : decision === "ACCEPTED"
        ? "accepted"
        : decision === "REJECTED"
          ? "rejected"
          : "failed";

  job.state.status = finalStatus;
  job.state.reason = reason || (decision ? null : "Claude output did not contain Decision: ACCEPTED or Decision: REJECTED");
  job.state.exit_code = exitCode;
  job.state.decision = decision;
  job.state.completed_at = nowIso();
  job.state.updated_at = nowIso();
  job.state.changed_files = await getChangedFiles(job.state.cwd);
  job.state.checks = {
    claude_exit_code: exitCode,
    stdout_bytes: Buffer.byteLength(job.stdout),
    stderr_bytes: Buffer.byteLength(job.stderr),
    output_has_decision: decision !== null
  };

  const reviewMarkdown = text || [
    "Decision: REJECTED",
    "",
    "Findings:",
    "- Claude Code produced no parseable review output."
  ].join("\n");

  await ensureParent(job.state.output_path);
  await fs.writeFile(job.state.output_path, reviewMarkdown.endsWith("\n") ? reviewMarkdown : reviewMarkdown + "\n", "utf8");
  await writeJson(job.state.state_path, job.state);
}

const startSchema = z.object({
  experiment_id: z.string().min(1),
  round: z.number().int().positive().default(1),
  max_rounds: z.number().int().positive().default(DEFAULT_MAX_ROUNDS),
  cwd: z.string().optional(),
  packet_path: z.string().min(1),
  output_path: z.string().min(1),
  allowed_dirs: z.array(z.string()).optional(),
  timeout_seconds: z.number().int().positive().default(DEFAULT_TIMEOUT_SECONDS),
  model: z.string().default(process.env.CLAUDE_WORKER_MODEL || DEFAULT_MODEL),
  claude_command: z.string().default(process.env.CLAUDE_WORKER_COMMAND || "claude"),
  permission_mode: z.enum(["plan", "default", "dontAsk", "auto", "acceptEdits"]).default("plan"),
  anthropic_base_url: z.string().default(process.env.ANTHROPIC_BASE_URL || DEFAULT_BASE_URL),
  anthropic_api_key_env: z.string().default(process.env.CLAUDE_WORKER_API_KEY_ENV || "DEEPSEEK_API_KEY"),
  bare: z.boolean().default(true),
  verbose: z.boolean().default(true),
  max_budget_usd: z.number().positive().optional(),
  dry_run: z.boolean().default(false)
});

server.registerTool(
  "start",
  {
    title: "Start Claude Code Worker",
    description: "Start an async Claude Code review worker for an experiment packet.",
    inputSchema: startSchema
  },
  async (input) => {
    const args = startSchema.parse(input);
    if (args.round > args.max_rounds) {
      return textResponse({
        server_version: SERVER_VERSION,
        status: "blocked",
        reason: "exceeded max review rounds",
        round: args.round,
        max_rounds: args.max_rounds
      });
    }

    const cwd = normalizeCwd(args.cwd);
    const allowedDirs = asArray(args.allowed_dirs, [cwd]);
    const packetPath = resolveScopedPath(cwd, args.packet_path, allowedDirs);
    const outputPath = resolveScopedPath(cwd, args.output_path, allowedDirs);
    const stateDir = resolveScopedPath(cwd, "experiments/.agent-state", allowedDirs);
    const jobId = `${args.experiment_id}.round-${args.round}.${randomUUID().slice(0, 8)}`;
    const statePath = path.join(stateDir, `${jobId}.json`);
    const logPath = path.join(stateDir, `${jobId}.log`);
    const changedFiles = await getChangedFiles(cwd);
    const packet = await fs.readFile(packetPath, "utf8");

    const state: JobState = {
      server_version: SERVER_VERSION,
      job_id: jobId,
      experiment_id: args.experiment_id,
      round: args.round,
      max_rounds: args.max_rounds,
      status: args.dry_run ? "pending" : "running",
      cwd,
      packet_path: packetPath,
      output_path: outputPath,
      log_path: logPath,
      state_path: statePath,
      pid: null,
      model: args.model,
      permission_mode: args.permission_mode,
      started_at: nowIso(),
      updated_at: nowIso(),
      completed_at: null,
      exit_code: null,
      decision: null,
      reason: args.dry_run ? "dry_run" : null,
      changed_files: changedFiles,
      checks: {
        packet_bytes: Buffer.byteLength(packet),
        deepseek_base_url: args.anthropic_base_url,
        dry_run: args.dry_run
      }
    };

    await writeJson(statePath, state);

    const prompt = buildReviewPrompt({
      experiment_id: args.experiment_id,
      round: args.round,
      output_path: outputPath,
      changed_files: changedFiles
    });

    const cliArgs = ["-p", "--output-format", "stream-json", "--permission-mode", args.permission_mode, "--model", args.model];
    if (args.verbose) cliArgs.push("--verbose");
    if (args.bare) cliArgs.push("--bare");
    cliArgs.push("--disallowedTools", "Edit,Write,MultiEdit,NotebookEdit");
    if (args.max_budget_usd) cliArgs.push("--max-budget-usd", String(args.max_budget_usd));
    cliArgs.push(prompt);

    if (args.dry_run) {
      return textResponse({ ...state, command: args.claude_command, args: cliArgs });
    }

    const env: NodeJS.ProcessEnv = { ...process.env, ANTHROPIC_BASE_URL: args.anthropic_base_url };
    const apiKey = process.env.ANTHROPIC_API_KEY || process.env[args.anthropic_api_key_env];
    if (apiKey) env.ANTHROPIC_API_KEY = apiKey;

    const child = spawn(args.claude_command, cliArgs, {
      cwd,
      env,
      shell: process.platform === "win32"
    });

    state.pid = child.pid ?? null;
    state.updated_at = nowIso();
    await writeJson(statePath, state);

    const job: JobRecord = { state, child, stdout: "", stderr: "" };
    jobs.set(jobId, job);

    const timeout = setTimeout(() => {
      if (job.state.status === "running") {
        job.state.status = "timeout";
        job.state.reason = "timeout";
        child.kill();
      }
    }, args.timeout_seconds * 1000);

    child.stdin.write(packet);
    child.stdin.end();

    child.stdout.on("data", async (chunk) => {
      const text = chunk.toString();
      job.stdout += text;
      await appendFileSafe(logPath, text);
    });

    child.stderr.on("data", async (chunk) => {
      const text = chunk.toString();
      job.stderr += text;
      await appendFileSafe(logPath, `\n[stderr]\n${text}`);
    });

    child.on("close", async (exitCode) => {
      clearTimeout(timeout);
      const status = job.state.status === "timeout" || job.state.status === "canceled" ? job.state.status : exitCode === 0 ? "running" : "failed";
      await finishJob(job, status, status === "failed" ? "claude process failed" : job.state.reason, exitCode);
    });

    return textResponse(state);
  }
);

const jobLookupSchema = z.object({
  job_id: z.string().min(1),
  cwd: z.string().optional(),
  include_diff: z.boolean().default(false)
});

async function loadState(jobId: string, cwd?: string) {
  const live = jobs.get(jobId);
  if (live) return live.state;
  const root = normalizeCwd(cwd);
  const stateDir = path.join(root, "experiments", ".agent-state");
  const statePath = path.join(stateDir, `${jobId}.json`);
  return await readJson<JobState>(statePath);
}

server.registerTool(
  "get",
  {
    title: "Get Worker Status",
    description: "Get a Claude worker job status, changed files, checks, and optional git diff stat.",
    inputSchema: jobLookupSchema
  },
  async (input) => {
    const args = jobLookupSchema.parse(input);
    const state = await loadState(args.job_id, args.cwd);
    if (!state) return textResponse({ server_version: SERVER_VERSION, status: "failed", reason: "job not found" });
    return textResponse({
      ...state,
      changed_files: await getChangedFiles(state.cwd),
      diff: await getDiff(state.cwd, args.include_diff)
    });
  }
);

server.registerTool(
  "tail",
  {
    title: "Tail Worker Log",
    description: "Read the end of a Claude worker stream-json log.",
    inputSchema: z.object({
      job_id: z.string().min(1),
      cwd: z.string().optional(),
      bytes: z.number().int().positive().default(DEFAULT_TAIL_BYTES)
    })
  },
  async (input) => {
    const args = z.object({
      job_id: z.string().min(1),
      cwd: z.string().optional(),
      bytes: z.number().int().positive().default(DEFAULT_TAIL_BYTES)
    }).parse(input);
    const state = await loadState(args.job_id, args.cwd);
    if (!state) return textResponse({ server_version: SERVER_VERSION, status: "failed", reason: "job not found" });
    let text = "";
    try {
      const buffer = await fs.readFile(state.log_path);
      text = buffer.subarray(Math.max(0, buffer.length - args.bytes)).toString("utf8");
    } catch {
      text = "";
    }
    return textResponse({ server_version: SERVER_VERSION, job_id: args.job_id, log_path: state.log_path, tail: text });
  }
);

server.registerTool(
  "wait",
  {
    title: "Wait For Worker",
    description: "Wait for a Claude worker to finish without killing long-running reasoning.",
    inputSchema: z.object({
      job_id: z.string().min(1),
      cwd: z.string().optional(),
      timeout_seconds: z.number().int().positive().default(60),
      poll_seconds: z.number().positive().default(2)
    })
  },
  async (input) => {
    const args = z.object({
      job_id: z.string().min(1),
      cwd: z.string().optional(),
      timeout_seconds: z.number().int().positive().default(60),
      poll_seconds: z.number().positive().default(2)
    }).parse(input);
    const deadline = Date.now() + args.timeout_seconds * 1000;
    let state = await loadState(args.job_id, args.cwd);
    while (state && state.status === "running" && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, args.poll_seconds * 1000));
      state = await loadState(args.job_id, args.cwd);
    }
    if (!state) return textResponse({ server_version: SERVER_VERSION, status: "failed", reason: "job not found" });
    return textResponse({ ...state, changed_files: await getChangedFiles(state.cwd) });
  }
);

server.registerTool(
  "cancel",
  {
    title: "Cancel Worker",
    description: "Cancel a running Claude worker job.",
    inputSchema: z.object({ job_id: z.string().min(1), cwd: z.string().optional() })
  },
  async (input) => {
    const args = z.object({ job_id: z.string().min(1), cwd: z.string().optional() }).parse(input);
    const job = jobs.get(args.job_id);
    const state = job?.state || (await loadState(args.job_id, args.cwd));
    if (!state) return textResponse({ server_version: SERVER_VERSION, status: "failed", reason: "job not found" });
    if (job?.child && state.status === "running") {
      state.status = "canceled";
      state.reason = "canceled by caller";
      state.updated_at = nowIso();
      job.child.kill();
      await writeJson(state.state_path, state);
    }
    return textResponse(state);
  }
);

server.registerTool(
  "doctor",
  {
    title: "Doctor",
    description: "Check local Claude Code worker prerequisites.",
    inputSchema: z.object({ cwd: z.string().optional(), claude_command: z.string().default(process.env.CLAUDE_WORKER_COMMAND || "claude") })
  },
  async (input) => {
    const args = z.object({ cwd: z.string().optional(), claude_command: z.string().default(process.env.CLAUDE_WORKER_COMMAND || "claude") }).parse(input);
    const cwd = normalizeCwd(args.cwd);
    const nodeVersion = process.version;
    const claudeVersion = await runCommand(args.claude_command, ["-v"], cwd);
    const gitStatus = await runCommand("git", ["status", "--short"], cwd);
    const hasKey = Boolean(process.env.ANTHROPIC_API_KEY || process.env.DEEPSEEK_API_KEY);
    return textResponse({
      server_version: SERVER_VERSION,
      cwd,
      node_version: nodeVersion,
      claude_command: args.claude_command,
      claude_ok: claudeVersion.exitCode === 0,
      claude_version: claudeVersion.stdout.trim() || claudeVersion.stderr.trim(),
      git_ok: gitStatus.exitCode === 0,
      deepseek_or_anthropic_key_present: hasKey,
      recommended_base_url: DEFAULT_BASE_URL,
      recommended_model: DEFAULT_MODEL,
      checks: {
        safe_default_permission_mode: "plan",
        bypass_permissions_default: false,
        state_dir: path.join(cwd, "experiments", ".agent-state")
      }
    });
  }
);

server.registerTool(
  "setup",
  {
    title: "Setup Instructions",
    description: "Return Codex Desktop config and environment setup snippets for this MCP server.",
    inputSchema: z.object({ cwd: z.string().optional(), server_path: z.string().optional() })
  },
  async (input) => {
    const args = z.object({ cwd: z.string().optional(), server_path: z.string().optional() }).parse(input);
    const cwd = normalizeCwd(args.cwd);
    const serverPath = path.resolve(args.server_path || path.join(cwd, "mcp", "claude-code-worker-mcp", "dist", "index.js"));
    const snippet = [
      "[mcp_servers.claude_code_worker]",
      "type = \"stdio\"",
      "command = \"node\"",
      `args = [${JSON.stringify(serverPath)}]`,
      `cwd = ${JSON.stringify(cwd)}`,
      "startup_timeout_sec = 40",
      "tool_timeout_sec = 1800",
      "",
      "[mcp_servers.claude_code_worker.env]",
      `ANTHROPIC_BASE_URL = ${JSON.stringify(DEFAULT_BASE_URL)}`,
      "CLAUDE_WORKER_MODEL = \"deepseek-v4-pro\"",
      "CLAUDE_WORKER_API_KEY_ENV = \"DEEPSEEK_API_KEY\""
    ].join("\n");
    return textResponse({
      server_version: SERVER_VERSION,
      codex_config_path: "%USERPROFILE%\\.codex\\config.toml",
      codex_config_snippet: snippet,
      powershell_env_example: "$env:DEEPSEEK_API_KEY = \"sk-...\"",
      notes: [
        "Do not put real API keys in the repository.",
        "Restart Codex Desktop after adding the MCP server.",
        "Use doctor first, then start with dry_run=true for the first smoke test."
      ]
    });
  }
);

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
