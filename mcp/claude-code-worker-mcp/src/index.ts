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
const DEFAULT_WORKER_MODE = "cli" as const;
const DEFAULT_MAX_ROUNDS = 3;
const DEFAULT_TIMEOUT_SECONDS = 900;
const DEFAULT_TAIL_BYTES = 64 * 1024;
const WORKER_MODES = ["cli", "manual", "external"] as const;
type WorkerMode = (typeof WORKER_MODES)[number];

type JobStatus = "pending" | "running" | "accepted" | "rejected" | "failed" | "timeout" | "canceled" | "blocked";

type JobState = {
  server_version: string;
  job_id: string;
  experiment_id: string;
  round: number;
  max_rounds: number;
  mode: WorkerMode;
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

async function removeFileIfExists(filePath: string) {
  try {
    await fs.unlink(filePath);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
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
  const result = await runCommand("git", ["status", "--short", "--", "."], cwd);
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
  const passDecisions = [1, 2, 3].map((pass) => {
    const match = text.match(new RegExp(`^\\s*Pass\\s+${pass}\\s+Decision\\s*:\\s*(ACCEPTED|REJECTED)\\b`, "im"));
    return match ? (match[1] as "ACCEPTED" | "REJECTED") : null;
  });
  const overallMatch = text.match(/^\s*Overall\s+Decision\s*:\s*(ACCEPTED|REJECTED)\b/im);

  if (overallMatch) {
    if (passDecisions.some((decision) => decision === null)) return null;
    const expected = passDecisions.every((decision) => decision === "ACCEPTED") ? "ACCEPTED" : "REJECTED";
    return overallMatch[1] === expected ? expected : null;
  }

  if (passDecisions.some((decision) => decision !== null)) return null;

  const legacyMatch = text.match(/^\s*Decision\s*:\s*(ACCEPTED|REJECTED)\b/im);
  return legacyMatch ? (legacyMatch[1] as "ACCEPTED" | "REJECTED") : null;
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
  packet_path: string;
  output_path: string;
  changed_files: string[];
}) {
  return [
    "你是 experiment-agent-workflow 的 Claude Code 审查工作器。",
    "只审查，不修改代码，不运行长实验。",
    "必须完成固定三轮审查，并在靠前位置返回以下四行：",
    "Pass 1 Decision: ACCEPTED 或 REJECTED",
    "Pass 2 Decision: ACCEPTED 或 REJECTED",
    "Pass 3 Decision: ACCEPTED 或 REJECTED",
    "Overall Decision: ACCEPTED 或 REJECTED",
    "",
    "只有三轮全部 ACCEPTED 时，Overall Decision 才能是 ACCEPTED。",
    "如果存在正确性 bug、数据泄漏、指标无效、可复现性缺口、无关改动、不安全权限或证据不足，拒绝对应轮次。",
    "",
    `实验：${args.experiment_id}`,
    `审查轮次：${args.round}`,
    `审查包路径：${args.packet_path}`,
    `MCP 会把你的最终 Markdown 写入：${args.output_path}`,
    "",
    "审查前可见的变更文件：",
    args.changed_files.length > 0 ? args.changed_files.join("\n") : "(none reported)",
    "",
    "审查包内容会通过 stdin 提供。优先基于审查包完成审查；只有证据不足时才读取项目文件或 Git 差异。"
  ].join("\n");
}

async function finishJob(job: JobRecord, status: JobStatus, reason: string | null, exitCode: number | null) {
  const text = extractClaudeText(job.stdout);
  const decision = parseDecision(text);
  const finalStatus: JobStatus =
    status === "timeout" || status === "canceled"
      ? status
      : decision === "REJECTED"
        ? "rejected"
      : decision === "ACCEPTED"
        ? status === "failed"
          ? "failed"
          : "accepted"
        : "failed";

  job.state.status = finalStatus;
  job.state.reason = finalStatus === "failed" ? reason || "Claude 输出没有包含可解析且一致的三轮审查决策" : null;
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

  const fallbackReviewMarkdown = [
    "Pass 1 Decision: REJECTED",
    "Pass 2 Decision: REJECTED",
    "Pass 3 Decision: REJECTED",
    "Overall Decision: REJECTED",
    "",
    "审查发现：",
    "- Claude Code 没有产生可解析的审查输出。"
  ].join("\n");
  const reviewMarkdown = decision ? text : fallbackReviewMarkdown;

  await ensureParent(job.state.output_path);
  await fs.writeFile(job.state.output_path, reviewMarkdown.endsWith("\n") ? reviewMarkdown : reviewMarkdown + "\n", "utf8");
  await writeJson(job.state.state_path, job.state);
}

async function refreshStateFromOutput(state: JobState) {
  if (state.status !== "pending" && state.status !== "running") return state;

  let output = "";
  try {
    const outputStat = await fs.stat(state.output_path);
    if (outputStat.mtimeMs + 1000 < new Date(state.started_at).getTime()) return state;
  } catch {
    return state;
  }

  try {
    output = await fs.readFile(state.output_path, "utf8");
  } catch {
    return state;
  }

  const decision = parseDecision(output);
  state.updated_at = nowIso();
  state.changed_files = await getChangedFiles(state.cwd);
  state.checks = {
    ...state.checks,
    output_bytes: Buffer.byteLength(output),
    output_has_decision: decision !== null
  };

  if (decision === "ACCEPTED") {
    state.status = "accepted";
    state.decision = "ACCEPTED";
    state.reason = null;
    state.completed_at = nowIso();
  } else if (decision === "REJECTED") {
    state.status = "rejected";
    state.decision = "REJECTED";
    state.reason = null;
    state.completed_at = nowIso();
  } else {
    state.status = "failed";
    state.reason = "审查输出已存在，但没有包含可解析且一致的三轮审查决策";
    state.completed_at = nowIso();
  }

  await writeJson(state.state_path, state);
  return state;
}

const startSchema = z.object({
  experiment_id: z.string().min(1),
  round: z.number().int().positive().default(1),
  max_rounds: z.number().int().positive().default(DEFAULT_MAX_ROUNDS),
  mode: z.enum(WORKER_MODES).default(DEFAULT_WORKER_MODE),
  cwd: z.string().optional(),
  packet_path: z.string().min(1),
  output_path: z.string().min(1),
  allowed_dirs: z.array(z.string()).optional(),
  timeout_seconds: z.number().int().positive().default(DEFAULT_TIMEOUT_SECONDS),
  model: z.string().optional(),
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
    title: "启动 Claude Code 审查工作器",
    description: "为实验审查包启动异步 Claude Code 审查任务。",
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
    if (!args.dry_run) await removeFileIfExists(outputPath);
    const stateDir =
      args.mode === "manual"
        ? resolveScopedPath(cwd, "experiments/.agent-queue/state", allowedDirs)
        : resolveScopedPath(cwd, "experiments/.agent-state", allowedDirs);
    const queueRoot = resolveScopedPath(cwd, "experiments/.agent-queue", allowedDirs);
    const jobId = `${args.experiment_id}.round-${args.round}.${randomUUID().slice(0, 8)}`;
    const statePath = path.join(stateDir, `${jobId}.json`);
    const logPath =
      args.mode === "manual"
        ? path.join(queueRoot, "logs", `${jobId}.log`)
        : path.join(stateDir, `${jobId}.log`);
    const requestPath = path.join(queueRoot, "inbox", `${jobId}.request.json`);
    const packetCopyPath = path.join(queueRoot, "inbox", `${jobId}.review-packet.md`);
    const changedFiles = await getChangedFiles(cwd);
    const packet = await fs.readFile(packetPath, "utf8");

    const state: JobState = {
      server_version: SERVER_VERSION,
      job_id: jobId,
      experiment_id: args.experiment_id,
      round: args.round,
      max_rounds: args.max_rounds,
      mode: args.mode,
      status: args.mode === "manual" || args.dry_run ? "pending" : "running",
      cwd,
      packet_path: packetPath,
      output_path: outputPath,
      log_path: logPath,
      state_path: statePath,
      pid: null,
      model: args.model || (args.mode === "external" ? process.env.CLAUDE_WORKER_MODEL || DEFAULT_MODEL : process.env.CLAUDE_WORKER_CLI_MODEL || "default"),
      permission_mode: args.permission_mode,
      started_at: nowIso(),
      updated_at: nowIso(),
      completed_at: null,
      exit_code: null,
      decision: null,
      reason: args.dry_run ? "dry_run" : null,
      changed_files: changedFiles,
      checks: {
        mode: args.mode,
        packet_bytes: Buffer.byteLength(packet),
        dry_run: args.dry_run,
        ...(args.mode === "manual"
          ? {
              request_path: requestPath,
              packet_copy_path: packetCopyPath,
              outbox_path: outputPath
            }
          : {
              cli_mode: args.mode === "cli",
              external_mode: args.mode === "external",
              ...(args.mode === "external" ? { deepseek_base_url: args.anthropic_base_url } : {})
            })
      }
    };

    await writeJson(statePath, state);

    if (args.mode === "manual") {
      const request = {
        server_version: SERVER_VERSION,
        job_id: jobId,
        experiment_id: args.experiment_id,
        round: args.round,
        max_rounds: args.max_rounds,
        status: state.status,
        cwd,
        packet_path: packetPath,
        packet_copy_path: packetCopyPath,
        output_path: outputPath,
        state_path: statePath,
        changed_files: changedFiles,
        instructions: [
          "当前 Claude Code 会话需要审查这个请求。",
          "除非用户明确要求，否则不要编辑文件。",
          "请把 Markdown 审查结果写入 output_path，并在靠前位置包含这四行：",
          "Pass 1 Decision: ACCEPTED 或 REJECTED",
          "Pass 2 Decision: ACCEPTED 或 REJECTED",
          "Pass 3 Decision: ACCEPTED 或 REJECTED",
          "Overall Decision: ACCEPTED 或 REJECTED"
        ]
      };

      if (!args.dry_run) {
        await ensureParent(packetCopyPath);
        await fs.writeFile(packetCopyPath, packet, "utf8");
        await writeJson(requestPath, request);
        await appendFileSafe(logPath, `manual request queued at ${nowIso()}\nrequest: ${requestPath}\noutput: ${outputPath}\n`);
      }

      return textResponse({ ...state, request_path: requestPath, packet_copy_path: packetCopyPath });
    }

    const prompt = buildReviewPrompt({
      experiment_id: args.experiment_id,
      round: args.round,
      packet_path: packetPath,
      output_path: outputPath,
      changed_files: changedFiles
    });

    const cliArgs = ["-p", "--output-format", "stream-json", "--permission-mode", args.permission_mode];
    const effectiveModel = args.model || (args.mode === "external" ? process.env.CLAUDE_WORKER_MODEL || DEFAULT_MODEL : process.env.CLAUDE_WORKER_CLI_MODEL);
    if (effectiveModel) cliArgs.push("--model", effectiveModel);
    if (args.verbose) cliArgs.push("--verbose");
    if (args.bare) cliArgs.push("--bare");
    if (args.mode === "cli") cliArgs.push("--add-dir", cwd);
    cliArgs.push("--tools=Read");
    cliArgs.push("--disallowedTools=Bash,Edit,Write,MultiEdit,NotebookEdit");
    if (args.max_budget_usd) cliArgs.push("--max-budget-usd", String(args.max_budget_usd));

    const stdinPayload = [
      prompt,
      "",
      "--- 审查包内容开始 ---",
      packet,
      "--- 审查包内容结束 ---"
    ].join("\n");

    if (args.dry_run) {
      return textResponse({ ...state, command: args.claude_command, args: cliArgs, stdin_bytes: Buffer.byteLength(stdinPayload) });
    }

    const env: NodeJS.ProcessEnv = { ...process.env };
    if (args.mode === "external") {
      env.ANTHROPIC_BASE_URL = args.anthropic_base_url;
      const apiKey = process.env.ANTHROPIC_API_KEY || process.env[args.anthropic_api_key_env];
      if (apiKey) env.ANTHROPIC_API_KEY = apiKey;
    }

    const job: JobRecord = { state, stdout: "", stderr: "" };
    jobs.set(jobId, job);

    setImmediate(() => {
      void (async () => {
        let child: ChildProcessWithoutNullStreams;
        try {
          child = spawn(args.claude_command, cliArgs, {
            cwd,
            env,
            shell: process.platform === "win32"
          });
        } catch (error) {
          job.state.status = "failed";
          job.state.reason = error instanceof Error ? error.message : String(error);
          job.state.updated_at = nowIso();
          job.state.completed_at = nowIso();
          await writeJson(statePath, job.state);
          return;
        }

        job.child = child;
        state.pid = child.pid ?? null;
        state.updated_at = nowIso();
        await writeJson(statePath, state);

        const timeout = setTimeout(() => {
          if (job.state.status === "running") {
            job.state.status = "timeout";
            job.state.reason = "timeout";
            child.kill();
          }
        }, args.timeout_seconds * 1000);

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

        child.stdin.write(stdinPayload);
        child.stdin.end();
      })();
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
  const candidates = [
    path.join(root, "experiments", ".agent-queue", "state", `${jobId}.json`),
    path.join(root, "experiments", ".agent-state", `${jobId}.json`)
  ];
  for (const statePath of candidates) {
    const state = await readJson<JobState>(statePath);
    if (state) return state;
  }
  return null;
}

server.registerTool(
  "get",
  {
    title: "读取审查任务状态",
    description: "读取 Claude 审查任务状态、变更文件、检查项和可选 Git diff 统计。",
    inputSchema: jobLookupSchema
  },
  async (input) => {
    const args = jobLookupSchema.parse(input);
    const loaded = await loadState(args.job_id, args.cwd);
    const state = loaded ? await refreshStateFromOutput(loaded) : null;
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
    title: "读取审查日志尾部",
    description: "读取 Claude 审查工作器 stream-json 日志的末尾内容。",
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
    if (!text) {
      try {
        text = await fs.readFile(state.output_path, "utf8");
      } catch {
        text = "";
      }
    }
    return textResponse({ server_version: SERVER_VERSION, job_id: args.job_id, log_path: state.log_path, tail: text });
  }
);

server.registerTool(
  "wait",
  {
    title: "等待审查任务结束",
    description: "等待 Claude 审查任务结束，不中断正在进行的推理。",
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
    let loaded = await loadState(args.job_id, args.cwd);
    let state = loaded ? await refreshStateFromOutput(loaded) : null;
    while (state && (state.status === "running" || state.status === "pending") && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, args.poll_seconds * 1000));
      loaded = await loadState(args.job_id, args.cwd);
      state = loaded ? await refreshStateFromOutput(loaded) : null;
    }
    if (!state) return textResponse({ server_version: SERVER_VERSION, status: "failed", reason: "job not found" });
    return textResponse({ ...state, changed_files: await getChangedFiles(state.cwd) });
  }
);

server.registerTool(
  "cancel",
  {
    title: "取消审查任务",
    description: "取消正在运行的 Claude 审查任务。",
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
    } else if (state.status === "running" || state.status === "pending") {
      state.status = "canceled";
      state.reason = "canceled by caller";
      state.updated_at = nowIso();
      state.completed_at = nowIso();
      await writeJson(state.state_path, state);
    }
    return textResponse(state);
  }
);

server.registerTool(
  "doctor",
  {
    title: "环境检查",
    description: "检查本地 Claude Code 审查工作器的运行前提。",
    inputSchema: z.object({ cwd: z.string().optional(), claude_command: z.string().default(process.env.CLAUDE_WORKER_COMMAND || "claude") })
  },
  async (input) => {
    const args = z.object({ cwd: z.string().optional(), claude_command: z.string().default(process.env.CLAUDE_WORKER_COMMAND || "claude") }).parse(input);
    const cwd = normalizeCwd(args.cwd);
    const nodeVersion = process.version;
    const claudeVersion = await runCommand(args.claude_command, ["-v"], cwd);
    const gitStatus = await runCommand("git", ["status", "--short"], cwd);
    const hasKey = Boolean(process.env.ANTHROPIC_API_KEY || process.env.DEEPSEEK_API_KEY);
    const queueRoot = path.join(cwd, "experiments", ".agent-queue");
    return textResponse({
      server_version: SERVER_VERSION,
      cwd,
      node_version: nodeVersion,
      default_mode: DEFAULT_WORKER_MODE,
      claude_command: args.claude_command,
      claude_ok: claudeVersion.exitCode === 0,
      claude_version: claudeVersion.stdout.trim() || claudeVersion.stderr.trim(),
      git_ok: gitStatus.exitCode === 0,
      deepseek_or_anthropic_key_present: hasKey,
      api_key_required_for_default_mode: false,
      api_key_required_for_cli_mode: false,
      api_key_required_for_external_mode: true,
      recommended_base_url: DEFAULT_BASE_URL,
      recommended_model: DEFAULT_MODEL,
      checks: {
        safe_default_permission_mode: "plan",
        bypass_permissions_default: false,
        queue_root: queueRoot,
        inbox_dir: path.join(queueRoot, "inbox"),
        outbox_dir: "output_path supplied to start",
        default_state_dir: path.join(cwd, "experiments", ".agent-state"),
        manual_state_dir: path.join(queueRoot, "state")
      }
    });
  }
);

server.registerTool(
  "setup",
  {
    title: "配置说明",
    description: "返回该 MCP 服务的 Codex Desktop 配置片段和环境说明。",
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
      "tool_timeout_sec = 1800"
    ].join("\n");
    return textResponse({
      server_version: SERVER_VERSION,
      codex_config_path: "%USERPROFILE%\\.codex\\config.toml",
      codex_config_snippet: snippet,
      cli_mode_example: "mode = \"cli\" 会通过 claude -p 使用本机 Claude CLI OAuth 登录状态，不需要 API key。",
      external_mode_env_example: "$env:DEEPSEEK_API_KEY = \"sk-...\"",
      notes: [
        "默认模式是 cli，会使用本机 Claude CLI OAuth 登录状态，不需要 API key。",
        "manual 模式保留为文件队列备用模式。",
        "只有 external 模式需要 DEEPSEEK_API_KEY 或 ANTHROPIC_API_KEY。",
        "加入 MCP 服务后需要重启 Codex Desktop。",
        "第一次冒烟测试建议先调用 doctor，再用 dry_run=true 调用 start。"
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
