import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { promises as fs } from "node:fs";
import path from "node:path";

const cwdArg = process.argv.slice(2).find((arg) => !arg.startsWith("--"));
const cwd = path.resolve(cwdArg || process.cwd());
const serverPath = path.resolve("dist/index.js");
const runCliE2E = process.argv.includes("--cli-e2e");
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function assertSmoke(condition, message) {
  if (!condition) {
    throw new Error(`Smoke failed: ${message}`);
  }
}

const client = new Client({ name: "claude-code-worker-smoke", version: "0.1.0" });
const transport = new StdioClientTransport({
  command: "node",
  args: [serverPath],
  cwd
});

await client.connect(transport);

const tools = await client.listTools();
const names = tools.tools.map((tool) => tool.name).sort();
console.log("tools:", names.join(", "));
for (const name of ["cancel", "doctor", "get", "setup", "start", "tail", "wait"]) {
  assertSmoke(names.includes(name), `missing tool ${name}`);
}

const doctor = await client.callTool({
  name: "doctor",
  arguments: { cwd }
});
console.log(JSON.stringify(doctor.structuredContent, null, 2));
assertSmoke(doctor.structuredContent?.default_mode === "cli", "default mode must be cli");
assertSmoke(doctor.structuredContent?.claude_ok === true, "claude command must be available");
assertSmoke(doctor.structuredContent?.api_key_required_for_default_mode === false, "default mode must not require an API key");
assertSmoke(doctor.structuredContent?.api_key_required_for_cli_mode === false, "cli mode must not require an API key");
assertSmoke(doctor.structuredContent?.api_key_required_for_external_mode === true, "external mode must require an API key");

const dryRun = await client.callTool({
  name: "start",
  arguments: {
    experiment_id: "EXP-000",
    round: 1,
    cwd,
    packet_path: "fixtures/EXP-000.review-packet.md",
    output_path: "fixtures/EXP-000.claude-review.md",
    allowed_dirs: [cwd],
    dry_run: true
  }
});
console.log(JSON.stringify({
  dry_run_status: dryRun.structuredContent?.status,
  dry_run_reason: dryRun.structuredContent?.reason,
  dry_run_mode: dryRun.structuredContent?.mode,
  dry_run_model: dryRun.structuredContent?.model,
  dry_run_permission_mode: dryRun.structuredContent?.permission_mode,
  dry_run_request_path: dryRun.structuredContent?.request_path
}, null, 2));
assertSmoke(dryRun.structuredContent?.status === "pending", "dry-run should create pending state");
assertSmoke(dryRun.structuredContent?.reason === "dry_run", "dry-run reason should be dry_run");
assertSmoke(dryRun.structuredContent?.mode === "cli", "dry-run should use default cli mode");
assertSmoke(dryRun.structuredContent?.permission_mode === "plan", "dry-run should use plan permission mode");

const manual = await client.callTool({
  name: "start",
  arguments: {
    mode: "manual",
    experiment_id: "EXP-SMOKE",
    round: 1,
    cwd,
    packet_path: "fixtures/EXP-000.review-packet.md",
    output_path: "fixtures/EXP-SMOKE.claude-review.md",
    allowed_dirs: [cwd]
  }
});

await fs.writeFile(
  path.join(cwd, "fixtures", "EXP-SMOKE.claude-review.md"),
  [
    "Pass 1 Decision: ACCEPTED",
    "Pass 2 Decision: ACCEPTED",
    "Pass 3 Decision: ACCEPTED",
    "Overall Decision: ACCEPTED",
    "",
    "Findings:",
    "- Manual queue fallback smoke result parsed successfully."
  ].join("\n") + "\n",
  "utf8"
);

const parsed = await client.callTool({
  name: "get",
  arguments: {
    job_id: manual.structuredContent?.job_id,
    cwd
  }
});

console.log(JSON.stringify({
  manual_status: parsed.structuredContent?.status,
  manual_decision: parsed.structuredContent?.decision,
  manual_mode: parsed.structuredContent?.mode
}, null, 2));
assertSmoke(parsed.structuredContent?.status === "accepted", "manual fallback should parse accepted state");
assertSmoke(parsed.structuredContent?.decision === "ACCEPTED", "manual fallback should parse ACCEPTED decision");
assertSmoke(parsed.structuredContent?.mode === "manual", "manual fallback should stay in manual mode");

if (runCliE2E) {
  const cli = await client.callTool({
    name: "start",
    arguments: {
      experiment_id: "EXP-CLI-E2E",
      round: 1,
      max_rounds: 3,
      cwd,
      packet_path: "fixtures/EXP-CLI-E2E.review-packet.md",
      output_path: "fixtures/EXP-CLI-E2E.claude-review.md",
      allowed_dirs: [cwd],
      timeout_seconds: 240,
      permission_mode: "plan",
      bare: true,
      verbose: true,
      max_budget_usd: 0.8
    }
  });

  console.log(JSON.stringify({
    cli_start_status: cli.structuredContent?.status,
    cli_mode: cli.structuredContent?.mode,
    cli_job_id: cli.structuredContent?.job_id
  }, null, 2));

  let current = cli.structuredContent;
  for (let index = 0; index < 48; index += 1) {
    await sleep(5000);
    const got = await client.callTool({
      name: "get",
      arguments: {
        job_id: cli.structuredContent?.job_id,
        cwd
      }
    });
    current = got.structuredContent;
    console.log(JSON.stringify({
      cli_poll: index + 1,
      cli_status: current?.status,
      cli_decision: current?.decision,
      cli_exit_code: current?.exit_code
    }, null, 2));
    if (!["running", "pending"].includes(current?.status)) break;
  }

  if (current?.status !== "accepted") {
    throw new Error(`CLI e2e smoke failed: status=${current?.status}, decision=${current?.decision}, exit_code=${current?.exit_code}`);
  }
}

await client.close();
