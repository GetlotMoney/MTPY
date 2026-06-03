import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { promises as fs } from "node:fs";
import path from "node:path";

const cwd = path.resolve(process.argv[2] || process.cwd());
const serverPath = path.resolve("dist/index.js");

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

const doctor = await client.callTool({
  name: "doctor",
  arguments: { cwd }
});
console.log(JSON.stringify(doctor.structuredContent, null, 2));

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

const manual = await client.callTool({
  name: "start",
  arguments: {
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
  "Decision: ACCEPTED\n\nFindings:\n- Manual queue smoke result parsed successfully.\n",
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

await client.close();
