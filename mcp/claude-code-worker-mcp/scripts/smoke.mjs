import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
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
  dry_run_model: dryRun.structuredContent?.model,
  dry_run_permission_mode: dryRun.structuredContent?.permission_mode
}, null, 2));

await client.close();
