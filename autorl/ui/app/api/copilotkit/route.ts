/**
 * CopilotKit runtime — server-side actions that call the AutoRL Python backend.
 *
 * The LLM (GPT) decides which action to call based on the conversation.
 * Actions:
 *   generate_plan   → POST /api/plan  on the Python backend
 *   start_training  → POST /api/run   on the Python backend
 *   get_status      → GET  /api/status/{run} on the Python backend
 *   get_results     → GET  /api/results/{run} on the Python backend
 */

import {
  CopilotRuntime,
  OpenAIAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { NextRequest } from "next/server";

const BACKEND = process.env.AUTORL_BACKEND_URL ?? "http://localhost:8000";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const runtime = new CopilotRuntime({
  actions: (): any[] => [
    // ── 1. Generate spawn plan ─────────────────────────────────────────────
    {
      name: "generate_plan",
      description:
        "Generate an AutoRL training spawn plan from the user's task description. " +
        "Call this first when the user describes what they want to train. " +
        "Returns a spawn plan with agent configs for the user to approve.",
      parameters: [
        {
          name: "task",
          type: "string",
          description:
            "The RL task to train (e.g. 'Train the best MuJoCo locomotion policy')",
          required: true,
        },
      ],
      handler: async ({ task }: { task: string }) => {
        const res = await fetch(`${BACKEND}/api/plan`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ task }),
        });
        if (!res.ok) throw new Error(`Plan generation failed: ${res.statusText}`);
        return res.json();
      },
    },

    // ── 2. Start training after user approves ──────────────────────────────
    {
      name: "start_training",
      description:
        "Start the AutoRL training swarm after the user has approved the spawn plan. " +
        "Launches all agents in parallel. Training runs in the background — " +
        "use get_status to track progress.",
      parameters: [
        {
          name: "task",
          type: "string",
          description: "Original task description",
          required: true,
        },
        {
          name: "run_dir",
          type: "string",
          description: "Run directory returned by generate_plan",
          required: true,
        },
        {
          name: "plan",
          type: "string",
          description: "JSON-stringified spawn plan array from generate_plan",
          required: true,
        },
      ],
      handler: async ({
        task,
        run_dir,
        plan,
      }: {
        task: string;
        run_dir: string;
        plan: string;
      }) => {
        const res = await fetch(`${BACKEND}/api/run`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ task, run_dir, plan: JSON.parse(plan) }),
        });
        if (!res.ok) throw new Error(`Failed to start training: ${res.statusText}`);
        return res.json();
      },
    },

    // ── 3. Poll live status ────────────────────────────────────────────────
    {
      name: "get_status",
      description:
        "Get the current training status: heartbeats for all agents, " +
        "any Sentinel interventions. Call this when the user asks for progress.",
      parameters: [
        {
          name: "run_name",
          type: "string",
          description: "Run name (directory basename) returned by start_training",
          required: true,
        },
      ],
      handler: async ({ run_name }: { run_name: string }) => {
        const res = await fetch(`${BACKEND}/api/status/${run_name}`);
        if (!res.ok) throw new Error(`Status check failed: ${res.statusText}`);
        return res.json();
      },
    },

    // ── 4. Get final results ───────────────────────────────────────────────
    {
      name: "get_results",
      description:
        "Get the final evaluation results and best model checkpoint path " +
        "once training is complete. Call when status shows 'completed'.",
      parameters: [
        {
          name: "run_name",
          type: "string",
          description: "Run name returned by start_training",
          required: true,
        },
      ],
      handler: async ({ run_name }: { run_name: string }) => {
        const res = await fetch(`${BACKEND}/api/results/${run_name}`);
        if (!res.ok) throw new Error(`Results fetch failed: ${res.statusText}`);
        return res.json();
      },
    },
  ],
});

const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
  runtime,
  serviceAdapter: new OpenAIAdapter({ model: "gpt-4o-mini" }),
  endpoint: "/api/copilotkit",
});

export const POST = handleRequest;
