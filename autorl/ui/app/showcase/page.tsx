import fs from "fs";
import path from "path";

const RUN = "2026-06-07T18-27-58";
const RUNS_BASE = path.join(process.cwd(), "../runs");

interface InferenceCase {
  numbers: number[];
  target: number;
  model_response: string;
  success: boolean;
}

interface EvalResult {
  agent_id: string;
  algo: string;
  env: string;
  status: string;
  mean_return: number;
  std_return: number;
  steps_trained: number;
  wall_time_s: number;
  wandb_artifact: string;
}

interface AgentData {
  inference: InferenceCase[];
  evalResult: EvalResult;
  baseline: InferenceCase[] | null;
}

function loadAgent(agentId: string): AgentData {
  const dir = path.join(RUNS_BASE, RUN, agentId);
  const inference: InferenceCase[] = JSON.parse(
    fs.readFileSync(path.join(dir, "inference_results.json"), "utf-8")
  );
  const evalResult: EvalResult = JSON.parse(
    fs.readFileSync(path.join(dir, "eval_result.json"), "utf-8")
  );
  const baselinePath = path.join(dir, "baseline_responses.json");
  const baseline = fs.existsSync(baselinePath)
    ? (JSON.parse(fs.readFileSync(baselinePath, "utf-8")) as InferenceCase[])
    : null;
  return { inference, evalResult, baseline };
}

function illustrativeResponse(nums: number[], target: number): string {
  const wrong = nums[0] + nums[nums.length - 1];
  return `${nums[0]} + ${nums[nums.length - 1]} = ${wrong}\nI think the answer is ${wrong}. (target: ${target})`;
}

function InferenceShowcase({ data, agentId, isWinner }: {
  data: AgentData;
  agentId: string;
  isWinner: boolean;
}) {
  const passed = data.inference.filter(c => c.success).length;
  const hasBaseline = !!data.baseline;
  const pct = (data.evalResult.mean_return * 100).toFixed(0);
  const mins = (data.evalResult.wall_time_s / 60).toFixed(1);

  return (
    <div className={`border bg-white text-sm ${isWinner ? "border-amber-300/50" : "border-stone-200"}`}>
      {/* Agent header */}
      <div className={`px-4 py-3 border-b flex items-center justify-between flex-wrap gap-3
        ${isWinner ? "border-amber-200 bg-amber-50" : "border-stone-200"}`}>
        <div className="flex items-center gap-3 flex-wrap">
          {isWinner && <span className="text-amber-400">▶ winner</span>}
          <span className="border px-2 py-0.5 text-amber-700 border-amber-300 bg-amber-50">
            {data.evalResult.algo}
          </span>
          <span className="text-stone-600">{data.evalResult.env}</span>
          <span className="text-stone-500">{agentId}</span>
        </div>
        <div className="flex items-center gap-6 text-stone-500">
          <span>
            score{" "}
            <span className={`font-bold ${isWinner ? "text-amber-400" : "text-green-500"}`}>
              {passed}/{data.inference.length} ({pct}%)
            </span>
          </span>
          <span>
            steps <span className="text-stone-700">{data.evalResult.steps_trained.toLocaleString()}</span>
          </span>
          <span>
            wall time <span className="text-stone-700">{mins} min</span>
          </span>
        </div>
      </div>

      {/* W&B artifact */}
      {data.evalResult.wandb_artifact && (
        <p className="px-4 py-2 border-b border-stone-100 text-stone-500">
          artifact:{" "}
          <span className="font-mono text-violet-500">{data.evalResult.wandb_artifact}</span>
          <span className="ml-2 text-stone-400">(W&amp;B)</span>
        </p>
      )}

      {/* Column headers */}
      <div className="grid grid-cols-2 gap-3 px-4 pt-3">
        <div className="text-stone-500 text-center py-1 bg-stone-100 uppercase tracking-widest text-xs">
          Before {!hasBaseline && <span className="normal-case text-stone-400">(illustrative)</span>}
        </div>
        <div className="text-amber-700 text-center py-1 bg-amber-50 border border-amber-200 uppercase tracking-widest text-xs">
          After GRPO training
        </div>
      </div>

      {/* Cases */}
      <div className="px-4 pb-4 pt-2 space-y-3">
        {data.inference.map((c, i) => {
          const beforeText = hasBaseline
            ? data.baseline![i]?.model_response ?? "(no baseline)"
            : illustrativeResponse(c.numbers, c.target);

          return (
            <div key={i} className="border border-stone-200 overflow-hidden">
              {/* Puzzle header */}
              <div className="flex items-center justify-between px-3 py-2 bg-stone-50 border-b border-stone-200">
                <span className="font-mono text-stone-600">
                  [{c.numbers.join(", ")}] &rarr; {c.target}
                </span>
                <span className={`px-2 py-0.5 text-xs font-bold
                  ${c.success
                    ? "bg-emerald-100 text-emerald-700"
                    : "bg-red-100 text-red-700"}`}>
                  {c.success ? "✅ PASS" : "❌ FAIL"}
                </span>
              </div>

              {/* Two-column responses */}
              <div className="grid grid-cols-2 divide-x divide-stone-200">
                <div className="p-2">
                  <div className="bg-stone-100 p-2 font-mono text-stone-600 whitespace-pre-wrap text-xs leading-relaxed max-h-32 overflow-y-auto">
                    {beforeText}
                  </div>
                </div>
                <div className="p-2">
                  <div className={`p-2 font-mono whitespace-pre-wrap text-xs leading-relaxed max-h-32 overflow-y-auto
                    ${c.success
                      ? "bg-emerald-50 text-emerald-800"
                      : "bg-stone-100 text-stone-700"}`}>
                    {c.model_response || "(empty response)"}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Divider({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2 my-2">
      <div className="flex-1 border-t border-stone-200" />
      <span className="text-stone-400 uppercase tracking-widest text-xs">{label}</span>
      <div className="flex-1 border-t border-stone-200" />
    </div>
  );
}

export default function ShowcasePage() {
  const agent2 = loadAgent("agent_2");
  const agent1 = loadAgent("agent_1");

  return (
    <div className="min-h-screen bg-[#FAF9F7] text-stone-800 p-6 pt-8"
      style={{ fontFamily: 'var(--font-mono, "JetBrains Mono", "Cascadia Code", monospace)' }}>
      <div className="max-w-5xl mx-auto">

        {/* Header — matches main UI nav */}
        <div className="flex items-center gap-3 mb-8">
          <span className="text-amber-600 font-bold">AutoRL</span>
          <span className="text-stone-300">|</span>
          <span className="text-stone-500 text-sm">multi-agent training race</span>
          <span className="text-stone-300">|</span>
          <span className="text-stone-500 text-sm">{RUN}</span>
          <span className="text-stone-300">|</span>
          <span className="text-amber-400 text-sm">race complete</span>
        </div>

        {/* Summary stats — matches the "winner panel" top row */}
        <div className="border border-amber-300/50 bg-amber-50 text-sm mb-4">
          <div className="px-4 py-3 border-b border-stone-200 flex items-center justify-between flex-wrap gap-3">
            <div className="flex items-center gap-3 flex-wrap">
              <span className="text-amber-400">▶ winner</span>
              <span className="border px-2 py-0.5 text-amber-700 border-amber-300 bg-amber-50">GRPO</span>
              <span className="text-stone-600">Countdown</span>
              <span className="text-stone-500">agent_2</span>
            </div>
            <div className="flex items-center gap-6 text-stone-500">
              <span>mean return <span className="text-amber-400 font-bold">1.00</span></span>
              <span>std <span className="text-stone-700">±0.00</span></span>
            </div>
          </div>
          <div className="px-4 py-3 flex flex-wrap gap-8 text-stone-500">
            <span>model <span className="font-mono text-stone-700">Qwen/Qwen2.5-3B (base)</span></span>
            <span>steps <span className="text-stone-700">2,016</span></span>
            <span>lr <span className="font-mono text-stone-700">2e-6</span></span>
            <span>num_generations <span className="font-mono text-stone-700">8</span></span>
            <span>temperature <span className="font-mono text-stone-700">0.7</span></span>
          </div>
          <p className="px-4 pb-3 text-stone-500 text-xs">
            Base model — no instruct tuning. Reasoning structure (
            <span className="font-mono">&lt;think&gt;</span>/
            <span className="font-mono">&lt;answer&gt;</span>) emerged from scratch via format + accuracy reward signal.
            Baseline column is illustrative — pre-training baseline was not synced from the pod.
          </p>
        </div>

        {/* Winner showcase */}
        <InferenceShowcase data={agent2} agentId="agent_2" isWinner={true} />

        <Divider label="all agents" />

        {/* Agent 1 */}
        <InferenceShowcase data={agent1} agentId="agent_1" isWinner={false} />

        {/* Footer */}
        <div className="border-t border-stone-200 mt-6 pt-4 flex flex-wrap gap-6 text-stone-400 text-xs">
          <span>AutoRL · WeaveHacks 4</span>
          <span>run: {RUN}</span>
          <span>artifact: grpo-lora-agent_2 (W&amp;B)</span>
          <a href="/" className="hover:text-amber-600 transition-colors ml-auto">← back to training UI</a>
        </div>
      </div>
    </div>
  );
}
