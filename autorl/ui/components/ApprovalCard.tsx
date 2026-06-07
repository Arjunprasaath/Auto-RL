"use client";

interface SpawnEntry {
  id: string;
  algo: string;
  env: string;
  exec: string;
  time_budget_min: number;
  hparams: Record<string, unknown>;
}

interface ApprovalCardProps {
  plan: SpawnEntry[];
  task: string;
  runDir: string;
  onApprove: () => void;
  onReject: () => void;
}

export function ApprovalCard({ plan, task, onApprove, onReject }: ApprovalCardProps) {
  const localAgents = plan.filter((e) => e.exec === "local");
  const cloudAgents = plan.filter((e) => e.exec === "runpod");

  return (
    <div className="bg-white border border-stone-200 rounded-xl p-5 max-w-lg w-full shadow-sm">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-8 h-8 rounded-full bg-violet-600 flex items-center justify-center text-sm font-bold text-white">
          RL
        </div>
        <div>
          <p className="font-semibold text-stone-800">AutoRL Orchestrator</p>
          <p className="text-sm text-stone-500">Ready to spawn {plan.length} agents</p>
        </div>
      </div>

      <div className="bg-stone-50 rounded-lg px-3 py-2 mb-4 text-sm text-stone-700 italic border border-stone-200">
        &ldquo;{task}&rdquo;
      </div>

      <div className="space-y-2 mb-5">
        {plan.map((entry) => {
          const lr = entry.hparams.lr as number;
          const isDangerous = lr >= 0.1;
          return (
            <div
              key={entry.id}
              className={`flex items-center justify-between rounded-lg px-3 py-2 text-sm
                ${isDangerous ? "bg-red-50 border border-red-200" : "bg-stone-50 border border-stone-200"}`}
            >
              <div className="flex items-center gap-2">
                <span className="font-mono text-violet-700">{entry.id}</span>
                <span className="text-stone-700">{entry.algo}</span>
                <span className="text-stone-500 text-xs">{entry.env}</span>
                {isDangerous && (
                  <span className="text-red-600 text-xs font-semibold">⚠ lr={lr} (sentinel bait)</span>
                )}
              </div>
              <div className="flex items-center gap-2 text-xs text-stone-600">
                <span>{entry.time_budget_min}m</span>
                <span className={`px-1.5 py-0.5 rounded ${entry.exec === "local" ? "bg-blue-100 text-blue-700" : "bg-orange-100 text-orange-700"}`}>
                  {entry.exec}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      <div className="text-sm text-stone-600 mb-5 space-y-0.5">
        <p>Local: {localAgents.length} agent{localAgents.length !== 1 ? "s" : ""} (Mac CPU)</p>
        {cloudAgents.length > 0 && (
          <p>Cloud: {cloudAgents.length} agent{cloudAgents.length !== 1 ? "s" : ""} (RunPod GPU)</p>
        )}
        <p className="text-amber-600 mt-1">
          Doom Loop Sentinel will monitor all agents for NaN / stale behavior.
        </p>
      </div>

      <div className="flex gap-3">
        <button
          onClick={onApprove}
          className="flex-1 bg-amber-600 hover:bg-amber-500 text-white font-semibold py-2 rounded-lg transition-colors text-sm"
        >
          Approve & Launch
        </button>
        <button
          onClick={onReject}
          className="flex-1 bg-stone-100 hover:bg-stone-200 text-stone-700 font-semibold py-2 rounded-lg transition-colors text-sm border border-stone-300"
        >
          Reject
        </button>
      </div>
    </div>
  );
}
