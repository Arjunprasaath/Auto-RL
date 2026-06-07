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
    <div className="bg-gray-900 border border-gray-700 rounded-xl p-5 max-w-lg w-full shadow-xl">
      {/* Header */}
      <div className="flex items-center gap-3 mb-4">
        <div className="w-8 h-8 rounded-full bg-violet-600 flex items-center justify-center text-sm font-bold">
          RL
        </div>
        <div>
          <p className="font-semibold text-gray-100">AutoRL Orchestrator</p>
          <p className="text-xs text-gray-400">Ready to spawn {plan.length} agents</p>
        </div>
      </div>

      {/* Task */}
      <div className="bg-gray-800 rounded-lg px-3 py-2 mb-4 text-sm text-gray-300 italic">
        &ldquo;{task}&rdquo;
      </div>

      {/* Agent list */}
      <div className="space-y-2 mb-5">
        {plan.map((entry) => {
          const lr = entry.hparams.lr as number;
          const isDangerous = lr >= 0.1;
          return (
            <div
              key={entry.id}
              className={`flex items-center justify-between rounded-lg px-3 py-2 text-sm
                ${isDangerous ? "bg-red-950 border border-red-800" : "bg-gray-800"}`}
            >
              <div className="flex items-center gap-2">
                <span className="font-mono text-violet-300">{entry.id}</span>
                <span className="text-gray-300">{entry.algo}</span>
                <span className="text-gray-500 text-xs">{entry.env}</span>
                {isDangerous && (
                  <span className="text-red-400 text-xs font-semibold">⚠ lr={lr} (sentinel bait)</span>
                )}
              </div>
              <div className="flex items-center gap-2 text-xs text-gray-400">
                <span>{entry.time_budget_min}m</span>
                <span className={`px-1.5 py-0.5 rounded ${entry.exec === "local" ? "bg-blue-900 text-blue-300" : "bg-orange-900 text-orange-300"}`}>
                  {entry.exec}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Summary */}
      <div className="text-xs text-gray-400 mb-5 space-y-0.5">
        <p>Local: {localAgents.length} agent{localAgents.length !== 1 ? "s" : ""} (Mac CPU)</p>
        {cloudAgents.length > 0 && (
          <p>Cloud: {cloudAgents.length} agent{cloudAgents.length !== 1 ? "s" : ""} (RunPod GPU)</p>
        )}
        <p className="text-yellow-400 mt-1">
          Doom Loop Sentinel will monitor all agents for NaN / stale behavior.
        </p>
      </div>

      {/* Buttons */}
      <div className="flex gap-3">
        <button
          onClick={onApprove}
          className="flex-1 bg-violet-600 hover:bg-violet-500 text-white font-semibold py-2 rounded-lg transition-colors text-sm"
        >
          Approve & Launch
        </button>
        <button
          onClick={onReject}
          className="flex-1 bg-gray-700 hover:bg-gray-600 text-gray-300 font-semibold py-2 rounded-lg transition-colors text-sm"
        >
          Reject
        </button>
      </div>
    </div>
  );
}
