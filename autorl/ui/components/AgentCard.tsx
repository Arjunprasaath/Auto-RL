"use client";

interface Heartbeat {
  agent_id: string;
  status: string;
  steps_completed: number;
  current_reward: number;
  loss?: number | null;
  anomaly?: string | null;
  timestamp: string;
}

interface PlanEntry {
  id: string;
  algo: string;
  env: string;
  hparams: Record<string, unknown>;
  time_budget_min: number;
}

interface AgentCardProps {
  entry: PlanEntry;
  heartbeat?: Heartbeat;
}

const STATUS_COLOR: Record<string, string> = {
  starting: "bg-blue-800 text-blue-200",
  training: "bg-green-900 text-green-300",
  completed: "bg-emerald-900 text-emerald-300",
  failed: "bg-red-900 text-red-300",
  restarted: "bg-yellow-900 text-yellow-300",
};

const ALGO_COLOR: Record<string, string> = {
  PPO: "text-violet-400",
  SAC: "text-cyan-400",
  A2C: "text-pink-400",
  GRPO: "text-orange-400",
};

export function AgentCard({ entry, heartbeat }: AgentCardProps) {
  const status = heartbeat?.status ?? "waiting";
  const hasAnomaly = heartbeat?.anomaly === "nan_loss";
  const lr = entry.hparams.lr as number;
  const isDangerous = lr >= 0.1;

  return (
    <div className={`rounded-xl p-4 border transition-all duration-300
      ${hasAnomaly
        ? "bg-red-950 border-red-700 shadow-red-900/50 shadow-lg"
        : status === "completed"
        ? "bg-emerald-950 border-emerald-800"
        : isDangerous
        ? "bg-red-950/50 border-red-900"
        : "bg-gray-900 border-gray-800"
      }`}>

      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="font-mono text-xs text-gray-400">{entry.id}</span>
          <span className={`font-bold text-sm ${ALGO_COLOR[entry.algo] ?? "text-gray-300"}`}>
            {entry.algo}
          </span>
          {isDangerous && (
            <span className="text-xs text-red-400 font-semibold">☠ lr={lr}</span>
          )}
        </div>
        <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${STATUS_COLOR[status] ?? "bg-gray-800 text-gray-400"}`}>
          {status}
        </span>
      </div>

      {/* Env */}
      <p className="text-xs text-gray-500 mb-3 truncate">{entry.env}</p>

      {/* Metrics */}
      {heartbeat ? (
        <div className="space-y-2">
          <div className="flex justify-between text-xs">
            <span className="text-gray-400">Steps</span>
            <span className="font-mono text-gray-200">
              {heartbeat.steps_completed.toLocaleString()}
            </span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-gray-400">Reward</span>
            <span className={`font-mono ${heartbeat.current_reward > 0 ? "text-green-400" : "text-gray-300"}`}>
              {heartbeat.current_reward.toFixed(1)}
            </span>
          </div>

          {/* Reward bar */}
          <div className="w-full bg-gray-800 rounded-full h-1.5 mt-2">
            <div
              className={`h-1.5 rounded-full transition-all duration-500 ${
                hasAnomaly ? "bg-red-500" :
                status === "completed" ? "bg-emerald-500" :
                "bg-violet-500"
              }`}
              style={{
                width: `${Math.min(100, Math.max(0, (heartbeat.steps_completed / (entry.time_budget_min * 60 * 100)) * 100))}%`
              }}
            />
          </div>

          {hasAnomaly && (
            <p className="text-xs text-red-400 font-semibold mt-1">
              ⚠ NaN loss — Sentinel intervening…
            </p>
          )}
        </div>
      ) : (
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <div className="w-2 h-2 rounded-full bg-gray-600 animate-pulse" />
          Waiting to start…
        </div>
      )}
    </div>
  );
}
