"use client";

interface EvalResult {
  agent_id: string;
  algo: string;
  env: string;
  status: string;
  mean_return: number;
  std_return: number;
  steps_trained: number;
  wall_time_s: number;
  checkpoint_path: string;
}

interface ResultsPanelProps {
  results: EvalResult[];
  best: EvalResult | null;
  runName: string;
}

const MEDAL = ["🥇", "🥈", "🥉"];

export function ResultsPanel({ results, best, runName }: ResultsPanelProps) {
  const completed = [...results]
    .filter((r) => r.status === "completed")
    .sort((a, b) => b.mean_return - a.mean_return);

  const failed = results.filter((r) => r.status !== "completed");

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-gray-100">Training Results</h2>
        <span className="text-xs text-gray-400 font-mono">{runName}</span>
      </div>

      {/* Best model highlight */}
      {best && (
        <div className="bg-emerald-950 border border-emerald-700 rounded-xl p-4">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-2xl">🏆</span>
            <div>
              <p className="font-bold text-emerald-300">{best.algo} wins!</p>
              <p className="text-xs text-gray-400">{best.env} · {best.agent_id}</p>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-3 text-center mt-3">
            <div>
              <p className="text-2xl font-bold text-emerald-300">
                {best.mean_return.toFixed(0)}
              </p>
              <p className="text-xs text-gray-400">mean return</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-gray-300">
                ±{best.std_return.toFixed(0)}
              </p>
              <p className="text-xs text-gray-400">std</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-gray-300">
                {(best.wall_time_s / 60).toFixed(1)}m
              </p>
              <p className="text-xs text-gray-400">wall time</p>
            </div>
          </div>
          <div className="mt-3 bg-black/30 rounded-lg p-2">
            <p className="text-xs text-gray-400 mb-1">Best checkpoint</p>
            <p className="text-xs font-mono text-emerald-400 break-all">
              {best.checkpoint_path}
            </p>
          </div>
        </div>
      )}

      {/* Leaderboard */}
      {completed.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
            Leaderboard
          </p>
          {completed.map((r, i) => (
            <div
              key={r.agent_id}
              className="flex items-center justify-between bg-gray-900 border border-gray-800 rounded-lg px-3 py-2"
            >
              <div className="flex items-center gap-2">
                <span className="text-lg w-6">{MEDAL[i] ?? "·"}</span>
                <div>
                  <span className="font-semibold text-sm text-gray-200">{r.algo}</span>
                  <span className="text-xs text-gray-500 ml-2">{r.agent_id}</span>
                </div>
              </div>
              <div className="text-right">
                <p className="font-mono text-sm text-gray-200">
                  {r.mean_return.toFixed(0)}
                </p>
                <p className="text-xs text-gray-500">±{r.std_return.toFixed(0)}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Failed agents */}
      {failed.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
            Eliminated by Sentinel
          </p>
          {failed.map((r) => (
            <div
              key={r.agent_id}
              className="flex items-center justify-between bg-red-950/40 border border-red-900/40 rounded-lg px-3 py-1.5"
            >
              <span className="text-xs text-gray-400">{r.agent_id} · {r.algo}</span>
              <span className="text-xs text-red-400">{r.status}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
