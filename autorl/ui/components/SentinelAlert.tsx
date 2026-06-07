"use client";

interface SentinelIntervention {
  timestamp: string;
  agent_id: string;
  failure_reason: string;
  failed_hparams: Record<string, unknown>;
  llm_suggested_hparams: Record<string, unknown>;
  outcome: string;
}

interface SentinelAlertProps {
  intervention: SentinelIntervention;
}

const REASON_LABEL: Record<string, string> = {
  nan_loss: "NaN loss — weights exploded",
  stale_heartbeat_nudge: "Stale heartbeat — nudging",
  stale_after_nudge: "Still stale after nudge — restarting",
  nan_loss_second_failure: "NaN on restart — killed permanently",
  stale_second_failure: "Still stale after restart — killed permanently",
};

const OUTCOME_COLOR: Record<string, string> = {
  pending: "text-amber-600",
  completed: "text-green-600",
  failed_again: "text-red-600",
  nudge_sent: "text-blue-600",
  killed_permanently: "text-red-700",
};

export function SentinelAlert({ intervention }: SentinelAlertProps) {
  const isPermanentKill = intervention.outcome === "killed_permanently";

  return (
    <div className={`border rounded-xl p-4 text-sm shadow-sm w-full max-w-lg
      ${isPermanentKill ? "bg-red-50 border-red-200" : "bg-amber-50 border-amber-200"}`}>
      <div className="flex items-start gap-3">
        <span className="text-base mt-0.5">{isPermanentKill ? "🔴" : "⚠️"}</span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-bold text-amber-700">Doom Loop Sentinel</span>
            <span className="text-stone-500 text-xs font-mono">{intervention.agent_id}</span>
          </div>

          <p className="text-stone-800 mb-2">
            {REASON_LABEL[intervention.failure_reason] ?? intervention.failure_reason}
          </p>

          <div className="bg-stone-100 rounded-lg p-2 text-xs font-mono space-y-1 mb-2">
            <div className="text-red-700">
              Failed: lr={String(intervention.failed_hparams.lr)}
              {intervention.failed_hparams.seed !== undefined && ` seed=${intervention.failed_hparams.seed}`}
            </div>
            {Object.keys(intervention.llm_suggested_hparams).length > 0 && (
              <div className="text-green-700">
                LLM suggests: {Object.entries(intervention.llm_suggested_hparams)
                  .map(([k, v]) => `${k}=${v}`)
                  .join(" ")}
              </div>
            )}
          </div>

          <div className="flex items-center justify-between text-xs">
            <span className="text-stone-500">
              {new Date(intervention.timestamp).toLocaleTimeString()}
            </span>
            <span className={OUTCOME_COLOR[intervention.outcome] ?? "text-stone-500"}>
              {intervention.outcome}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

export function SentinelAlertList({ log }: { log: SentinelIntervention[] }) {
  if (log.length === 0) return null;
  return (
    <div className="space-y-3">
      <p className="text-xs font-semibold text-stone-500 uppercase tracking-wider">
        Sentinel Interventions
      </p>
      {log.map((entry, i) => (
        <SentinelAlert key={i} intervention={entry} />
      ))}
    </div>
  );
}
