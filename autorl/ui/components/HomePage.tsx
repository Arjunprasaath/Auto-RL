"use client";

import { useState, useRef, useEffect, useCallback } from "react";

const BACKEND = "http://localhost:8000";
const POLL_MS = 2_000;

const SUGGESTIONS = [
  "Train the best MuJoCo locomotion policy",
  "Race PPO vs SAC on HalfCheetah-v5",
  "Train an agent on Hopper-v5",
  "Train a Countdown reasoning agent with GRPO",
];

// ── Types ──────────────────────────────────────────────────────────────────────

interface SpawnEntry {
  id: string; algo: string; env: string; exec: string;
  time_budget_min: number; hparams: Record<string, unknown>;
}
interface Heartbeat {
  agent_id: string; status: string; steps_completed: number;
  current_reward: number; anomaly?: string | null; timestamp: string;
}
interface SentinelEntry {
  timestamp: string; agent_id: string; failure_reason: string;
  failed_hparams: Record<string, unknown>;
  llm_suggested_hparams: Record<string, unknown>; outcome: string;
}
interface EvalResult {
  agent_id: string; algo: string; env: string; status: string;
  mean_return: number; std_return: number; checkpoint_path: string;
}
interface HistPt { steps: number; reward: number; seg: number; }
type Phase = "idle" | "planning" | "plan_ready" | "launching" | "racing" | "done" | "error";
type AnimState = "vanish" | "appear" | "idle";

// ── Style maps ─────────────────────────────────────────────────────────────────

const ALGO_STYLE: Record<string, { border: string; badge: string; bar: string; rgb: string }> = {
  PPO:  { border: "border-violet-600",  badge: "bg-violet-900 text-violet-300",  bar: "bg-violet-500",  rgb: "#8b5cf6" },
  SAC:  { border: "border-cyan-600",    badge: "bg-cyan-900 text-cyan-300",      bar: "bg-cyan-500",    rgb: "#06b6d4" },
  A2C:  { border: "border-pink-600",    badge: "bg-pink-900 text-pink-300",      bar: "bg-pink-500",    rgb: "#ec4899" },
  GRPO: { border: "border-orange-600",  badge: "bg-orange-900 text-orange-300",  bar: "bg-orange-500",  rgb: "#f97316" },
};
const DEF_STYLE = { border: "border-gray-700", badge: "bg-gray-800 text-gray-300", bar: "bg-gray-500", rgb: "#6b7280" };
const as = (algo: string) => ALGO_STYLE[algo] ?? DEF_STYLE;
const SEG_COLORS = ["#8b5cf6", "#06b6d4", "#34d399", "#fbbf24", "#f472b6"];

// ── Mini SVG reward chart ──────────────────────────────────────────────────────

function MiniChart({ history, algoRgb, hasNaN }: { history: HistPt[]; algoRgb: string; hasNaN: boolean }) {
  const W = 360, H = 72;
  if (history.length < 2) return (
    <div className="w-full h-[72px] flex items-center justify-center">
      <p className="text-xs text-gray-700">Waiting for data…</p>
    </div>
  );

  const minS = Math.min(...history.map(p => p.steps));
  const maxS = Math.max(...history.map(p => p.steps));
  const minR = Math.min(...history.map(p => p.reward));
  const maxR = Math.max(...history.map(p => p.reward));
  const xS = (s: number) => ((s - minS) / (maxS - minS || 1)) * W;
  const yS = (r: number) => H - 4 - ((r - minR) / (maxR - minR || 1)) * (H - 8);

  const segs = new Map<number, HistPt[]>();
  for (const p of history) {
    if (!segs.has(p.seg)) segs.set(p.seg, []);
    segs.get(p.seg)!.push(p);
  }
  const maxSeg = Math.max(...history.map(p => p.seg));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-[72px]" preserveAspectRatio="none">
      <defs>
        {[...segs.keys()].map(seg => {
          const isFailSeg = seg === maxSeg && hasNaN;
          const color = isFailSeg ? "#ef4444" : (SEG_COLORS[seg % SEG_COLORS.length] ?? algoRgb);
          return (
            <linearGradient key={seg} id={`grad-${seg}-${algoRgb.replace("#","")}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity="0.25" />
              <stop offset="100%" stopColor={color} stopOpacity="0" />
            </linearGradient>
          );
        })}
      </defs>
      {[...segs.entries()].map(([seg, pts]) => {
        const isFailSeg = seg === maxSeg && hasNaN;
        const color = isFailSeg ? "#ef4444" : (SEG_COLORS[seg % SEG_COLORS.length] ?? algoRgb);
        const pathD = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${xS(p.steps).toFixed(1)} ${yS(p.reward).toFixed(1)}`).join(" ");
        const areaD = pts.length > 1
          ? `${pathD} L ${xS(pts[pts.length-1].steps).toFixed(1)} ${H} L ${xS(pts[0].steps).toFixed(1)} ${H} Z`
          : "";
        return (
          <g key={seg}>
            {areaD && <path d={areaD} fill={`url(#grad-${seg}-${algoRgb.replace("#","")})`} />}
            <path d={pathD} fill="none" stroke={color} strokeWidth="2"
              strokeLinecap="round" strokeLinejoin="round"
              strokeDasharray={isFailSeg ? "4 3" : undefined} />
            {seg > 0 && pts.length > 0 && (
              <circle cx={xS(pts[0].steps)} cy={yS(pts[0].reward)} r="4"
                fill={color} stroke="#0f172a" strokeWidth="1.5" />
            )}
          </g>
        );
      })}
      <text x="2" y="10" fontSize="9" fill="#4b5563">{maxR.toFixed(0)}</text>
      <text x="2" y={H - 2} fontSize="9" fill="#4b5563">{minR.toFixed(0)}</text>
    </svg>
  );
}

// ── Plan loading screen ────────────────────────────────────────────────────────

const PLAN_STEPS = [
  "Orchestrator analysing task…",
  "Evaluating algorithms (PPO · SAC · A2C · GRPO)…",
  "Designing agent lineup…",
  "Setting hyperparameters…",
  "Validating spawn plan…",
];
function PlanningScreen({ task }: { task: string }) {
  const [step, setStep] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setStep(s => Math.min(s + 1, PLAN_STEPS.length - 1)), 3000);
    return () => clearInterval(id);
  }, []);
  return (
    <div className="w-full max-w-xl space-y-6">
      <div className="bg-gray-900 border border-gray-800 rounded-xl px-4 py-3">
        <p className="text-xs text-gray-500 mb-1">Your task</p>
        <p className="text-sm text-gray-200 italic">&ldquo;{task}&rdquo;</p>
      </div>
      <div className="space-y-3">
        {PLAN_STEPS.map((label, i) => {
          const done = i < step, active = i === step;
          return (
            <div key={i} className="flex items-center gap-3">
              <div className={`w-5 h-5 rounded-full flex items-center justify-center shrink-0 text-xs font-bold
                ${done ? "bg-emerald-600 text-white" : active ? "border-2 border-violet-500 border-t-transparent animate-spin" : "border border-gray-700"}`}>
                {done && "✓"}
              </div>
              <p className={`text-sm ${done ? "text-gray-600 line-through" : active ? "text-gray-200" : "text-gray-600"}`}>{label}</p>
            </div>
          );
        })}
      </div>
      <p className="text-xs text-gray-600 text-center">Orchestrator LLM generating lineup — ~10–20 s</p>
    </div>
  );
}

// ── Pre-launch agent card ─────────────────────────────────────────────────────

function AgentLineupCard({ entry, index }: { entry: SpawnEntry; index: number }) {
  const st = as(entry.algo);
  const lr = entry.hparams.lr as number;
  const isDoom = lr >= 0.1;
  const hparams: [string, string][] = [
    ["lr", String(lr)],
    ["seed", String(entry.hparams.seed ?? "—")],
    ...(entry.hparams.n_steps != null ? [["n_steps", String(entry.hparams.n_steps)] as [string,string]] : []),
    ...(entry.hparams.gamma   != null ? [["gamma",   String(entry.hparams.gamma)]   as [string,string]] : []),
    ...(entry.hparams.model   != null ? [["model",   String(entry.hparams.model).split("/").pop() ?? ""] as [string,string]] : []),
  ];
  return (
    <div className={`bg-gray-900 border-l-4 ${st.border} rounded-xl p-4 space-y-3 ${isDoom ? "ring-1 ring-red-800/60" : ""}`}>
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs font-mono text-gray-500">#{index + 1}</span>
          <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${st.badge}`}>{entry.algo}</span>
          {entry.exec === "runpod" && <span className="text-xs px-2 py-0.5 rounded-full bg-yellow-900/60 text-yellow-400 border border-yellow-800/50">RunPod GPU</span>}
          {isDoom && <span className="text-xs px-2 py-0.5 rounded-full bg-red-900/60 text-red-400 border border-red-800/50">☠ doom bait</span>}
        </div>
        <span className="text-xs text-gray-500 shrink-0">{entry.time_budget_min} min</span>
      </div>
      <p className="text-sm font-semibold text-gray-200 truncate">{entry.env}</p>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
        {hparams.map(([k, v]) => (
          <div key={k} className="flex items-center gap-1.5">
            <span className="text-xs text-gray-500 w-16 shrink-0">{k}</span>
            <span className={`text-xs font-mono ${k === "lr" && isDoom ? "text-red-400 font-bold" : "text-gray-300"}`}>{v}</span>
          </div>
        ))}
      </div>
      {isDoom && <p className="text-xs text-red-400 border-t border-red-900/40 pt-2">Sentinel will intercept — expects NaN loss</p>}
    </div>
  );
}

// ── Live agent card ───────────────────────────────────────────────────────────

function LiveAgentCard({
  entry, hb, history, sentinelEntries, animState, onInfer, inferring, hasVideo,
}: {
  entry: SpawnEntry;
  hb?: Heartbeat;
  history: HistPt[];
  sentinelEntries: SentinelEntry[];
  animState: AnimState;
  onInfer: () => void;
  inferring: boolean;
  hasVideo: boolean;
}) {
  const st = as(entry.algo);
  const hasNaN    = hb?.anomaly === "nan_loss";
  const status    = hb?.status ?? "waiting";
  const maxSteps  = entry.time_budget_min * 60 * 100;
  const pct       = hb ? Math.min(100, (hb.steps_completed / maxSteps) * 100) : 0;
  const currentSeg = history.length > 0 ? history[history.length - 1].seg : 0;
  const latestIntervention = sentinelEntries.at(-1);
  const canInfer  = entry.algo !== "GRPO";

  const STATUS_COLOR: Record<string, string> = {
    training: "bg-green-900 text-green-300",
    completed: "bg-emerald-900 text-emerald-300",
    failed: "bg-red-900 text-red-300",
    restarted: "bg-yellow-900 text-yellow-300",
  };
  const animClass = animState === "vanish" ? "agent-vanish" : animState === "appear" ? "agent-appear" : "";

  return (
    <div className={`bg-gray-900 border-l-4 ${st.border} rounded-xl p-4 space-y-3
      ${hasNaN ? "ring-1 ring-red-700" : status === "completed" ? "ring-1 ring-emerald-800" : ""}
      ${animClass}`}>

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 flex-wrap">
          <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${st.badge}`}>{entry.algo}</span>
          <span className="text-xs font-mono text-gray-500">{entry.id}</span>
          {currentSeg > 0 && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-yellow-900/60 text-yellow-400 border border-yellow-800/40">
              restart #{currentSeg}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {canInfer && (
            <button onClick={onInfer} disabled={inferring}
              className={`text-xs px-2 py-0.5 rounded-lg font-semibold transition-colors
                ${inferring ? "bg-gray-800 text-gray-500 cursor-not-allowed" :
                  "bg-gray-800 text-gray-400 hover:bg-violet-900 hover:text-violet-300 border border-gray-700"}`}>
              {inferring ? "⏳ recording…" : "▶ infer"}
            </button>
          )}
          <span className={`text-xs px-2 py-0.5 rounded-full ${STATUS_COLOR[status] ?? "bg-gray-800 text-gray-500"}`}>
            {status}
          </span>
        </div>
      </div>

      <p className="text-xs text-gray-500 truncate">{entry.env}</p>

      {latestIntervention && currentSeg > 0 && (
        <div className="bg-yellow-950/40 border border-yellow-800/40 rounded-lg px-3 py-2 text-xs">
          <p className="text-yellow-400 font-semibold mb-1">GPT config (restart #{currentSeg})</p>
          <p className="font-mono text-gray-300">
            {Object.entries(latestIntervention.llm_suggested_hparams).map(([k, v]) => `${k}=${v}`).join(" · ")}
          </p>
        </div>
      )}

      {hb ? (
        <div className="space-y-2">
          <div className="flex justify-between text-xs">
            <span className="text-gray-400">Steps</span>
            <span className="font-mono text-gray-200">{hb.steps_completed.toLocaleString()}</span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-gray-400">Reward</span>
            <span className={`font-mono font-bold ${hb.current_reward > 0 ? "text-green-400" : "text-gray-300"}`}>
              {hb.current_reward.toFixed(1)}
            </span>
          </div>
          <div className="w-full bg-gray-800 rounded-full h-1">
            <div className={`h-1 rounded-full transition-all duration-700
              ${hasNaN ? "bg-red-500" : status === "completed" ? "bg-emerald-500" : st.bar}`}
              style={{ width: `${pct}%` }} />
          </div>
          {hasNaN && <p className="text-xs text-red-400 font-semibold animate-pulse">⚠ NaN loss — Sentinel intervening…</p>}
        </div>
      ) : (
        <div className="flex items-center gap-2 text-xs text-gray-600">
          <div className="w-2 h-2 rounded-full bg-gray-700 animate-pulse" /> Starting…
        </div>
      )}

      {history.length >= 2 && (
        <div className="border-t border-gray-800 pt-2">
          <MiniChart history={history} algoRgb={st.rgb} hasNaN={hasNaN} />
          {currentSeg > 0 && (
            <div className="flex items-center gap-3 mt-1 text-xs text-gray-600 flex-wrap">
              {Array.from({ length: currentSeg + 1 }, (_, i) => (
                <span key={i} className="flex items-center gap-1">
                  <span className="w-3 h-0.5 inline-block rounded" style={{ backgroundColor: SEG_COLORS[i % SEG_COLORS.length] }} />
                  {i === 0 ? "orig" : `restart ${i}`}
                </span>
              ))}
              <span className="flex items-center gap-1">
                <span className="w-3 h-0.5 inline-block rounded bg-red-500" />NaN
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Live leaderboard ──────────────────────────────────────────────────────────

const ENV_SCALE_NOTE: Record<string, string> = {
  "HalfCheetah-v5": "max ~12 000",
  "HalfCheetah-v4": "max ~12 000",
  "Hopper-v5":      "max ~3 500",
  "Hopper-v4":      "max ~3 500",
  "Ant-v5":         "max ~8 000",
  "Walker2d-v5":    "max ~6 000",
  "Humanoid-v5":    "max ~8 000",
};

function Leaderboard({
  plan, heartbeats, results, sentinel, phase,
}: {
  plan: SpawnEntry[];
  heartbeats: Heartbeat[];
  results: EvalResult[];
  sentinel: SentinelEntry[];
  phase: "racing" | "done";
}) {
  // During race use current_reward; when done use mean_return from eval
  const rows = plan.map(entry => {
    const hb = heartbeats.find(h => h.agent_id === entry.id);
    const res = results.find(r => r.agent_id === entry.id);
    const sentCount = sentinel.filter(s => s.agent_id === entry.id).length;
    return {
      id: entry.id,
      algo: entry.algo,
      env: entry.env,
      score: phase === "done" && res ? res.mean_return : (hb?.current_reward ?? -Infinity),
      status: hb?.status ?? "waiting",
      restarts: sentCount,
      done: phase === "done" && res?.status === "completed",
      failed: phase === "done" && res?.status !== "completed",
    };
  });

  // Group by env, sort by score within each group
  const byEnv = new Map<string, typeof rows>();
  for (const r of rows) {
    if (!byEnv.has(r.env)) byEnv.set(r.env, []);
    byEnv.get(r.env)!.push(r);
  }
  for (const grp of byEnv.values()) grp.sort((a, b) => b.score - a.score);

  const MEDALS = ["🥇", "🥈", "🥉"];
  const multiEnv = byEnv.size > 1;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-bold text-gray-200 uppercase tracking-wider">Leaderboard</h2>
        <span className={`text-xs px-2 py-0.5 rounded-full ${phase === "done" ? "bg-emerald-900 text-emerald-300" : "bg-green-900 text-green-300 animate-pulse"}`}>
          {phase === "done" ? "final" : "live"}
        </span>
      </div>

      {multiEnv && (
        <div className="bg-amber-950/40 border border-amber-800/40 rounded-lg px-3 py-2 text-xs text-amber-300">
          <p className="font-semibold mb-0.5">⚠ Different environments</p>
          <p className="text-amber-400/70">Scores below are <em>not</em> directly comparable across envs — grouped separately.</p>
        </div>
      )}

      {[...byEnv.entries()].map(([env, grp]) => (
        <div key={env} className="space-y-2">
          <div className="flex items-center justify-between">
            <p className="text-xs font-semibold text-gray-400">{env}</p>
            {ENV_SCALE_NOTE[env] && (
              <p className="text-xs text-gray-600">{ENV_SCALE_NOTE[env]}</p>
            )}
          </div>
          {grp.map((row, i) => {
            const st = as(row.algo);
            return (
              <div key={row.id}
                className={`flex items-center gap-2 rounded-lg px-3 py-2 border transition-all duration-500
                  ${row.failed ? "bg-red-950/30 border-red-900/30 opacity-60" :
                    i === 0 ? "bg-gray-800 border-gray-700 ring-1 ring-violet-700/40" :
                    "bg-gray-900 border-gray-800"}`}>
                <span className="text-base w-6 shrink-0">{row.failed ? "✕" : (MEDALS[i] ?? "·")}</span>
                <span className={`text-xs font-bold px-1.5 py-0.5 rounded-full shrink-0 ${st.badge}`}>{row.algo}</span>
                <span className="text-xs text-gray-400 font-mono flex-1 truncate">{row.id}</span>
                {row.restarts > 0 && (
                  <span className="text-xs text-yellow-500 shrink-0">↩{row.restarts}</span>
                )}
                <span className={`text-xs font-mono font-bold shrink-0
                  ${row.failed ? "text-red-500" : row.score > 0 ? "text-green-400" : "text-gray-400"}`}>
                  {row.score === -Infinity ? "—" : row.score.toFixed(0)}
                </span>
              </div>
            );
          })}
        </div>
      ))}

      {phase === "racing" && (
        <p className="text-xs text-gray-700 text-center pt-1">Live reward · updates every 2 s</p>
      )}
    </div>
  );
}

// ── Sentinel banner ───────────────────────────────────────────────────────────

function SentinelBanner({ entry }: { entry: SentinelEntry }) {
  const isKilled = entry.outcome === "killed_permanently";
  return (
    <div className={`flex items-start gap-3 rounded-xl p-3 border text-sm
      ${isKilled ? "bg-red-950 border-red-700" : "bg-amber-950 border-amber-700"}`}>
      <span className="text-lg mt-0.5">{isKilled ? "🔴" : "⚠️"}</span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1 flex-wrap">
          <span className="font-bold text-amber-300 text-xs">Sentinel</span>
          <span className="font-mono text-xs text-gray-400">{entry.agent_id}</span>
          <span className="text-xs text-gray-500">{new Date(entry.timestamp).toLocaleTimeString()}</span>
        </div>
        <div className="bg-black/30 rounded-lg p-2 font-mono text-xs space-y-1">
          <p className="text-red-300">Failed: lr={String(entry.failed_hparams.lr)}</p>
          {Object.keys(entry.llm_suggested_hparams).length > 0 && (
            <p className="text-green-300">GPT → {Object.entries(entry.llm_suggested_hparams).map(([k,v]) => `${k}=${v}`).join(" ")}</p>
          )}
        </div>
        <p className={`text-xs mt-1 font-semibold
          ${entry.outcome === "completed" ? "text-green-400" : entry.outcome === "killed_permanently" ? "text-red-400" : "text-yellow-400"}`}>
          → {entry.outcome}
        </p>
      </div>
    </div>
  );
}

// ── Video modal ───────────────────────────────────────────────────────────────

function VideoModal({ url, agentId, onClose }: { url: string; agentId: string; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
      onClick={onClose}>
      <div className="bg-gray-900 border border-gray-700 rounded-2xl p-5 max-w-2xl w-full mx-4 space-y-4"
        onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between">
          <div>
            <p className="font-bold text-gray-100">Agent inference</p>
            <p className="text-xs text-gray-500 font-mono">{agentId}</p>
          </div>
          <button onClick={onClose}
            className="text-gray-500 hover:text-gray-200 text-xl leading-none">✕</button>
        </div>
        <video src={url} controls autoPlay loop
          className="w-full rounded-xl bg-black max-h-[60vh] object-contain">
          Your browser does not support HTML video.
        </video>
        <p className="text-xs text-gray-600">One deterministic episode recorded from the saved checkpoint.</p>
      </div>
    </div>
  );
}

// ── Header ────────────────────────────────────────────────────────────────────

function Header() {
  return (
    <div className="flex items-center gap-3 mb-10">
      <div className="w-10 h-10 rounded-xl bg-violet-600 flex items-center justify-center font-bold text-lg select-none">RL</div>
      <div>
        <p className="text-xl font-bold text-gray-100">AutoRL</p>
        <p className="text-xs text-gray-500">Multi-agent training race</p>
      </div>
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────────

export default function HomePage() {
  const [task,       setTask]       = useState("");
  const [phase,      setPhase]      = useState<Phase>("idle");
  const [plan,       setPlan]       = useState<SpawnEntry[]>([]);
  const [runName,    setRunName]    = useState("");
  const [runDir,     setRunDir]     = useState("");
  const [heartbeats, setHeartbeats] = useState<Heartbeat[]>([]);
  const [sentinel,   setSentinel]   = useState<SentinelEntry[]>([]);
  const [results,    setResults]    = useState<EvalResult[]>([]);
  const [best,       setBest]       = useState<EvalResult | null>(null);
  const [errorMsg,   setErrorMsg]   = useState("");

  const [history,     setHistory]     = useState<Record<string, HistPt[]>>({});
  const [animStates,  setAnimStates]  = useState<Record<string, AnimState>>({});
  const [inferring,   setInferring]   = useState<Record<string, boolean>>({});
  const [videos,      setVideos]      = useState<Record<string, string>>({});
  const [videoModal,  setVideoModal]  = useState<{ agentId: string; url: string } | null>(null);

  const prevSentinelCount = useRef<Record<string, number>>({});
  const pollRef           = useRef<ReturnType<typeof setInterval> | null>(null);
  const textareaRef       = useRef<HTMLTextAreaElement>(null);

  // ── Polling ──────────────────────────────────────────────────────────────────

  const startPolling = useCallback((name: string) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${BACKEND}/api/status/${name}`);
        if (!res.ok) return;
        const data = await res.json();
        const hbs: Heartbeat[]      = data.heartbeats   ?? [];
        const slog: SentinelEntry[] = data.sentinel_log ?? [];
        setHeartbeats(hbs);
        setSentinel(slog);

        const sentByAgent: Record<string, number> = {};
        for (const e of slog) sentByAgent[e.agent_id] = (sentByAgent[e.agent_id] ?? 0) + 1;

        const newAnimStates: Record<string, AnimState> = {};
        for (const [agentId, cnt] of Object.entries(sentByAgent)) {
          const prev = prevSentinelCount.current[agentId] ?? 0;
          if (cnt > prev) {
            newAnimStates[agentId] = "vanish";
            setTimeout(() => {
              setAnimStates(a => ({ ...a, [agentId]: "appear" }));
              setTimeout(() => setAnimStates(a => ({ ...a, [agentId]: "idle" })), 600);
            }, 600);
          }
        }
        prevSentinelCount.current = sentByAgent;
        if (Object.keys(newAnimStates).length > 0) setAnimStates(a => ({ ...a, ...newAnimStates }));

        setHistory(prev => {
          const next = { ...prev };
          for (const hb of hbs) {
            const seg = sentByAgent[hb.agent_id] ?? 0;
            const pts = next[hb.agent_id] ?? [];
            const last = pts.at(-1);
            if (!last || last.steps !== hb.steps_completed) {
              next[hb.agent_id] = [...pts, { steps: hb.steps_completed, reward: hb.current_reward, seg }];
            }
          }
          return next;
        });

        if (data.status === "completed" || data.status === "failed") {
          clearInterval(pollRef.current!);
          const rRes = await fetch(`${BACKEND}/api/results/${name}`);
          if (rRes.ok) {
            const rData = await rRes.json();
            setResults(rData.results ?? []);
            setBest(rData.best ?? null);
          }
          setPhase("done");
        }
      } catch { /* ignore */ }
    }, POLL_MS);
  }, []);

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  // ── Handlers ─────────────────────────────────────────────────────────────────

  const handleGeneratePlan = async () => {
    if (!task.trim()) return;
    setPhase("planning"); setErrorMsg("");
    try {
      const res = await fetch(`${BACKEND}/api/plan`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task: task.trim() }),
      });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const data = await res.json();
      setPlan(data.plan); setRunName(data.run_name); setRunDir(data.run_dir);
      setPhase("plan_ready");
    } catch (e) { setErrorMsg(e instanceof Error ? e.message : String(e)); setPhase("error"); }
  };

  const handleLaunch = async () => {
    setPhase("launching");
    try {
      const res = await fetch(`${BACKEND}/api/run`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task, run_dir: runDir, plan }),
      });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      setPhase("racing"); startPolling(runName);
    } catch (e) { setErrorMsg(e instanceof Error ? e.message : String(e)); setPhase("error"); }
  };

  const handleInfer = async (agentId: string) => {
    // Always infer fresh from the best checkpoint at this moment
    setInferring(p => ({ ...p, [agentId]: true }));
    try {
      const res = await fetch(`${BACKEND}/api/infer`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_name: runName, agent_id: agentId }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail ?? res.statusText);
      }
      const data = await res.json();
      setVideoModal({ agentId, url: `${BACKEND}/api/video/${data.filename}` });
    } catch (e) {
      alert(`Inference failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setInferring(p => ({ ...p, [agentId]: false }));
    }
  };

  const handleReset = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    setPhase("idle"); setTask(""); setPlan([]); setRunName(""); setRunDir("");
    setHeartbeats([]); setSentinel([]); setResults([]); setBest(null); setErrorMsg("");
    setHistory({}); setAnimStates({}); setInferring({}); setVideos({}); setVideoModal(null);
    prevSentinelCount.current = {};
    setTimeout(() => textareaRef.current?.focus(), 50);
  };

  // ── Derived ──────────────────────────────────────────────────────────────────

  const hbById = Object.fromEntries(heartbeats.map(h => [h.agent_id, h]));
  const sentByAgent = sentinel.reduce<Record<string, SentinelEntry[]>>((acc, e) => {
    acc[e.agent_id] = [...(acc[e.agent_id] ?? []), e];
    return acc;
  }, {});

  // ── Render ────────────────────────────────────────────────────────────────────

  const isCentered = !["racing", "done"].includes(phase);

  return (
    <div className={`min-h-screen bg-gray-950 ${isCentered ? "flex flex-col items-center justify-center p-6 py-12" : "p-6 pt-8"}`}>
      {videoModal && (
        <VideoModal url={videoModal.url} agentId={videoModal.agentId} onClose={() => setVideoModal(null)} />
      )}

      <Header />

      {/* ── IDLE / ERROR ── */}
      {(phase === "idle" || phase === "error") && (
        <div className="w-full max-w-xl space-y-4">
          <p className="text-center text-gray-300 text-lg font-medium">What do you want to train?</p>
          <div className="relative">
            <textarea ref={textareaRef} autoFocus value={task}
              onChange={e => setTask(e.target.value)}
              onKeyDown={e => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleGeneratePlan(); }}
              rows={3} placeholder="Describe the RL task…"
              className="w-full bg-gray-900 border border-gray-700 focus:border-violet-500 rounded-xl px-4 py-3 text-gray-100 placeholder-gray-600 text-sm resize-none outline-none transition-colors" />
            <p className="absolute bottom-2 right-3 text-xs text-gray-600">⌘ Enter</p>
          </div>
          <div className="flex flex-wrap gap-2">
            {SUGGESTIONS.map(s => (
              <button key={s} onClick={() => setTask(s)}
                className="text-xs px-3 py-1.5 rounded-full bg-gray-800 hover:bg-gray-700 text-gray-400 hover:text-gray-200 border border-gray-700 transition-colors">{s}</button>
            ))}
          </div>
          {phase === "error" && (
            <div className="bg-red-950 border border-red-800 rounded-xl p-3 text-sm text-red-300">
              <p className="font-semibold mb-1">Error</p>
              <p className="font-mono text-xs break-all">{errorMsg}</p>
              <p className="text-xs text-red-400 mt-2">Backend running? <code className="font-mono">bash ui/agent/start.sh</code></p>
            </div>
          )}
          <button onClick={handleGeneratePlan} disabled={!task.trim()}
            className="w-full bg-violet-600 hover:bg-violet-500 disabled:bg-gray-800 disabled:text-gray-600 text-white font-semibold py-3 rounded-xl transition-colors text-sm">
            Generate Agent Lineup →
          </button>
        </div>
      )}

      {/* ── PLANNING ── */}
      {phase === "planning" && <PlanningScreen task={task} />}

      {/* ── PLAN READY ── */}
      {phase === "plan_ready" && (
        <div className="w-full max-w-2xl space-y-5">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold text-gray-100">Agent Lineup</h2>
            <span className="text-xs text-gray-500 font-mono">{runName}</span>
          </div>
          <p className="text-sm text-gray-400 italic">&ldquo;{task}&rdquo;</p>
          <div className="grid grid-cols-3 gap-3">
            {[["Agents", plan.length], ["Local", plan.filter(e=>e.exec==="local").length], ["RunPod", plan.filter(e=>e.exec==="runpod").length]].map(([l,v]) => (
              <div key={String(l)} className="bg-gray-900 border border-gray-800 rounded-xl p-3 text-center">
                <p className="text-2xl font-bold text-gray-100">{v}</p>
                <p className="text-xs text-gray-500 mt-0.5">{l}</p>
              </div>
            ))}
          </div>
          {plan.some(e => (e.hparams.lr as number) >= 0.1) && (
            <div className="flex items-start gap-3 bg-amber-950/40 border border-amber-800/50 rounded-xl p-3">
              <span className="text-amber-400 text-lg mt-0.5">⚠</span>
              <div>
                <p className="text-sm font-semibold text-amber-300">Doom Loop Sentinel active</p>
                <p className="text-xs text-gray-400 mt-0.5">One agent has a dangerously high LR — Sentinel will detect the NaN crash and ask GPT to suggest a recovery config.</p>
              </div>
            </div>
          )}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {plan.map((e, i) => <AgentLineupCard key={e.id} entry={e} index={i} />)}
          </div>
          <div className="flex gap-3 pt-1">
            <button onClick={handleReset}
              className="flex-1 bg-gray-800 hover:bg-gray-700 text-gray-400 font-semibold py-3 rounded-xl transition-colors text-sm">
              ← Change task
            </button>
            <button onClick={handleLaunch}
              className="flex-1 bg-violet-600 hover:bg-violet-500 text-white font-semibold py-3 rounded-xl transition-colors text-sm">
              🚀 Launch Race
            </button>
          </div>
        </div>
      )}

      {/* ── LAUNCHING ── */}
      {phase === "launching" && (
        <div className="flex flex-col items-center gap-4 text-center">
          <div className="w-12 h-12 rounded-full border-2 border-violet-500 border-t-transparent animate-spin" />
          <p className="text-gray-200 font-medium">Launching {plan.length} agents…</p>
          <p className="text-xs text-gray-500">Swarm runner starting — dashboard will appear shortly</p>
        </div>
      )}

      {/* ── RACING — two-panel layout ── */}
      {phase === "racing" && (
        <div className="w-full max-w-7xl">
          {/* top bar */}
          <div className="flex items-center justify-between mb-5">
            <div className="flex items-center gap-3">
              <h2 className="text-lg font-bold text-gray-100">Race Live</h2>
              <span className="text-xs px-2 py-0.5 rounded-full bg-green-900 text-green-300 animate-pulse">training</span>
            </div>
            <div className="text-xs text-gray-500 space-x-3">
              <span>{heartbeats.filter(h=>h.status==="training").length} running</span>
              <span>{heartbeats.filter(h=>h.status==="completed").length} done</span>
              <span className="font-mono">{runName}</span>
              <button onClick={handleReset}
                className="ml-2 text-xs px-2 py-0.5 rounded bg-gray-800 hover:bg-gray-700 text-gray-400 border border-gray-700">
                ✕ reset
              </button>
            </div>
          </div>
          <p className="text-xs text-gray-500 italic mb-5">&ldquo;{task}&rdquo;</p>

          <div className="flex gap-5 items-start">
            {/* LEFT: scrollable agent cascade */}
            <div className="flex-1 min-w-0 space-y-4 max-h-[calc(100vh-220px)] overflow-y-auto pr-1">
              {plan.map(e => (
                <LiveAgentCard key={e.id}
                  entry={e}
                  hb={hbById[e.id]}
                  history={history[e.id] ?? []}
                  sentinelEntries={sentByAgent[e.id] ?? []}
                  animState={animStates[e.id] ?? "idle"}
                  onInfer={() => handleInfer(e.id)}
                  inferring={!!inferring[e.id]}
                  hasVideo={!!videos[e.id]}
                />
              ))}

              {/* Sentinel banners at bottom of left column */}
              {sentinel.length > 0 && (
                <div className="space-y-3 pt-2">
                  <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Sentinel Interventions</p>
                  {sentinel.map((e, i) => <SentinelBanner key={i} entry={e} />)}
                </div>
              )}
              <p className="text-xs text-gray-700 text-center py-2">Refreshing every 2 s</p>
            </div>

            {/* RIGHT: sticky leaderboard */}
            <div className="w-72 xl:w-80 shrink-0 sticky top-6">
              <div className="bg-gray-900 border border-gray-800 rounded-2xl p-4">
                <Leaderboard
                  plan={plan}
                  heartbeats={heartbeats}
                  results={[]}
                  sentinel={sentinel}
                  phase="racing"
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── DONE — two-panel layout ── */}
      {phase === "done" && (
        <div className="w-full max-w-7xl">
          <div className="flex items-center justify-between mb-5">
            <h2 className="text-lg font-bold text-gray-100">Race Complete 🏁</h2>
            <button onClick={handleReset}
              className="text-xs px-3 py-1.5 rounded-lg bg-violet-600 hover:bg-violet-500 text-white font-semibold transition-colors">
              ← Train again
            </button>
          </div>

          <div className="flex gap-5 items-start">
            {/* LEFT: agent cards with infer buttons */}
            <div className="flex-1 min-w-0 space-y-4 max-h-[calc(100vh-180px)] overflow-y-auto pr-1">
              {best && (
                <div className="bg-emerald-950 border border-emerald-700 rounded-xl p-4">
                  <div className="flex items-center gap-3 mb-3">
                    <span className="text-3xl">🏆</span>
                    <div>
                      <p className="font-bold text-emerald-300 text-lg">{best.algo} wins!</p>
                      <p className="text-xs text-gray-400">{best.env} · {best.agent_id}</p>
                    </div>
                    <button onClick={() => handleInfer(best.agent_id)}
                      disabled={!!inferring[best.agent_id]}
                      className="ml-auto text-xs px-3 py-1.5 rounded-lg bg-emerald-800 hover:bg-emerald-700 text-emerald-200 font-semibold transition-colors disabled:opacity-50">
                      {inferring[best.agent_id] ? "⏳ Recording…" : "▶ Watch inference"}
                    </button>
                  </div>
                  <div className="grid grid-cols-2 gap-3 text-center mb-3">
                    <div className="bg-black/20 rounded-lg p-2">
                      <p className="text-2xl font-bold text-emerald-300">{best.mean_return.toFixed(0)}</p>
                      <p className="text-xs text-gray-400">mean return</p>
                    </div>
                    <div className="bg-black/20 rounded-lg p-2">
                      <p className="text-2xl font-bold text-gray-300">±{best.std_return.toFixed(0)}</p>
                      <p className="text-xs text-gray-400">std</p>
                    </div>
                  </div>
                  {history[best.agent_id]?.length >= 2 && (
                    <div className="bg-black/20 rounded-lg p-2">
                      <MiniChart history={history[best.agent_id]} algoRgb={as(best.algo).rgb} hasNaN={false} />
                    </div>
                  )}
                </div>
              )}

              {/* All agent cards with infer */}
              {plan.map(e => (
                <LiveAgentCard key={e.id}
                  entry={e}
                  hb={hbById[e.id]}
                  history={history[e.id] ?? []}
                  sentinelEntries={sentByAgent[e.id] ?? []}
                  animState="idle"
                  onInfer={() => handleInfer(e.id)}
                  inferring={!!inferring[e.id]}
                  hasVideo={!!videos[e.id]}
                />
              ))}

              {sentinel.length > 0 && (
                <div className="space-y-3 pt-2">
                  <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Sentinel Interventions</p>
                  {sentinel.map((e, i) => <SentinelBanner key={i} entry={e} />)}
                </div>
              )}
            </div>

            {/* RIGHT: final leaderboard */}
            <div className="w-72 xl:w-80 shrink-0 sticky top-6">
              <div className="bg-gray-900 border border-gray-800 rounded-2xl p-4">
                <Leaderboard
                  plan={plan}
                  heartbeats={heartbeats}
                  results={results}
                  sentinel={sentinel}
                  phase="done"
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
