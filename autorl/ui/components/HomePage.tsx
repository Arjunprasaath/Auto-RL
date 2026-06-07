"use client";

import { useState, useRef, useEffect, useCallback } from "react";

const BACKEND = "http://localhost:8000";
const POLL_MS = 2_000;

const SUGGESTIONS = [
  "Train the best MuJoCo locomotion policy",
  "Race PPO vs SAC on HalfCheetah-v5",
  "Train an agent on Hopper-v5",
  "Train a Countdown reasoning agent with GRPO",
  "Race PPO and A2C on ALE/Pong-v5",
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
  doctor_stderr?: string;
  rationale?: string;
}
interface EvalResult {
  agent_id: string; algo: string; env: string; status: string;
  mean_return: number; std_return: number; checkpoint_path: string;
}
interface InferenceCase {
  numbers: number[]; target: number; prompt?: string;
  model_response: string; success: boolean;
}
interface GrpoInferData { results: InferenceCase[]; baseline: InferenceCase[]; }
interface HistPt { steps: number; reward: number; seg: number; }
type Phase = "idle" | "planning" | "plan_ready" | "launching" | "racing" | "done" | "error";
type AnimState = "vanish" | "appear" | "idle";

// ── Env-family detection ───────────────────────────────────────────────────────

type EnvFamily = "mujoco" | "classic" | "toytext" | "box2d" | "atari" | "grpo";

const ATARI_KEYWORDS = [
  "pong","breakout","spaceinvaders","asteroids","qbert","montezuma","mspacman",
  "beamrider","enduro","pitfall","venture","videopinball","atlantis","assault",
  "alien","amidar","kangaroo","krull","battlezone","berzerk","centipede",
  "choppercommand","crazyclimber","defender","demonattack","doubledunk",
  "fishingderby","freeway","frostbite","gopher","gravitar","hero","icehockey",
  "jamesbond","nameThisGame","phoenix","privateeye","roadrunner","robotank",
  "seaquest","skiing","solaris","stargunner","tennis","timepilot","tutankham",
  "upndown","wizard",
];

function detectEnvFamily(env: string): EnvFamily {
  if (env === "Countdown") return "grpo";
  const e = env.toLowerCase();
  if (["frozenlake", "taxi", "cliffwalking", "blackjack"].some(k => e.includes(k))) return "toytext";
  if (["lunarlander", "bipedalwalker", "carracing"].some(k => e.includes(k))) return "box2d";
  if (["halfcheetah", "hopper", "ant", "walker2d", "swimmer", "humanoid",
       "reacher", "pusher", "invertedpendulum"].some(k => e.includes(k))) return "mujoco";
  if (e.startsWith("ale/") || ATARI_KEYWORDS.some(k => e.includes(k))) return "atari";
  return "classic";
}

function isBinaryRewardEnv(env: string): boolean {
  const e = env.toLowerCase();
  return ["frozenlake", "blackjack"].some(k => e.includes(k));
}

const ENV_FAMILY_META: Record<EnvFamily, {
  label: string; icon: string; fps: string; color: string;
  desc: string; inferLabel: string; episodeNote: string;
}> = {
  mujoco:  { label: "MuJoCo",     icon: "⬡", fps: "30 fps", color: "text-amber-400",
             desc: "Continuous control — physics simulation",
             inferLabel: "▶ watch",  episodeNote: "1 deterministic episode" },
  classic: { label: "Classic",    icon: "◈", fps: "30 fps", color: "text-gray-300",
             desc: "Classic control task",
             inferLabel: "▶ run",    episodeNote: "3 deterministic episodes" },
  toytext: { label: "Grid World", icon: "⊞", fps: "4 fps",  color: "text-green-400",
             desc: "Discrete grid world",
             inferLabel: "▶ replay", episodeNote: "5 complete episodes" },
  box2d:   { label: "Box2D",      icon: "◉", fps: "30 fps", color: "text-blue-400",
             desc: "Physics-based 2D simulation",
             inferLabel: "▶ fly",    episodeNote: "2 complete episodes" },
  atari:   { label: "Atari",      icon: "▦", fps: "30 fps", color: "text-red-400",
             desc: "Arcade pixel game — CNN policy",
             inferLabel: "▶ play",   episodeNote: "2 game episodes" },
  grpo:    { label: "Countdown",  icon: "∑", fps: "—",      color: "text-amber-500",
             desc: "LLM arithmetic reasoning (no video)",
             inferLabel: "",          episodeNote: "" },
};

// ── Style maps ─────────────────────────────────────────────────────────────────

const ALGO_STYLE: Record<string, { accent: string; tag: string; bar: string; rgb: string }> = {
  PPO:  { accent: "text-violet-400",  tag: "text-violet-300 border-violet-700",  bar: "bg-violet-500",  rgb: "#8b5cf6" },
  SAC:  { accent: "text-cyan-400",    tag: "text-cyan-300 border-cyan-700",      bar: "bg-cyan-500",    rgb: "#06b6d4" },
  A2C:  { accent: "text-pink-400",    tag: "text-pink-300 border-pink-700",      bar: "bg-pink-500",    rgb: "#ec4899" },
  GRPO: { accent: "text-amber-400",   tag: "text-amber-300 border-amber-700",    bar: "bg-amber-500",   rgb: "#f97316" },
};
const DEF_STYLE = { accent: "text-gray-400", tag: "text-gray-300 border-gray-700", bar: "bg-gray-500", rgb: "#6b7280" };
const as = (algo: string) => ALGO_STYLE[algo] ?? DEF_STYLE;
const SEG_COLORS = ["#8b5cf6", "#06b6d4", "#34d399", "#fbbf24", "#f472b6"];

// ── Mini SVG reward chart ──────────────────────────────────────────────────────

function MiniChart({
  history, algoRgb, hasNaN, family, env,
}: {
  history: HistPt[]; algoRgb: string; hasNaN: boolean; family: EnvFamily; env: string;
}) {
  const W = 360, H = 64;

  if (family === "toytext" && isBinaryRewardEnv(env)) {
    const successes = history.filter(p => p.reward > 0).length;
    const rate = history.length > 0 ? successes / history.length : 0;
    return (
      <div className="w-full h-[72px] flex flex-col justify-center gap-2 px-1">
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">success rate</span>
          <span className="text-green-400">{(rate * 100).toFixed(0)}%</span>
        </div>
        <div className="w-full bg-[#1a1a1a] h-1.5">
          <div className="h-1.5 bg-green-500 transition-all duration-700" style={{ width: `${rate * 100}%` }} />
        </div>
        <p className="text-xs text-gray-600">{successes}/{history.length} episodes</p>
      </div>
    );
  }

  if (history.length < 2) return (
    <div className="w-full h-[72px] flex items-center">
      <p className="text-sm text-gray-700">awaiting data…</p>
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
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-[64px]" preserveAspectRatio="none">
      {[...segs.entries()].map(([seg, pts]) => {
        const isFailSeg = seg === maxSeg && hasNaN;
        const color = isFailSeg ? "#ef4444" : (SEG_COLORS[seg % SEG_COLORS.length] ?? algoRgb);
        const pathD = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${xS(p.steps).toFixed(1)} ${yS(p.reward).toFixed(1)}`).join(" ");
        return (
          <g key={seg}>
            <path d={pathD} fill="none" stroke={color} strokeWidth="1.5"
              strokeLinecap="round" strokeLinejoin="round"
              strokeDasharray={isFailSeg ? "4 3" : undefined} />
            {seg > 0 && pts.length > 0 && (
              <circle cx={xS(pts[0].steps)} cy={yS(pts[0].reward)} r="3"
                fill={color} stroke="#0d0d0d" strokeWidth="1" />
            )}
          </g>
        );
      })}
      <text x="2" y="10" fontSize="8" fill="#555" fontFamily="monospace">{maxR.toFixed(0)}</text>
      <text x="2" y={H - 2} fontSize="8" fill="#555" fontFamily="monospace">{minR.toFixed(0)}</text>
    </svg>
  );
}

// ── Planning screen ────────────────────────────────────────────────────────────

const PLAN_STEPS = [
  "analysing task",
  "selecting algorithms",
  "designing agent lineup",
  "setting hyperparameters",
  "validating spawn plan",
];

function PlanningScreen({ task }: { task: string }) {
  const [step, setStep] = useState(0);
  const [dots, setDots] = useState(".");
  useEffect(() => {
    const id = setInterval(() => setStep(s => Math.min(s + 1, PLAN_STEPS.length - 1)), 3000);
    return () => clearInterval(id);
  }, []);
  useEffect(() => {
    const id = setInterval(() => setDots(d => d.length >= 3 ? "." : d + "."), 400);
    return () => clearInterval(id);
  }, []);
  return (
    <div className="w-full max-w-xl space-y-5">
      <div className="border border-[#2a2a2a] bg-[#111] px-4 py-3">
        <p className="text-xs text-gray-600 mb-1">task</p>
        <p className="text-base text-gray-300">&ldquo;{task}&rdquo;</p>
      </div>
      <div className="space-y-3">
        {PLAN_STEPS.map((label, i) => {
          const done = i < step, active = i === step;
          return (
            <div key={i} className="flex items-center gap-3 text-sm">
              <span className={done ? "text-amber-500" : active ? "text-amber-500" : "text-gray-700"}>
                {done ? "✓" : active ? "›" : "·"}
              </span>
              <span className={done ? "text-gray-600 line-through" : active ? "text-gray-200" : "text-gray-700"}>
                {label}{active ? dots : ""}
              </span>
            </div>
          );
        })}
      </div>
      <p className="text-xs text-gray-700">orchestrator LLM · ~10–20 s</p>
    </div>
  );
}

// ── Editable pre-launch agent card ────────────────────────────────────────────

function AgentLineupCard({
  entry, index, onDelete, onUpdate,
}: {
  entry: SpawnEntry; index: number;
  onDelete: () => void; onUpdate: (u: SpawnEntry) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState<SpawnEntry>(entry);
  const st = as(entry.algo);
  const lr = entry.hparams.lr as number;
  const isDoom = lr >= 0.1;

  function setHp(key: string, raw: string) {
    const num = parseFloat(raw);
    setDraft(d => ({ ...d, hparams: { ...d.hparams, [key]: isNaN(num) ? raw : num } }));
  }

  const hpList = [
    ["lr",       String(entry.hparams.lr)],
    ["seed",     String(entry.hparams.seed ?? "—")],
    ...(entry.hparams.n_steps  != null ? [["n_steps",  String(entry.hparams.n_steps)]]  : []),
    ...(entry.hparams.gamma    != null ? [["gamma",    String(entry.hparams.gamma)]]    : []),
    ...(entry.hparams.ent_coef != null ? [["ent_coef", String(entry.hparams.ent_coef)]] : []),
    ...(entry.hparams.model    != null ? [["model",    String(entry.hparams.model).split("/").pop() ?? ""]] : []),
  ] as [string, string][];

  return (
    <div className={`border border-[#2a2a2a] bg-[#111] text-sm ${isDoom ? "border-red-900/60" : ""}`}>
      {/* top bar */}
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-[#2a2a2a]">
        <div className="flex items-center gap-2.5">
          <span className="text-gray-600 text-xs">#{index + 1}</span>
          <span className={`border px-2 py-0.5 text-sm ${st.tag}`}>{entry.algo}</span>
          <span className="text-gray-300">{entry.env}</span>
          {isDoom && <span className="text-red-500">☠</span>}
        </div>
        <div className="flex items-center gap-3 text-gray-500">
          <span className="text-xs">{entry.time_budget_min}m</span>
          <button onClick={() => { setDraft(entry); setEditing(e => !e); }}
            className="hover:text-amber-400 transition-colors">{editing ? "cancel" : "edit"}</button>
          <button onClick={onDelete} className="hover:text-red-400 transition-colors text-base leading-none">×</button>
        </div>
      </div>

      {/* hparams row */}
      {!editing && (
        <div className="px-3 py-2.5 flex flex-wrap gap-x-5 gap-y-1.5">
          {hpList.map(([k, v]) => (
            <span key={k} className={k === "lr" && isDoom ? "text-red-400" : "text-gray-400"}>
              <span className="text-gray-600">{k}=</span>{v}
            </span>
          ))}
          {isDoom && <span className="text-red-500 ml-auto text-xs">sentinel bait</span>}
        </div>
      )}

      {/* edit form */}
      {editing && (
        <div className="px-3 py-3 space-y-3 border-t border-[#2a2a2a]">
          <div className="grid grid-cols-2 gap-x-6 gap-y-2">
            <div className="flex items-center gap-2">
              <span className="text-gray-600 w-20 text-xs">budget</span>
              <input type="number" step="1" min="1" max="30"
                value={draft.time_budget_min}
                onChange={e => setDraft(d => ({ ...d, time_budget_min: parseInt(e.target.value) || d.time_budget_min }))}
                className="flex-1 bg-[#0d0d0d] border border-[#333] px-2 py-1 text-sm text-gray-200 outline-none focus:border-amber-600" />
            </div>
            {(["lr","seed","gamma","ent_coef","n_steps"] as const)
              .filter(k => draft.hparams[k] != null)
              .map(k => (
                <div key={k} className="flex items-center gap-2">
                  <span className="text-gray-600 w-20 text-xs">{k}</span>
                  <input type="number" step={k === "seed" ? "1" : k === "n_steps" ? "256" : "any"}
                    value={String(draft.hparams[k])}
                    onChange={e => setHp(k, e.target.value)}
                    className="flex-1 bg-[#0d0d0d] border border-[#333] px-2 py-1 text-sm text-gray-200 outline-none focus:border-amber-600" />
                </div>
              ))}
          </div>
          <div className="flex gap-2 pt-1">
            <button onClick={() => { onUpdate(draft); setEditing(false); }}
              className="flex-1 bg-amber-600 hover:bg-amber-500 text-black font-bold py-1.5 transition-colors text-sm">
              save
            </button>
            <button onClick={() => { setDraft(entry); setEditing(false); }}
              className="flex-1 border border-[#333] hover:border-gray-500 text-gray-400 py-1.5 transition-colors text-sm">
              cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Live agent card helpers ────────────────────────────────────────────────────

function rewardLabel(family: EnvFamily, env: string): string {
  if (family === "grpo") return "accuracy";
  if (family === "toytext") {
    if (env.toLowerCase().includes("blackjack")) return "win rate";
    if (isBinaryRewardEnv(env)) return "success";
    return "reward";
  }
  if (family === "atari") return "score";
  return "reward";
}

function formatLiveReward(reward: number, family: EnvFamily, env: string): string {
  if (family === "grpo") return `${(reward * 100).toFixed(1)}%`;
  if (family === "toytext" && isBinaryRewardEnv(env)) return `${(reward * 100).toFixed(1)}%`;
  return reward.toFixed(1);
}

// ── Live agent card ───────────────────────────────────────────────────────────

function LiveAgentCard({
  entry, hb, history, sentinelEntries, animState, onInfer, inferring,
}: {
  entry: SpawnEntry; hb?: Heartbeat; history: HistPt[];
  sentinelEntries: SentinelEntry[]; animState: AnimState;
  onInfer: () => void; inferring: boolean; hasVideo: boolean;
}) {
  const st = as(entry.algo);
  const family = detectEnvFamily(entry.env);
  const fmeta  = ENV_FAMILY_META[family];
  const hasNaN    = hb?.anomaly === "nan_loss";
  const status    = hb?.status ?? "waiting";
  const stepsPerSec = family === "toytext" ? 50 : family === "box2d" ? 60 : family === "atari" ? 10 : 100;
  const maxSteps    = entry.time_budget_min * 60 * stepsPerSec;
  const pct         = hb ? Math.min(100, (hb.steps_completed / maxSteps) * 100) : 0;
  const currentSeg  = history.length > 0 ? history[history.length - 1].seg : 0;
  const latestIntervention = sentinelEntries.at(-1);
  const isGrpo = entry.algo === "GRPO";

  const statusDot =
    status === "training"  ? <span className="text-green-400 animate-pulse">●</span> :
    status === "completed" ? <span className="text-amber-400">✓</span> :
    status === "failed"    ? <span className="text-red-400">✗</span> :
    status === "restarted" ? <span className="text-yellow-400">↩</span> :
                             <span className="text-gray-600">○</span>;

  const animClass = animState === "vanish" ? "agent-vanish" : animState === "appear" ? "agent-appear" : "";

  return (
    <div className={`border border-[#2a2a2a] bg-[#111] text-sm
      ${hasNaN ? "border-red-900/60" : status === "completed" ? "border-[#2a3a2a]" : ""}
      ${animClass}`}>

      {/* header */}
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-[#2a2a2a]">
        <div className="flex items-center gap-2.5">
          {statusDot}
          <span className={`border px-2 py-0.5 ${st.tag}`}>{entry.algo}</span>
          <span className="text-gray-400">{entry.id}</span>
          {currentSeg > 0 && <span className="text-yellow-500 text-xs">restart×{currentSeg}</span>}
        </div>
        <div className="flex items-center gap-2.5">
          <button onClick={onInfer} disabled={inferring}
            className={`px-2.5 py-1 border transition-colors text-sm
              ${inferring
                ? "border-[#222] text-gray-600 cursor-not-allowed"
                : "border-[#333] text-gray-500 hover:border-amber-600 hover:text-amber-400"}`}>
            {inferring
              ? (isGrpo ? "loading…" : "recording…")
              : (isGrpo ? "test" : "infer")}
          </button>
          <span className="text-gray-600">{status}</span>
        </div>
      </div>

      {/* env row */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-[#1a1a1a]">
        <span className="text-gray-400">{entry.env}</span>
        <span className={`${fmeta.color} text-xs`}>{fmeta.icon} {fmeta.label}</span>
      </div>

      {/* doctor intervention */}
      {latestIntervention && latestIntervention.failure_reason === "environment_setup_error" && (
        <div className="px-3 py-2 border-b border-[#2a2a2a] bg-[#0d1a1a]">
          <p className="text-cyan-400 mb-1 text-xs">
            env-doctor {latestIntervention.outcome === "fixed_retrying" ? "fixed" : "failed"}
          </p>
          {(latestIntervention.llm_suggested_hparams?.fix_commands as string[] | undefined)?.map((c, i) => (
            <p key={i} className="text-gray-400 text-xs">$ {c}</p>
          ))}
        </div>
      )}
      {latestIntervention && currentSeg > 0 && latestIntervention.failure_reason !== "environment_setup_error" && (
        <div className="px-3 py-2 border-b border-[#2a2a2a] bg-[#1a1500]">
          <p className="text-yellow-500 mb-0.5 text-xs">sentinel restart #{currentSeg}</p>
          <p className="text-gray-400 text-xs">
            {Object.entries(latestIntervention.llm_suggested_hparams).map(([k, v]) => `${k}=${v}`).join(" · ")}
          </p>
        </div>
      )}

      {/* metrics */}
      {hb ? (
        <div className="px-3 py-2.5 space-y-2">
          <div className="flex gap-6">
            <span className="text-gray-600">steps <span className="text-gray-200">{hb.steps_completed.toLocaleString()}</span></span>
            <span className="text-gray-600">{rewardLabel(family, entry.env)} <span className={`font-bold ${hb.current_reward > 0 ? "text-green-400" : "text-gray-300"}`}>
              {formatLiveReward(hb.current_reward, family, entry.env)}</span>
            </span>
          </div>
          <div className="w-full bg-[#1a1a1a] h-1">
            <div className={`h-1 transition-all duration-700 ${hasNaN ? "bg-red-500" : status === "completed" ? "bg-amber-500" : st.bar}`}
              style={{ width: `${pct}%` }} />
          </div>
          {hasNaN && <p className="text-red-400 text-xs">nan loss — sentinel intervening…</p>}
        </div>
      ) : (
        <div className="px-3 py-2.5 text-gray-600">starting…</div>
      )}

      {/* chart */}
      {history.length >= 2 && (
        <div className="border-t border-[#1a1a1a] px-2 py-1.5">
          <MiniChart history={history} algoRgb={st.rgb} hasNaN={hasNaN} family={family} env={entry.env} />
          {currentSeg > 0 && (
            <div className="flex items-center gap-3 mt-1 text-xs text-gray-600 flex-wrap">
              {Array.from({ length: currentSeg + 1 }, (_, i) => (
                <span key={i} className="flex items-center gap-1">
                  <span className="w-3 h-px inline-block" style={{ backgroundColor: SEG_COLORS[i % SEG_COLORS.length] }} />
                  {i === 0 ? "orig" : `r${i}`}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Leaderboard ────────────────────────────────────────────────────────────────

function formatScore(score: number, env: string): string {
  if (score === -Infinity) return "—";
  const family = detectEnvFamily(env);
  if (family === "grpo") return `${(score * 100).toFixed(0)}%`;
  if (family === "toytext" && isBinaryRewardEnv(env)) return `${(score * 100).toFixed(0)}%`;
  if (score < 0) return score.toFixed(1);
  return score.toFixed(0);
}

const ENV_SCALE_NOTE: Record<string, string> = {
  "HalfCheetah-v5": "max ~12 000", "Hopper-v5": "max ~3 500", "Ant-v5": "max ~8 000",
  "Walker2d-v5": "max ~6 000", "Humanoid-v5": "max ~8 000", "HumanoidStandup-v5": "max ~200 000",
  "Swimmer-v5": "max ~360", "InvertedPendulum-v5": "max ~1 000", "InvertedDoublePendulum-v5": "max ~9 000",
  "CartPole-v1": "max 500", "Pendulum-v1": "max 0 (less neg = better)",
  "MountainCar-v0": "max 0 (less neg = better)", "MountainCarContinuous-v0": "max ~95",
  "LunarLander-v3": "max ~300", "BipedalWalker-v3": "max ~300",
  "FrozenLake-v1": "success %", "FrozenLake8x8-v1": "success %",
  "Taxi-v3": "max ~8", "CliffWalking-v1": "always neg",
  "ALE/Pong-v5": "max +21", "ALE/Breakout-v5": "max ~800", "ALE/SpaceInvaders-v5": "max ~10 000",
};

function Leaderboard({
  plan, heartbeats, results, sentinel, phase,
}: {
  plan: SpawnEntry[]; heartbeats: Heartbeat[]; results: EvalResult[];
  sentinel: SentinelEntry[]; phase: "racing" | "done";
}) {
  const VALID_STATUSES = new Set(["completed", "early_stopped", "race_dropout"]);
  const rows = plan.map(entry => {
    const hb  = heartbeats.find(h => h.agent_id === entry.id);
    const res = results.find(r => r.agent_id === entry.id);
    const sentCount = sentinel.filter(s => s.agent_id === entry.id).length;
    return {
      id: entry.id, algo: entry.algo, env: entry.env,
      score: phase === "done" && res ? res.mean_return : (hb?.current_reward ?? -Infinity),
      status: hb?.status ?? "waiting", restarts: sentCount,
      done:      phase === "done" && res != null && VALID_STATUSES.has(res.status),
      failed:    phase === "done" && res != null && !VALID_STATUSES.has(res.status),
      earlyStop: phase === "done" && res?.status === "early_stopped",
      raceDrop:  phase === "done" && res?.status === "race_dropout",
    };
  });

  const byEnv = new Map<string, typeof rows>();
  for (const r of rows) {
    if (!byEnv.has(r.env)) byEnv.set(r.env, []);
    byEnv.get(r.env)!.push(r);
  }
  for (const grp of byEnv.values()) grp.sort((a, b) => b.score - a.score);
  const multiEnv = byEnv.size > 1;

  return (
    <div className="text-sm space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-gray-500 uppercase tracking-widest text-xs">leaderboard</span>
        <span className={`text-xs ${phase === "done" ? "text-amber-400" : "text-green-400 animate-pulse"}`}>
          {phase === "done" ? "final" : "live"}
        </span>
      </div>

      {multiEnv && (
        <p className="text-yellow-600 text-xs">scores across different envs are not comparable</p>
      )}

      {[...byEnv.entries()].map(([env, grp]) => (
        <div key={env} className="space-y-1">
          <div className="flex items-center justify-between text-xs">
            <span className="text-gray-500">{env}</span>
            {ENV_SCALE_NOTE[env] && <span className="text-gray-700">{ENV_SCALE_NOTE[env]}</span>}
          </div>
          {grp.map((row, i) => {
            const st = as(row.algo);
            const rankIcon = row.failed ? "✗" : row.earlyStop ? "⏭" : row.raceDrop ? "↩" :
                             i === 0 ? "▶" : i === 1 ? "›" : "·";
            return (
              <div key={row.id}
                className={`flex items-center gap-2.5 px-2.5 py-2 border
                  ${row.failed  ? "border-[#2a1a1a] text-gray-600" :
                    i === 0     ? "border-[#2a2a1a] bg-[#141408]" :
                                  "border-[#1e1e1e]"}`}>
                <span className={i === 0 && !row.failed ? "text-amber-400" : "text-gray-600"}>{rankIcon}</span>
                <span className={`border px-1.5 ${st.tag}`}>{row.algo}</span>
                <span className="text-gray-500 flex-1 truncate">{row.id}</span>
                {row.restarts > 0 && <span className="text-yellow-600 text-xs">↩{row.restarts}</span>}
                <span className={`font-bold ${row.failed ? "text-red-600" : row.score > 0 ? "text-green-400" : "text-gray-400"}`}>
                  {formatScore(row.score, row.env)}
                </span>
              </div>
            );
          })}
        </div>
      ))}

      {phase === "racing" && (
        <p className="text-gray-700 text-xs">updates every 2 s</p>
      )}
    </div>
  );
}

// ── Sentinel banner ───────────────────────────────────────────────────────────

function SentinelBanner({ entry }: { entry: SentinelEntry }) {
  const isDoctor  = entry.failure_reason === "environment_setup_error";
  const fixOk     = entry.outcome === "fixed_retrying";
  const isKilled  = entry.outcome === "killed_permanently";
  const cmds: string[] = (entry.llm_suggested_hparams?.fix_commands as string[]) ?? [];

  if (isDoctor) {
    return (
      <div className={`border text-sm ${fixOk ? "border-cyan-900 bg-[#0a1515]" : "border-orange-900 bg-[#150f00]"}`}>
        <div className="px-3 py-2.5 border-b border-[#2a2a2a] flex items-center gap-2">
          <span className={fixOk ? "text-cyan-400" : "text-orange-400"}>env-doctor</span>
          <span className="text-gray-500">{entry.agent_id}</span>
          <span className="text-gray-700 ml-auto text-xs">{new Date(entry.timestamp).toLocaleTimeString()}</span>
        </div>
        {cmds.length > 0 && (
          <div className="px-3 py-2 space-y-1">
            {cmds.map((c, i) => <p key={i} className="text-gray-400">$ {c}</p>)}
          </div>
        )}
        {entry.rationale && <p className="px-3 pb-2 text-gray-600">{entry.rationale}</p>}
        <p className={`px-3 pb-2 ${fixOk ? "text-cyan-400" : "text-red-400"}`}>
          → {fixOk ? "fixed — retrying" : "fix failed"}
        </p>
      </div>
    );
  }

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
          {entry.failed_hparams && (
            <p className="text-red-300">Failed: {Object.entries(entry.failed_hparams).map(([k,v]) => `${k}=${v}`).join(" ")}</p>
          )}
          {entry.llm_suggested_hparams && Object.keys(entry.llm_suggested_hparams).length > 0 && (
            <p className="text-green-300">GPT → {Object.entries(entry.llm_suggested_hparams).map(([k,v]) => `${k}=${v}`).join(" ")}</p>
          )}
        </div>
        <p className={`text-xs mt-1 font-semibold
          ${entry.outcome === "completed" ? "text-green-400" : isKilled ? "text-red-400" : "text-yellow-400"}`}>
          → {entry.outcome}
        </p>
      </div>
    </div>
  );
}

// ── Inference showcase ─────────────────────────────────────────────────────────

function InferenceShowcase({ cases, baseline, agentId, wandbArtifact }: {
  cases: InferenceCase[]; baseline?: InferenceCase[]; agentId: string; wandbArtifact?: string;
}) {
  const passed = cases.filter(c => c.success).length;
  const hasBaseline = baseline && baseline.length > 0;

  return (
    <div className="text-sm space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-sm font-semibold text-gray-300">
          {hasBaseline ? "Before / After Training — " : "Inference Showcase — "}
          <span className="font-mono text-orange-400">{agentId}</span>
        </p>
        <span className="text-xs text-gray-500">{passed}/{cases.length} correct after training</span>
      </div>
      {wandbArtifact && (
        <p className="text-xs text-gray-500">
          Model artifact: <span className="font-mono text-violet-400">{wandbArtifact}</span> (W&B)
        </p>
      )}

      {hasBaseline ? (
        <div className="space-y-3">
          {/* Column headers */}
          <div className="grid grid-cols-2 gap-3">
            <div className="text-xs font-semibold text-gray-400 text-center py-1 bg-gray-800/50 rounded-lg">
              Before (base model)
            </div>
            <div className="text-xs font-semibold text-orange-400 text-center py-1 bg-orange-950/30 rounded-lg border border-orange-900/40">
              After (trained)
            </div>
          </div>
          {cases.map((c, i) => {
            const b = baseline[i];
            return (
              <div key={i} className="rounded-xl border border-gray-700 bg-gray-900/40 overflow-hidden">
                {/* Case header */}
                <div className="flex items-center justify-between px-3 py-2 bg-gray-800/60 border-b border-gray-700">
                  <span className="text-xs font-mono text-gray-400">
                    [{c.numbers.join(", ")}] → {c.target}
                  </span>
                  <span className={`text-xs font-bold px-2 py-0.5 rounded-full
                    ${c.success ? "bg-emerald-900 text-emerald-300" : "bg-red-900 text-red-300"}`}>
                    {c.success ? "✅ PASS" : "❌ FAIL"}
                  </span>
                </div>
                {/* Two-column responses */}
                <div className="grid grid-cols-2 divide-x divide-gray-700">
                  <div className="p-2">
                    <div className="bg-black/30 rounded-lg p-2 text-xs font-mono text-gray-500 whitespace-pre-wrap max-h-40 overflow-y-auto">
                      {b?.model_response || "(no baseline)"}
                    </div>
                  </div>
                  <div className="p-2">
                    <div className={`rounded-lg p-2 text-xs font-mono whitespace-pre-wrap max-h-40 overflow-y-auto
                      ${c.success
                        ? "bg-emerald-950/40 text-emerald-200"
                        : "bg-black/30 text-gray-300"}`}>
                      {c.model_response || "(empty response)"}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="space-y-2">
          {cases.map((c, i) => (
            <div key={i} className={`rounded-xl border p-3 space-y-2
              ${c.success ? "bg-emerald-950/40 border-emerald-800" : "bg-red-950/40 border-red-800"}`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-sm">{c.success ? "✅" : "❌"}</span>
                  <span className="text-xs font-mono text-gray-400">
                    [{c.numbers.join(", ")}] → {c.target}
                  </span>
                </div>
                <span className={`text-xs font-bold px-2 py-0.5 rounded-full
                  ${c.success ? "bg-emerald-900 text-emerald-300" : "bg-red-900 text-red-300"}`}>
                  {c.success ? "PASS" : "FAIL"}
                </span>
              </div>
              <div className="bg-black/30 rounded-lg p-2 text-xs font-mono text-gray-300 whitespace-pre-wrap max-h-32 overflow-y-auto">
                {c.model_response || "(empty response)"}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Video modal ───────────────────────────────────────────────────────────────

function VideoModal({ url, agentId, envId, envFamily, onClose }: {
  url: string; agentId: string; envId: string; envFamily: EnvFamily; onClose: () => void;
}) {
  const meta = ENV_FAMILY_META[envFamily];
  const isCompact = envFamily === "toytext" || envFamily === "atari";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/90"
      onClick={onClose}>
      <div className={`bg-[#0d0d0d] border border-[#333] p-4 w-full mx-4 space-y-3 ${isCompact ? "max-w-md" : "max-w-2xl"}`}
        onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between border-b border-[#2a2a2a] pb-3">
          <div>
            <span className={`text-base ${meta.color}`}>{meta.icon} {meta.label}</span>
            <span className="text-gray-500 ml-3 text-sm">{agentId} · {envId}</span>
          </div>
          <button onClick={onClose} className="text-gray-600 hover:text-gray-300 transition-colors">✕</button>
        </div>
        <div className="flex gap-4 text-xs text-gray-500">
          <span>playback <span className="text-gray-300">{meta.fps}</span></span>
          <span>episodes <span className="text-gray-300">{meta.episodeNote}</span></span>
          <span className="text-gray-600">{meta.desc}</span>
        </div>
        <video src={url} controls autoPlay loop
          className={`w-full bg-black object-contain ${isCompact ? "max-h-[40vh]" : "max-h-[60vh]"}`}
          style={(envFamily === "toytext" || envFamily === "atari") ? { imageRendering: "pixelated" } as React.CSSProperties : undefined}>
          Your browser does not support HTML video.
        </video>
        <p className="text-xs text-gray-700">deterministic policy rollout from saved checkpoint</p>
      </div>
    </div>
  );
}

// ── Divider helper ────────────────────────────────────────────────────────────

function Divider({ label }: { label?: string }) {
  if (!label) return <div className="border-t border-[#2a2a2a]" />;
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 border-t border-[#2a2a2a]" />
      <span className="text-xs text-gray-700 uppercase tracking-widest">{label}</span>
      <div className="flex-1 border-t border-[#2a2a2a]" />
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
  const [videoModal,  setVideoModal]  = useState<{ agentId: string; url: string; envId: string; envFamily: EnvFamily } | null>(null);
  const [grpoInfer,   setGrpoInfer]   = useState<Record<string, GrpoInferData>>({});
  const [wandbArtifacts, setWandbArtifacts] = useState<Record<string, string>>({});
  const [hfRepoUrl,   setHfRepoUrl]   = useState("");
  const [hfSnippet,   setHfSnippet]   = useState("");
  const [snippetCopied, setSnippetCopied] = useState(false);

  const prevSentinelCount = useRef<Record<string, number>>({});
  const pollRef           = useRef<ReturnType<typeof setInterval> | null>(null);
  const sseRef            = useRef<EventSource | null>(null);
  const textareaRef       = useRef<HTMLTextAreaElement>(null);

  const handleStatusData = useCallback((data: {
    status?: string; heartbeats?: Heartbeat[]; sentinel_log?: SentinelEntry[]; plan?: SpawnEntry[];
  }, name: string) => {
    const hbs: Heartbeat[]        = data.heartbeats   ?? [];
    const slog: SentinelEntry[]   = data.sentinel_log ?? [];
    const serverPlan: SpawnEntry[] = data.plan        ?? [];
    setHeartbeats(hbs); setSentinel(slog);
    setPlan(prev => serverPlan.length > prev.length ? serverPlan : prev);

    const sentByAgent: Record<string, number> = {};
    for (const e of slog) sentByAgent[e.agent_id] = (sentByAgent[e.agent_id] ?? 0) + 1;

    const newAnim: Record<string, AnimState> = {};
    for (const [agentId, cnt] of Object.entries(sentByAgent)) {
      if (cnt > (prevSentinelCount.current[agentId] ?? 0)) {
        newAnim[agentId] = "vanish";
        setTimeout(() => {
          setAnimStates(a => ({ ...a, [agentId]: "appear" }));
          setTimeout(() => setAnimStates(a => ({ ...a, [agentId]: "idle" })), 600);
        }, 600);
      }
    }
    prevSentinelCount.current = sentByAgent;
    if (Object.keys(newAnim).length > 0) setAnimStates(a => ({ ...a, ...newAnim }));

    setHistory(prev => {
      const next = { ...prev };
      for (const hb of hbs) {
        const seg = sentByAgent[hb.agent_id] ?? 0;
        const pts = next[hb.agent_id] ?? [];
        const last = pts.at(-1);
        if (!last || last.steps !== hb.steps_completed)
          next[hb.agent_id] = [...pts, { steps: hb.steps_completed, reward: hb.current_reward, seg }];
      }
      return next;
    });

    if (data.status === "completed" || data.status === "failed") {
      stopLive();
      fetch(`${BACKEND}/api/results/${name}`)
        .then(r => r.ok ? r.json() : null)
        .then(rData => {
          if (rData) {
            setResults(rData.results ?? []); setBest(rData.best ?? null);
            if (rData.hf_repo_url) setHfRepoUrl(rData.hf_repo_url);
            if (rData.hf_code_snippet) setHfSnippet(rData.hf_code_snippet);
          }
          setPhase("done");
        })
        .catch(() => setPhase("done"));
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const stopLive = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    if (sseRef.current)  { sseRef.current.close(); sseRef.current = null; }
  }, []);

  const startSSE = useCallback((name: string) => {
    stopLive();
    fetch(`${BACKEND}/api/status/${name}`).then(r => r.ok ? r.json() : null)
      .then(data => { if (data) handleStatusData(data, name); }).catch(() => {});
    const es = new EventSource(`${BACKEND}/api/stream/${name}`);
    sseRef.current = es;
    es.addEventListener("heartbeat", (e: MessageEvent) => {
      try {
        const hb: Heartbeat = JSON.parse(e.data);
        setHeartbeats(prev => {
          const idx = prev.findIndex(h => h.agent_id === hb.agent_id);
          return idx >= 0 ? prev.map((h, i) => i === idx ? hb : h) : [...prev, hb];
        });
        setHistory(prev => {
          const pts = prev[hb.agent_id] ?? [];
          const last = pts.at(-1);
          if (!last || last.steps !== hb.steps_completed)
            return { ...prev, [hb.agent_id]: [...pts, { steps: hb.steps_completed, reward: hb.current_reward, seg: 0 }] };
          return prev;
        });
        if (Math.random() < 0.1) {
          fetch(`${BACKEND}/api/status/${name}`).then(r => r.ok ? r.json() : null)
            .then(data => { if (data) handleStatusData(data, name); }).catch(() => {});
        }
      } catch { /* ignore */ }
    });
    es.addEventListener("no_redis", () => { es.close(); sseRef.current = null; startPolling(name); });
    es.onerror = () => { es.close(); sseRef.current = null; startPolling(name); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [handleStatusData, stopLive]);

  const startPolling = useCallback((name: string) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${BACKEND}/api/status/${name}`);
        if (!res.ok) return;
        handleStatusData(await res.json(), name);
      } catch { /* ignore */ }
    }, POLL_MS);
  }, [handleStatusData]);

  const startLive = useCallback((name: string) => { startSSE(name); }, [startSSE]);
  useEffect(() => () => { stopLive(); }, [stopLive]);

  useEffect(() => {
    if (phase === "racing" && runName && runDir && plan.length > 0)
      localStorage.setItem("autorl_active_run", JSON.stringify({ runName, runDir, task, plan }));
  }, [phase, runName, runDir, task, plan]);

  useEffect(() => {
    if (phase === "done" || phase === "idle") localStorage.removeItem("autorl_active_run");
  }, [phase]);

  useEffect(() => {
    if (phase === "done" && best?.algo === "GRPO" && !grpoInfer[best.agent_id]?.results?.length)
      handleGrpoInfer(best.agent_id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, best]);

  useEffect(() => {
    const saved = localStorage.getItem("autorl_active_run");
    if (!saved) return;
    try {
      const { runName: sn, runDir: sd, task: st, plan: sp } = JSON.parse(saved);
      if (sn && sd && sp?.length) {
        setRunName(sn); setRunDir(sd); setTask(st ?? ""); setPlan(sp);
        setPhase("racing"); startLive(sn);
      }
    } catch { localStorage.removeItem("autorl_active_run"); }
  }, [startLive]);

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
      setPhase("racing"); startLive(runName);
    } catch (e) { setErrorMsg(e instanceof Error ? e.message : String(e)); setPhase("error"); }
  };

  const handleGrpoInfer = async (agentId: string) => {
    if (grpoInfer[agentId]?.results?.length) return;
    setInferring(p => ({ ...p, [agentId]: true }));
    try {
      const res = await fetch(`${BACKEND}/api/inference/${runName}/${agentId}`);
      if (!res.ok) throw new Error((await res.json().catch(() => ({ detail: res.statusText }))).detail);
      const data = await res.json();
      setGrpoInfer(p => ({ ...p, [agentId]: { results: data.results ?? [], baseline: data.baseline ?? [] } }));
      if (data.wandb_artifact) setWandbArtifacts(p => ({ ...p, [agentId]: data.wandb_artifact }));
    } catch (e) { alert(`Inference failed: ${e instanceof Error ? e.message : String(e)}`); }
    finally { setInferring(p => ({ ...p, [agentId]: false })); }
  };

  const handleInfer = async (agentId: string) => {
    const entry = plan.find(e => e.id === agentId);
    if (entry?.algo === "GRPO") return handleGrpoInfer(agentId);
    setInferring(p => ({ ...p, [agentId]: true }));
    const envId = entry?.env ?? "";
    try {
      const res = await fetch(`${BACKEND}/api/infer`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_name: runName, agent_id: agentId }),
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({ detail: res.statusText }))).detail);
      const data = await res.json();
      setVideoModal({ agentId, url: `${BACKEND}/api/video/${data.filename}`, envId, envFamily: detectEnvFamily(envId) });
    } catch (e) { alert(`Inference failed: ${e instanceof Error ? e.message : String(e)}`); }
    finally { setInferring(p => ({ ...p, [agentId]: false })); }
  };

  const handleReset = () => {
    stopLive();
    if (runName) fetch(`${BACKEND}/api/cancel/${runName}`, { method: "POST" }).catch(() => {});
    localStorage.removeItem("autorl_active_run");
    setPhase("idle"); setTask(""); setPlan([]); setRunName(""); setRunDir("");
    setHeartbeats([]); setSentinel([]); setResults([]); setBest(null); setErrorMsg("");
    setHistory({}); setAnimStates({}); setInferring({}); setVideos({}); setVideoModal(null);
    setGrpoInfer({}); setWandbArtifacts({});
    setHfRepoUrl(""); setHfSnippet(""); setSnippetCopied(false);
    prevSentinelCount.current = {};
    setTimeout(() => textareaRef.current?.focus(), 50);
  };

  // ── Derived ──────────────────────────────────────────────────────────────────

  const hbById = Object.fromEntries(heartbeats.map(h => [h.agent_id, h]));
  const sentByAgent = sentinel.reduce<Record<string, SentinelEntry[]>>((acc, e) => {
    acc[e.agent_id] = [...(acc[e.agent_id] ?? []), e]; return acc;
  }, {});

  const isCentered = !["racing", "done"].includes(phase);

  // ── Render ────────────────────────────────────────────────────────────────────

  return (
    <div className={`min-h-screen bg-[#0d0d0d] text-gray-200
      ${isCentered ? "flex flex-col items-center justify-center p-6 py-12" : "p-6 pt-8"}`}>

      {videoModal && (
        <VideoModal url={videoModal.url} agentId={videoModal.agentId}
          envId={videoModal.envId} envFamily={videoModal.envFamily}
          onClose={() => setVideoModal(null)} />
      )}

      {/* ── Header ── */}
      <div className={`flex items-center gap-3 ${isCentered ? "mb-10" : "mb-8"}`}>
        <span className="text-amber-500 text-xl font-bold">AutoRL</span>
        <span className="text-[#333]">|</span>
        <span className="text-gray-600 text-base">multi-agent training race</span>
        {!isCentered && runName && (
          <>
            <span className="text-[#333]">|</span>
            <span className="text-gray-600 text-sm">{runName}</span>
          </>
        )}
      </div>

      {/* ── IDLE / ERROR ── */}
      {(phase === "idle" || phase === "error") && (
        <div className="w-full max-w-xl space-y-4">
          <p className="text-gray-400 text-base">What do you want to train?</p>
          <div className="relative border border-[#2a2a2a] focus-within:border-amber-600 transition-colors">
            <textarea ref={textareaRef} autoFocus value={task}
              onChange={e => setTask(e.target.value)}
              onKeyDown={e => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleGeneratePlan(); }}
              rows={3} placeholder="describe the rl task…"
              className="w-full bg-[#111] px-4 py-3 text-gray-100 placeholder-gray-700 text-base resize-none outline-none" />
            <p className="absolute bottom-2 right-3 text-xs text-gray-700">⌘ enter</p>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {SUGGESTIONS.map(s => (
              <button key={s} onClick={() => setTask(s)}
                className="text-sm px-2.5 py-1 border border-[#2a2a2a] hover:border-amber-700 text-gray-600 hover:text-amber-400 transition-colors">
                {s}
              </button>
            ))}
          </div>
          {phase === "error" && (
            <div className="border border-red-900 bg-[#150000] px-4 py-3">
              <p className="text-red-400 mb-1 text-sm">error</p>
              <p className="text-gray-500 text-sm break-all">{errorMsg}</p>
              <p className="text-gray-700 mt-2 text-xs">backend running? <code>bash ui/agent/start.sh</code></p>
            </div>
          )}
          <button onClick={handleGeneratePlan} disabled={!task.trim()}
            className="w-full bg-amber-600 hover:bg-amber-500 disabled:bg-[#1a1a1a] disabled:text-gray-700 text-black font-bold py-3 transition-colors text-base">
            generate lineup →
          </button>
        </div>
      )}

      {/* ── PLANNING ── */}
      {phase === "planning" && <PlanningScreen task={task} />}

      {/* ── PLAN READY ── */}
      {phase === "plan_ready" && (
        <div className="w-full max-w-2xl space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-gray-300 text-base">agent lineup</span>
            <span className="text-gray-600 text-sm">{runName}</span>
          </div>
          <p className="text-gray-600 text-sm italic">&ldquo;{task}&rdquo;</p>

          <div className="flex gap-4 text-sm text-gray-600">
            <span><span className="text-gray-200">{plan.length}</span> agents</span>
            <span><span className="text-gray-200">{plan.filter(e=>e.exec==="local").length}</span> local</span>
            <span><span className="text-gray-200">{plan.filter(e=>e.exec==="runpod").length}</span> runpod</span>
          </div>

          {plan.some(e => (e.hparams.lr as number) >= 0.1) && (
            <div className="border border-yellow-900 bg-[#14100a] px-3 py-2 text-xs">
              <span className="text-yellow-500">sentinel active</span>
              <span className="text-gray-600 ml-2">one agent has a dangerously high lr — doom loop expected</span>
            </div>
          )}

          <p className="text-sm text-gray-700">edit or remove agents before launching</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {plan.map((e, i) => (
              <AgentLineupCard key={e.id} entry={e} index={i}
                onDelete={() => setPlan(p => p.filter(a => a.id !== e.id))}
                onUpdate={u => setPlan(p => p.map(a => a.id === u.id ? u : a))} />
            ))}
          </div>
          {plan.length === 0 && (
            <p className="text-gray-600 text-sm text-center py-4">all agents removed</p>
          )}
          <div className="flex gap-2 pt-1">
            <button onClick={handleReset}
              className="flex-1 border border-[#333] hover:border-gray-500 text-gray-500 hover:text-gray-300 py-3 text-base transition-colors">
              ← back
            </button>
            <button onClick={handleLaunch} disabled={plan.length === 0}
              className="flex-1 bg-amber-600 hover:bg-amber-500 disabled:bg-[#1a1a1a] disabled:text-gray-700 text-black font-bold py-3 text-base transition-colors">
              launch race ({plan.length})
            </button>
          </div>
        </div>
      )}

      {/* ── LAUNCHING ── */}
      {phase === "launching" && (
        <div className="text-center space-y-3">
          <p className="text-amber-400 animate-pulse text-base">launching {plan.length} agents…</p>
          <p className="text-gray-700 text-sm">swarm runner starting</p>
        </div>
      )}

      {/* ── RACING ── */}
      {phase === "racing" && (
        <div className="w-full max-w-7xl">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3 text-base">
              <span className="text-green-400 animate-pulse">●</span>
              <span className="text-gray-200">race live</span>
              <span className="text-gray-600 text-sm">
                {heartbeats.filter(h=>h.status==="training").length} running ·{" "}
                {heartbeats.filter(h=>h.status==="completed").length} done
              </span>
            </div>
            <button onClick={handleReset}
              className="text-sm border border-[#333] hover:border-red-900 text-gray-500 hover:text-red-400 px-3 py-1.5 transition-colors">
              × stop &amp; reset
            </button>
          </div>
          <p className="text-gray-700 text-sm italic mb-4">&ldquo;{task}&rdquo;</p>

          <div className="flex gap-5 items-start">
            <div className="flex-1 min-w-0 space-y-3 max-h-[calc(100vh-200px)] overflow-y-auto pr-1">
              {plan.map(e => (
                <LiveAgentCard key={e.id} entry={e} hb={hbById[e.id]}
                  history={history[e.id] ?? []} sentinelEntries={sentByAgent[e.id] ?? []}
                  animState={animStates[e.id] ?? "idle"}
                  onInfer={() => handleInfer(e.id)} inferring={!!inferring[e.id]} hasVideo={!!videos[e.id]} />
              ))}
              {sentinel.length > 0 && (
                <div className="space-y-2 pt-2">
                  <Divider label="interventions" />
                  {sentinel.map((e, i) => <SentinelBanner key={i} entry={e} />)}
                </div>
              )}
              <p className="text-gray-700 text-xs text-center py-2">polling every 2 s</p>
            </div>
            <div className="w-72 xl:w-80 shrink-0 sticky top-6 border border-[#2a2a2a] bg-[#111] p-4">
              <Leaderboard plan={plan} heartbeats={heartbeats} results={[]} sentinel={sentinel} phase="racing" />
            </div>
          </div>
        </div>
      )}

      {/* ── DONE ── */}
      {phase === "done" && (
        <div className="w-full max-w-7xl">
          <div className="flex items-center justify-between mb-4">
            <span className="text-amber-400 text-base">race complete</span>
            <button onClick={handleReset}
              className="text-sm border border-[#333] hover:border-amber-600 text-gray-500 hover:text-amber-400 px-3 py-1.5 transition-colors">
              ← train again
            </button>
          </div>

          <div className="flex gap-5 items-start">
            <div className="flex-1 min-w-0 space-y-3 max-h-[calc(100vh-160px)] overflow-y-auto pr-1">

              {/* winner panel */}
              {best && (
                <div className="border border-amber-900/50 bg-[#14100a] text-sm">
                  <div className="px-4 py-3 border-b border-[#2a2a2a] flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="text-amber-400">▶ winner</span>
                      <span className={`border px-2 py-0.5 ${as(best.algo).tag}`}>{best.algo}</span>
                      <span className="text-gray-400">{best.env}</span>
                      <span className="text-gray-600">{best.agent_id}</span>
                    </div>
                    <button onClick={() => handleInfer(best.agent_id)} disabled={!!inferring[best.agent_id]}
                      className="border border-[#333] hover:border-amber-600 text-gray-500 hover:text-amber-400 px-2.5 py-1 transition-colors disabled:opacity-40">
                      {inferring[best.agent_id]
                        ? (best.algo === "GRPO" ? "⏳ Loading…" : "⏳ Recording…")
                        : (best.algo === "GRPO"
                          ? (grpoInfer[best.agent_id]?.results?.length ? "✅ Before/After loaded" : "🧪 View before/after")
                          : "▶ Watch inference")}
                    </button>
                  </div>
                  <div className="px-4 py-3 flex gap-8 border-b border-[#1a1a1a]">
                    <span>mean return <span className="text-amber-400 font-bold">{best.mean_return.toFixed(2)}</span></span>
                    <span>std <span className="text-gray-300">±{best.std_return.toFixed(2)}</span></span>
                  </div>

                  {wandbArtifacts[best.agent_id] && (
                    <p className="px-4 py-2 text-gray-600 border-b border-[#1a1a1a]">
                      artifact: <span className="text-violet-400">{wandbArtifacts[best.agent_id]}</span>
                    </p>
                  )}

                  {hfRepoUrl && (
                    <div className="px-4 py-3 border-b border-[#1a1a1a] flex items-center justify-between">
                      <span className="text-gray-400 truncate">{hfRepoUrl}</span>
                      <a href={hfRepoUrl} target="_blank" rel="noopener noreferrer"
                        className="ml-3 shrink-0 border border-[#333] hover:border-amber-600 text-gray-500 hover:text-amber-400 px-2.5 py-1 transition-colors">
                        view ↗
                      </a>
                    </div>
                  )}

                  {hfSnippet && (
                    <div>
                      <div className="px-4 py-2 border-b border-[#1a1a1a] flex items-center justify-between">
                        <span className="text-gray-600">standalone usage</span>
                        <button onClick={() => {
                          navigator.clipboard.writeText(hfSnippet).then(() => {
                            setSnippetCopied(true);
                            setTimeout(() => setSnippetCopied(false), 2000);
                          });
                        }} className="text-xs border border-[#333] hover:border-amber-600 text-gray-600 hover:text-amber-400 px-2 py-0.5 transition-colors">
                          {snippetCopied ? "copied" : "copy"}
                        </button>
                      </div>
                      <pre className="px-4 py-3 text-gray-400 overflow-x-auto max-h-48 leading-relaxed text-xs">{hfSnippet}</pre>
                    </div>
                  )}

                  {history[best.agent_id]?.length >= 2 && (
                    <div className="border-t border-[#1a1a1a] px-2 py-1.5">
                      <MiniChart history={history[best.agent_id]} algoRgb={as(best.algo).rgb} hasNaN={false}
                        family={detectEnvFamily(best.env)} env={best.env} />
                    </div>
                  )}
                </div>
              )}

              {/* GRPO before/after showcase for winning agent */}
              {best && grpoInfer[best.agent_id]?.results?.length ? (
                <InferenceShowcase
                  cases={grpoInfer[best.agent_id].results}
                  baseline={grpoInfer[best.agent_id].baseline}
                  agentId={best.agent_id}
                  wandbArtifact={wandbArtifacts[best.agent_id]}
                />
              ) : null}

              <Divider label="all agents" />

              {plan.map(e => (
                <LiveAgentCard key={e.id} entry={e} hb={hbById[e.id]}
                  history={history[e.id] ?? []} sentinelEntries={sentByAgent[e.id] ?? []}
                  animState="idle" onInfer={() => handleInfer(e.id)}
                  inferring={!!inferring[e.id]} hasVideo={!!videos[e.id]} />
              ))}

              {sentinel.length > 0 && (
                <div className="space-y-2 pt-2">
                  <Divider label="sentinel log" />
                  {sentinel.map((e, i) => <SentinelBanner key={i} entry={e} />)}
                </div>
              )}
            </div>

            <div className="w-72 xl:w-80 shrink-0 sticky top-6 border border-[#2a2a2a] bg-[#111] p-4">
              <Leaderboard plan={plan} heartbeats={heartbeats} results={results} sentinel={sentinel} phase="done" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
