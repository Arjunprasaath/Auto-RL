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
}
interface EvalResult {
  agent_id: string; algo: string; env: string; status: string;
  mean_return: number; std_return: number; checkpoint_path: string;
}
interface HistPt { steps: number; reward: number; seg: number; }
type Phase = "idle" | "planning" | "plan_ready" | "launching" | "racing" | "done" | "error" | "wm_training" | "reward_design";
type AnimState = "vanish" | "appear" | "idle";

interface DatasetMeta {
  obs_cols: string[]; act_cols: string[]; reward_col: string;
  next_obs_cols: string[]; done_col: string;
  obs_dim: number; act_dim: number;
  act_type: "discrete" | "continuous"; act_n: number | null;
  reward_min: number; reward_max: number;
  n_samples: number; hidden_sizes: number[];
  dataset_path: string; initial_states_path: string;
  source_env?: string;        // real gym env id inferred from HF config (e.g. "Ant-v5")
  _size_reasoning?: string;   // from dataset_size_agent
  _split_used?: string;
}

interface AgentLogEntry {
  agent: string;       // "dataset_size" | "arch_search" | "algo_selector" | "hparam"
  decision: string;    // short one-liner
  reasoning: string;   // longer explanation
  timestamp: number;
}

interface WMRolloutStep {
  step: number;
  true_reward: number;
  pred_reward: number;
  true_obs: number[];
  pred_obs: number[];
  true_done?: number;
  pred_done?: number;
  obs_mse_step?: number;
}

interface WMEval {
  n_val_samples: number;
  obs_mse: number;
  reward_mse: number;
  reward_mae: number;
  done_accuracy: number;
  val_loss: number;
  obs_dims_plotted: number;
  one_step_rollout: WMRolloutStep[];
  open_loop_rollout: WMRolloutStep[];
}

interface RewardMsg {
  role: "user" | "assistant";
  text: string;
  code?: string;
  explanation?: string;
}

// ── Env-family detection ───────────────────────────────────────────────────────

type EnvFamily = "mujoco" | "classic" | "toytext" | "box2d" | "atari" | "grpo" | "worldmodel";

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
  if (env === "WorldModel-v0") return "worldmodel";
  const e = env.toLowerCase();
  if (["frozenlake", "taxi", "cliffwalking", "blackjack"].some(k => e.includes(k))) return "toytext";
  if (["lunarlander", "bipedalwalker", "carracing"].some(k => e.includes(k))) return "box2d";
  if (["halfcheetah", "hopper", "ant", "walker2d", "swimmer", "humanoid",
       "reacher", "pusher", "invertedpendulum"].some(k => e.includes(k))) return "mujoco";
  if (e.startsWith("ale/") || ATARI_KEYWORDS.some(k => e.includes(k))) return "atari";
  return "classic";
}

/** FrozenLake and Blackjack have binary 0/1 (or +1/−1) outcomes — show success-rate bar.
 *  Taxi (−200..+8) and CliffWalking (always negative) are NOT binary; use line chart. */
function isBinaryRewardEnv(env: string): boolean {
  const e = env.toLowerCase();
  return ["frozenlake", "blackjack"].some(k => e.includes(k));
}

const ENV_FAMILY_META: Record<EnvFamily, {
  label: string; icon: string; fps: string; color: string;
  desc: string; inferLabel: string; episodeNote: string;
}> = {
  mujoco:  {
    label: "MuJoCo",      icon: "🤖", fps: "30 fps", color: "text-violet-400",
    desc: "Continuous control — physics simulation",
    inferLabel: "▶ watch",   episodeNote: "1 deterministic episode",
  },
  classic: {
    label: "Classic",     icon: "🎮", fps: "30 fps", color: "text-cyan-400",
    desc: "Classic control task",
    inferLabel: "▶ run",     episodeNote: "3 deterministic episodes",
  },
  toytext: {
    label: "Grid World",  icon: "🧩", fps: "4 fps",  color: "text-emerald-400",
    desc: "Discrete grid world — 4 fps, each frame = one move",
    inferLabel: "▶ replay",  episodeNote: "5 complete episodes",
  },
  box2d:   {
    label: "Box2D",       icon: "🚀", fps: "30 fps", color: "text-orange-400",
    desc: "Physics-based 2D simulation",
    inferLabel: "▶ fly",     episodeNote: "2 complete episodes",
  },
  atari:   {
    label: "Atari",       icon: "👾", fps: "30 fps", color: "text-red-400",
    desc: "Arcade pixel game — CNN policy, discrete actions",
    inferLabel: "▶ play",   episodeNote: "2 game episodes",
  },
  grpo:    {
    label: "Countdown",   icon: "🔢", fps: "—",      color: "text-yellow-400",
    desc: "LLM arithmetic reasoning task (no video)",
    inferLabel: "",          episodeNote: "",
  },
  worldmodel: {
    label: "World Model", icon: "🧠", fps: "—",      color: "text-teal-400",
    desc: "Custom dataset — agents train inside a learned simulator",
    inferLabel: "",          episodeNote: "",
  },
};

// ── Style maps ─────────────────────────────────────────────────────────────────

const ALGO_STYLE: Record<string, { border: string; badge: string; bar: string; rgb: string }> = {
  PPO:         { border: "border-violet-600",  badge: "bg-violet-900 text-violet-300",  bar: "bg-violet-500",  rgb: "#8b5cf6" },
  SAC:         { border: "border-cyan-600",    badge: "bg-cyan-900 text-cyan-300",      bar: "bg-cyan-500",    rgb: "#06b6d4" },
  A2C:         { border: "border-pink-600",    badge: "bg-pink-900 text-pink-300",      bar: "bg-pink-500",    rgb: "#ec4899" },
  GRPO:        { border: "border-orange-600",  badge: "bg-orange-900 text-orange-300",  bar: "bg-orange-500",  rgb: "#f97316" },
  WORLD_MODEL: { border: "border-teal-600",    badge: "bg-teal-900 text-teal-300",      bar: "bg-teal-500",    rgb: "#14b8a6" },
};
const DEF_STYLE = { border: "border-gray-700", badge: "bg-gray-800 text-gray-300", bar: "bg-gray-500", rgb: "#6b7280" };
const as = (algo: string) => ALGO_STYLE[algo] ?? DEF_STYLE;
const SEG_COLORS = ["#8b5cf6", "#06b6d4", "#34d399", "#fbbf24", "#f472b6"];

// ── Mini SVG reward chart ──────────────────────────────────────────────────────

function MiniChart({
  history, algoRgb, hasNaN, family, env,
}: {
  history: HistPt[]; algoRgb: string; hasNaN: boolean; family: EnvFamily; env: string;
}) {
  const W = 360, H = 72;

  // Binary-outcome toy-text envs (FrozenLake): show success-rate bar.
  // Taxi / CliffWalking have continuous/non-binary rewards — use the line chart instead.
  if (family === "toytext" && isBinaryRewardEnv(env)) {
    const label = "Success rate";
    const successes = history.filter(p => p.reward > 0).length;
    const rate = history.length > 0 ? successes / history.length : 0;
    return (
      <div className="w-full h-[72px] flex flex-col justify-center gap-2 px-1">
        <div className="flex justify-between text-xs">
          <span className="text-gray-500">{label}</span>
          <span className="font-bold text-emerald-400">{(rate * 100).toFixed(0)}%</span>
        </div>
        <div className="w-full bg-gray-800 rounded-full h-2">
          <div className="h-2 rounded-full bg-emerald-500 transition-all duration-700"
            style={{ width: `${rate * 100}%` }} />
        </div>
        <p className="text-xs text-gray-600">{successes} / {history.length} polled episodes reached goal</p>
      </div>
    );
  }

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

// ── World-model eval charts (pred vs ground truth on val set) ───────────────

function WMDualLineChart({
  steps, trueKey, predKey, label, trueColor = "#34d399", predColor = "#f472b6",
}: {
  steps: WMRolloutStep[];
  trueKey: "true_reward" | "true_obs";
  predKey: "pred_reward" | "pred_obs";
  label: string;
  trueColor?: string;
  predColor?: string;
  dim?: number;
}) {
  const W = 360, H = 80;
  if (!steps.length) return (
    <div className="h-[80px] flex items-center justify-center text-xs text-gray-600">No rollout data</div>
  );

  const getVal = (s: WMRolloutStep, key: typeof trueKey, dim = 0) =>
    key.endsWith("_obs") ? (s[key as "true_obs"]?.[dim] ?? 0) : (s[key as "true_reward"] ?? 0);

  const dim = 0;
  const trueVals = steps.map(s => getVal(s, trueKey, dim));
  const predVals = steps.map(s => getVal(s, predKey, dim));
  const all = [...trueVals, ...predVals];
  const minV = Math.min(...all);
  const maxV = Math.max(...all);
  const xS = (i: number) => (i / Math.max(steps.length - 1, 1)) * W;
  const yS = (v: number) => H - 6 - ((v - minV) / (maxV - minV || 1)) * (H - 12);

  const toPath = (vals: number[]) =>
    vals.map((v, i) => `${i === 0 ? "M" : "L"}${xS(i).toFixed(1)},${yS(v).toFixed(1)}`).join(" ");

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-gray-500">{label}</span>
        <div className="flex gap-3 text-xs">
          <span style={{ color: trueColor }}>— true</span>
          <span style={{ color: predColor }}>- - pred</span>
        </div>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-[80px]" preserveAspectRatio="none">
        <path d={toPath(trueVals)} fill="none" stroke={trueColor} strokeWidth="1.5" />
        <path d={toPath(predVals)} fill="none" stroke={predColor} strokeWidth="1.5" strokeDasharray="4 3" />
      </svg>
    </div>
  );
}

function WMEvalPanel({ eval: wmEval }: { eval: WMEval }) {
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  return (
    <div className="bg-gray-900 border border-teal-700/40 rounded-xl p-4 space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-teal-300 uppercase tracking-wider">
          World Model — Validation Test
        </span>
        <span className="text-xs text-gray-500 font-mono">{wmEval.n_val_samples.toLocaleString()} holdout samples</span>
      </div>

      <div className="grid grid-cols-4 gap-2 text-center">
        {[
          { label: "Obs MSE",    value: wmEval.obs_mse.toFixed(4),      good: wmEval.obs_mse < 0.1 },
          { label: "Reward MAE", value: wmEval.reward_mae.toFixed(4),   good: wmEval.reward_mae < 0.5 },
          { label: "Done Acc",   value: pct(wmEval.done_accuracy),       good: wmEval.done_accuracy > 0.8 },
          { label: "Val Loss",   value: wmEval.val_loss.toFixed(4),     good: wmEval.val_loss < 1.0 },
        ].map(m => (
          <div key={m.label} className="bg-gray-800/60 rounded-lg py-2 px-1">
            <p className="text-xs text-gray-600">{m.label}</p>
            <p className={`font-mono text-sm font-bold ${m.good ? "text-teal-400" : "text-gray-300"}`}>{m.value}</p>
          </div>
        ))}
      </div>

      <div className="space-y-3">
        <WMDualLineChart
          steps={wmEval.one_step_rollout}
          trueKey="true_reward" predKey="pred_reward"
          label="One-step reward — true vs predicted (val set)"
        />
        <WMDualLineChart
          steps={wmEval.one_step_rollout}
          trueKey="true_obs" predKey="pred_obs"
          label={`Obs dim 0 — one-step prediction`}
        />
        {wmEval.open_loop_rollout.length > 0 && (
          <WMDualLineChart
            steps={wmEval.open_loop_rollout}
            trueKey="true_reward" predKey="pred_reward"
            label="Open-loop rollout — reward drift (pred state feeds forward)"
            predColor="#fbbf24"
          />
        )}
      </div>

      <p className="text-xs text-gray-600">
        Tested on 20% holdout split. Open-loop shows compounding error when the model feeds its own predictions.
      </p>
    </div>
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
  const family = detectEnvFamily(entry.env);
  const fmeta  = ENV_FAMILY_META[family];
  const hparams: [string, string][] = [
    ["lr", String(lr)],
    ["seed", String(entry.hparams.seed ?? "—")],
    ...(entry.hparams.n_steps != null ? [["n_steps", String(entry.hparams.n_steps)] as [string,string]] : []),
    ...(entry.hparams.gamma   != null ? [["gamma",   String(entry.hparams.gamma)]   as [string,string]] : []),
    ...(entry.hparams.ent_coef!= null ? [["ent_coef",String(entry.hparams.ent_coef)]as [string,string]] : []),
    ...(entry.hparams.model   != null ? [["model",   String(entry.hparams.model).split("/").pop() ?? ""] as [string,string]] : []),
  ];
  return (
    <div className={`bg-gray-900 border-l-4 ${st.border} rounded-xl p-4 space-y-3 ${isDoom ? "ring-1 ring-red-800/60" : ""}`}>
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs font-mono text-gray-500">#{index + 1}</span>
          <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${st.badge}`}>{entry.algo}</span>
          <span className={`text-xs px-1.5 py-0.5 rounded-full bg-gray-800 border border-gray-700 ${fmeta.color}`}>
            {fmeta.icon} {fmeta.label}
          </span>
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

// ── Live agent card helpers ────────────────────────────────────────────────────

function rewardLabel(family: EnvFamily, env: string): string {
  if (family === "grpo") return "Accuracy";
  if (family === "toytext") {
    if (env.toLowerCase().includes("blackjack")) return "Win Rate";
    if (isBinaryRewardEnv(env)) return "Success";
    return "Reward";  // Taxi, CliffWalking
  }
  if (family === "atari") return "Score";
  return "Reward";
}

function formatLiveReward(reward: number, family: EnvFamily, env: string): string {
  if (family === "grpo") return `${(reward * 100).toFixed(1)}%`;
  if (family === "toytext" && isBinaryRewardEnv(env)) return `${(reward * 100).toFixed(1)}%`;
  return reward.toFixed(1);
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
  const family = detectEnvFamily(entry.env);
  const fmeta  = ENV_FAMILY_META[family];
  const hasNaN    = hb?.anomaly === "nan_loss";
  const status    = hb?.status ?? "waiting";
  // Step rate estimates: MuJoCo ~100, Box2D ~60, Toy Text ~50, Atari ~10
  const stepsPerSec =
    family === "toytext" ? 50 :
    family === "box2d"   ? 60 :
    family === "atari"   ? 10 :
    100;
  const maxSteps    = entry.time_budget_min * 60 * stepsPerSec;
  const pct         = hb ? Math.min(100, (hb.steps_completed / maxSteps) * 100) : 0;
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
              {inferring ? "⏳ recording…" : fmeta.inferLabel}
            </button>
          )}
          <span className={`text-xs px-2 py-0.5 rounded-full ${STATUS_COLOR[status] ?? "bg-gray-800 text-gray-500"}`}>
            {status}
          </span>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <p className="text-xs text-gray-500 truncate flex-1">{entry.env}</p>
        <span className={`text-xs shrink-0 ${fmeta.color}`}>{fmeta.icon} {fmeta.label}</span>
      </div>

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
            <span className="text-gray-400">{rewardLabel(family, entry.env)}</span>
            <span className={`font-mono font-bold ${hb.current_reward > 0 ? "text-green-400" : "text-gray-300"}`}>
              {formatLiveReward(hb.current_reward, family, entry.env)}
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
          <MiniChart history={history} algoRgb={st.rgb} hasNaN={hasNaN} family={family} env={entry.env} />
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

// ── Live leaderboard helpers ───────────────────────────────────────────────────

function formatScore(score: number, env: string): string {
  if (score === -Infinity) return "—";
  const family = detectEnvFamily(env);
  if (family === "grpo") return `${(score * 100).toFixed(0)}%`;
  if (family === "toytext" && isBinaryRewardEnv(env)) return `${(score * 100).toFixed(0)}%`;
  if (score < 0) return score.toFixed(1);
  return score.toFixed(0);
}

// ── Live leaderboard ──────────────────────────────────────────────────────────

const ENV_SCALE_NOTE: Record<string, string> = {
  // MuJoCo
  "HalfCheetah-v5":         "max ~12 000",
  "HalfCheetah-v4":         "max ~12 000",
  "Hopper-v5":              "max ~3 500",
  "Hopper-v4":              "max ~3 500",
  "Ant-v5":                 "max ~8 000",
  "Walker2d-v5":            "max ~6 000",
  "Humanoid-v5":            "max ~8 000",
  "HumanoidStandup-v5":     "max ~200 000",
  "Swimmer-v5":             "max ~360",
  "Reacher-v5":             "max ~0  (less negative = better)",
  "Pusher-v5":              "max ~0  (less negative = better)",
  "InvertedPendulum-v5":    "max ~1 000",
  "InvertedDoublePendulum-v5": "max ~9 000",
  // Classic Control
  "CartPole-v1":              "max 500",
  "Pendulum-v1":              "max 0  (less negative = better)",
  "MountainCar-v0":           "max 0  (less negative = better)",
  "MountainCarContinuous-v0": "max ~95",
  "Acrobot-v1":               "max 0  (less negative = better)",
  // Box2D
  "LunarLander-v3":           "max ~300",
  "LunarLanderContinuous-v3": "max ~300",
  "BipedalWalker-v3":         "max ~300",
  // Toy Text — scores shown as %
  "FrozenLake-v1":    "success rate (0–100%)",
  "FrozenLake8x8-v1": "success rate (0–100%)",
  "Taxi-v3":          "max ~8  (−200 = worst; closer to +8 = better)",
  "CliffWalking-v1":  "always negative (max −13; less negative = better)",
  "Blackjack-v1":     "win rate (0–100%)",
  // Atari ALE
  "ALE/Pong-v5":           "max +21  (min −21)",
  "ALE/Breakout-v5":       "max ~800",
  "ALE/SpaceInvaders-v5":  "max ~10 000",
  "ALE/MsPacman-v5":       "max ~30 000",
  "ALE/Qbert-v5":          "max ~14 000",
  "ALE/Enduro-v5":         "max ~1 000",
  "ALE/Seaquest-v5":       "max ~20 000",
  "ALE/BeamRider-v5":      "max ~6 000",
  "ALE/Asteroids-v5":      "max ~10 000",
  "ALE/Freeway-v5":        "max 30",
  // World Model (custom dataset)
  "WorldModel-v0":         "reward scale depends on your dataset",
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
                  {formatScore(row.score, row.env)}
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

function VideoModal({
  url, agentId, envId, envFamily, onClose,
}: {
  url: string; agentId: string; envId: string; envFamily: EnvFamily; onClose: () => void;
}) {
  const meta = ENV_FAMILY_META[envFamily];
  const isToyText = envFamily === "toytext";
  const isPixelArt = envFamily === "toytext" || envFamily === "atari";
  const isCompact   = envFamily === "toytext" || envFamily === "atari";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
      onClick={onClose}>
      <div className={`bg-gray-900 border border-gray-700 rounded-2xl p-5 w-full mx-4 space-y-4
          ${isCompact ? "max-w-md" : "max-w-2xl"}`}
        onClick={e => e.stopPropagation()}>

        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-3xl">{meta.icon}</span>
            <div>
              <div className="flex items-center gap-2">
                <p className="font-bold text-gray-100">Agent inference</p>
                <span className={`text-xs px-2 py-0.5 rounded-full bg-gray-800 border border-gray-700 ${meta.color}`}>
                  {meta.label}
                </span>
              </div>
              <p className="text-xs text-gray-500 font-mono">{agentId} · {envId}</p>
            </div>
          </div>
          <button onClick={onClose}
            className="text-gray-500 hover:text-gray-200 text-xl leading-none">✕</button>
        </div>

        {/* Stats bar */}
        <div className="flex items-center gap-4 bg-gray-800/60 rounded-lg px-3 py-2">
          <div className="text-center">
            <p className="text-xs text-gray-500">Playback</p>
            <p className={`text-sm font-bold ${meta.color}`}>{meta.fps}</p>
          </div>
          <div className="w-px h-8 bg-gray-700" />
          <div className="text-center">
            <p className="text-xs text-gray-500">Episodes</p>
            <p className="text-sm font-bold text-gray-200">{meta.episodeNote}</p>
          </div>
          <div className="w-px h-8 bg-gray-700" />
          <div className="flex-1">
            <p className="text-xs text-gray-400">{meta.desc}</p>
          </div>
        </div>

        {/* Video — toy text gets a smaller constrained box to keep pixels readable */}
        <video src={url} controls autoPlay loop
          className={`w-full rounded-xl bg-black object-contain
            ${isCompact ? "max-h-[40vh]" : "max-h-[60vh]"}`}
          style={isPixelArt ? { imageRendering: "pixelated" } as React.CSSProperties : undefined}>
          Your browser does not support HTML video.
        </video>

        <p className="text-xs text-gray-600 text-center">
          Deterministic policy rollout from the saved checkpoint.
        </p>
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
  const [videoModal,  setVideoModal]  = useState<{ agentId: string; url: string; envId: string; envFamily: EnvFamily } | null>(null);

  // ── Dataset / World-Model mode ────────────────────────────────────────────
  const [mode,         setMode]         = useState<"gym" | "dataset">("gym");
  const [datasetMeta,  setDatasetMeta]  = useState<DatasetMeta | null>(null);
  const [uploadStatus, setUploadStatus] = useState<"idle" | "uploading" | "done" | "error">("idle");
  const [hfInput,      setHfInput]      = useState("");
  const [wmStatus,     setWmStatus]     = useState<string | null>(null);
  const [agentLog,     setAgentLog]     = useState<AgentLogEntry[]>([]);

  // WM trainer live heartbeat (epoch, val_loss, total_epochs)
  const [wmHeartbeat, setWmHeartbeat] = useState<Record<string, number | null>>({});
  const [wmEval,      setWmEval]      = useState<WMEval | null>(null);

  // Reward design chat state
  const [rewardMsgs,    setRewardMsgs]    = useState<RewardMsg[]>([]);
  const [rewardCode,    setRewardCode]    = useState<string>("");
  const [rewardInput,   setRewardInput]   = useState<string>("");
  const [rewardBusy,    setRewardBusy]    = useState<boolean>(false);
  const [rewardApplying, setRewardApplying] = useState<boolean>(false);
  const rewardChatRef = useRef<HTMLDivElement>(null);

  const prevSentinelCount = useRef<Record<string, number>>({});
  const pollRef           = useRef<ReturnType<typeof setInterval> | null>(null);
  const textareaRef       = useRef<HTMLTextAreaElement>(null);
  const autoRendered      = useRef<boolean>(false);

  // ── Polling ──────────────────────────────────────────────────────────────────

  const startPolling = useCallback((name: string) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${BACKEND}/api/status/${name}`);
        if (!res.ok) return;
        const data = await res.json();
        const hbs: Heartbeat[]        = data.heartbeats   ?? [];
        const slog: SentinelEntry[]   = data.sentinel_log ?? [];
        const serverPlan: SpawnEntry[] = data.plan        ?? [];
        setHeartbeats(hbs);
        setSentinel(slog);
        // Sync plan from server in case the UI missed agents (e.g. after server restart)
        setPlan(prev => serverPlan.length > prev.length ? serverPlan : prev);

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

        // World-model pipeline: wm_status goes planning → training → done, then status → running
        if (data.wm_status) setWmStatus(data.wm_status);
        if (data.agent_log?.length) setAgentLog(data.agent_log);
        if (data.wm_heartbeat) setWmHeartbeat(data.wm_heartbeat);
        if (data.wm_eval) setWmEval(data.wm_eval);
        // Keep wm_training phase active during RL racing so users see the full pipeline;
        // only regular gym runs transition to "racing".
        if (data.status === "running" || data.status === "racing") {
          setPhase(p => p === "wm_training" ? "wm_training" : "racing");
          if (data.plan?.length) setPlan(prev => data.plan.length > prev.length ? data.plan : prev);
        }

        if (data.status === "completed" || data.status === "failed") {
          clearInterval(pollRef.current!);
          const rRes = await fetch(`${BACKEND}/api/results/${name}`);
          if (rRes.ok) {
            const rData = await rRes.json();
            setResults(rData.results ?? []);
            setBest(rData.best ?? null);
            // Auto-render best RL agent once on completion (never wm_trainer)
            const bestAgent = rData.best;
            const isRlAgent = bestAgent
              && bestAgent.agent_id !== "wm_trainer"
              && bestAgent.algo?.toUpperCase() !== "WORLD_MODEL";
            if (isRlAgent && !autoRendered.current) {
              autoRendered.current = true;
              setTimeout(
                () => handleInfer(bestAgent.agent_id, name, rData.plan ?? []),
                800,
              );
            }
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

  const handleInfer = async (agentId: string, _runName?: string, _plan?: SpawnEntry[], envOverride?: string) => {
    const rn    = _runName ?? runName;
    const pl    = _plan    ?? plan;
    const inferKey = envOverride ? `${agentId}_real` : agentId;
    setInferring(p => ({ ...p, [inferKey]: true }));
    const entry    = pl.find(e => e.id === agentId);
    const envId    = envOverride ?? entry?.env ?? "";
    const envFamily = detectEnvFamily(envId);
    try {
      const res = await fetch(`${BACKEND}/api/infer`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_name: rn, agent_id: agentId, env_override: envOverride ?? null }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail ?? res.statusText);
      }
      const data = await res.json();
      const videoUrl = `${BACKEND}/api/video/${data.filename}`;
      setVideos(v => ({ ...v, [inferKey]: videoUrl }));
      setVideoModal({ agentId, url: videoUrl, envId, envFamily });
    } catch (e) {
      alert(`Inference failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setInferring(p => ({ ...p, [inferKey]: false }));
    }
  };

  const handleUploadDataset = async (file: File) => {
    setUploadStatus("uploading"); setErrorMsg("");
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${BACKEND}/api/upload-dataset`, { method: "POST", body: form });
      if (!res.ok) { const e = await res.json().catch(() => ({ detail: res.statusText })); throw new Error(e.detail ?? res.statusText); }
      setDatasetMeta(await res.json()); setUploadStatus("done");
    } catch (e) { setUploadStatus("error"); setErrorMsg(e instanceof Error ? e.message : String(e)); }
  };

  const handleHFDataset = async () => {
    if (!hfInput.trim()) return;
    setUploadStatus("uploading"); setErrorMsg("");
    try {
      // Support "owner/dataset:config_name" syntax
      const raw = hfInput.trim();
      const colonIdx = raw.lastIndexOf(":");
      const slashIdx = raw.indexOf("/");
      // Only treat colon as config separator when it comes after the slash
      const hasConfig = colonIdx > slashIdx && colonIdx !== -1;
      const dataset_name = hasConfig ? raw.slice(0, colonIdx) : raw;
      const config_name  = hasConfig ? raw.slice(colonIdx + 1) : undefined;
      const res = await fetch(`${BACKEND}/api/hf-dataset`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_name, config_name }),
      });
      if (!res.ok) { const e = await res.json().catch(() => ({ detail: res.statusText })); throw new Error(e.detail ?? res.statusText); }
      setDatasetMeta(await res.json()); setUploadStatus("done");
    } catch (e) { setUploadStatus("error"); setErrorMsg(e instanceof Error ? e.message : String(e)); }
  };

  // ── Reward design helpers ────────────────────────────────────────────────────

  // Build history array to send to the backend (raw LLM format)
  const rewardHistory = (msgs: RewardMsg[]): {role: string; content: string}[] =>
    msgs.map(m => ({
      role: m.role,
      content: m.role === "assistant"
        ? JSON.stringify({ message: m.text, code: m.code ?? "", explanation: m.explanation ?? "" })
        : m.text,
    }));

  const sendRewardMessage = async (userText: string, msgs: RewardMsg[]) => {
    if (!datasetMeta) return;
    setRewardBusy(true);
    const nextMsgs: RewardMsg[] = userText
      ? [...msgs, { role: "user", text: userText }]
      : msgs;
    if (userText) setRewardMsgs(nextMsgs);
    try {
      const res = await fetch(`${BACKEND}/api/design-reward`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ meta: datasetMeta, history: rewardHistory(msgs), message: userText }),
      });
      if (!res.ok) { const e = await res.json().catch(() => ({ detail: res.statusText })); throw new Error(e.detail ?? res.statusText); }
      const data = await res.json();
      const updated: RewardMsg[] = [...nextMsgs, {
        role: "assistant", text: data.message, code: data.code, explanation: data.explanation,
      }];
      setRewardMsgs(updated);
      setRewardCode(data.code);
      setTimeout(() => rewardChatRef.current?.scrollTo({ top: 99999, behavior: "smooth" }), 80);
    } catch (e) { setErrorMsg(e instanceof Error ? e.message : String(e)); }
    finally { setRewardBusy(false); }
  };

  const handleEnterRewardDesign = () => {
    setPhase("reward_design");
    setRewardMsgs([]); setRewardCode(""); setRewardInput("");
    // Auto-fire first LLM message
    sendRewardMessage("", []);
  };

  const handleApproveReward = async () => {
    if (!datasetMeta || !rewardCode) return;
    setRewardApplying(true); setErrorMsg("");
    try {
      // Apply reward to dataset
      const applyRes = await fetch(`${BACKEND}/api/apply-reward`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ meta: datasetMeta, reward_code: rewardCode }),
      });
      if (!applyRes.ok) { const e = await applyRes.json().catch(() => ({ detail: applyRes.statusText })); throw new Error(e.detail ?? applyRes.statusText); }
      const updatedMeta = await applyRes.json();
      setDatasetMeta(updatedMeta);
      // Launch WM pipeline with updated meta
      await _launchWorldModel(updatedMeta);
    } catch (e) { setErrorMsg(e instanceof Error ? e.message : String(e)); setPhase("error"); }
    finally { setRewardApplying(false); }
  };

  const _launchWorldModel = async (meta: DatasetMeta) => {
    setPhase("wm_training"); setErrorMsg("");
    const res = await fetch(`${BACKEND}/api/world-model-plan`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ meta }),
    });
    if (!res.ok) { const e = await res.json().catch(() => ({ detail: res.statusText })); throw new Error(e.detail ?? res.statusText); }
    const data = await res.json();
    setRunName(data.run_name); setWmStatus("planning");
    startPolling(data.run_name);
  };

  const handleStartWorldModel = async () => {
    if (!datasetMeta) return;
    try { await _launchWorldModel(datasetMeta); }
    catch (e) { setErrorMsg(e instanceof Error ? e.message : String(e)); setPhase("error"); }
  };

  const handleReset = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    setPhase("idle"); setTask(""); setPlan([]); setRunName(""); setRunDir("");
    setHeartbeats([]); setSentinel([]); setResults([]); setBest(null); setErrorMsg("");
    setHistory({}); setAnimStates({}); setInferring({}); setVideos({}); setVideoModal(null);
    setDatasetMeta(null); setUploadStatus("idle"); setHfInput(""); setWmStatus(null); setAgentLog([]);
    setRewardMsgs([]); setRewardCode(""); setRewardInput(""); setRewardBusy(false); setRewardApplying(false);
    setWmHeartbeat({}); setWmEval(null);
    prevSentinelCount.current = {};
    autoRendered.current = false;
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
        <VideoModal
          url={videoModal.url}
          agentId={videoModal.agentId}
          envId={videoModal.envId}
          envFamily={videoModal.envFamily}
          onClose={() => setVideoModal(null)}
        />
      )}

      <Header />

      {/* ── IDLE / ERROR ── */}
      {(phase === "idle" || phase === "error") && (
        <div className="w-full max-w-xl space-y-4">
          {/* Mode toggle */}
          <div className="flex bg-gray-900 border border-gray-800 rounded-xl p-1 gap-1">
            {(["gym", "dataset"] as const).map(m => (
              <button key={m} onClick={() => { setMode(m); setDatasetMeta(null); setUploadStatus("idle"); setErrorMsg(""); }}
                className={`flex-1 text-sm font-semibold py-2 rounded-lg transition-colors
                  ${mode === m ? "bg-violet-600 text-white" : "text-gray-400 hover:text-gray-200"}`}>
                {m === "gym" ? "🎮 Gym Task" : "📊 Custom Dataset"}
              </button>
            ))}
          </div>

          {mode === "gym" ? (
            <>
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
              <button onClick={handleGeneratePlan} disabled={!task.trim()}
                className="w-full bg-violet-600 hover:bg-violet-500 disabled:bg-gray-800 disabled:text-gray-600 text-white font-semibold py-3 rounded-xl transition-colors text-sm">
                Generate Agent Lineup →
              </button>
            </>
          ) : (
            /* ── Dataset upload panel ── */
            <div className="space-y-4">
              <p className="text-center text-gray-300 text-lg font-medium">Upload your RL dataset</p>
              <p className="text-xs text-gray-500 text-center">Provide transitions: (obs, action, reward, next_obs, done) — CSV, JSON, or parquet</p>

              {/* File upload */}
              <label className={`flex flex-col items-center gap-2 border-2 border-dashed rounded-xl px-4 py-6 cursor-pointer transition-colors
                ${uploadStatus === "uploading" ? "border-violet-700 bg-violet-950/20" : "border-gray-700 hover:border-violet-600 bg-gray-900 hover:bg-gray-900/80"}`}>
                <span className="text-3xl">📁</span>
                <span className="text-sm text-gray-400">
                  {uploadStatus === "uploading" ? "Inspecting…" : "Drop file here or click to browse"}
                </span>
                <span className="text-xs text-gray-600">CSV · JSON · JSONL · parquet</span>
                <input type="file" className="hidden" accept=".csv,.json,.jsonl,.parquet"
                  disabled={uploadStatus === "uploading"}
                  onChange={e => { if (e.target.files?.[0]) handleUploadDataset(e.target.files[0]); }} />
              </label>

              <div className="flex items-center gap-3">
                <div className="flex-1 h-px bg-gray-800" />
                <span className="text-xs text-gray-600">or download from HuggingFace</span>
                <div className="flex-1 h-px bg-gray-800" />
              </div>

              {/* HuggingFace input */}
              <div className="flex gap-2">
                <input value={hfInput} onChange={e => setHfInput(e.target.value)}
                  onKeyDown={e => { if (e.key === "Enter") handleHFDataset(); }}
                  placeholder="owner/dataset:config  (e.g. jat-project/jat-dataset:mujoco-ant)"
                  className="flex-1 bg-gray-900 border border-gray-700 focus:border-violet-500 rounded-xl px-3 py-2 text-sm text-gray-100 placeholder-gray-600 outline-none transition-colors" />
                <button onClick={handleHFDataset} disabled={!hfInput.trim() || uploadStatus === "uploading"}
                  className="px-4 py-2 text-sm font-semibold bg-gray-800 hover:bg-gray-700 disabled:opacity-40 text-gray-300 rounded-xl border border-gray-700 transition-colors whitespace-nowrap">
                  {uploadStatus === "uploading" ? "⏳" : "Download"}
                </button>
              </div>

              {/* Dataset meta preview */}
              {datasetMeta && uploadStatus === "done" && (
                <div className="bg-teal-950/40 border border-teal-800/50 rounded-xl p-4 space-y-3">
                  <p className="text-sm font-semibold text-teal-300">Dataset inspected</p>
                  <div className="grid grid-cols-2 gap-x-6 gap-y-1.5 text-xs">
                    <div className="text-gray-500">Samples</div>
                    <div className="text-gray-200 font-mono">{datasetMeta.n_samples.toLocaleString()}</div>
                    <div className="text-gray-500">Obs dim</div>
                    <div className="text-gray-200 font-mono">{datasetMeta.obs_dim}</div>
                    <div className="text-gray-500">Action</div>
                    <div className="text-gray-200 font-mono">
                      {datasetMeta.act_type === "discrete" ? `discrete (${datasetMeta.act_n} actions)` : `continuous (dim ${datasetMeta.act_dim})`}
                    </div>
                    <div className="text-gray-500">Reward range</div>
                    <div className="text-gray-200 font-mono">{datasetMeta.reward_min.toFixed(2)} – {datasetMeta.reward_max.toFixed(2)}</div>
                    <div className="text-gray-500">World model</div>
                    <div className="text-gray-200 font-mono">MLP {datasetMeta.hidden_sizes.join("×")}</div>
                    {datasetMeta._split_used && (
                      <>
                        <div className="text-gray-500">Split used</div>
                        <div className="text-gray-400 font-mono text-xs">{datasetMeta._split_used}</div>
                      </>
                    )}
                  </div>
                  {datasetMeta._size_reasoning && (
                    <div className="flex gap-2 bg-gray-900/60 rounded-lg p-2.5">
                      <span className="text-base shrink-0">📦</span>
                      <p className="text-xs text-gray-400 leading-relaxed">
                        <span className="text-green-400 font-semibold">Data Sizer: </span>
                        {datasetMeta._size_reasoning}
                      </p>
                    </div>
                  )}
                  <p className="text-xs text-gray-500">
                    Multi-agent planner will choose architecture, algorithms &amp; hyperparameters before training.
                  </p>
                  <div className="flex gap-2">
                    <button onClick={handleEnterRewardDesign}
                      className="flex-1 bg-violet-700 hover:bg-violet-600 text-white font-semibold py-2.5 rounded-xl transition-colors text-sm">
                      ✏️ Design Reward
                    </button>
                    <button onClick={handleStartWorldModel}
                      className="flex-1 bg-teal-600 hover:bg-teal-500 text-white font-semibold py-2.5 rounded-xl transition-colors text-sm">
                      🧠 Train →
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {phase === "error" && (
            <div className="bg-red-950 border border-red-800 rounded-xl p-3 text-sm text-red-300">
              <p className="font-semibold mb-1">Error</p>
              <p className="font-mono text-xs break-all">{errorMsg}</p>
              <p className="text-xs text-red-400 mt-2">Backend running? <code className="font-mono">bash ui/agent/start.sh</code></p>
            </div>
          )}
        </div>
      )}

      {/* ── REWARD DESIGN ── */}
      {phase === "reward_design" && datasetMeta && (
        <div className="w-full max-w-2xl space-y-4">
          {/* Header */}
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold text-gray-100">Design Reward Function</h2>
            <button onClick={() => setPhase("idle")}
              className="text-xs text-gray-500 hover:text-gray-300 transition-colors">← Back</button>
          </div>

          {/* Dataset context chip */}
          <div className="flex flex-wrap gap-2 text-xs">
            {[
              ["obs_dim", datasetMeta.obs_dim],
              ["act", datasetMeta.act_type === "discrete" ? `discrete ×${datasetMeta.act_n}` : `continuous ×${datasetMeta.act_dim}`],
              ["reward", `${datasetMeta.reward_min.toFixed(2)} – ${datasetMeta.reward_max.toFixed(2)}`],
              ["samples", datasetMeta.n_samples.toLocaleString()],
            ].map(([k, v]) => (
              <span key={String(k)} className="bg-gray-800 border border-gray-700 rounded-lg px-2 py-1 text-gray-400">
                <span className="text-gray-600">{k}: </span>{v}
              </span>
            ))}
          </div>

          {/* Chat messages */}
          <div ref={rewardChatRef}
            className="bg-gray-950 border border-gray-800 rounded-xl p-4 space-y-4 max-h-80 overflow-y-auto">
            {rewardMsgs.length === 0 && (
              <div className="flex items-center gap-2 text-gray-600 text-sm">
                <div className="w-4 h-4 rounded-full border border-violet-500 border-t-transparent animate-spin" />
                Analysing dataset…
              </div>
            )}
            {rewardMsgs.map((msg, i) => (
              <div key={i} className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : ""}`}>
                <div className={`shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-sm
                  ${msg.role === "assistant" ? "bg-violet-800 text-violet-200" : "bg-gray-700 text-gray-300"}`}>
                  {msg.role === "assistant" ? "🤖" : "You"}
                </div>
                <div className={`flex-1 space-y-2 ${msg.role === "user" ? "items-end" : ""}`}>
                  <p className={`text-sm leading-relaxed rounded-xl px-3 py-2 inline-block max-w-full
                    ${msg.role === "assistant"
                      ? "bg-gray-900 text-gray-200 text-left"
                      : "bg-violet-900/60 text-violet-100 text-right ml-auto"}`}>
                    {msg.text}
                  </p>
                  {msg.code && (
                    <pre className="text-xs bg-gray-900 border border-gray-700 rounded-lg p-3 overflow-x-auto
                      text-green-300 font-mono leading-relaxed">
                      {msg.code}
                    </pre>
                  )}
                  {msg.explanation && (
                    <p className="text-xs text-gray-500 italic">{msg.explanation}</p>
                  )}
                </div>
              </div>
            ))}
            {rewardBusy && (
              <div className="flex items-center gap-2 text-gray-600 text-sm">
                <div className="w-4 h-4 rounded-full border border-violet-500 border-t-transparent animate-spin" />
                Thinking…
              </div>
            )}
          </div>

          {/* Input + send */}
          <div className="flex gap-2">
            <input
              value={rewardInput}
              onChange={e => setRewardInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === "Enter" && !e.shiftKey && rewardInput.trim() && !rewardBusy) {
                  const txt = rewardInput.trim();
                  setRewardInput("");
                  sendRewardMessage(txt, rewardMsgs);
                }
              }}
              disabled={rewardBusy}
              placeholder="Ask for changes… (e.g. add a penalty for large actions)"
              className="flex-1 bg-gray-900 border border-gray-700 focus:border-violet-500 rounded-xl px-3 py-2 text-sm text-gray-100 placeholder-gray-600 outline-none transition-colors disabled:opacity-40"
            />
            <button
              disabled={!rewardInput.trim() || rewardBusy}
              onClick={() => { const txt = rewardInput.trim(); setRewardInput(""); sendRewardMessage(txt, rewardMsgs); }}
              className="px-4 py-2 text-sm font-semibold bg-gray-800 hover:bg-gray-700 disabled:opacity-40 text-gray-300 rounded-xl border border-gray-700 transition-colors">
              Send
            </button>
          </div>

          {/* Approve */}
          <button
            onClick={handleApproveReward}
            disabled={!rewardCode || rewardBusy || rewardApplying}
            className="w-full bg-teal-600 hover:bg-teal-500 disabled:opacity-40 text-white font-semibold py-3 rounded-xl transition-colors text-sm flex items-center justify-center gap-2">
            {rewardApplying
              ? <><div className="w-4 h-4 rounded-full border-2 border-white border-t-transparent animate-spin" /> Applying reward & launching…</>
              : "✅ Approve & Train World Model →"}
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

      {/* ── WM TRAINING (planning + training phases) ── */}
      {phase === "wm_training" && (() => {
        const epoch       = (wmHeartbeat.steps_completed as number) ?? 0;
        const totalEpochs = (wmHeartbeat.total_epochs   as number) ?? 200;
        const valLoss     = (wmHeartbeat.loss            as number | null) ?? null;
        const pct         = totalEpochs > 0 ? Math.min(100, (epoch / totalEpochs) * 100) : 0;
        const isPlanning  = wmStatus === "planning" || (wmStatus !== "training" && wmStatus !== "done" && wmStatus !== "evaluating" && wmStatus !== "eval_done" && agentLog.length < 3);
        const isTraining  = wmStatus === "training";
        const isEvaluating = wmStatus === "evaluating";
        const hasEval     = wmStatus === "eval_done" || !!wmEval;

        const AGENT_META: Record<string, { icon: string; label: string; color: string }> = {
          arch_search:   { icon: "🏗️",  label: "Arch Agent",     color: "text-violet-400" },
          algo_selector: { icon: "🎯",  label: "Algo Selector",  color: "text-blue-400"   },
          hparam:        { icon: "⚙️",  label: "Hparam Agent",   color: "text-amber-400"  },
          dataset_size:  { icon: "📦",  label: "Data Sizer",     color: "text-green-400"  },
          system:        { icon: "⚡",  label: "System",         color: "text-gray-400"   },
        };

        return (
          <div className="w-full max-w-2xl space-y-5">

            {/* Header */}
            <div className="flex items-center gap-3">
              {best ? (
                <div className="w-10 h-10 rounded-full bg-teal-800/60 flex items-center justify-center shrink-0 text-xl">🏆</div>
              ) : (
                <div className={`w-10 h-10 rounded-full border-2 shrink-0 animate-spin ${
                  isPlanning ? "border-violet-500 border-t-transparent"
                  : plan.length > 0 ? "border-violet-400 border-t-transparent"
                  : "border-teal-500 border-t-transparent"
                }`} />
              )}
              <div>
                <p className="text-gray-100 font-semibold">
                  {best ? "World Model Pipeline Complete"
                   : isPlanning ? "Multi-Agent Planning…"
                   : plan.length > 0 ? "RL Agents Racing in World Model…"
                   : isEvaluating ? "Testing World Model on Holdout Data…"
                   : hasEval && !plan.length ? "World Model Validated — Launching RL Agents…"
                   : isTraining ? "Training World Model…"
                   : wmStatus === "done" ? "World Model Trained — Running Tests…"
                   : "Training World Model…"}
                </p>
                <p className="text-xs text-gray-500">
                  {best
                    ? `Best: ${best.agent_id} · ${best.algo} · return ${best.mean_return?.toFixed(3) ?? "—"}`
                    : isPlanning
                      ? "Agents are deciding architecture, algorithms & hyperparameters"
                      : plan.length > 0
                        ? `${plan.length} algorithms racing inside the learned simulator`
                        : isEvaluating
                          ? "Evaluating one-step and open-loop predictions on 20% holdout"
                          : hasEval && wmEval
                            ? `Val obs MSE ${wmEval.obs_mse.toFixed(4)} · reward MAE ${wmEval.reward_mae.toFixed(4)} · starting PPO/SAC/A2C…`
                            : isTraining
                              ? `Phase 2 of 4 — learning dynamics from ${datasetMeta?.n_samples?.toLocaleString() ?? "?"} transitions`
                              : "Preparing world model pipeline…"}
                </p>
              </div>
            </div>

            {/* Phase indicator strip */}
            <div className="flex gap-1 text-xs">
              {[
                { key: "plan",  label: "1 · Plan",        done: agentLog.length >= 3 || isTraining || hasEval || plan.length > 0 },
                { key: "train", label: "2 · Train WM",    done: wmStatus === "done" || wmStatus === "evaluating" || hasEval || plan.length > 0 },
                { key: "test",  label: "3 · Test WM",   done: hasEval || plan.length > 0 },
                { key: "race",  label: "4 · Race RL",     done: !!best },
              ].map(s => {
                const active = !s.done && (
                  (s.key === "plan"  && isPlanning)   ||
                  (s.key === "train" && isTraining)   ||
                  (s.key === "test"  && (isEvaluating || (wmStatus === "done" && !hasEval))) ||
                  (s.key === "race"  && plan.length > 0 && !best)
                );
                return (
                  <div key={s.key}
                    className={`flex-1 rounded-lg py-1.5 text-center font-medium border transition-colors ${
                      s.done   ? "bg-teal-900/50 border-teal-700 text-teal-300"
                      : active ? "bg-violet-900/40 border-violet-700 text-violet-300"
                               : "bg-gray-900 border-gray-800 text-gray-600"
                    }`}>
                    {s.done ? "✓ " : ""}{s.label}
                  </div>
                );
              })}
            </div>

            {/* Agent reasoning log */}
            {agentLog.length > 0 && (
              <div className="bg-gray-900/80 border border-gray-800 rounded-xl overflow-hidden">
                <div className="px-4 py-2 border-b border-gray-800 flex items-center gap-2">
                  <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Agent Reasoning</span>
                  <span className="text-xs text-gray-600">{agentLog.length} decisions</span>
                </div>
                <div className="divide-y divide-gray-800/60 max-h-56 overflow-y-auto">
                  {agentLog.map((entry, i) => {
                    const m = AGENT_META[entry.agent] ?? { icon: "🤖", label: entry.agent, color: "text-gray-400" };
                    return (
                      <div key={i} className="px-4 py-3 group">
                        <div className="flex items-start gap-2">
                          <span className="text-base shrink-0">{m.icon}</span>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-0.5">
                              <span className={`text-xs font-semibold ${m.color}`}>{m.label}</span>
                              <span className="text-xs text-gray-600">
                                {new Date(entry.timestamp * 1000).toLocaleTimeString()}
                              </span>
                            </div>
                            <p className="text-xs text-gray-200 font-medium">{entry.decision}</p>
                            <p className="text-xs text-gray-500 mt-0.5 hidden group-hover:block leading-relaxed">
                              {entry.reasoning}
                            </p>
                          </div>
                          <span className="text-teal-500 text-xs shrink-0">✓</span>
                        </div>
                      </div>
                    );
                  })}
                  {isPlanning && agentLog.length < 3 && (
                    <div className="px-4 py-3 flex items-center gap-2">
                      <div className="w-4 h-4 rounded-full border-2 border-violet-500 border-t-transparent animate-spin shrink-0" />
                      <span className="text-xs text-gray-500">
                        {agentLog.length === 0 ? "arch_search deciding architecture…"
                         : agentLog.length === 1 ? "algo_selector choosing algorithms…"
                         : "hparam_agent tuning hyperparameters…"}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* WM training progress (shown once training starts) */}
            {(isTraining || wmStatus === "done" || isEvaluating || hasEval) && (
              <div className="bg-gray-900 border border-teal-800/50 rounded-xl p-4 space-y-3">
                {/* Header row */}
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                    World Model Training
                  </span>
                  <span className="font-mono text-xs text-teal-300">
                    {isTraining ? `epoch ${epoch} / ${totalEpochs}`
                      : "✓ trained"}
                  </span>
                </div>

                {/* Real progress bar */}
                <div className="relative w-full bg-gray-800 rounded-full h-2 overflow-hidden">
                  <div
                    className="h-2 rounded-full bg-gradient-to-r from-teal-600 to-teal-400 transition-all duration-700"
                    style={{ width: isTraining ? `${Math.max(pct, 2)}%` : "100%" }}
                  />
                  {isTraining && pct < 100 && (
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent animate-pulse rounded-full" />
                  )}
                </div>

                {/* Val loss + epoch counters */}
                <div className="grid grid-cols-3 gap-3 text-center">
                  <div>
                    <p className="text-xs text-gray-600">Epoch</p>
                    <p className="font-mono text-sm font-bold text-gray-200">{epoch || "—"}</p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-600">Progress</p>
                    <p className="font-mono text-sm font-bold text-teal-400">
                      {isTraining ? (epoch > 0 ? `${pct.toFixed(0)}%` : "—") : "100%"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-600">Val Loss</p>
                    <p className={`font-mono text-sm font-bold ${
                      valLoss === null ? "text-gray-600"
                      : valLoss < 0.1  ? "text-teal-400"
                      : valLoss < 0.5  ? "text-yellow-400"
                      : "text-gray-300"}`}>
                      {valLoss !== null ? valLoss.toFixed(4) : "—"}
                    </p>
                  </div>
                </div>

                {datasetMeta && (
                  <p className="text-xs text-gray-600">
                    {datasetMeta.n_samples.toLocaleString()} transitions · obs {datasetMeta.obs_dim}D · {datasetMeta.act_type}
                  </p>
                )}
              </div>
            )}

            {/* WM eval results — shown after training, before RL swarm */}
            {isEvaluating && !wmEval && (
              <div className="bg-gray-900 border border-teal-800/50 rounded-xl p-4 flex items-center gap-3">
                <div className="w-5 h-5 rounded-full border-2 border-teal-500 border-t-transparent animate-spin shrink-0" />
                <p className="text-xs text-gray-400">Running one-step &amp; open-loop evaluation on holdout transitions…</p>
              </div>
            )}
            {wmEval && <WMEvalPanel eval={wmEval} />}

            {/* Phase 4 — RL agents racing inside the world model */}
            {plan.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full border border-violet-500 border-t-transparent animate-spin shrink-0" />
                  <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                    Phase 4 · RL Agents Racing in World Model
                  </span>
                </div>
                <div className="grid grid-cols-1 gap-2">
                  {plan.map(entry => {
                    const hb      = heartbeats.find(h => h.agent_id === entry.id);
                    const vidUrl  = videos[entry.id];
                    const isInfer = !!inferring[entry.id];
                    const isDone  = hb?.status === "completed";
                    return (
                      <div key={entry.id}
                        className="bg-gray-900 border border-gray-800 rounded-xl px-4 py-3 flex items-center gap-3">
                        <div className={`w-2 h-2 rounded-full shrink-0 ${
                          isDone ? "bg-teal-400" : hb ? "bg-violet-400 animate-pulse" : "bg-gray-700"
                        }`} />
                        <div className="flex-1 min-w-0">
                          <p className="text-xs font-semibold text-gray-300">
                            {entry.id} · <span className="text-violet-400">{entry.algo}</span>
                          </p>
                          {hb ? (
                            <p className="text-xs text-gray-500 font-mono">
                              reward {hb.current_reward.toFixed(3)}
                              {hb.loss !== null && hb.loss !== undefined ? `  loss ${(hb.loss as number).toFixed(3)}` : ""}
                            </p>
                          ) : (
                            <p className="text-xs text-gray-600">waiting…</p>
                          )}
                        </div>
                        {/* Render buttons */}
                        <div className="flex flex-col items-end gap-1 shrink-0">
                          {/* World model trajectory render */}
                          {vidUrl ? (
                            <button onClick={() => {
                              const envFamily = detectEnvFamily(entry.env);
                              setVideoModal({ agentId: entry.id, url: vidUrl, envId: entry.env, envFamily });
                            }}
                              className="text-xs text-teal-400 hover:text-teal-300 font-semibold transition-colors">
                              ▶ watch
                            </button>
                          ) : (isDone || hb) ? (
                            <button onClick={() => handleInfer(entry.id)} disabled={isInfer}
                              className="text-xs text-gray-500 hover:text-violet-400 disabled:opacity-40 font-semibold transition-colors">
                              {isInfer ? "⏳" : "▶ WM plot"}
                            </button>
                          ) : null}
                          {/* Real env render — only when source_env is known */}
                          {(isDone || hb) && datasetMeta?.source_env && (() => {
                            const realKey = `${entry.id}_real`;
                            const realVid = videos[realKey];
                            const isRealInfer = !!inferring[realKey];
                            return realVid ? (
                              <button onClick={() => {
                                const ef = detectEnvFamily(datasetMeta.source_env!);
                                setVideoModal({ agentId: entry.id, url: realVid, envId: datasetMeta.source_env!, envFamily: ef });
                              }}
                                className="text-xs text-amber-400 hover:text-amber-300 font-semibold transition-colors">
                                🦾 watch real
                              </button>
                            ) : (
                              <button onClick={() => handleInfer(entry.id, undefined, undefined, datasetMeta.source_env)}
                                disabled={isRealInfer}
                                className="text-xs text-gray-600 hover:text-amber-400 disabled:opacity-40 font-semibold transition-colors">
                                {isRealInfer ? "⏳" : `🦾 ${datasetMeta.source_env}`}
                              </button>
                            );
                          })()}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Pre-race hint when plan not yet available */}
            {plan.length === 0 && hasEval && (
              <p className="text-xs text-gray-600 text-center animate-pulse">
                Launching{" "}
                {agentLog.find(e => e.agent === "algo_selector")?.decision?.replace("Competing: ", "") ?? "PPO / SAC / A2C"}{" "}
                inside the validated world model…
              </p>
            )}
            {plan.length === 0 && !hasEval && agentLog.length >= 3 && wmStatus !== "training" && wmStatus !== "evaluating" && (
              <p className="text-xs text-gray-600 text-center">
                Phase 3 will test the world model on holdout data, then race{" "}
                {agentLog.find(e => e.agent === "algo_selector")?.decision?.replace("Competing: ", "") ?? "PPO / SAC / A2C"}.
              </p>
            )}

            {/* ── Pipeline complete — best agent summary ── */}
            {best && (
              <div className="bg-teal-950/40 border border-teal-700/50 rounded-xl p-4 space-y-3">
                <div className="flex items-center gap-2">
                  <span className="text-teal-400 text-lg">🏆</span>
                  <p className="text-sm font-bold text-teal-300">Pipeline Complete</p>
                </div>
                <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs">
                  <span className="text-gray-500">Best agent</span>
                  <span className="font-mono text-gray-200">{best.agent_id} · {best.algo}</span>
                  <span className="text-gray-500">Mean return</span>
                  <span className="font-mono text-teal-400 font-bold">{best.mean_return?.toFixed(3) ?? "—"}</span>
                </div>
                <div className="flex gap-2">
                  {/* World model trajectory video */}
                  {videos[best.agent_id] ? (
                    <button onClick={() => {
                      const envFamily = detectEnvFamily(best.env ?? "WorldModel-v0");
                      setVideoModal({ agentId: best.agent_id, url: videos[best.agent_id], envId: best.env ?? "WorldModel-v0", envFamily });
                    }}
                      className="flex-1 bg-teal-700 hover:bg-teal-600 text-white text-sm font-semibold py-2 rounded-xl transition-colors">
                      ▶ Watch WM Plot
                    </button>
                  ) : (
                    <button onClick={() => handleInfer(best.agent_id)}
                      disabled={!!inferring[best.agent_id]}
                      className="flex-1 bg-teal-700 hover:bg-teal-600 disabled:opacity-40 text-white text-sm font-semibold py-2 rounded-xl transition-colors flex items-center justify-center gap-2">
                      {inferring[best.agent_id]
                        ? <><div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" /> Rendering…</>
                        : "▶ WM Plot"}
                    </button>
                  )}
                  {/* Real env video — only when source env is known */}
                  {datasetMeta?.source_env && (() => {
                    const realKey = `${best.agent_id}_real`;
                    const realVid = videos[realKey];
                    const isRI    = !!inferring[realKey];
                    return realVid ? (
                      <button onClick={() => {
                        const ef = detectEnvFamily(datasetMeta.source_env!);
                        setVideoModal({ agentId: best.agent_id, url: realVid, envId: datasetMeta.source_env!, envFamily: ef });
                      }}
                        className="flex-1 bg-amber-700 hover:bg-amber-600 text-white text-sm font-semibold py-2 rounded-xl transition-colors">
                        🦾 Watch Real
                      </button>
                    ) : (
                      <button onClick={() => handleInfer(best.agent_id, undefined, undefined, datasetMeta.source_env)}
                        disabled={isRI}
                        className="flex-1 bg-amber-800/70 hover:bg-amber-700 disabled:opacity-40 text-amber-200 text-sm font-semibold py-2 rounded-xl transition-colors flex items-center justify-center gap-2">
                        {isRI
                          ? <><div className="w-4 h-4 border-2 border-amber-200 border-t-transparent rounded-full animate-spin" /> Rendering…</>
                          : `🦾 ${datasetMeta.source_env}`}
                      </button>
                    );
                  })()}
                </div>
              </div>
            )}

            <button onClick={handleReset}
              className="text-xs text-gray-600 hover:text-gray-400 underline mx-auto block transition-colors">
              {best ? "← Start new run" : "Cancel"}
            </button>
          </div>
        );
      })()}

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
          {task && <p className="text-xs text-gray-500 italic mb-5">&ldquo;{task}&rdquo;</p>}
          {!task && datasetMeta && (
            <p className="text-xs text-gray-500 mb-5">
              🧠 World model trained on {datasetMeta.n_samples.toLocaleString()} samples — agents racing inside the learned simulator
            </p>
          )}

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
                      {inferring[best.agent_id]
                        ? "⏳ Recording…"
                        : `${ENV_FAMILY_META[detectEnvFamily(best.env)].icon} Watch inference`}
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
