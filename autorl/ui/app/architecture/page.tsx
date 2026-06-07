"use client";

const C = {
  bg:       "#1C1A17",
  surface:  "#242118",
  surfaceB: "#2A2720",
  border:   "#38342C",
  borderHi: "#504A3E",
  coral:    "#D4714E",
  amber:    "#C8925A",
  text:     "#EDE8DF",
  textDim:  "#8C8378",
  textFaint:"#524E48",
  violet:   "#8B76D4",
  cyan:     "#4EAEC0",
  red:      "#C84848",
  green:    "#52A86C",
  wandb:    "#E8A020",
  weave:    "#6890D8",
  openai:   "#18A878",
};

export default function ArchitecturePage() {
  return (
    <div className="min-h-screen" style={{ background: C.bg, color: C.text, fontFamily: "'Inter', system-ui, sans-serif" }}>

      {/* Top bar */}
      <div className="flex items-center justify-between px-6 py-3"
        style={{ borderBottom: `1px solid ${C.border}`, background: C.surface }}>
        <div className="flex items-center gap-2">
          <div className="w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold"
            style={{ background: C.coral, color: "#fff" }}>A</div>
          <span className="font-semibold text-sm">AutoRL</span>
          <span className="text-xs" style={{ color: C.textFaint }}>/</span>
          <span className="text-xs" style={{ color: C.textDim }}>Multi-Agent Architecture</span>
        </div>
      </div>

      {/* Main diagram — single centered column */}
      <div className="max-w-3xl mx-auto px-6 py-8 space-y-0">

        {/* ── USER ── */}
        <AgentBox color={C.textDim} badge="Human" title="User" center>
          <Bullet>Types a natural-language RL task · sees live charts, sentinel alerts, final results</Bullet>
        </AgentBox>

        <Flow color={C.textDim} label="task prompt" />

        {/* ── ORCHESTRATOR ── */}
        <AgentBox color={C.coral} badge="LLM Agent" title="Orchestrator Agent" center>
          <Bullet>Calls <Tag color={C.openai}>OpenAI</Tag> to generate a SpawnPlan — picks algos (PPO / SAC / A2C / GRPO), envs, and hparams</Bullet>
          <Bullet>Spawns <b>N agents in parallel</b> — any number, local or RunPod · seeds one with bad lr to demo Sentinel recovery · traced to <Tag color={C.weave}>Weave</Tag></Bullet>
        </AgentBox>

        {/* Fork */}
        <div className="flex items-start">
          <ForkArrow label="N × SpawnPlan entries · exec=local" color={C.coral} left />
          <ForkArrow label="N × SpawnPlan entries · exec=runpod" color={C.coral} />
        </div>

        {/* ── TRAINING AGENTS ROW ── */}
        <div className="grid grid-cols-2 gap-3">

          <AgentBox color={C.violet} badge="SB3 · Local · N agents" title="PPO / SAC / A2C">
            <Bullet>Each agent runs as an asyncio subprocess · trains in 5k-step chunks</Bullet>
            <Bullet>Early-stop on stagnation · race dropout vs peers · warm-starts from <Tag color={C.wandb}>W&B</Tag></Bullet>
            <Bullet>Logs metrics + checkpoint to <Tag color={C.wandb}>W&B</Tag> · traces to <Tag color={C.weave}>Weave</Tag></Bullet>
          </AgentBox>

          <AgentBox color={C.cyan} badge="RunPod · GPU · N agents" title="GRPO Agent">
            <Bullet>Each agent gets its own GPU pod (H100 / RTX PRO 6000) · fine-tunes Qwen2.5-3B + LoRA</Bullet>
            <Bullet>Reward: format (&lt;think&gt; tags) + accuracy · demonstrates reasoning emergence</Bullet>
            <Bullet>LoRA adapter → <Tag color={C.wandb}>W&B</Tag> · before/after traces → <Tag color={C.weave}>Weave</Tag></Bullet>
          </AgentBox>
        </div>

        {/* ── CONNECTION BAND: Training ↔ Sentinel ── */}
        <ConnBand>
          <ConnRow color={C.green} dir="up">
            <b>heartbeat</b> &#123;steps, reward, anomaly&#125; — every agent writes this every 2s; Sentinel reads all of them
          </ConnRow>
          <ConnRow color={C.red} dir="down">
            <b>nudge</b> &#123;new_lr&#125; via <b>Redis pub/sub</b> (file fallback) · <b>kill + restart</b> with LLM-suggested hparams
          </ConnRow>
        </ConnBand>

        {/* ── SENTINEL ── */}
        <div className="rounded-2xl overflow-hidden"
          style={{ border: `2px solid #3C1818`, boxShadow: `0 0 20px #C8484812` }}>
          <div className="px-4 py-2.5 flex items-center gap-2"
            style={{ background: "#180A0A", borderBottom: `1px solid #3C1818` }}>
            <div className="relative flex-shrink-0">
              <div className="w-2 h-2 rounded-full" style={{ background: C.red }} />
              <div className="absolute inset-0 w-2 h-2 rounded-full animate-ping" style={{ background: C.red, opacity: 0.4 }} />
            </div>
            <span className="text-xs font-bold" style={{ color: C.red }}>Sentinel Agent</span>
            <span className="text-xs px-2 py-0.5 rounded-full ml-1"
              style={{ background: `${C.red}18`, color: C.red, border: `1px solid ${C.red}30` }}>
              LLM loop · always-on
            </span>
            <span className="text-xs ml-auto" style={{ color: "#E0505060" }}>
              watches all N agents · runs until training completes
            </span>
          </div>
          <div className="px-4 py-3 grid grid-cols-2 gap-4" style={{ background: "#130606" }}>
            <div className="space-y-1">
              <Bullet small>Polls every agent's heartbeat every 2s</Bullet>
              <Bullet small>Calls <Tag color={C.openai}>OpenAI</Tag> to suggest recovery hparams on failure</Bullet>
              <Bullet small>Logs every decision → <b>sentinel_log.json</b> → UI banners · <Tag color={C.weave}>Weave</Tag></Bullet>
            </div>
            <div className="space-y-1.5">
              {[
                { t: "NaN loss",    a: "OpenAI → kill → restart",      c: C.red     },
                { t: "Stale 10m",  a: "OpenAI → nudge via Redis",      c: "#D08040" },
                { t: "Stale 20m",  a: "OpenAI → kill → restart",       c: C.red     },
                { t: "2nd failure",a: "kill permanently",               c: C.textDim },
              ].map(({ t, a, c }) => (
                <div key={t} className="flex items-center gap-2 text-xs">
                  <span className="font-medium w-20 flex-shrink-0" style={{ color: c }}>{t}</span>
                  <span style={{ color: C.textDim }}>→ {a}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ── CONNECTION BAND: Training → Evaluator ── */}
        <ConnBand>
          <ConnRow color={C.amber} dir="down">
            <b>eval_result</b> &#123;mean_return, std_return, steps, status&#125; — from every agent on completion
          </ConnRow>
        </ConnBand>

        {/* ── EVALUATOR ── */}
        <AgentBox color={C.amber} badge="LLM Agent" title="Evaluator Agent" center>
          <Bullet>Calls <Tag color={C.openai}>OpenAI</Tag> to rank all N agents by mean_return + stability · restarted agents that finish count as wins · traced to <Tag color={C.weave}>Weave</Tag></Bullet>
        </AgentBox>

        <Flow color={C.amber} label="rankings + best agent" />

        {/* ── RESULTS ── */}
        <div className="rounded-xl overflow-hidden"
          style={{ border: `1.5px solid ${C.borderHi}`, background: C.surface }}>
          <div className="px-4 py-2" style={{ borderBottom: `1px solid ${C.border}`, background: C.surfaceB }}>
            <span className="text-xs font-semibold" style={{ color: C.text }}>Results &amp; Demo</span>
          </div>
          <div className="grid grid-cols-2 divide-x" style={{ borderColor: C.border }}>
            <div className="px-4 py-3 space-y-1.5">
              <p className="text-xs font-semibold mb-1" style={{ color: C.textDim }}>Inference</p>
              <Bullet>Best SB3 agent → <b>live rollout</b> rendered as MP4 in the UI</Bullet>
              <Bullet>Best GRPO agent → <b>before/after showcase</b> (base model vs &lt;think&gt; reasoning)</Bullet>
            </div>
            <div className="px-4 py-3 space-y-1.5">
              <p className="text-xs font-semibold mb-1" style={{ color: C.textDim }}>Storage</p>
              <Bullet><Tag color={C.wandb}>W&B</Tag> — all checkpoints + LoRA adapters · warm-start source</Bullet>
              <Bullet><Tag color="#E8601C">HuggingFace</Tag> — best model auto-pushed · usage snippet in UI</Bullet>
              <Bullet><Tag color={C.weave}>Weave</Tag> — all LLM traces + inference results</Bullet>
            </div>
          </div>
        </div>

        {/* ── OBSERVABILITY ── */}
        <div className="pt-8 pb-2">
          <div className="flex items-center gap-3 mb-4">
            <div className="h-px flex-1" style={{ background: C.border }} />
            <span className="text-xs" style={{ color: C.textFaint }}>external services &amp; observability</span>
            <div className="h-px flex-1" style={{ background: C.border }} />
          </div>
          <div className="grid grid-cols-3 gap-3">

            <ObsBox color={C.openai} icon="⬡" title="OpenAI API">
              <ObsRow dir="in"  label="Prompts from Orchestrator" sub="spawn planning" />
              <ObsRow dir="in"  label="Prompts from Sentinel"     sub="recovery hparams" />
              <ObsRow dir="in"  label="Prompts from Evaluator"    sub="agent ranking" />
              <ObsRow dir="out" label="Completions → all three agents" />
            </ObsBox>

            <ObsBox color={C.wandb} icon="W" title="Weights & Biases">
              <ObsRow dir="in"  label="Metrics from PPO/SAC/A2C"  sub="reward curves, loss" />
              <ObsRow dir="in"  label="Metrics from GRPO"         sub="reward, steps" />
              <ObsRow dir="in"  label="Model artifacts"           sub="model.zip, LoRA adapter" />
              <ObsRow dir="out" label="Warm-start checkpoint → Training Agents" />
            </ObsBox>

            <ObsBox color={C.weave} icon="◈" title="Weave (W&B)">
              <ObsRow dir="in" label="LLM traces from Orchestrator" sub="inputs + outputs" />
              <ObsRow dir="in" label="LLM traces from Sentinel"     sub="failure + recovery" />
              <ObsRow dir="in" label="LLM traces from Evaluator"    sub="ranking" />
              <ObsRow dir="in" label="Training steps from all agents" />
              <ObsRow dir="in" label="GRPO before/after responses" />
            </ObsBox>

          </div>
        </div>

        <div className="h-8" />
      </div>
    </div>
  );
}

// ── Primitives ─────────────────────────────────────────────────────────────

function AgentBox({ color, badge, title, children, center }: {
  color: string; badge: string; title: string;
  children: React.ReactNode; center?: boolean;
}) {
  return (
    <div className={`rounded-xl overflow-hidden ${center ? "mx-auto max-w-lg" : ""}`}
      style={{ border: `1.5px solid ${color}40`, background: C.surface }}>
      <div className="flex items-center gap-2 px-4 py-2.5"
        style={{ background: `${color}10`, borderBottom: `1px solid ${color}25` }}>
        <span className="text-xs px-2 py-0.5 rounded-full font-medium"
          style={{ background: `${color}20`, color, border: `1px solid ${color}35` }}>
          {badge}
        </span>
        <span className="text-sm font-semibold" style={{ color }}>{title}</span>
      </div>
      <div className="px-4 py-3 space-y-1.5">{children}</div>
    </div>
  );
}

function Bullet({ children, small }: { children: React.ReactNode; small?: boolean }) {
  return (
    <div className="flex items-start gap-2">
      <span className="mt-1 flex-shrink-0 w-1 h-1 rounded-full" style={{ background: C.borderHi }} />
      <p className={small ? "text-xs leading-relaxed" : "text-xs leading-relaxed"}
        style={{ color: C.textDim }}>{children}</p>
    </div>
  );
}

function Tag({ children, color }: { children: React.ReactNode; color: string }) {
  return (
    <span className="inline-flex items-center px-1.5 py-0 rounded font-medium text-xs"
      style={{ background: `${color}18`, color, border: `1px solid ${color}30` }}>
      {children}
    </span>
  );
}

function Flow({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex flex-col items-center py-1">
      <div className="w-px h-3" style={{ background: color, opacity: 0.4 }} />
      <span className="text-xs px-2.5 py-0.5 rounded-full my-1 font-medium"
        style={{ background: `${color}12`, color, border: `1px solid ${color}30` }}>
        ↓ {label}
      </span>
      <div className="w-px h-3" style={{ background: color, opacity: 0.4 }} />
    </div>
  );
}

function ForkArrow({ label, color, left }: { label: string; color: string; left?: boolean }) {
  return (
    <div className={`flex-1 flex flex-col ${left ? "items-end pr-1.5" : "items-start pl-1.5"}`}>
      <div className="w-px h-3" style={{ background: color, opacity: 0.4 }} />
      <span className="text-xs px-2 py-0.5 rounded-full my-1"
        style={{ background: `${color}12`, color, border: `1px solid ${color}25`, maxWidth: "90%",
          textAlign: left ? "right" : "left" }}>
        ↓ {label}
      </span>
    </div>
  );
}

function ConnBand({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-xl overflow-hidden my-2"
      style={{ border: `1px solid ${C.border}`, background: C.surfaceB }}>
      <div className="px-4 py-0.5">
        <p className="text-xs font-medium py-1" style={{ color: C.textFaint }}>
          communication channel
        </p>
      </div>
      <div style={{ borderTop: `1px solid ${C.border}` }}>
        {children}
      </div>
    </div>
  );
}

function ConnRow({ color, dir, children }: {
  color: string; dir: "up" | "down"; children: React.ReactNode;
}) {
  const arrow = dir === "up" ? "↑" : "↓";
  return (
    <div className="flex items-start gap-3 px-4 py-2.5"
      style={{ borderBottom: `1px solid ${C.border}` }}>
      <div className="flex items-center gap-1 flex-shrink-0 w-14 mt-0.5">
        <span className="font-bold text-sm" style={{ color }}>{arrow}</span>
        <span className="text-xs font-medium" style={{ color }}>
          {dir === "up" ? "to sentinel" : "to agents"}
        </span>
      </div>
      <p className="text-xs leading-relaxed" style={{ color: C.textDim }}>{children}</p>
    </div>
  );
}

function ObsBox({ color, icon, title, children }: {
  color: string; icon: string; title: string; children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl overflow-hidden"
      style={{ border: `1.5px solid ${color}35`, background: C.surface }}>
      <div className="flex items-center gap-2 px-3 py-2"
        style={{ background: `${color}10`, borderBottom: `1px solid ${color}20` }}>
        <span className="w-5 h-5 flex items-center justify-center rounded text-xs font-bold"
          style={{ background: `${color}25`, color }}>{icon}</span>
        <span className="text-xs font-semibold" style={{ color }}>{title}</span>
      </div>
      <div className="px-3 py-2.5 space-y-1.5">{children}</div>
    </div>
  );
}

function ObsRow({ dir, label, sub }: { dir: "in" | "out"; label: string; sub?: string }) {
  const color = dir === "in" ? C.textDim : C.green;
  return (
    <div className="flex items-start gap-1.5 text-xs">
      <span className="flex-shrink-0 mt-0.5 font-medium" style={{ color }}>
        {dir === "in" ? "←" : "→"}
      </span>
      <span style={{ color: C.textDim }}>
        {label}{sub && <span style={{ color: C.textFaint }}> · {sub}</span>}
      </span>
    </div>
  );
}
