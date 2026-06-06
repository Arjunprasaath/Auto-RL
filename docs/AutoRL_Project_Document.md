# AutoRL
## Multi-Agent Reinforcement Learning Orchestration
**WeaveHacks — Weights & Biases Multi-Agent Hackathon**

| Duration | Team | Local | Cloud |
|---|---|---|---|
| 1.5 days (36 hours) | 2 engineers | Mac M3 Pro — MuJoCo | RunPod GPU — LLM RL |

---

## 1. Executive Summary

AutoRL is a multi-agent system where a user describes an RL task in natural language through a CopilotKit UI, and an Orchestrator agent autonomously decides the environment, selects competing algorithms, spawns N independent Training Agents that each start, monitor, and complete a training run, then hands results to an Evaluator agent that picks the best-performing model — which is shown **in action live in the UI** (MuJoCo video render or Countdown live solve).

A dedicated **Doom Loop Sentinel** agent monitors all spawned agents in real time. On detecting a stuck agent, it first nudges it with a corrective prompt. If the agent remains stuck, the Sentinel kills and restarts it with a modified configuration. Every agent is traced end-to-end by W&B Weave.

> **Why it fits WeaveHacks:** Genuinely autonomous agents that make decisions, monitor progress, and adapt · Doom Loop Sentinel with escalating intervention · live training race visible in CopilotKit UI · Weave traces show the full multi-agent graph

---

## 1.1 What Makes This Multi-Agent (Not a Pipeline)

A pipeline is a fixed sequence of function calls. AutoRL is multi-agent because each agent makes autonomous decisions during its lifecycle:

- **Orchestrator:** Interprets the user task, decides environment family, selects which algorithms to race, determines N, chooses execution targets — none of this is hard-coded. Different prompts produce genuinely different spawn plans.
- **Training Agents:** Each agent starts its training job, monitors metrics during the run (checking for divergence, NaN losses, plateau detection), and writes heartbeats so the Sentinel can verify liveness. They are not fire-and-forget scripts.
- **Doom Loop Sentinel:** A persistent supervisor spawned at the start. It watches all Training Agent heartbeats and Weave traces. On detecting a stuck agent it escalates: nudge → kill + restart → kill only. It makes independent intervention decisions.
- **Evaluator:** Reads all results, reasons about which metrics matter for this task, and produces a justified ranking — not a fixed formula.

---

## 1.2 End-to-End Flow

1. User types a task in CopilotKit chat: *"train an agent to run fast in a physics sim, and an LLM that learns to solve arithmetic puzzles"*
2. Orchestrator decides: MuJoCo HalfCheetah for locomotion (local Mac), Countdown puzzle for LLM reasoning (RunPod). Selects PPO + SAC for MuJoCo, GRPO with different seeds for Countdown. Emits `spawn_plan.json`.
3. CopilotKit shows an approval card: *"Spawning 4 agents: 2 local MuJoCo, 2 RunPod GRPO. Approve?"* User confirms.
4. Doom Loop Sentinel is spawned and begins watching.
5. N Training Agents start in parallel — each connects to its execution target, launches training, monitors progress, streams metrics to Weave.
6. Fixed time budget expires (10 min MuJoCo, 20 min Countdown GRPO). Agents write `eval_result.json`.
7. Orchestrator spawns Evaluator. Evaluator compares results within each environment family, picks winners.
8. Best MuJoCo model renders a video. Best Countdown model solves 5 test puzzles live in the UI — both shown in CopilotKit.

---

## 1.3 Scope

| Item | In Scope |
|---|---|
| Environments | HalfCheetah-v5, Hopper-v5 (local Mac); Countdown puzzle (RunPod) |
| MuJoCo algorithms | PPO, SAC, A2C via Stable-Baselines3 |
| LLM RL algorithm | GRPO (cold-start, no SFT) with different seeds on Qwen2.5-3B-Instruct |
| N agents | Default N=4: 2 MuJoCo local + 2 Countdown RunPod |
| LLM RL story | Base 3B model fails at hard Countdown variants → GRPO trains to 60-70% in 20 min |
| Time budget | Fixed: 10 min MuJoCo, 20 min Countdown GRPO |
| UI | CopilotKit: chat input, approval gate, live race dashboard, model-in-action viewer |
| Observability | W&B Weave: @weave.op traces, online evals, Evaluation leaderboard |
| Final output | Best model shown in action in UI + Weave leaderboard + run_report.md |
| Out of scope | Maze navigation, SFT warmup, TD3, continuous online learning |

---

## 2. Why Countdown, Not Maze

Maze navigation required a two-stage pipeline: SFT first (2-4 hours on 500k mazes), then GRPO. That consumed most of the 36-hour budget before a single RL step ran.

**Countdown is the right environment because:**

- The task: given numbers like `[4, 7, 2, 9]`, use `+, -, ×, ÷` to reach a target like `24`
- **No SFT required.** Qwen2.5-3B-Instruct already knows arithmetic format; it just lacks multi-step planning
- **Verifiable reward = 4 lines of code:** parse the model's output expression, evaluate it, check if it equals the target
- **Dataset on HuggingFace:** `zouxuhong/Countdown-Tasks-3to4` — one line to load
- **Fast convergence:** 20-30 min on RTX 4090 with LoRA, documented results show 3B cold-start GRPO goes from ~51% → 67%
- **Emergent reasoning is visible:** the model's responses get longer and show step-by-step thinking as GRPO trains — you can watch this live in Weave
- **Demo moment:** show base model failing `[4, 7, 2, 9] → 24`, then trained model solving it with explicit chain-of-thought

---

## 3. Architecture

### 3.1 System Overview

Five agent roles: Orchestrator (1), Training Agent (N), Doom Loop Sentinel (1), Evaluator (1), and a CopilotKit UI layer. Agents communicate through files (`spawn_plan.json`, `eval_result.json`, `heartbeat.json`) and Weave traces — never directly. The Orchestrator is the sole agent that spawns and despawns other agents.

### 3.2 Agent Catalogue

#### Orchestrator
The central decision-maker. Takes user prompt from CopilotKit, decides environment family (MuJoCo or Countdown), selects algorithms, assigns execution targets, emits `spawn_plan.json`. After training, spawns Evaluator and triggers model-in-action viewer.

| | |
|---|---|
| **In** | User prompt via CopilotKit chat |
| **Out** | `spawn_plan.json` · spawns Training Agents, Sentinel, Evaluator |
| **Exec** | `@weave.op` · persistent through entire run · manages agent lifecycle |

#### Training Agent ×N
Responsible for one complete training lifecycle: (1) connect to execution target, (2) start training with assigned algo/env/hparams, (3) monitor metrics every 30s — check for NaN losses, reward plateau, divergence, (4) write `heartbeat.json` every 60s, (5) on completion write `eval_result.json`.

| | |
|---|---|
| **In** | One entry from `spawn_plan.json` |
| **Out** | `eval_result.json` · `heartbeat.json` (every 60s) · checkpoint files · Weave online eval |
| **Exec** | `@weave.op` · local asyncio subprocess (MuJoCo) or RunPod SSH (Countdown) · fixed time budget |

#### Doom Loop Sentinel
Persistent supervisor spawned at pipeline start. Monitors all Training Agents via heartbeat files and Weave traces. Escalation: nudge → kill + restart → kill only.

| | |
|---|---|
| **In** | All `heartbeat.json` files · Weave trace stream |
| **Out** | Corrective nudge · kill signal · restart spawn request to Orchestrator |
| **Exec** | `@weave.op` · persistent · reads heartbeats every 30s |

#### Evaluator
Spawned after all Training Agents complete or timeout. Groups results by environment family. Uses an LLM call to reason about why the winner won. Pushes to Weave Evaluation with custom scorers.

| | |
|---|---|
| **In** | N × `eval_result.json` · Weave trace summaries |
| **Out** | `rankings.json` · Weave Evaluation leaderboard · written rationale per env family |
| **Exec** | `@weave.op` · Weave Evaluation API · LLM call for rationale |

#### CopilotKit UI Layer
Human-agent interface. Chat input, approval card before spawning, live race dashboard (`useCoAgentStateRender`), Sentinel alerts, model-in-action viewer (MuJoCo video + Countdown live solve).

| | |
|---|---|
| **In** | User input · Orchestrator state updates · Sentinel alerts |
| **Out** | User approval · task prompt to Orchestrator |
| **Exec** | Next.js + CopilotKit React SDK · AG-UI protocol · `useCoAgent`, `useCopilotAction` hooks |

---

### 3.3 Data Contracts

#### `spawn_plan.json` — Orchestrator → Training Agents

```json
[
  {"id":"agent_1","algo":"PPO","env":"HalfCheetah-v5","exec":"local","time_budget_min":10,"hparams":{"lr":3e-4}},
  {"id":"agent_2","algo":"SAC","env":"HalfCheetah-v5","exec":"local","time_budget_min":10,"hparams":{"lr":3e-4}},
  {"id":"agent_3","algo":"GRPO","env":"Countdown","exec":"runpod","time_budget_min":20,"hparams":{"model":"Qwen/Qwen2.5-3B-Instruct","seed":42}},
  {"id":"agent_4","algo":"GRPO","env":"Countdown","exec":"runpod","time_budget_min":20,"hparams":{"model":"Qwen/Qwen2.5-3B-Instruct","seed":123,"lr":1.0}}
]
```

> `agent_4` has `lr=1.0` deliberately — this guarantees divergence to trigger the Sentinel for the demo.

#### `heartbeat.json` — Training Agent → Sentinel (every 60s)

```json
{
  "agent_id": "agent_1",
  "timestamp": "2026-06-07T14:32:00Z",
  "status": "training",
  "steps_completed": 12000,
  "current_reward": 2341.5,
  "loss": 0.043,
  "anomaly": null
}
```

#### `eval_result.json` — Training Agent → Evaluator

```json
{
  "agent_id": "agent_3",
  "algo": "GRPO",
  "env": "Countdown",
  "status": "completed",
  "mean_return": 0.67,
  "std_return": 0.21,
  "steps_trained": 3200,
  "wall_time_s": 1180,
  "weave_run_id": "abc123",
  "checkpoint_path": "/workspace/checkpoints/agent_3/"
}
```

---

### 3.4 Doom Loop Sentinel — Escalation Protocol

| Condition | Action | Effect |
|---|---|---|
| Heartbeat stale > 2 min | **NUDGE:** write `nudge.json` with new hparams (halved lr, new seed) | Agent retries with modified params; Sentinel logs to Weave |
| Still stale > 4 min after nudge | **KILL + RESTART:** terminate process, spawn replacement with modified config | New agent starts fresh; old agent marked `status=restarted` |
| Replacement also stalls > 4 min | **KILL ONLY:** terminate permanently | Marked `status=failed`; Evaluator works with fewer results |
| `anomaly: "nan_loss"` in heartbeat | **Immediate KILL + RESTART** (skip nudge) | NaN is unrecoverable; restart with `lr=3e-4` |

> **Why one Sentinel, not per-agent guards:** A single Sentinel has global view — it can detect 3 of 4 agents stuck (RunPod might be down) and make a systemic decision rather than each guard acting independently.

---

### 3.5 Execution Environments

| Property | MuJoCo / Gymnasium (local Mac M3) | Countdown GRPO (RunPod GPU) |
|---|---|---|
| Task | Physics locomotion | Arithmetic puzzle: reach target using given numbers |
| Algorithms | PPO, SAC, A2C (Stable-Baselines3) | GRPO cold-start, no SFT (trl library) |
| Model | MLP policy (SB3 default) | Qwen2.5-3B-Instruct + LoRA |
| Dataset | Gymnasium environments | `zouxuhong/Countdown-Tasks-3to4` (HuggingFace) |
| Time budget | 10 min per agent | 20 min per agent |
| Reward | Dense (physics return) | Binary: expression evaluates to target → 1.0, else 0.0 |
| Baseline | Random policy: ~0 return | Base 3B model: ~51% on 3-arg, ~2% on 5-arg Countdown |
| Expected result | PPO ~3k, SAC ~5k return on HalfCheetah | GRPO: ~67% on 3-arg Countdown after training |
| Best model in action | Rendered video clip of trained agent | Live solve of 5 test puzzles in CopilotKit UI |
| Weave logging | Every 1k steps via SB3 callback | Every GRPO step via trl callback |

---

### 3.6 Weave Integration

| Feature | How AutoRL uses it |
|---|---|
| `@weave.op` | Every agent method is a named trace node visible in Weave UI |
| Online evals | Training Agents push reward/accuracy every 30s; visible as live charts |
| `weave.Evaluation` | Evaluator pushes rankings as scored Evaluation with per-env leaderboard |
| Trace tree | Judges see: Orchestrator → [N parallel Training Agents + Sentinel] → Evaluator |
| Sentinel interventions | Every nudge/kill/restart is a Weave trace event |

### 3.7 CopilotKit Integration

| Feature | How AutoRL uses it |
|---|---|
| `CopilotChat` | User types task description; Orchestrator responds with its plan |
| `useCopilotAction` (approval) | Before spawning: card shows N agents, envs, estimated cost |
| `useCoAgentStateRender` | Live race dashboard: agent cards with status, reward, time remaining |
| `useCopilotAction` (sentinel) | Alert card when Sentinel intervenes: "Agent 4 stuck. Restarting with lr=3e-4." |
| Generative UI (results) | MuJoCo: video player. Countdown: animated step-by-step puzzle solver. |

---

## 4. Project Plan (36 Hours)

### 4.1 Success Criteria

- User types a task in CopilotKit and sees N agent cards appear on the race dashboard, updating in real time
- At least 2 MuJoCo agents and 1 Countdown agent produce `eval_result.json` files on the Weave leaderboard
- The best MuJoCo model plays a rendered video in the UI; the best Countdown model solves test puzzles live
- The Doom Loop Sentinel demonstrates at least one intervention visible in Weave traces

### 4.2 Milestones

| Phase | Time | Person A | Person B |
|---|---|---|---|
| Scaffold | Hours 0–4 | Hour 0: schema lock. CopilotKit app scaffolded. Orchestrator shell. Weave init. Pydantic validation. | Hour 0: schema lock. SB3 PPO + SAC scripts. `eval_result.json` + `heartbeat.json` writers. MuJoCo verified on M3. |
| Core agents | Hours 4–8 | Orchestrator: prompt → env decision → algo selection → `spawn_plan.json`. Approval card wired. Sentinel skeleton. | Weave SB3 callback (every 1k steps). RunPod pod pre-warmed. Countdown dataset downloaded. GRPO training script skeleton. |
| Integration | Hours 8–14 | Sentinel escalation logic. `@weave.op` on all agents. asyncio swarm runner. CopilotKit race dashboard. | GRPO training script complete with heartbeat writer. Base model baseline recorded (~51% on 3-arg Countdown). |
| Eval pipeline | Hours 14–20 | **N=2 local-only MUST be green by Hour 16.** Evaluator agent with LLM rationale. Reporter. | GRPO verified end-to-end on RunPod. Integration: all 4 agents producing `eval_result.json`. |
| Polish | Hours 20–28 | Full N=4 race. CopilotKit model-in-action viewer. Sentinel demo triggered deliberately. | MuJoCo video render script. Countdown live solve output. Baseline numbers validated. |
| Demo prep | Hours 28–36 | Weave public. README. Demo video recorded by Hour 32. Demo dry run at Hour 30 (both). | Before/after Countdown screenshots. Contingency N=2 fallback ready. Pod alive until submission. |

### 4.3 Critical Path

The critical path is: **RunPod pod ready (Hour 2) → GRPO script tested (Hour 12) → N=4 integration (Hour 18)**. If GRPO is not verified by Hour 12, fall back to N=2 local-only at Hour 16.

> **Hard gate — Hour 16:** N=2 local-only (PPO + SAC on HalfCheetah) MUST work end-to-end through the full pipeline: Orchestrator → Training Agents → Sentinel → Evaluator → CopilotKit UI. This is the minimum viable demo.

### 4.4 Risk Register

| Risk | Probability | Mitigation |
|---|---|---|
| RunPod pod startup > 10 min | Medium | Pre-warm at Hour 0; keep alive until submission. Budget: ~$9 for 20h on RTX 4090 ($0.44/hr) |
| GRPO does not converge in 20 min | Low | Even partial improvement (51% → 55%) is measurable. Use `seed=42` (documented to work) as primary |
| 3B model OOM on RTX 4090 | Low | Use 4-bit quantization + LoRA (r=16). 3B with LoRA fits in ~10GB VRAM |
| CopilotKit AG-UI integration issues | Medium | Fall back to terminal + Weave URL only. CopilotKit is additive, not critical path |
| MJX-JAX broken on M3 | Low | Standard MuJoCo CPU. Slower but functional in 10-min budget |
| Sentinel never fires | High | `agent_4` has `lr=1.0` deliberately — guaranteed NaN within 30s |
| Demo network failure | Low | Pre-record 2-min video by Hour 32 |
| API costs exceed budget | Low | Use claude-haiku for planning. Reserve sonnet for Evaluator. Budget $15 total |

---

## 5. Two-Person Workload Split

### 5.1 Division Philosophy

Person A owns the agent brains, the UI, and the supervisor. Person B owns the training scripts, compute targets, and model-in-action outputs. The interface boundary is `spawn_plan.json` (A writes, B reads), `eval_result.json` (B writes, A reads), and `heartbeat.json` (B writes, A reads via Sentinel).

### 5.2 Shared Checkpoints

| Checkpoint | When | What to verify |
|---|---|---|
| Schema lock | Hour 0 | `spawn_plan.json`, `eval_result.json`, `heartbeat.json` schemas written to SCHEMA.md |
| Local smoke test | Hour 5 | Person B: PPO trains 1k steps, both JSON files written correctly. Person A: Sentinel reads heartbeat. |
| Weave trace review | Hour 10 | Both open Weave. Orchestrator + TrainingAgent nodes visible. Online eval chart streaming. |
| N=2 local green | **Hour 16** | PPO + SAC → Evaluator → rankings → CopilotKit shows results. **MINIMUM VIABLE DEMO.** |
| N=4 full race | Hour 22 | Add 2 RunPod GRPO agents. Leaderboard shows 4 rows. Sentinel demo triggered. |
| Demo dry run | Hour 30 | Full cold-start demo. Time it. Record video immediately after. |

### 5.3 Git Workflow

- Single repo, two branches: `person-a` and `person-b` off `main`
- Merge to `main` only at shared checkpoints
- Person A owns: `orchestrator/`, `agents/sentinel.py`, `evaluator/`, `ui/`
- Person B owns: `training/`, `environments/`, `runpod/`, `model_viewer/`
- Shared: `SCHEMA.md`, `pyproject.toml`, `.env.template`, `README.md`

---

## 6. Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | Python 3.12 · asyncio for parallel agent swarm |
| LLM calls | Claude Haiku for planning · Claude Sonnet for Evaluator rationale |
| RL framework (MuJoCo) | Stable-Baselines3 · Gymnasium · MuJoCo 3.x · MJX-JAX |
| RL framework (LLM) | trl `GRPOTrainer` · Qwen2.5-3B-Instruct + LoRA · **no SFT** |
| LLM RL environment | Countdown puzzle · `zouxuhong/Countdown-Tasks-3to4` (HuggingFace) |
| Reward function | Binary: `eval(expression) == target` |
| Cloud compute | RunPod Python SDK · RTX 4090 ($0.44/hr) |
| Observability | W&B Weave (`@weave.op`, `weave.Evaluation`, online evals) |
| UI | CopilotKit + Next.js · AG-UI protocol |
| Agent supervision | Doom Loop Sentinel · heartbeat monitoring · escalation ladder |
| Video render | SB3 `render_mode=rgb_array` · imageio → mp4 |
| Validation | Pydantic models for all JSON contracts |

---

## 7. Repository Structure

```
autorl/
├── SCHEMA.md                     ← data contracts
├── pyproject.toml
├── .env.template                 ← WANDB_API_KEY, RUNPOD_API_KEY, ANTHROPIC_API_KEY
├── orchestrator/
│   ├── main.py                   ← entry point; asyncio event loop
│   ├── orchestrator_agent.py     ← prompt → env decision → algo selection → spawn
│   ├── swarm_runner.py           ← asyncio.gather() over N agents + sentinel
│   └── schemas.py                ← Pydantic models for all JSON contracts
├── agents/
│   ├── training_agent.py         ← start, monitor, heartbeat, complete lifecycle
│   └── sentinel.py               ← doom loop detection + escalation ladder
├── evaluator/
│   ├── evaluator_agent.py        ← rank within env family, LLM rationale
│   └── reporter.py               ← run_report.md generation
├── training/                     ← PERSON B
│   ├── train_ppo.py
│   ├── train_sac.py
│   ├── train_a2c.py
│   ├── train_grpo_countdown.py   ← GRPO cold-start, no SFT
│   └── callbacks/
│       ├── weave_callback.py     ← SB3 → Weave online eval
│       └── heartbeat_writer.py   ← background thread, writes every 60s
├── environments/                 ← PERSON B
│   └── countdown_env.py          ← puzzle generation + reward function
├── runpod/                       ← PERSON B
│   ├── pod_manager.py
│   └── teardown.py
├── model_viewer/                 ← PERSON B
│   ├── render_mujoco.py          ← SB3 checkpoint → mp4
│   └── countdown_inference.py   ← GRPO checkpoint → live puzzle solve JSON
└── ui/                           ← PERSON A
    ├── app/
    │   ├── page.tsx
    │   └── api/copilotkit/route.ts
    ├── components/
    │   ├── RaceDashboard.tsx
    │   ├── ApprovalCard.tsx
    │   ├── SentinelAlert.tsx
    │   └── ModelViewer.tsx
    └── agent/
        └── middleware.py          ← CopilotKitMiddleware wrapping Orchestrator
```

---

## 8. Demo Script (2.5 minutes)

Practice until it runs exactly 2.5 min cold. Record video by Hour 32.

| Time | Action |
|---|---|
| 0:00 – 0:20 | Open CopilotKit UI. Type: *"Train a fast runner in a physics sim and an LLM that learns to solve arithmetic puzzles."* Orchestrator responds with its plan in chat. |
| 0:20 – 0:35 | Approval card: *"Spawning 4 agents: 2 local MuJoCo (PPO, SAC), 2 RunPod Countdown (GRPO seed=42, GRPO seed=123 lr=1.0). Est. cost: $0.15. Approve?"* Click Approve. |
| 0:35 – 1:10 | Race dashboard: 4 agent cards. SAC reward climbing faster than PPO. GRPO seed=42 accuracy ticking up from 51%. GRPO seed=123 (lr=1.0) already showing NaN in heartbeat. |
| 1:10 – 1:30 | **Sentinel fires:** alert card — *"Agent 4 (GRPO lr=1.0): NaN loss detected. Killing and restarting with lr=3e-4."* Show Weave trace node for the intervention. |
| 1:30 – 1:50 | Training ends. Evaluator runs. Leaderboard: SAC wins on HalfCheetah, GRPO seed=42 wins on Countdown. Show LLM-generated rationale. |
| 1:50 – 2:10 | **MONEY SHOT:** MuJoCo video plays — trained HalfCheetah running. Switch to Countdown solver — model given `[4, 7, 2, 9] → 24` and works through step-by-step chain of thought. |
| 2:10 – 2:30 | Open Weave trace tree. Point out the multi-agent graph: Orchestrator → [4 Training Agents + Sentinel] → Evaluator. Show Sentinel intervention node. *"Change the prompt, get a different race."* |
