# Auto-RL

Multi-agent reinforcement learning orchestration for **WeaveHacks**. Describe an RL task in natural language — a swarm of agents plans, trains, monitors, and ranks competing models, with full [Weave](https://wandb.ai) tracing and a live CopilotKit UI.

---

## What's built

| Component | Status | Entry point |
|-----------|--------|-------------|
| **Orchestrator** — LLM generates `spawn_plan.json` | ✅ | `orchestrator/orchestrator_agent.py` |
| **Swarm Runner** — asyncio launcher for all agents | ✅ | `orchestrator/swarm_runner.py` |
| **Training agents** — PPO / SAC / A2C (MuJoCo) | ✅ | `training/train_ppo.py` etc. |
| **GRPO / Countdown** — language-model reward training | ✅ | `training/train_grpo_countdown.py` |
| **Doom Loop Sentinel** — LLM-based watchdog | ✅ | `agents/sentinel.py` |
| **CopilotKit UI** — chat + live race dashboard | ✅ | `ui/` |
| **Evaluator** | 🔜 | `evaluator/` |

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.12 |
| Node.js | ≥ 20 |
| npm | ≥ 10 |

---

## Setup

### 1 — Python environment

```bash
cd autorl
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify MuJoCo:

```bash
python -c "import gymnasium; e=gymnasium.make('HalfCheetah-v5'); e.reset(); print('OK')"
```

### 2 — Environment variables

```bash
cp .env.template .env
# then fill in the values
```

| Key | Required | Used for |
|-----|----------|----------|
| `OPENAI_API_KEY` | ✅ | Orchestrator LLM, Sentinel LLM, CopilotKit chat |
| `WANDB_API_KEY` | ✅ | Weave / W&B tracing |
| `RUNPOD_API_KEY` | optional | Cloud GPU for GRPO |
| `WEAVE_PROJECT` | optional | Weave project name (default: `autorl`) |

`.env` is git-ignored — never commit your keys.

### 3 — UI (Node.js packages)

```bash
cd autorl/ui
npm install
```

---

## Running the full pipeline

### Option A — Full swarm via the UI (recommended)

**Terminal 1** — Python FastAPI backend:

```bash
cd autorl
bash ui/agent/start.sh
# → FastAPI running at http://localhost:8000
```

**Terminal 2** — Next.js frontend:

```bash
cd autorl/ui
npm run dev
# → UI running at http://localhost:3000
```

Open **http://localhost:3000** and type a task, e.g.:

> "Train the best MuJoCo locomotion policy"

The chat LLM will:
1. Generate a spawn plan and show an **Approval Card**
2. After you click **Approve & Launch**, start all training agents in parallel
3. Stream live **Agent Cards** (steps, reward, anomaly flags) every 5 s
4. Show **Sentinel Alerts** whenever the Doom Loop Sentinel intervenes
5. Display the **leaderboard and best checkpoint path** when training ends

---

### Option B — Orchestrator + swarm from the terminal

```bash
cd autorl
source .venv/bin/activate

# Step 1: Generate spawn plan
python orchestrator/orchestrator_agent.py "Train the best MuJoCo locomotion policy"

# Step 2: Run the swarm (reads runs/latest/spawn_plan.json automatically)
python orchestrator/swarm_runner.py
```

---

## Individual components

### Train a single agent

```bash
# PPO — quick smoke test (~2 min)
python training/train_ppo.py --agent-id test_ppo --env-id HalfCheetah-v5 --time-budget 120

# SAC — continuous-action envs only
python training/train_sac.py --agent-id test_sac --env-id HalfCheetah-v5 --time-budget 120

# A2C
python training/train_a2c.py --agent-id test_a2c --env-id HalfCheetah-v5 --time-budget 120
```

Outputs land in `runs/latest/{agent-id}/`:

| File | Contents |
|------|----------|
| `heartbeat.json` | Live status, updated every ~30 s |
| `eval_result.json` | Final metrics (mean/std return, steps, wall time) |
| `model.zip` | Saved checkpoint |

### Doom Loop Sentinel (standalone)

The Sentinel runs automatically inside the swarm. To inspect what it does:

```bash
# After a swarm run, check the intervention log:
cat runs/latest/sentinel_log.json
```

Each entry records: failure reason, original hparams, LLM-suggested hparams, and outcome.

### Render a trained model to video

```bash
python model_viewer/render_mujoco.py \
  --checkpoint runs/latest/agent_1/model.zip --algo PPO \
  --output runs/latest/best.mp4
```

### Weave tracing

Tracing is automatic when `WANDB_API_KEY` is set. Each run prints its trace URL. Disable for quick local tests:

```bash
WEAVE_DISABLED=1 python training/train_ppo.py --agent-id test --time-budget 30
```

---

## Project structure

```
autorl/
├── orchestrator/
│   ├── orchestrator_agent.py   # LLM → spawn_plan.json
│   ├── swarm_runner.py         # asyncio swarm launcher
│   └── device.py               # resolve cpu/mps/runpod
├── agents/
│   ├── sentinel.py             # LLM-based Doom Loop Sentinel
│   └── training_agent.py       # subprocess wrapper (kill/restart)
├── training/
│   ├── train_ppo.py
│   ├── train_sac.py
│   ├── train_a2c.py
│   └── train_grpo_countdown.py
├── ui/
│   ├── agent/
│   │   ├── middleware.py       # FastAPI backend (:8000)
│   │   └── start.sh            # convenience start script
│   ├── app/
│   │   ├── api/copilotkit/
│   │   │   └── route.ts        # CopilotKit runtime + server actions
│   │   ├── layout.tsx          # CopilotKit provider
│   │   └── page.tsx            # entrypoint (SSR-disabled)
│   ├── components/
│   │   ├── HomePage.tsx        # two-panel UI (chat + dashboard)
│   │   ├── ApprovalCard.tsx    # spawn plan approval
│   │   ├── AgentCard.tsx       # live per-agent status
│   │   ├── SentinelAlert.tsx   # LLM intervention cards
│   │   └── ResultsPanel.tsx    # leaderboard + best checkpoint
│   └── package.json
├── runs/                       # created at runtime
│   └── latest -> YYYY-MM-DD_HH-MM-SS/
├── .env.template
└── requirements.txt
```

---

## Data contracts

The JSON files produced by training (`heartbeat.json`, `eval_result.json`, `spawn_plan.json`, `sentinel_log.json`) are defined by Pydantic models in `orchestrator/orchestrator_agent.py`. See `docs/AutoRL_Project_Document.md` for the full schema.
