# Auto-RL

Multi-agent reinforcement learning orchestration for **WeaveHacks**. The end goal
is to describe an RL task in natural language and have a swarm of agents plan,
train, monitor, and rank competing models — with full
[Weave](https://wandb.ai) tracing.

## What works today

This repo is under active development. The functionality currently built and
verified is the **local MuJoCo training path**:

- **PPO training** on MuJoCo (`training/train_ppo.py`)
- **SAC training** on MuJoCo (`training/train_sac.py`)
- **MuJoCo video rendering** of a trained checkpoint (`model_viewer/render_mujoco.py`)
- **Weave/W&B tracing** of training runs
- **Data-contract schemas** (`orchestrator/schemas.py`, documented in `SCHEMA.md`)

Everything below describes how to use these pieces. The orchestrator, sentinel,
evaluator, RunPod/GRPO path, and UI are not built yet.

## Setup

Requires **Python ≥ 3.12**.

```bash
cd autorl
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify MuJoCo works:

```bash
python -c "import gymnasium; e=gymnasium.make('HalfCheetah-v5'); e.reset(); print('OK')"
```

### Required dependency not yet in the repo

`training/train_ppo.py` and `training/train_sac.py` import `HeartbeatWriter` from
`training/callbacks/heartbeat_writer.py`, which is not yet committed. Until that
file is present, the training scripts raise `ModuleNotFoundError` on import.
(`render_mujoco.py` has no such dependency and works standalone.)

## Environment variables

Tracing needs a W&B key. Copy the template and fill it in:

```bash
cp .env.template .env
```

| Key | Used for |
|-----|----------|
| `WANDB_API_KEY` | Weave/W&B tracing |
| `WEAVE_PROJECT` | Weave project name (default: `autorl`) |

`.env` is git-ignored — never commit your keys.

## Usage

All commands run from the `autorl/` package root.

### Train (PPO / SAC on MuJoCo)

```bash
# PPO — quick smoke test (~2 min)
python training/train_ppo.py --agent-id test_ppo --env-id HalfCheetah-v5 --time-budget 120

# SAC — continuous-action envs only; expect a replay warmup
python training/train_sac.py --agent-id test_sac --env-id HalfCheetah-v5 --time-budget 120

# Full budget (10 min)
python training/train_ppo.py --agent-id agent_1 --time-budget 600
```

CLI args (both scripts): `--agent-id` (required), `--env-id` (default
`HalfCheetah-v5`), `--time-budget` (seconds, default 600), `--lr` (default 3e-4),
`--seed` (default 42), `--results-dir` (default `./results`).

Outputs land in `results/{agent-id}/`:
- `heartbeat.json` — live status, updated every 60s
- `eval_result.json` — final metrics
- `model.zip` — saved checkpoint

### Weave tracing

Tracing is automatic when `WANDB_API_KEY` is set; each run prints its trace URL.
Disable it for quick local runs:

```bash
WEAVE_DISABLED=1 python training/train_ppo.py --agent-id test --time-budget 30
```

### Render a trained model to video

```bash
python model_viewer/render_mujoco.py \
  --checkpoint results/test_sac/model.zip --algo SAC \
  --output results/best_mujoco.mp4
```

`--algo` must match how the checkpoint was trained (PPO / SAC / A2C).

## Data contracts

The JSON files produced by training (`heartbeat.json`, `eval_result.json`) are
defined by Pydantic models in `orchestrator/schemas.py`. See `SCHEMA.md` for the
full contract.
