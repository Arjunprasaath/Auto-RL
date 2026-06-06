# AutoRL Data Contracts

This document defines the JSON schemas used for communication between Person A (orchestration) and Person B (training).

## Overview

| File | Writer | Reader | Frequency |
|------|--------|--------|-----------|
| `spawn_plan.json` | Orchestrator (A) | Training Agents (B) | Once at start |
| `heartbeat.json` | Training Agents (B) | Sentinel (A) | Every 60s |
| `eval_result.json` | Training Agents (B) | Evaluator (A) | Once at completion |
| `nudge.json` | Sentinel (A) | Training Agents (B) | On intervention |

---

## spawn_plan.json

**Location:** `./spawn_plan.json`

Emitted by the Orchestrator after parsing the user's task. Each entry defines one training agent to spawn.

### Schema

```python
class SpawnPlanEntry(BaseModel):
    id: str                                    # "agent_1", "agent_2", etc.
    algo: str                                  # "PPO", "SAC", "A2C", "GRPO"
    env: str                                   # "HalfCheetah-v5", "Hopper-v5", "Countdown"
    exec: Literal["local", "runpod"]           # Execution target
    time_budget_min: int                       # 10 for MuJoCo, 20 for Countdown
    hparams: dict = {}                         # Algorithm-specific hyperparameters
```

### Example

```json
[
  {
    "id": "agent_1",
    "algo": "PPO",
    "env": "HalfCheetah-v5",
    "exec": "local",
    "time_budget_min": 10,
    "hparams": {"lr": 3e-4, "seed": 42}
  },
  {
    "id": "agent_2",
    "algo": "SAC",
    "env": "HalfCheetah-v5",
    "exec": "local",
    "time_budget_min": 10,
    "hparams": {"lr": 3e-4, "seed": 42}
  },
  {
    "id": "agent_3",
    "algo": "GRPO",
    "env": "Countdown",
    "exec": "runpod",
    "time_budget_min": 20,
    "hparams": {"model": "Qwen/Qwen2.5-3B-Instruct", "seed": 42, "lr": 1e-6}
  },
  {
    "id": "agent_4",
    "algo": "GRPO",
    "env": "Countdown",
    "exec": "runpod",
    "time_budget_min": 20,
    "hparams": {"model": "Qwen/Qwen2.5-3B-Instruct", "seed": 123, "lr": 1.0}
  }
]
```

> Note: `agent_4` has `lr=1.0` deliberately to guarantee NaN divergence for Sentinel demo.

---

## heartbeat.json

**Location:** `./results/{agent_id}/heartbeat.json`

Written by each Training Agent every 60 seconds. Read by the Doom Loop Sentinel to detect stuck agents.

### Schema

```python
class Heartbeat(BaseModel):
    agent_id: str
    timestamp: datetime                        # ISO 8601 format
    status: Literal["starting", "training", "completed", "failed", "restarted"]
    steps_completed: int = 0
    current_reward: float = 0.0
    loss: Optional[float] = None
    anomaly: Optional[str] = None              # "nan_loss", "plateau", or null
```

### Example

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

### Anomaly Values

| Value | Meaning | Sentinel Response |
|-------|---------|-------------------|
| `null` | Normal operation | Continue monitoring |
| `"nan_loss"` | Loss became NaN | Immediate kill + restart |
| `"plateau"` | Reward hasn't improved for 5+ updates | Nudge with new seed |

---

## eval_result.json

**Location:** `./results/{agent_id}/eval_result.json`

Written by each Training Agent upon completion (or timeout/failure). Read by the Evaluator to rank results.

### Schema

```python
class EvalResult(BaseModel):
    agent_id: str
    algo: str
    env: str
    status: Literal["completed", "failed", "timed_out", "restarted"]
    mean_return: float = 0.0
    std_return: float = 0.0
    steps_trained: int = 0
    wall_time_s: float = 0.0
    weave_run_id: str = ""
    checkpoint_path: str = ""
```

### Example

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

## nudge.json

**Location:** `./results/{agent_id}/nudge.json`

Written by the Sentinel when an agent appears stuck but hasn't exceeded the kill threshold. The Training Agent should check for this file every 60s and apply new hyperparameters if found.

### Schema

```python
class NudgeConfig(BaseModel):
    lr: float                                  # New learning rate (typically current / 2)
    seed: int                                  # New random seed
    message: str = ""                          # Optional explanation
```

### Example

```json
{
  "lr": 1.5e-4,
  "seed": 999,
  "message": "Heartbeat stale for 2+ minutes. Adjusting hyperparameters."
}
```

### Training Agent Behavior

1. Check for `nudge.json` every 60 seconds during training
2. If found:
   - Apply new `lr` and `seed`
   - Delete the `nudge.json` file
   - Continue training with new parameters
3. Update next heartbeat with new status

---

## Directory Structure

Each race gets its own timestamped **run directory** under `runs/`, which
co-locates the spawn plan and all per-agent outputs so re-running never
overwrites a previous run. The Orchestrator mints the run dir and threads it to
training scripts via `--results-dir`.

```
autorl/
└── runs/
    ├── latest -> 2026-06-06T13-41-02   # symlink to most recent run
    └── 2026-06-06T13-41-02/
        ├── spawn_plan.json             # Written by Orchestrator
        ├── rankings.json               # Written by Evaluator
        ├── run_report.md               # Written by Reporter
        ├── agent_1/
        │   ├── heartbeat.json          # Updated every 60s
        │   ├── eval_result.json        # Written on completion
        │   ├── nudge.json              # Written by Sentinel (if needed)
        │   └── model.zip               # Model checkpoint
        ├── agent_2/
        │   └── ...
        └── ...
```

> Note: training scripts take `--results-dir`, so paths like
> `heartbeat.json` / `eval_result.json` resolve to `<run_dir>/<agent_id>/...`.
> Their relative layout within an agent folder is unchanged.
