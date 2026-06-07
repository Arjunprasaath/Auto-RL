"""Phase 2.1 — task description -> spawn_plan.json (OpenAI Agents SDK + Weave)."""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from typing import Literal, Optional

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from dotenv import load_dotenv

from orchestrator.device import is_mps, resolve_grpo_device, resolve_sb3_device

load_dotenv(os.path.join(_PKG_ROOT, ".env"))

import weave
from agents import Agent, AgentOutputSchema, Runner
from pydantic import BaseModel, Field

RUNS_DIR = os.path.join(_PKG_ROOT, "runs")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini-2026-03-17")


class SpawnPlanEntry(BaseModel):
    id: str
    algo: str
    env: str
    exec: Literal["local", "runpod"]
    time_budget_min: int
    hparams: dict = Field(default_factory=dict)


class Heartbeat(BaseModel):
    agent_id: str
    timestamp: datetime
    status: Literal["starting", "training", "completed", "failed", "restarted"]
    steps_completed: int = 0
    current_reward: float = 0.0
    loss: Optional[float] = None
    anomaly: Optional[str] = None


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


class NudgeConfig(BaseModel):
    lr: float
    seed: int
    message: str = ""


def create_run_dir(run_id: str | None = None, base: str = RUNS_DIR) -> str:
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = os.path.join(base, run_id)
    os.makedirs(run_dir, exist_ok=True)
    try:
        latest = os.path.join(base, "latest")
        if os.path.islink(latest) or os.path.exists(latest):
            os.remove(latest)
        os.symlink(os.path.basename(run_dir), latest)
    except OSError:
        pass
    return run_dir


def _countdown_exec() -> str:
    return "local" if is_mps() else "runpod"


def _build_instructions() -> str:
    countdown_exec = _countdown_exec()
    device_note = (
        "Apple Silicon MPS is available — Countdown GRPO runs locally on MPS (exec=local)."
        if countdown_exec == "local"
        else "Countdown GRPO requires a cloud GPU (exec=runpod)."
    )
    return f"""You are the AutoRL Orchestrator. Read the user's task and return a spawn plan:
a list of training agents to run in parallel. Each agent must explore a DIFFERENT config —
never spawn N copies of the same algo/env/hparams. Your goal is to spawn agents with DIVERSE
configurations so the race is informative.

Compute backend: {device_note}

## Output format
Return JSON matching this schema exactly:
{{
  "entries": [
    {{
      "id": "agent_1",
      "algo": "PPO",
      "env": "HalfCheetah-v5",
      "exec": "local",
      "time_budget_min": 2,
      "hparams": {{ "lr": 0.0003, "gamma": 0.99, "n_steps": 2048, "ent_coef": 0.0, "seed": 42 }}
    }},
    {{
      "id": "agent_2",
      "algo": "GRPO",
      "env": "Countdown",
      "exec": "{countdown_exec}",
      "time_budget_min": 5,
      "hparams": {{
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "lr": 0.000001,
        "num_generations": 4,
        "temperature": 1.0,
        "seed": 123
      }}
    }}
  ]
}}

Required fields on every entry:
- id: string, unique, sequential — "agent_1", "agent_2", ...
- algo: one of "PPO", "SAC", "A2C", "GRPO"
- env: a valid Gymnasium environment id string (see families below)
- exec: exactly "local" or "runpod" (all Gymnasium/SB3 envs MUST be "local")
- time_budget_min: integer minutes (see below)
- hparams: object of numeric/string hyperparameters (never omit; use {{}} only if truly none apply)

## Environment families

### Gymnasium / Stable-Baselines3 environments (exec MUST be "local", time_budget_min: 2)

Use algo "PPO", "SAC", or "A2C". Pick the env that best matches the user's task.
You may use ANY valid Gymnasium environment id. Common ones grouped by type:

**MuJoCo continuous control** (all algos work):
- HalfCheetah-v5, Hopper-v5, Ant-v5, Walker2d-v5, Swimmer-v5
- Humanoid-v5, HumanoidStandup-v5, Reacher-v5, Pusher-v5
- InvertedPendulum-v5, InvertedDoublePendulum-v5

**Classic Control** (PPO and A2C work; SAC requires continuous action space):
- CartPole-v1 (discrete → PPO/A2C only)
- MountainCar-v0 (discrete → PPO/A2C only)
- MountainCarContinuous-v0 (continuous → all algos)
- Pendulum-v1 (continuous → all algos)
- Acrobot-v1 (discrete → PPO/A2C only)

**Box2D** (requires `pip install gymnasium[box2d]`):
- LunarLander-v3 (discrete → PPO/A2C only)
- LunarLanderContinuous-v3 (continuous → all algos)
- BipedalWalker-v3 (continuous → all algos)
- CarRacing-v3 (continuous, pixel obs → PPO/A2C only, use CnnPolicy)

**IMPORTANT action-space rules:**
- SAC ONLY supports continuous (Box) action spaces — never assign SAC to CartPole, MountainCar-v0, LunarLander-v3, Acrobot, or any other discrete env.
- PPO and A2C support both discrete and continuous action spaces.
- For pixel-observation envs (CarRacing), set hparams policy="CnnPolicy"; for all others use "MlpPolicy" (default).

**SB3 hparams (for any Gymnasium env):**
- Required: lr (float, default 3e-4), gamma (float, default 0.99), seed (int, unique per algo)
- Optional for PPO/A2C: n_steps (int rollout length, default 2048; try 512 or 4096), ent_coef (0.0–0.05)
- Optional for SAC: ent_coef (0.0–0.1)
- Do NOT add n_steps for SAC-only agents.

### Countdown arithmetic puzzle (exec MUST be "{countdown_exec}", time_budget_min: 5)
- env: exactly "Countdown"
- algo: exactly "GRPO"
- Task: use given numbers with +, -, *, / to reach a target number
- Required hparams:
  - model: always "Qwen/Qwen2.5-3B-Instruct"
  - lr: float (default 1e-6; vary slightly, e.g. 5e-7 vs 2e-6)
  - seed: int, MUST differ for every agent
- Optional hparams: num_generations (default 4; try 8), temperature (0.7–1.0)

## Planning rules
1. Spawn a maximum of 10 agents unless the user explicitly asks for a different count.
2. Match agents to the user task — infer the most appropriate Gymnasium env(s) from the description.
   If the user names a specific env (e.g. "CartPole", "Ant"), use it exactly.
3. Vary algo, env, lr, n_steps, and ent_coef across agents so the race is informative.
4. Every agent with the same algo MUST have a different seed.
5. Include EXACTLY ONE agent anywhere in the plan with hparams.lr = 1.0 (Sentinel fault-tolerance demo).
   All other agents must use sensible learning rates (never 1.0 except that one agent).
6. Never use SAC for discrete-action environments.
"""


class SpawnPlan(BaseModel):
    entries: list[SpawnPlanEntry]


_orchestrator = Agent(
    name="Orchestrator",
    instructions=_build_instructions(),
    model=OPENAI_MODEL,
    output_type=AgentOutputSchema(SpawnPlan, strict_json_schema=False),
)

_G = "Qwen/Qwen2.5-3B-Instruct"
_SB3_ALGOS = frozenset({"PPO", "SAC", "A2C"})
_DISCRETE_ONLY_ALGOS = frozenset({"PPO", "A2C"})  # SAC requires continuous action spaces
_COUNTDOWN_EXEC = _countdown_exec()
_DEFAULT_PLAN = [
    SpawnPlanEntry(id="agent_1", algo="PPO", env="HalfCheetah-v5", exec="local", time_budget_min=2,
                   hparams={"lr": 3e-4, "gamma": 0.99, "n_steps": 2048, "seed": 42}),
    SpawnPlanEntry(id="agent_2", algo="SAC", env="HalfCheetah-v5", exec="local", time_budget_min=2,
                   hparams={"lr": 1e-3, "gamma": 0.99, "seed": 7}),
    SpawnPlanEntry(id="agent_3", algo="A2C", env="Hopper-v5", exec="local", time_budget_min=2,
                   hparams={"lr": 7e-4, "gamma": 0.99, "n_steps": 512, "seed": 99}),
    SpawnPlanEntry(id="agent_4", algo="PPO", env="HalfCheetah-v5", exec="local", time_budget_min=2,
                   hparams={"lr": 1.0, "gamma": 0.99, "n_steps": 4096, "seed": 77}),
]


def _validate_plan(entries: list) -> list[SpawnPlanEntry]:
    """Parse each entry through SpawnPlanEntry and enforce orchestrator rules."""
    if not entries:
        raise ValueError("empty plan")
    plan = [SpawnPlanEntry.model_validate(e) for e in entries]
    if len({e.id for e in plan}) != len(plan):
        raise ValueError("duplicate agent ids")
    seeds: dict[str, set] = {}
    for e in plan:
        if e.env == "Countdown":
            # GRPO / LLM path
            if e.algo != "GRPO" or e.time_budget_min != 5:
                raise ValueError(f"{e.id}: Countdown needs algo=GRPO, time_budget_min=5")
            if e.exec not in ("local", "runpod"):
                raise ValueError(f"{e.id}: Countdown exec must be local or runpod")
            if e.exec != _COUNTDOWN_EXEC:
                raise ValueError(
                    f"{e.id}: Countdown needs exec={_COUNTDOWN_EXEC!r} on this machine "
                    f"(sb3={resolve_sb3_device()}, grpo={resolve_grpo_device()})"
                )
        else:
            # Any Gymnasium environment via Stable-Baselines3
            if e.algo not in _SB3_ALGOS:
                raise ValueError(f"{e.id}: Gymnasium envs need algo PPO/SAC/A2C, got {e.algo!r}")
            if e.exec != "local":
                raise ValueError(f"{e.id}: Gymnasium/SB3 envs must use exec=local")
            if e.time_budget_min != 2:
                raise ValueError(f"{e.id}: Gymnasium/SB3 envs need time_budget_min=2")
        seed = e.hparams.get("seed")
        if seed is None:
            raise ValueError(f"{e.id}: hparams.seed required")
        if seed in seeds.setdefault(e.algo, set()):
            raise ValueError(f"{e.id}: duplicate seed {seed} for algo {e.algo}")
        seeds[e.algo].add(seed)
    return plan


def _finalize(plan: list[SpawnPlanEntry], path: str) -> list[SpawnPlanEntry]:
    if not any(e.hparams.get("lr") == 1.0 for e in plan):
        plan[-1].hparams["lr"] = 1.0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump([e.model_dump() for e in plan], f, indent=2)
    return plan


@weave.op(name="Orchestrator")
async def create_spawn_plan(task: str, path: str) -> list[SpawnPlanEntry]:
    prompt = task
    for attempt in range(2):
        try:
            raw = (await Runner.run(_orchestrator, prompt)).final_output.entries
            plan = _validate_plan(raw)
            print(f"[orchestrator] {len(plan)} agents via {OPENAI_MODEL}")
            return _finalize(plan, path)
        except Exception as e:  # noqa: BLE001
            print(f"[orchestrator] attempt {attempt + 1} failed: {e}")
            prompt = f"{task}\n\nPrevious attempt failed: {e}. Fix the JSON."
    print("[orchestrator] using hard-coded default plan")
    return _finalize(_validate_plan(list(_DEFAULT_PLAN)), path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task", nargs="?", default="Train the best MuJoCo locomotion policy.")
    args = p.parse_args()
    if os.environ.get("WANDB_API_KEY") and not os.environ.get("WEAVE_DISABLED"):
        try:
            weave.init(os.environ.get("WEAVE_PROJECT", "autorl"))
        except Exception as e:  # noqa: BLE001
            print(f"[weave] init skipped ({e})")
    run_dir = create_run_dir()
    path = os.path.join(run_dir, "spawn_plan.json")
    plan = asyncio.run(create_spawn_plan(args.task, path))
    print(
        f"[orchestrator] sb3={resolve_sb3_device()} grpo={resolve_grpo_device()} "
        f"run dir: {run_dir}\n{json.dumps([e.model_dump() for e in plan], indent=2)}"
    )


if __name__ == "__main__":
    main()
