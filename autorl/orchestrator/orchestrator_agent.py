"""
Orchestrator agent (Phase 2.1) — built on the OpenAI Agents SDK.

Takes a natural-language task description, asks an LLM to decide which RL
experiments to race, and emits a validated `spawn_plan.json` (a list of
SpawnPlanEntry). On any failure it retries once, then falls back to a
hard-coded default plan so the pipeline always proceeds.

Run standalone:
    python orchestrator/orchestrator_agent.py "train a fast cheetah"
"""

import argparse
import asyncio
import json
import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(_PKG_ROOT, ".env"))

import weave
from agents import Agent, AgentOutputSchema, Runner
from pydantic import BaseModel

from orchestrator.run_context import create_run_dir, spawn_plan_path
from orchestrator.schemas import SpawnPlanEntry

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


class SpawnPlan(BaseModel):
    """Wrapper so the agent returns a top-level object (required for structured output)."""

    entries: list[SpawnPlanEntry]


INSTRUCTIONS = """You are the AutoRL Orchestrator. Given a user task description, \
decide which RL experiments to run. Your goal is to spawn agents with DIVERSE \
configurations so the race is informative — not N copies of the same run.

Available environments:
- MuJoCo (exec: local): HalfCheetah-v5, Hopper-v5
  Algorithms: PPO, SAC, A2C (Stable-Baselines3)
  Time budget: 10 minutes per agent (time_budget_min: 10)
- Countdown arithmetic puzzle (exec: runpod):
  Task: use given numbers with +,-,*,/ to reach a target number
  Algorithms: GRPO (Qwen2.5-3B-Instruct)
  Time budget: 20 minutes per agent (time_budget_min: 20)

For MuJoCo agents, set these in hparams:
- lr: learning rate (3e-4 default; vary across agents — try 1e-4 or 1e-3 for diversity)
- gamma: discount factor (0.99 for locomotion; 0.95 for shorter-horizon tasks)
- n_steps: PPO/A2C rollout length (2048 default; try 512 or 4096 for variety; ignored by SAC)
- ent_coef: entropy bonus (0.0–0.05; higher encourages more exploration)
- seed: random seed (must differ across same-algo agents)

For GRPO agents, set these in hparams:
- model: base LLM (always "Qwen/Qwen2.5-3B-Instruct")
- lr: learning rate (1e-6 default; vary slightly, e.g. 5e-7 vs 2e-6)
- num_generations: group size — completions sampled per prompt (4 default; try 8 for more signal)
- temperature: sampling temperature during rollouts (0.7–1.0)
- seed: must differ across agents

Rules:
- Produce entries matching the SpawnPlanEntry schema.
- id must be unique: "agent_1", "agent_2", ...
- Vary lr, n_steps, and ent_coef across same-algo agents so each agent explores a different region.
- Include exactly one agent with hparams.lr = 1.0 to test fault tolerance (Sentinel demo).
- Default to N=4 unless the user specifies otherwise.
"""


def default_plan() -> list[SpawnPlanEntry]:
    """Hard-coded N=4 fallback (2 local MuJoCo + 2 RunPod GRPO) with diverse hparams."""
    return [
        SpawnPlanEntry(id="agent_1", algo="PPO", env="HalfCheetah-v5", exec="local",
                       time_budget_min=10,
                       hparams={"lr": 3e-4, "gamma": 0.99, "n_steps": 2048,
                                "ent_coef": 0.0, "seed": 42}),
        SpawnPlanEntry(id="agent_2", algo="SAC", env="HalfCheetah-v5", exec="local",
                       time_budget_min=10,
                       hparams={"lr": 1e-3, "gamma": 0.99,
                                "ent_coef": 0.01, "seed": 7}),
        SpawnPlanEntry(id="agent_3", algo="GRPO", env="Countdown", exec="runpod",
                       time_budget_min=20,
                       hparams={"model": "Qwen/Qwen2.5-3B-Instruct", "lr": 1e-6,
                                "num_generations": 4, "temperature": 1.0, "seed": 42}),
        SpawnPlanEntry(id="agent_4", algo="GRPO", env="Countdown", exec="runpod",
                       time_budget_min=20,
                       hparams={"model": "Qwen/Qwen2.5-3B-Instruct", "lr": 1.0,
                                "num_generations": 8, "temperature": 0.8, "seed": 123}),
    ]


def _ensure_sentinel_demo(plan: list[SpawnPlanEntry]) -> list[SpawnPlanEntry]:
    """Guarantee exactly one agent has lr=1.0 so the Sentinel demo always fires."""
    if not any(e.hparams.get("lr") == 1.0 for e in plan):
        plan[-1].hparams["lr"] = 1.0
    return plan


def _write_plan(plan: list[SpawnPlanEntry], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump([e.model_dump() for e in plan], f, indent=2)


_orchestrator = Agent(
    name="Orchestrator",
    instructions=INSTRUCTIONS,
    model=OPENAI_MODEL,
    # hparams is a free-form dict -> incompatible with strict JSON schema.
    output_type=AgentOutputSchema(SpawnPlan, strict_json_schema=False),
)


@weave.op(name="Orchestrator")
async def create_spawn_plan(task: str, path: str) -> list[SpawnPlanEntry]:
    """Turn a task description into a validated spawn plan written to `path`."""
    prompt = task
    for attempt in range(2):
        try:
            result = await Runner.run(_orchestrator, prompt)
            plan = result.final_output.entries
            if not plan:
                raise ValueError("empty plan")
            plan = _ensure_sentinel_demo(plan)
            _write_plan(plan, path)
            print(f"[orchestrator] {len(plan)} agents planned via {OPENAI_MODEL}")
            return plan
        except Exception as e:  # noqa: BLE001
            print(f"[orchestrator] attempt {attempt + 1} failed: {e}")
            prompt = f"{task}\n\nPrevious attempt failed: {e}. Return a valid plan."

    plan = _ensure_sentinel_demo(default_plan())
    _write_plan(plan, path)
    print("[orchestrator] using hard-coded default plan")
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs="?", default="Train the best MuJoCo locomotion policy.")
    args = parser.parse_args()

    if os.environ.get("WANDB_API_KEY") and not os.environ.get("WEAVE_DISABLED"):
        try:
            weave.init(os.environ.get("WEAVE_PROJECT", "autorl"))
        except Exception as e:  # noqa: BLE001
            print(f"[weave] init skipped ({e})")

    run_dir = create_run_dir()
    plan = asyncio.run(create_spawn_plan(args.task, spawn_plan_path(run_dir)))
    print(f"[orchestrator] run dir: {run_dir}")
    print(json.dumps([e.model_dump() for e in plan], indent=2))


if __name__ == "__main__":
    main()
