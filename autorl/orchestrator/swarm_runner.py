"""Phase 2.2 — asyncio swarm over spawn_plan entries."""

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

from orchestrator.device import resolve_sb3_device
from agents.sentinel import run_sentinel
from agents.training_agent import run_training_agent
from orchestrator.orchestrator_agent import EvalResult, RUNS_DIR, SpawnPlanEntry


def load_spawn_plan(run_dir: str) -> list[SpawnPlanEntry]:
    with open(os.path.join(run_dir, "spawn_plan.json")) as f:
        return [SpawnPlanEntry.model_validate(e) for e in json.load(f)]


def _collect_results(plan: list[SpawnPlanEntry], results_dir: str) -> list[EvalResult]:
    results = []
    for entry in plan:
        path = os.path.join(results_dir, entry.id, "eval_result.json")
        if os.path.exists(path):
            with open(path) as f:
                results.append(EvalResult.model_validate(json.load(f)))
    return results


@weave.op(name="SwarmRunner")
async def run_swarm(plan: list[SpawnPlanEntry], results_dir: str) -> list[EvalResult]:
    stop_event = asyncio.Event()
    sentinel_task = asyncio.create_task(
        run_sentinel([e.id for e in plan], results_dir, stop_event)
    )
    training_tasks = [
        asyncio.create_task(run_training_agent(entry, results_dir))
        for entry in plan
    ]

    max_budget = max(e.time_budget_min for e in plan) + 2  # 2 min grace
    await asyncio.wait(training_tasks, timeout=max_budget * 60)

    stop_event.set()
    await sentinel_task
    return _collect_results(plan, results_dir)


async def _main(run_dir: str) -> list[EvalResult]:
    plan = load_spawn_plan(run_dir)
    sb3 = resolve_sb3_device()
    print(f"[swarm] sb3={sb3} — launching {len(plan)} agents in {run_dir}")
    results = await run_swarm(plan, run_dir)
    print(f"[swarm] collected {len(results)}/{len(plan)} eval results")
    return results


def main():
    p = argparse.ArgumentParser(description="Run training swarm from spawn_plan.json")
    p.add_argument("--run-dir", default=os.path.join(RUNS_DIR, "latest"),
                   help="Run directory containing spawn_plan.json")
    args = p.parse_args()

    if os.environ.get("WANDB_API_KEY") and not os.environ.get("WEAVE_DISABLED"):
        try:
            weave.init(os.environ.get("WEAVE_PROJECT", "autorl"))
        except Exception as e:  # noqa: BLE001
            print(f"[weave] init skipped ({e})")

    results = asyncio.run(_main(args.run_dir))
    print(json.dumps([r.model_dump() for r in results], indent=2))


if __name__ == "__main__":
    main()
