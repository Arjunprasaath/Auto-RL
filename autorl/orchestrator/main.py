"""Phase 4.1 — AutoRL pipeline entry point.

Chains: create_spawn_plan → run_swarm → evaluate_results → render best model.

Usage (from autorl/ directory):
    python orchestrator/main.py
    python orchestrator/main.py "Race PPO vs SAC on HalfCheetah-v5"
    python orchestrator/main.py --task "Train a Hopper agent" --run-id my_run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(_PKG_ROOT, ".env"))

import weave

from orchestrator.device import resolve_grpo_device, resolve_sb3_device
from orchestrator.orchestrator_agent import (
    EvalResult,
    SpawnPlanEntry,
    create_run_dir,
    create_spawn_plan,
)
from orchestrator.swarm_runner import run_swarm
from evaluator.evaluator_agent import evaluate_results


# ── Helpers ───────────────────────────────────────────────────────────────────


def _render_best_mujoco(best_result: dict, run_dir: str) -> str | None:
    script_path = os.path.join(_PKG_ROOT, "model_viewer", "render_model.py")
    checkpoint_path = best_result.get("checkpoint_path")
    env_id = best_result.get("env")

    if not checkpoint_path or not env_id or not os.path.exists(checkpoint_path):
        print("[pipeline] best MuJoCo model missing checkpoint, skipping render")
        return None

    out_video = os.path.join(run_dir, "best_model.mp4")
    print(f"[pipeline] rendering {env_id} from {checkpoint_path}...")

    cmd = [
        sys.executable,
        script_path,
        "--checkpoint", checkpoint_path,
        "--env", env_id,
        "--output", out_video,
    ]
    try:
        subprocess.run(cmd, check=True)
        return out_video
    except subprocess.CalledProcessError as e:
        print(f"[pipeline] render failed: {e}")
        return None


def _render_best_countdown(best_result: dict, run_dir: str) -> str | None:
    """Render an example generation for the best Countdown (GRPO) model."""
    script_path = os.path.join(_PKG_ROOT, "model_viewer", "render_countdown.py")
    checkpoint_path = best_result.get("checkpoint_path")
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print("[pipeline] best Countdown model missing checkpoint, skipping render")
        return None

    out_json = os.path.join(run_dir, "best_countdown_example.json")
    print(f"[pipeline] rendering Countdown example from {checkpoint_path}...")

    cmd = [
        sys.executable,
        script_path,
        "--checkpoint", checkpoint_path,
        "--output", out_json,
    ]
    try:
        subprocess.run(cmd, check=True)
        return out_json
    except subprocess.CalledProcessError as e:
        print(f"[pipeline] render failed: {e}")
        return None


# ── Pipeline ──────────────────────────────────────────────────────────────────


async def pipeline(task: str, run_dir: str) -> dict:
    """Full AutoRL pipeline: orchestrate → train → evaluate → render."""

    # ── Step 1: generate spawn plan ──────────────────────────────────────────
    plan_path = os.path.join(run_dir, "spawn_plan.json")
    print(f"\n{'='*60}")
    print(f"[pipeline] step 1 — orchestrator")
    print(f"{'='*60}")
    plan: list[SpawnPlanEntry] = await create_spawn_plan(task, plan_path)
    print(f"[pipeline] {len(plan)} agents planned → {plan_path}")

    # ── Step 2: run swarm (training + sentinel) ───────────────────────────────
    print(f"\n{'='*60}")
    print(f"[pipeline] step 2 — swarm ({resolve_sb3_device()} / {resolve_grpo_device()})")
    print(f"{'='*60}")
    results: list[EvalResult] = await run_swarm(plan, run_dir)
    print(f"[pipeline] swarm done — {len(results)}/{len(plan)} results collected")

    # ── Write results to Redis history for future Orchestrator RAG ────────────
    try:
        from coordination.redis_coordinator import coordinator as _coord
        plan_by_id = {e.id: e for e in plan}
        for r in results:
            hp = plan_by_id[r.agent_id].hparams if r.agent_id in plan_by_id else {}
            _coord.record_run_result(r.algo, r.env, hp, r.mean_return, r.status)
        print(f"[pipeline] {len(results)} result(s) recorded to Redis history")
    except Exception as _e:  # noqa: BLE001
        print(f"[pipeline] Redis history write skipped ({_e})")

    # ── Step 3: evaluate and rank ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[pipeline] step 3 — evaluator")
    print(f"{'='*60}")
    rankings: dict = await evaluate_results(results, run_dir)

    # ── Step 4: render best models ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[pipeline] step 4 — model render")
    print(f"{'='*60}")
    
    mujoco_rankings = rankings.get("MuJoCo", [])
    countdown_rankings = rankings.get("Countdown", [])
    
    video_path: str | None = None
    if mujoco_rankings:
        best_entry = mujoco_rankings[0]
        best_agent_id = best_entry.get("agent_id")
        best_result = next(
            (r.model_dump() for r in results if r.agent_id == best_agent_id), None
        )
        if best_result:
            video_path = _render_best_mujoco(best_result, run_dir)
            
    example_path: str | None = None
    if countdown_rankings:
        best_entry = countdown_rankings[0]
        best_agent_id = best_entry.get("agent_id")
        best_result = next(
            (r.model_dump() for r in results if r.agent_id == best_agent_id), None
        )
        if best_result:
            example_path = _render_best_countdown(best_result, run_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "run_dir": run_dir,
        "task": task,
        "n_agents": len(plan),
        "n_results": len(results),
        "rankings": rankings,
        "video_path": video_path,
        "example_path": example_path,
    }

    summary_path = os.path.join(run_dir, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    _print_summary(rankings, video_path, example_path)
    return summary


def _print_summary(rankings: dict, video_path: str | None, example_path: str | None = None) -> None:
    print(f"\n{'='*60}")
    print("[pipeline] COMPLETE")
    print(f"{'='*60}")
    
    for group, entries in rankings.items():
        if entries:
            best = entries[0]
            print(f"Best {group} model: {best.get('algo')} ({best.get('agent_id')}) — return={best.get('mean_return')}")
            print(f"  Rationale: {best.get('rationale', '')[:100]}...")
            
    if video_path:
        print(f"\nMuJoCo Render saved to: {video_path}")
    if example_path:
        print(f"\nCountdown Example saved to: {example_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="AutoRL full pipeline")
    p.add_argument("task", nargs="?",
                   default="Train the best MuJoCo locomotion policy.",
                   help="Natural-language task description")
    p.add_argument("--run-id", default=None,
                   help="Optional run ID (default: timestamp)")
    args = p.parse_args()

    if os.environ.get("WANDB_API_KEY") and not os.environ.get("WEAVE_DISABLED"):
        try:
            weave.init(os.environ.get("WEAVE_PROJECT", "autorl"))
        except Exception as e:  # noqa: BLE001
            print(f"[weave] init skipped ({e})")

    from orchestrator.orchestrator_agent import RUNS_DIR
    run_dir = create_run_dir(args.run_id, base=RUNS_DIR)
    print(f"[pipeline] run dir: {run_dir}")

    result = asyncio.run(pipeline(args.task, run_dir))
    print(f"\n[pipeline] summary written → {result['run_dir']}/pipeline_summary.json")


if __name__ == "__main__":
    main()
