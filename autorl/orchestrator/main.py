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


def _render_best_mujoco(best: dict, run_dir: str) -> str | None:
    """Render an MP4 for the best MuJoCo agent; return the output path or None."""
    ckpt = best.get("checkpoint_path", "")
    algo = best.get("algo", "")
    env  = best.get("env", "")

    if not ckpt or not os.path.exists(ckpt) or algo not in ("PPO", "SAC", "A2C"):
        print(f"[main] skipping video render — checkpoint not available ({ckpt!r})")
        return None

    output = os.path.join(run_dir, "best_mujoco.mp4")
    # Also mirror to results/ for the UI's /api/video endpoint
    results_copy = os.path.join(_PKG_ROOT, "results", "best_mujoco.mp4")

    render_script = os.path.join(_PKG_ROOT, "model_viewer", "render_mujoco.py")
    cmd = [
        sys.executable, render_script,
        "--checkpoint", ckpt,
        "--env-id",     env,
        "--algo",       algo,
        "--output",     output,
        "--n-steps",    "500",
    ]
    print(f"[main] rendering best model → {output}")
    try:
        subprocess.run(cmd, check=True, cwd=_PKG_ROOT)
        # Copy to results/ so the UI can find it immediately
        os.makedirs(os.path.dirname(results_copy), exist_ok=True)
        import shutil
        shutil.copy2(output, results_copy)
        print(f"[main] video saved: {output}")
        return output
    except subprocess.CalledProcessError as e:
        print(f"[main] render failed (exit {e.returncode}) — continuing without video")
        return None


# ── Pipeline ──────────────────────────────────────────────────────────────────


@weave.op(name="AutoRL_Pipeline")
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

    # ── Step 3: evaluate and rank ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[pipeline] step 3 — evaluator")
    print(f"{'='*60}")
    rankings: dict = await evaluate_results(results, run_dir)

    # ── Step 4: render best MuJoCo model ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[pipeline] step 4 — model render")
    print(f"{'='*60}")
    mujoco_rankings = rankings.get("MuJoCo", [])
    video_path: str | None = None
    if mujoco_rankings:
        best_entry = mujoco_rankings[0]
        best_agent_id = best_entry.get("agent_id")
        best_result = next(
            (r.model_dump() for r in results if r.agent_id == best_agent_id), None
        )
        if best_result:
            video_path = _render_best_mujoco(best_result, run_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "run_dir": run_dir,
        "task": task,
        "n_agents": len(plan),
        "n_results": len(results),
        "rankings": rankings,
        "video_path": video_path,
    }

    summary_path = os.path.join(run_dir, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    _print_summary(rankings, video_path)
    return summary


def _print_summary(rankings: dict, video_path: str | None) -> None:
    print(f"\n{'='*60}")
    print("[pipeline] COMPLETE")
    print(f"{'='*60}")
    for group, entries in rankings.items():
        if not entries:
            continue
        print(f"\n{group}:")
        for e in entries:
            rank    = e.get("rank", "?")
            aid     = e.get("agent_id", "?")
            algo    = e.get("algo", "?")
            rat     = e.get("rationale", "")[:80]
            print(f"  {rank}. {algo} ({aid}) — {rat}")
    if video_path:
        print(f"\nVideo: {video_path}")


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
