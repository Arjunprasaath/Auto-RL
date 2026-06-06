"""
Run context helpers — one timestamped directory per AutoRL race.

A "run" co-locates the spawn plan, per-agent results, rankings, and report:

    runs/2026-06-06T13-41-02/
    ├── spawn_plan.json
    ├── rankings.json
    ├── run_report.md
    └── agent_1/
        ├── heartbeat.json
        ├── eval_result.json
        └── model.zip

The orchestrator mints the run dir and threads it through create_spawn_plan()
and the swarm runner (as --results-dir), so re-running never clobbers prior runs.
"""

import os
from datetime import datetime, timezone

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(_PKG_ROOT, "runs")


def new_run_id() -> str:
    """Filesystem-safe UTC timestamp, e.g. 2026-06-06T13-41-02."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")


def create_run_dir(run_id: str | None = None, base: str = RUNS_DIR) -> str:
    """Create runs/<run_id>/ (minting a timestamp if needed) and return its path.

    Also refreshes a `runs/latest` symlink pointing at the new run, for
    convenience. Returns the absolute run directory path.
    """
    run_id = run_id or new_run_id()
    run_dir = os.path.join(base, run_id)
    os.makedirs(run_dir, exist_ok=True)
    _update_latest(base, run_dir)
    return run_dir


def spawn_plan_path(run_dir: str) -> str:
    return os.path.join(run_dir, "spawn_plan.json")


def _update_latest(base: str, run_dir: str) -> None:
    latest = os.path.join(base, "latest")
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            os.remove(latest)
        os.symlink(os.path.basename(run_dir), latest)
    except OSError:
        # Symlinks may be unavailable (e.g. some filesystems); non-fatal.
        pass
