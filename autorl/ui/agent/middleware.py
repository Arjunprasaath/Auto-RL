"""AutoRL FastAPI backend — the bridge between CopilotKit and the AutoRL pipeline.

Endpoints consumed by the Next.js CopilotKit runtime (via server-side actions):
  POST /api/plan          → generate spawn plan from user task
  POST /api/run           → start swarm (non-blocking, returns run_dir)
  GET  /api/status/{run}  → poll live heartbeats for all agents
  GET  /api/results/{run} → eval_result.json + sentinel_log.json when done

Run with:
  cd autorl
  source .venv/bin/activate
  uvicorn ui.agent.middleware:app --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

_PKG_ROOT = Path(__file__).parent.parent.parent  # autorl/
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from dotenv import load_dotenv
load_dotenv(_PKG_ROOT / ".env")

import weave
from orchestrator.orchestrator_agent import SpawnPlanEntry, create_run_dir, create_spawn_plan
from orchestrator.swarm_runner import run_swarm

app = FastAPI(title="AutoRL Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Init Weave once ───────────────────────────────────────────────────────────

if os.environ.get("WANDB_API_KEY") and not os.environ.get("WEAVE_DISABLED"):
    try:
        weave.init(os.environ.get("WEAVE_PROJECT", "autorl"))
    except Exception as _e:
        print(f"[backend] weave init skipped ({_e})")

# ── In-memory run registry ────────────────────────────────────────────────────

_runs: dict[str, dict[str, Any]] = {}  # run_name → {plan, task, status, results}


# ── Request / response models ─────────────────────────────────────────────────

class PlanRequest(BaseModel):
    task: str


class RunRequest(BaseModel):
    task: str
    run_dir: str
    plan: list[dict]


# ── Helpers ───────────────────────────────────────────────────────────────────

RUNS_BASE = _PKG_ROOT / "runs"
VIDEO_DIR = _PKG_ROOT / "ui" / "agent" / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)


def _run_name(run_dir: str) -> str:
    return Path(run_dir).name


def _read_heartbeats(run_dir: str) -> list[dict]:
    """Read every agent heartbeat.json in the run directory."""
    out = []
    p = Path(run_dir)
    if not p.exists():
        return out
    for agent_dir in sorted(p.iterdir()):
        hb_path = agent_dir / "heartbeat.json"
        if hb_path.exists():
            try:
                out.append(json.loads(hb_path.read_text()))
            except Exception:
                pass
    return out


def _read_results(run_dir: str) -> list[dict]:
    """Read every eval_result.json in the run directory."""
    out = []
    p = Path(run_dir)
    for agent_dir in sorted(p.iterdir()):
        r_path = agent_dir / "eval_result.json"
        if r_path.exists():
            try:
                out.append(json.loads(r_path.read_text()))
            except Exception:
                pass
    return out


def _read_sentinel_log(run_dir: str) -> list[dict]:
    log_path = Path(run_dir) / "sentinel_log.json"
    if log_path.exists():
        try:
            return json.loads(log_path.read_text())
        except Exception:
            pass
    return []


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/plan")
async def generate_plan(req: PlanRequest) -> dict:
    """Generate a spawn plan from a user task description."""
    print(f"\n{'='*60}")
    print(f"[backend] ✓ task received: \"{req.task}\"")
    print(f"{'='*60}\n")
    run_dir = create_run_dir()
    plan_path = os.path.join(run_dir, "spawn_plan.json")
    plan = await create_spawn_plan(req.task, plan_path)

    run_name = _run_name(run_dir)
    _runs[run_name] = {
        "task": req.task,
        "run_dir": run_dir,
        "plan": [e.model_dump() for e in plan],
        "status": "pending_approval",
        "results": [],
    }

    return {
        "run_dir": run_dir,
        "run_name": run_name,
        "plan": [e.model_dump() for e in plan],
    }


@app.post("/api/run")
async def start_run(req: RunRequest) -> dict:
    """Kick off the training swarm.

    Non-blocking: launches the swarm as a background asyncio task.
    The caller should poll /api/status/{run_name} for progress.
    """
    run_name = _run_name(req.run_dir)
    plan = [SpawnPlanEntry.model_validate(e) for e in req.plan]

    if run_name not in _runs:
        _runs[run_name] = {
            "task": req.task,
            "run_dir": req.run_dir,
            "plan": req.plan,
            "status": "running",
            "results": [],
        }
    else:
        _runs[run_name]["status"] = "running"

    async def _run_and_store() -> None:
        try:
            results = await run_swarm(plan, req.run_dir)
            _runs[run_name]["results"] = [r.model_dump() for r in results]
            _runs[run_name]["status"] = "completed"
        except Exception as e:
            print(f"[backend] swarm failed: {e}")
            _runs[run_name]["status"] = "failed"
            _runs[run_name]["error"] = str(e)

    asyncio.create_task(_run_and_store())

    return {"run_name": run_name, "status": "running"}


@app.get("/api/status/{run_name}")
def get_status(run_name: str) -> dict:
    """Return live heartbeats + sentinel log for the given run.

    Frontend polls this every 5 s to update the race dashboard.
    """
    run = _runs.get(run_name)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_name}' not found")

    run_dir = run["run_dir"]
    heartbeats = _read_heartbeats(run_dir)
    sentinel_log = _read_sentinel_log(run_dir)

    return {
        "run_name": run_name,
        "status": run["status"],
        "plan": run["plan"],
        "heartbeats": heartbeats,
        "sentinel_log": sentinel_log,
    }


@app.get("/api/results/{run_name}")
def get_results(run_name: str) -> dict:
    """Return final eval results + best checkpoint once the swarm is done."""
    run = _runs.get(run_name)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_name}' not found")

    run_dir = run["run_dir"]
    results = _read_results(run_dir)
    sentinel_log = _read_sentinel_log(run_dir)

    # Pick best by mean_return.
    completed = [r for r in results if r.get("status") == "completed"]
    best = max(completed, key=lambda r: r.get("mean_return", -1e9)) if completed else None

    return {
        "run_name": run_name,
        "status": run["status"],
        "results": results,
        "best": best,
        "sentinel_log": sentinel_log,
    }


# ── Inference endpoints ───────────────────────────────────────────────────────

class InferRequest(BaseModel):
    run_name: str
    agent_id: str


@app.post("/api/infer")
async def infer_agent(req: InferRequest) -> dict:
    """Record one episode for the given agent and return the mp4 filename.

    Runs render_mujoco.py in a *subprocess* (not a thread) so MuJoCo can
    initialise its OpenGL context on the subprocess's main thread — required
    on macOS, which forbids OpenGL from non-main threads.
    """
    run = _runs.get(req.run_name)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{req.run_name}' not found")

    plan: list[dict] = run.get("plan", [])
    entry = next((e for e in plan if e["id"] == req.agent_id), None)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Agent '{req.agent_id}' not in plan")

    algo     = entry["algo"]
    env_name = entry["env"]
    run_dir  = run["run_dir"]

    if algo not in ("PPO", "SAC", "A2C"):
        raise HTTPException(status_code=400, detail=f"Inference not supported for algo '{algo}' (only PPO/SAC/A2C)")

    model_path = Path(run_dir) / req.agent_id / "model.zip"
    if not model_path.exists():
        raise HTTPException(status_code=400, detail=f"No checkpoint yet for {req.agent_id} — still training?")

    filename    = f"{req.agent_id}_{int(time.time())}.mp4"
    output_path = str(VIDEO_DIR / filename)

    render_script = str(_PKG_ROOT / "model_viewer" / "render_mujoco.py")
    proc = await asyncio.create_subprocess_exec(
        sys.executable, render_script,
        "--checkpoint", str(model_path),
        "--env-id",     env_name,
        "--algo",       algo,
        "--output",     output_path,
        "--n-steps",    "1000",
        cwd=str(_PKG_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        detail = stderr.decode(errors="replace").strip().splitlines()[-1] if stderr else "render failed"
        raise HTTPException(status_code=500, detail=detail)

    return {"filename": filename, "url": f"/api/video/{filename}"}


@app.get("/api/video/{filename}")
async def serve_video(filename: str) -> FileResponse:
    """Serve a recorded inference video."""
    path = VIDEO_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(str(path), media_type="video/mp4", filename=filename)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
