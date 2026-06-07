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

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
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


class HFDatasetRequest(BaseModel):
    dataset_name: str
    split: str = "train"
    config_name: str | None = None


class WMPlanRequest(BaseModel):
    meta: dict           # DatasetMeta serialised as dict
    time_budget_min: int = 5


class RewardDesignRequest(BaseModel):
    meta: dict                        # DatasetMeta serialised as dict
    history: list[dict] = []          # [{"role": "user"|"assistant", "content": str}]
    message: str = ""                 # latest user message ("" = trigger first suggestion)


class ApplyRewardRequest(BaseModel):
    meta: dict                        # DatasetMeta serialised as dict
    reward_code: str                  # approved reward_fn Python source


# ── Helpers ───────────────────────────────────────────────────────────────────

RUNS_BASE    = _PKG_ROOT / "runs"
VIDEO_DIR    = _PKG_ROOT / "ui" / "agent" / "videos"
DATASET_DIR  = _PKG_ROOT / "datasets"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Map HuggingFace dataset config names → real Gymnasium env IDs
_CONFIG_TO_ENV: dict[str, str] = {
    # MuJoCo locomotion
    "mujoco-ant":                   "Ant-v5",
    "mujoco-halfcheetah":           "HalfCheetah-v5",
    "mujoco-cheetah":               "HalfCheetah-v5",
    "mujoco-hopper":                "Hopper-v5",
    "mujoco-walker2d":              "Walker2d-v5",
    "mujoco-humanoid":              "Humanoid-v5",
    "mujoco-swimmer":               "Swimmer-v5",
    "mujoco-reacher":               "Reacher-v5",
    "mujoco-pusher":                "Pusher-v5",
    "mujoco-inverteddoublependulum": "InvertedDoublePendulum-v5",
    "mujoco-invertedpendulum":      "InvertedPendulum-v5",
    # Classic control
    "cartpole":                     "CartPole-v1",
    "mountaincar":                  "MountainCar-v0",
    "mountaincarcontinuous":        "MountainCarContinuous-v0",
    "acrobot":                      "Acrobot-v1",
    "pendulum":                     "Pendulum-v1",
    # Atari (ALE)
    "atari-pong":                   "ALE/Pong-v5",
    "atari-breakout":               "ALE/Breakout-v5",
    "atari-spaceinvaders":          "ALE/SpaceInvaders-v5",
    "atari-asteroids":              "ALE/Asteroids-v5",
}


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
    all_hbs      = _read_heartbeats(run_dir)
    # Expose wm_trainer heartbeat separately so the UI can show epoch / val_loss
    wm_heartbeat = next((h for h in all_hbs if h.get("agent_id") == "wm_trainer"), None)
    heartbeats   = [h for h in all_hbs if h.get("agent_id") != "wm_trainer"]
    sentinel_log = _read_sentinel_log(run_dir)

    # Read agent_log.json from disk to pick up incremental planner writes
    agent_log_path = Path(run_dir) / "agent_log.json"
    try:
        disk_log = json.loads(agent_log_path.read_text()) if agent_log_path.exists() else []
    except Exception:
        disk_log = []
    # Merge: use whichever is longer
    mem_log = run.get("agent_log", [])
    agent_log = disk_log if len(disk_log) >= len(mem_log) else mem_log

    return {
        "run_name":    run_name,
        "status":      run["status"],
        "wm_status":   run.get("wm_status"),   # "planning" | "training" | "done" | "failed" | None
        "plan":        run["plan"],
        "heartbeats":  heartbeats,
        "wm_heartbeat": wm_heartbeat,          # wm_trainer heartbeat (epoch, val_loss, total_epochs)
        "sentinel_log": sentinel_log,
        "agent_log":   agent_log,              # multi-agent planner reasoning
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


# ── Env-family helpers ────────────────────────────────────────────────────────

_ATARI_KEYWORDS = (
    "pong", "breakout", "spaceinvaders", "asteroids", "qbert", "montezuma",
    "mspacman", "beamrider", "enduro", "pitfall", "venture", "videopinball",
    "atlantis", "assault", "alien", "amidar", "kangaroo", "krull", "battlezone",
    "berzerk", "centipede", "choppercommand", "crazyclimber", "defender",
    "demonattack", "doubledunk", "fishingderby", "freeway", "frostbite",
    "gopher", "gravitar", "hero", "icehockey", "jamesbond", "phoenix",
    "privateeye", "roadrunner", "robotank", "seaquest", "skiing", "solaris",
    "stargunner", "tennis", "timepilot", "tutankham", "upndown", "wizard",
)


def _detect_env_family(env_id: str) -> str:
    """Mirror of the frontend detectEnvFamily — used to pick render args."""
    e = env_id.lower()
    if any(k in e for k in ("frozenlake", "taxi", "cliffwalking", "blackjack")):
        return "toytext"
    if any(k in e for k in ("lunarlander", "bipedalwalker", "carracing")):
        return "box2d"
    if any(k in e for k in ("halfcheetah", "hopper", "ant", "walker2d", "swimmer",
                             "humanoid", "reacher", "pusher", "invertedpendulum")):
        return "mujoco"
    if e.startswith("ale/") or any(k in e for k in _ATARI_KEYWORDS):
        return "atari"
    return "classic"


def _render_extra_args(env_id: str) -> list[str]:
    """Return family-appropriate renderer CLI args for n_steps / n_episodes."""
    if env_id == "WorldModel-v0":
        return ["--n-steps", "500", "--n-episodes", "3"]
    family = _detect_env_family(env_id)
    if family == "toytext":
        # Grid-world episodes are very short — collect 5 complete ones
        return ["--n-episodes", "5"]
    if family == "mujoco":
        # One full robot episode (up to 1 000 steps = ~33 s at 30 fps)
        return ["--n-steps", "1000", "--n-episodes", "1"]
    if family == "box2d":
        # Two landings / walks
        return ["--n-steps", "800", "--n-episodes", "2"]
    if family == "atari":
        # Atari episodes can be long; record 2 complete games
        return ["--n-steps", "2000", "--n-episodes", "2"]
    # Classic Control: a few short episodes
    return ["--n-steps", "500", "--n-episodes", "3"]


# ── Inference endpoints ───────────────────────────────────────────────────────

class InferRequest(BaseModel):
    run_name: str
    agent_id: str
    env_override: str | None = None   # render in this real env instead of WorldModel-v0


@app.post("/api/infer")
async def infer_agent(req: InferRequest) -> dict:
    """Record an episode (or episodes) for the given agent and return the mp4.

    Runs render_mujoco.py in a *subprocess* (not a thread) so MuJoCo can
    initialise its OpenGL context on the subprocess's main thread — required
    on macOS, which forbids OpenGL from non-main threads.

    The number of steps / episodes recorded is chosen automatically per
    environment family (MuJoCo / Classic / Box2D / Grid World).
    """
    run = _runs.get(req.run_name)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{req.run_name}' not found")

    if req.agent_id == "wm_trainer":
        raise HTTPException(
            status_code=400,
            detail="wm_trainer is the world model trainer — it has no RL policy to render. "
                   "Use one of the RL agents (agent_1, agent_2, agent_3) instead."
        )

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

    # env_override: render the WM-trained policy in the original real env instead
    render_env_name = req.env_override if req.env_override else env_name
    filename    = f"{req.agent_id}_{int(time.time())}.mp4"
    output_path = str(VIDEO_DIR / filename)
    env_family  = _detect_env_family(render_env_name)

    from orchestrator.device import subprocess_env
    render_env = subprocess_env()
    if render_env_name == "WorldModel-v0":
        # WorldModel render: pass checkpoint paths as env vars
        wm_ckpt = entry["hparams"].get("wm_checkpoint", "")
        wm_meta = entry["hparams"].get("wm_meta", "")
        if wm_ckpt:
            render_env["WORLD_MODEL_CHECKPOINT"] = wm_ckpt
        if wm_meta:
            render_env["WORLD_MODEL_META"] = wm_meta
        env_family = "worldmodel"
    # else: real env render — no extra vars needed; MuJoCo loads via gymnasium normally

    render_script = str(_PKG_ROOT / "model_viewer" / "render_mujoco.py")
    proc = await asyncio.create_subprocess_exec(
        sys.executable, render_script,
        "--checkpoint", str(model_path),
        "--env-id",     render_env_name,
        "--algo",       algo,
        "--output",     output_path,
        *_render_extra_args(render_env_name),
        cwd=str(_PKG_ROOT),
        env=render_env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        detail = stderr.decode(errors="replace").strip().splitlines()[-1] if stderr else "render failed"
        raise HTTPException(status_code=500, detail=detail)

    return {"filename": filename, "url": f"/api/video/{filename}", "env_family": env_family}


@app.get("/api/video/{filename}")
async def serve_video(filename: str) -> FileResponse:
    """Serve a recorded inference video."""
    path = VIDEO_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(str(path), media_type="video/mp4", filename=filename)


# ── Dataset & World-Model endpoints ──────────────────────────────────────────


@app.post("/api/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)) -> dict:
    """Accept a CSV / JSON / parquet upload, inspect it, return DatasetMeta."""
    import shutil

    from agents.dataset_inspector_agent import inspect as ds_inspect

    suffix = Path(file.filename or "dataset.csv").suffix or ".csv"
    ds_path  = DATASET_DIR / f"upload_{int(time.time())}{suffix}"
    init_path = DATASET_DIR / f"initial_states_{int(time.time())}.parquet"

    with open(ds_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        meta = ds_inspect(str(ds_path), str(init_path))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Dataset inspection failed: {exc}")

    return meta.model_dump()


@app.post("/api/hf-dataset")
async def hf_dataset(req: HFDatasetRequest) -> dict:
    """Download a HuggingFace dataset, inspect it, return DatasetMeta."""
    # Validate: must look like owner/dataset-name, not a shell command
    name = req.dataset_name.strip()
    if not name or " " in name or not "/" in name:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid dataset name {name!r}. Expected format: owner/dataset-name (e.g. edbeeching/decision_transformer_gym_replay_hopper)"
        )

    from agents.dataset_inspector_agent import (
        DatasetMeta,
        download_from_huggingface,
        inspect as ds_inspect,
    )
    from agents.wm_planner_agent import recommend_dataset_size

    ds_dir    = DATASET_DIR / f"hf_{int(time.time())}"
    ds_dir.mkdir(parents=True, exist_ok=True)
    init_path = str(ds_dir / "initial_states.parquet")

    # ── Dataset size agent: decide how much to download ──────────────────────
    effective_split = req.split  # fallback if agent fails
    size_reasoning  = ""
    try:
        size_rec = await asyncio.to_thread(
            recommend_dataset_size, name, req.config_name
        )
        effective_split = size_rec.split
        size_reasoning  = size_rec.reasoning
        print(f"[hf-dataset] size_agent → split={effective_split!r}  "
              f"~{size_rec.n_samples_estimate:,} transitions")
    except Exception as exc:
        print(f"[hf-dataset] size_agent failed ({exc}), using requested split={effective_split!r}")

    try:
        ds_path = await asyncio.to_thread(
            download_from_huggingface,
            name,
            effective_split,
            str(ds_dir),
            req.config_name,
        )
        meta = await asyncio.to_thread(ds_inspect, ds_path, init_path)
    except Exception as exc:
        import traceback
        print(f"[hf-dataset] ERROR for {name!r}: {exc}")
        traceback.print_exc()
        raise HTTPException(status_code=422, detail=f"HF dataset error: {exc}")

    # Infer source env from config name (e.g. "mujoco-ant" → "Ant-v5")
    source_env = None
    if req.config_name:
        source_env = _CONFIG_TO_ENV.get(req.config_name.lower().replace("-", "-"))
        if not source_env:
            # Fuzzy match: strip "mujoco-" prefix and try gymnasium directly
            for key, val in _CONFIG_TO_ENV.items():
                if key in req.config_name.lower():
                    source_env = val
                    break

    result = meta.model_dump()
    result["_size_reasoning"] = size_reasoning   # extra field for UI display
    result["_split_used"]     = effective_split
    if source_env:
        result["source_env"] = source_env
    return result


@app.post("/api/design-reward")
async def design_reward_endpoint(req: RewardDesignRequest) -> dict:
    """One turn of the reward-design LLM conversation.

    Returns {message, code, explanation} — the frontend adds this to its
    chat history and shows the code block to the user.
    """
    from agents.dataset_inspector_agent import DatasetMeta
    from agents.reward_designer_agent import design_reward

    meta   = DatasetMeta.model_validate(req.meta)
    result = await asyncio.to_thread(design_reward, meta, req.history, req.message)
    return result.model_dump()


@app.post("/api/apply-reward")
async def apply_reward_endpoint(req: ApplyRewardRequest) -> dict:
    """Apply the approved reward function to the dataset and return updated DatasetMeta.

    Rewrites the reward column in-place over a copy of the dataset parquet, then
    returns a new DatasetMeta whose dataset_path points to the rewritten file and
    whose reward_min/max reflect the new distribution.
    """
    from agents.dataset_inspector_agent import DatasetMeta
    from agents.reward_designer_agent import apply_reward_fn

    meta     = DatasetMeta.model_validate(req.meta)
    out_path = str(Path(meta.dataset_path).parent / "dataset_custom_reward.parquet")

    try:
        new_path = await asyncio.to_thread(
            apply_reward_fn, meta.dataset_path, meta, req.reward_code, out_path
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to apply reward: {exc}")

    # Compute updated reward stats from the rewritten file
    import pandas as pd
    df       = pd.read_parquet(new_path)
    new_meta = meta.model_copy(update={
        "dataset_path": new_path,
        "reward_min":   float(df[meta.reward_col].min()),
        "reward_max":   float(df[meta.reward_col].max()),
    })
    return new_meta.model_dump()


@app.post("/api/world-model-plan")
async def world_model_plan(req: WMPlanRequest) -> dict:
    """Start a two-phase world-model pipeline.

    Phase 1 (background): trains the world model on the dataset.
    Phase 2 (background): runs PPO / SAC / A2C inside the learned simulator.

    Returns immediately with run_name — poll /api/status/{run_name} for progress.
    """
    from agents.dataset_inspector_agent import DatasetMeta
    from orchestrator.orchestrator_agent import SpawnPlanEntry

    meta      = DatasetMeta.model_validate(req.meta)
    run_dir   = create_run_dir()
    run_name  = _run_name(run_dir)

    _runs[run_name] = {
        "task":      f"World model — {Path(meta.dataset_path).name}",
        "run_dir":   run_dir,
        "plan":      [],          # populated once WM training finishes
        "status":    "wm_training",
        "wm_status": "planning",  # planning → training → done
        "results":   [],
        "agent_log": [],          # populated as planner agents complete
    }

    asyncio.create_task(_run_wm_pipeline(run_name, run_dir, meta, req.time_budget_min))

    return {"run_name": run_name, "status": "wm_training"}


async def _run_wm_pipeline(
    run_name: str, run_dir: str, meta, budget_min: int
) -> None:
    """Background task: plan → train WM (phase 1) → run SB3 swarm (phase 2).

    Phase 0: Multi-agent planner decides architecture and algorithm lineup.
    Phase 1: Train the world model with planner-recommended architecture.
    Phase 2: Race SB3 algorithms inside the learned world model.
    """
    from agents.dataset_inspector_agent import DatasetMeta
    from agents.wm_planner_agent import run_planning
    from orchestrator.orchestrator_agent import SpawnPlanEntry
    from orchestrator.swarm_runner import run_swarm

    # ── Phase 0: multi-agent planning ──────────────────────────────────────
    print(f"[backend] wm_pipeline: phase 0 — multi-agent planning for {run_name}")
    _runs[run_name]["wm_status"] = "planning"

    try:
        decisions = await asyncio.to_thread(run_planning, meta, run_dir)
        _runs[run_name]["agent_log"] = [e.model_dump() for e in decisions.agent_log]
        print(
            f"[backend] wm_pipeline: planning done — "
            f"arch={decisions.arch.hidden_sizes} {decisions.arch.activation} "
            f"algos={[a.algo for a in decisions.algos]}"
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[backend] wm_pipeline: planning failed ({exc}) — using defaults")
        # Fallback to sensible defaults so the pipeline continues
        from agents.wm_planner_agent import (
            ArchDecision, AlgoEntry, HparamEntry, PlannerDecisions, AgentLogEntry
        )
        decisions = PlannerDecisions(
            arch=ArchDecision(
                hidden_sizes=meta.hidden_sizes,
                activation="silu",
                dropout=0.0,
                reasoning="Fallback defaults (planner failed)",
            ),
            algos=[
                AlgoEntry(algo="PPO", agent_id="agent_1", reasoning="fallback"),
                AlgoEntry(algo="SAC" if meta.act_type == "continuous" else "A2C",
                          agent_id="agent_2", reasoning="fallback"),
                AlgoEntry(algo="A2C", agent_id="agent_3", reasoning="sentinel"),
            ],
            hparams=[
                HparamEntry(algo="PPO",  agent_id="agent_1",
                            hparams={"lr": 3e-4, "gamma": 0.99, "n_steps": 512, "seed": 42},
                            reasoning="fallback"),
                HparamEntry(algo="SAC" if meta.act_type == "continuous" else "A2C",
                            agent_id="agent_2",
                            hparams={"lr": 1e-3, "gamma": 0.99, "seed": 7},
                            reasoning="fallback"),
                HparamEntry(algo="A2C", agent_id="agent_3",
                            hparams={"lr": 1.0, "gamma": 0.99, "n_steps": 256, "seed": 99},
                            reasoning="doom-loop sentinel"),
            ],
            agent_log=[AgentLogEntry(
                agent="system", decision="Using default config (planner unavailable)",
                reasoning=str(exc), timestamp=time.time(),
            )],
        )
        _runs[run_name]["agent_log"] = [e.model_dump() for e in decisions.agent_log]

    # ── Phase 1: train world model ──────────────────────────────────────────
    wm_dir    = Path(run_dir) / "wm_trainer"
    wm_dir.mkdir(parents=True, exist_ok=True)
    meta_path = str(wm_dir / "dataset_meta.json")
    with open(meta_path, "w") as f:
        f.write(meta.model_dump_json())

    arch = decisions.arch
    wm_cmd = [
        sys.executable, str(_PKG_ROOT / "training" / "train_world_model.py"),
        "--agent-id",     "wm_trainer",
        "--dataset-path", meta.dataset_path,
        "--meta-path",    meta_path,
        "--results-dir",  run_dir,
        "--time-budget",  str(budget_min * 60),
        "--hidden-sizes", *[str(s) for s in arch.hidden_sizes],
        "--activation",   arch.activation,
        "--dropout",      str(arch.dropout),
    ]
    _runs[run_name]["wm_status"] = "training"
    print(f"[backend] wm_pipeline: phase 1 — training world model in {run_dir}")
    proc = await asyncio.create_subprocess_exec(
        *wm_cmd, cwd=str(_PKG_ROOT),
        stdout=None,                        # inherit → visible in server log
        stderr=asyncio.subprocess.PIPE,     # capture for error reporting only
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        err = stderr.decode(errors="replace").strip() if stderr else "unknown error"
        print(f"[backend] wm_pipeline: world model training failed:\n{err}")
        _runs[run_name]["status"]    = "failed"
        _runs[run_name]["wm_status"] = "failed"
        return

    ckpt_path = str(wm_dir / "wm_checkpoint.pt")
    if not Path(ckpt_path).exists():
        print(f"[backend] wm_pipeline: checkpoint not found at {ckpt_path}")
        _runs[run_name]["status"]    = "failed"
        _runs[run_name]["wm_status"] = "failed"
        return

    _runs[run_name]["wm_status"] = "done"
    print(f"[backend] wm_pipeline: phase 1 done — checkpoint at {ckpt_path}")

    # ── Phase 2: SB3 swarm — planner-chosen algos + hparams ─────────────────
    wm_hparams = {"wm_checkpoint": ckpt_path, "wm_meta": meta_path}

    # Build a lookup of hparams by agent_id
    hp_map = {h.agent_id: h.hparams for h in decisions.hparams}

    plan: list[SpawnPlanEntry] = []
    for algo_entry in decisions.algos:
        hp = {**hp_map.get(algo_entry.agent_id, {}), **wm_hparams}
        plan.append(SpawnPlanEntry(
            id=algo_entry.agent_id,
            algo=algo_entry.algo,
            env="WorldModel-v0",
            exec="local",
            time_budget_min=budget_min,
            hparams=hp,
        ))

    _runs[run_name]["plan"]   = [e.model_dump() for e in plan]
    _runs[run_name]["status"] = "running"

    import json as _json
    plan_path = Path(run_dir) / "spawn_plan.json"
    plan_path.write_text(_json.dumps([e.model_dump() for e in plan], indent=2))

    try:
        results = await run_swarm(plan, run_dir)
        _runs[run_name]["results"] = [r.model_dump() for r in results]
        _runs[run_name]["status"]  = "completed"
    except Exception as exc:
        print(f"[backend] wm_pipeline: swarm failed: {exc}")
        _runs[run_name]["status"] = "failed"
        _runs[run_name]["error"]  = str(exc)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
