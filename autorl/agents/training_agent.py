"""Wrap training scripts as asyncio subprocesses (Phase 2.3).

Manages the lifecycle of individual training agent processes:
  - run_training_agent   launch a training script as a subprocess
  - kill_training_agent  SIGTERM → SIGKILL a running process
  - restart_training_agent  kill then re-launch with new hparams

Local algo scripts live in training/train_{algo}.py.
Remote (RunPod) execution is handled via runpod_client.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import weave

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from orchestrator.device import is_mps, resolve_grpo_device, resolve_sb3_device, subprocess_env
from orchestrator.orchestrator_agent import EvalResult, SpawnPlanEntry

PROCESSES: dict[str, asyncio.subprocess.Process] = {}
_POD_LOCK = asyncio.Lock()
_LOCAL_ALGOS = frozenset({"PPO", "SAC", "A2C", "GRPO"})


# ─── Command builders ─────────────────────────────────────────────────────────


def _cmd(entry: SpawnPlanEntry, results_dir: str, hp: dict | None = None) -> list[str]:
    h = {**entry.hparams, **(hp or {})}
    return [
        sys.executable, os.path.join("training", f"train_{entry.algo.lower()}.py"),
        "--agent-id", entry.id,
        "--env-id", entry.env,
        "--time-budget", str(entry.time_budget_min * 60),
        "--lr", str(h.get("lr", 3e-4)),
        "--seed", str(h.get("seed", 42)),
        "--results-dir", results_dir,
        "--device", resolve_sb3_device(),
    ]


def _grpo_cmd(entry: SpawnPlanEntry, results_dir: str, hp: dict | None = None) -> list[str]:
    h = {**entry.hparams, **(hp or {})}
    return [
        sys.executable, os.path.join("training", "train_grpo_countdown.py"),
        "--agent-id", entry.id,
        "--time-budget", str(entry.time_budget_min * 60),
        "--lr", str(h.get("lr", 1e-6)),
        "--seed", str(h.get("seed", 42)),
        "--num-generations", str(h.get("num_generations", 4)),
        "--temperature", str(h.get("temperature", 1.0)),
        "--results-dir", results_dir,
        "--device", resolve_grpo_device(),
    ]


def _write_failed(entry: SpawnPlanEntry, results_dir: str) -> None:
    path = os.path.join(results_dir, entry.id, "eval_result.json")
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(EvalResult(
            agent_id=entry.id, algo=entry.algo, env=entry.env, status="failed",
        ).model_dump(), f)


# ─── RunPod helpers ───────────────────────────────────────────────────────────


async def _ensure_pod() -> str:
    async with _POD_LOCK:
        from runpod_client import pod_manager as pm

        if not os.environ.get("RUNPOD_API_KEY"):
            raise RuntimeError("RUNPOD_API_KEY not set")
        pod_id = os.environ.get("RUNPOD_POD_ID") or pm.POD_ID
        if pod_id:
            return pod_id
        pod_id = await asyncio.to_thread(pm.create_training_pod)
        if not await asyncio.to_thread(pm.wait_for_pod, pod_id):
            raise RuntimeError("RunPod failed to start")
        await asyncio.to_thread(pm.install_dependencies, pod_id)
        if not await asyncio.to_thread(pm.verify_dependencies, pod_id):
            raise RuntimeError("RunPod dependency check failed")
        return pod_id


def _grpo_remote_cmd(entry: SpawnPlanEntry, remote_results: str, hp: dict) -> str:
    from runpod_client.pod_manager import VENV_PYTHON

    return (
        f"{VENV_PYTHON} /workspace/training/train_grpo_countdown.py "
        f"--agent-id {entry.id} "
        f"--time-budget {entry.time_budget_min * 60} "
        f"--lr {hp.get('lr', 1e-6)} "
        f"--seed {hp.get('seed', 42)} "
        f"--num-generations {hp.get('num_generations', 4)} "
        f"--temperature {hp.get('temperature', 1.0)} "
        f"--results-dir {remote_results}"
    )


async def _run_runpod(entry: SpawnPlanEntry, results_dir: str, hp: dict | None = None) -> int:
    from runpod_client.pod_manager import scp_from_pod, ssh_exec

    pod_id = await _ensure_pod()
    h = {**entry.hparams, **(hp or {})}
    run_name = os.path.basename(os.path.abspath(results_dir))
    remote_results = f"/workspace/results/{run_name}"
    local_agent = os.path.join(results_dir, entry.id)
    os.makedirs(local_agent, exist_ok=True)

    await asyncio.to_thread(ssh_exec, pod_id, f"mkdir -p {remote_results}")
    cmd = _grpo_remote_cmd(entry, remote_results, h)
    print(f"[{entry.id}] runpod: {cmd}")
    timeout = entry.time_budget_min * 60 + 300
    try:
        await asyncio.to_thread(ssh_exec, pod_id, cmd, timeout)
        code = 0
    except Exception as e:  # noqa: BLE001
        print(f"[{entry.id}] runpod training failed: {e}")
        code = 1

    for fname in ("eval_result.json", "heartbeat.json"):
        try:
            await asyncio.to_thread(
                scp_from_pod,
                pod_id,
                f"{remote_results}/{entry.id}/{fname}",
                os.path.join(local_agent, fname),
            )
        except Exception as e:  # noqa: BLE001
            print(f"[{entry.id}] scp {fname} failed: {e}")

    if code != 0 or not os.path.exists(os.path.join(local_agent, "eval_result.json")):
        _write_failed(entry, results_dir)
        return 1
    print(f"[{entry.id}] runpod done")
    return 0


# ─── Public API ───────────────────────────────────────────────────────────────


async def kill_training_agent(agent_id: str) -> bool:
    """Terminate a running training subprocess. Returns True if a process was killed."""
    proc = PROCESSES.get(agent_id)
    if not proc or proc.returncode is not None:
        return False
    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=10)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
    return True


@weave.op(name="TrainingAgent")
async def run_training_agent(
    entry: SpawnPlanEntry,
    results_dir: str,
    hparams_override: dict | None = None,
) -> int:
    """Launch a training script as an asyncio subprocess and wait for it to finish."""
    os.makedirs(os.path.join(results_dir, entry.id), exist_ok=True)

    if entry.exec == "runpod" and is_mps() and entry.algo.upper() == "GRPO":
        print(f"[{entry.id}] MPS available — running GRPO locally instead of RunPod")
        entry = entry.model_copy(update={"exec": "local"})

    if entry.exec == "local":
        if entry.algo.upper() not in _LOCAL_ALGOS:
            raise ValueError(f"{entry.id}: no local script for algo {entry.algo}")
        cmd = (
            _grpo_cmd(entry, results_dir, hparams_override)
            if entry.algo.upper() == "GRPO"
            else _cmd(entry, results_dir, hparams_override)
        )
        print(
            f"[{entry.id}] launch "
            f"(sb3={resolve_sb3_device()}, grpo={resolve_grpo_device()}): "
            f"{' '.join(cmd[1:])}"
        )
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=_PKG_ROOT, env=subprocess_env()
        )
        PROCESSES[entry.id] = proc
        try:
            code = await proc.wait()
        finally:
            PROCESSES.pop(entry.id, None)
        if code != 0:
            _write_failed(entry, results_dir)
        print(f"[{entry.id}] exit {code}")
        return code

    if entry.exec == "runpod":
        if entry.algo.upper() != "GRPO":
            raise ValueError(f"{entry.id}: runpod only supports GRPO")
        return await _run_runpod(entry, results_dir, hparams_override)

    raise ValueError(f"{entry.id}: unknown exec {entry.exec!r}")


async def restart_training_agent(
    entry: SpawnPlanEntry,
    results_dir: str,
    lr: float = 3e-4,
    seed: int | None = None,
) -> int:
    """Kill a running agent and relaunch it with a new lr and seed."""
    await kill_training_agent(entry.id)
    override = {"lr": lr, "seed": seed if seed is not None else entry.hparams.get("seed", 42) + 1000}
    return await run_training_agent(entry, results_dir, hparams_override=override)
