"""Wrap training scripts as asyncio subprocesses (Phase 2.3).

Manages the lifecycle of individual training agent processes:
  - run_training_agent   launch a training script as a subprocess
  - kill_training_agent  SIGTERM → SIGKILL a running process
  - restart_training_agent  kill then re-launch with new hparams

Local algo scripts (PPO, SAC, A2C) live in training/train_{algo}.py.
Remote GRPO execution is handled via pod_manager.runpod_agent.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from orchestrator.device import resolve_sb3_device, subprocess_env
from orchestrator.orchestrator_agent import EvalResult, SpawnPlanEntry

PROCESSES: dict[str, asyncio.subprocess.Process] = {}
_LOCAL_ALGOS = frozenset({"PPO", "SAC", "A2C"})


# ─── Command builders ─────────────────────────────────────────────────────────


def _cmd(entry: SpawnPlanEntry, results_dir: str, hp: dict | None = None) -> list[str]:
    h = {**entry.hparams, **(hp or {})}
    cmd = [
        sys.executable, os.path.join("training", f"train_{entry.algo.lower()}.py"),
        "--agent-id", entry.id,
        "--env-id", entry.env,
        "--time-budget", str(entry.time_budget_min * 60),
        "--lr", str(h.get("lr", 3e-4)),
        "--seed", str(h.get("seed", 42)),
        "--results-dir", results_dir,
        "--device", resolve_sb3_device(),
    ]
    if "n_steps" in h:
        cmd += ["--n-steps", str(h["n_steps"])]
    if "ent_coef" in h:
        cmd += ["--ent-coef", str(h["ent_coef"])]
    if "gamma" in h:
        cmd += ["--gamma", str(h["gamma"])]
    if "policy" in h:
        cmd += ["--policy", str(h["policy"])]
    return cmd


def _write_failed(entry: SpawnPlanEntry, results_dir: str) -> None:
    path = os.path.join(results_dir, entry.id, "eval_result.json")
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(EvalResult(
            agent_id=entry.id, algo=entry.algo, env=entry.env, status="failed",
        ).model_dump(), f)


# ─── Public API ───────────────────────────────────────────────────────────────


async def kill_training_agent(agent_id: str) -> bool:
    """Terminate a running training subprocess or RunPod pod. Returns True if killed."""
    proc = PROCESSES.get(agent_id)
    if proc and proc.returncode is None:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        return True

    from pod_manager.runpod_agent import ACTIVE_PODS
    from pod_manager.pod_manager import terminate_pod

    pod_id = ACTIVE_PODS.pop(agent_id, None)
    if pod_id:
        terminate_pod(pod_id)
        print(f"[{agent_id}] RunPod pod {pod_id} terminated by sentinel")
        return True

    return False


async def run_training_agent(
    entry: SpawnPlanEntry,
    results_dir: str,
    hparams_override: dict | None = None,
) -> int:
    """Launch a training script as an asyncio subprocess and wait for it to finish."""
    os.makedirs(os.path.join(results_dir, entry.id), exist_ok=True)

    if entry.exec == "local":
        if entry.algo.upper() not in _LOCAL_ALGOS:
            raise ValueError(f"{entry.id}: no local script for algo {entry.algo}")
        cmd = _cmd(entry, results_dir, hparams_override)
        print(
            f"[{entry.id}] launch (device={resolve_sb3_device()}): "
            f"{' '.join(cmd[1:])}"
        )
        async def _launch(retry: bool = False) -> tuple[int, str]:
            """Start the cmd, capture stderr for the doctor, return (exit_code, stderr)."""
            label = f"{entry.id}{'[retry]' if retry else ''}"
            p = await asyncio.create_subprocess_exec(
                *cmd, cwd=_PKG_ROOT, env=subprocess_env(),
                stderr=asyncio.subprocess.PIPE,  # capture for doctor; stdout stays inherited
            )
            PROCESSES[entry.id] = p
            try:
                _, stderr_bytes = await p.communicate()
                code = p.returncode or 0
            finally:
                PROCESSES.pop(entry.id, None)
            stderr_text = stderr_bytes.decode(errors="replace") if stderr_bytes else ""
            # Echo stderr to server console on failure (was swallowed by PIPE)
            if code != 0 and stderr_text.strip():
                print(f"[{label}] stderr tail:\n{stderr_text[-1500:]}")
            return code, stderr_text

        code, stderr_text = await _launch()

        if code != 0 and stderr_text:
            from agents.env_doctor_agent import diagnose_and_fix
            result = diagnose_and_fix(stderr_text, entry.id, results_dir)
            if result.fixed:
                print(f"[{entry.id}] doctor applied fix — retrying")
                code, _ = await _launch(retry=True)

        if code != 0:
            _write_failed(entry, results_dir)
        print(f"[{entry.id}] exit {code}")
        return code

    if entry.exec == "runpod":
        if entry.algo.upper() != "GRPO":
            raise ValueError(f"{entry.id}: runpod only supports GRPO")
        from pod_manager.runpod_agent import run_grpo_on_runpod

        print(f"[{entry.id}] dispatching to RunPod via runpod_agent...")
        result = await asyncio.to_thread(
            run_grpo_on_runpod,
            entry,
            local_results_dir=results_dir,
            terminate_after=True,
        )
        if result.get("status") == "completed":
            return 0
        _write_failed(entry, results_dir)
        return 1

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
