"""
End-to-end RunPod dispatcher for a single GRPO agent.

`run_grpo_on_runpod(entry)` is the only public function.
It handles the full lifecycle for one SpawnPlanEntry with exec=="runpod":

    create pod → wait → install deps → SCP code → SSH train → poll heartbeat
    → SCP results back → terminate pod

Called by swarm/runner.py for every runpod entry in spawn_plan.json.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PKG_ROOT, ".env"))
except ImportError:
    pass

import weave

from pod_manager.pod_manager import (
    VENV_PYTHON,
    create_training_pod,
    get_pod_ssh_info,
    install_dependencies,
    scp_from_pod,
    ssh_exec,
    terminate_pod,
    wait_for_pod,
)
from orchestrator.schemas import SpawnPlanEntry

REMOTE_WORKSPACE = "/workspace"
REMOTE_RESULTS = f"{REMOTE_WORKSPACE}/results"


def _scp_to_pod(local_path: str, remote_path: str, host: str, port: int) -> None:
    """SCP a local file or directory to the pod."""
    flags = ["-r"] if Path(local_path).is_dir() else []
    subprocess.run(
        [
            "scp", *flags,
            "-P", str(port),
            "-o", "StrictHostKeyChecking=no",
            local_path,
            f"root@{host}:{remote_path}",
        ],
        check=True,
    )


def _upload_training_code(pod_id: str) -> None:
    """
    Upload the training/ and environments/ packages to /workspace/ on the pod.
    This makes `train_grpo_countdown.py` importable as written.
    """
    host, port = get_pod_ssh_info(pod_id)
    autorl_dir = _PKG_ROOT  # …/autorl/

    print(f"[runpod_agent] Uploading training code to pod {pod_id}...")
    ssh_exec(pod_id, f"mkdir -p {REMOTE_WORKSPACE}/training/callbacks {REMOTE_WORKSPACE}/environments")

    for src, dest in [
        (f"{autorl_dir}/training/", f"{REMOTE_WORKSPACE}/training/"),
        (f"{autorl_dir}/environments/", f"{REMOTE_WORKSPACE}/environments/"),
    ]:
        _scp_to_pod(src, dest, host, port)

    print("[runpod_agent] Code upload complete.")


def _build_train_cmd(entry: SpawnPlanEntry) -> str:
    """
    Convert a SpawnPlanEntry into the CLI command to run on the pod.
    Prefixes WANDB_API_KEY and WEAVE_PROJECT so per-step charts and Weave
    tracing work from inside the pod without a .env file there.
    """
    h = entry.hparams
    time_budget_s = entry.time_budget_min * 60

    # Env vars forwarded inline so they're available to the training process
    env_prefix = ""
    for var in ("WANDB_API_KEY", "WEAVE_PROJECT", "OPENAI_API_KEY"):
        val = os.environ.get(var)
        if val:
            env_prefix += f"{var}={val} "

    parts = [
        env_prefix + VENV_PYTHON,
        f"{REMOTE_WORKSPACE}/training/train_grpo_countdown.py",
        f"--agent-id {entry.id}",
        f"--time-budget {time_budget_s}",
        f"--lr {h.get('lr', 1e-6)}",
        f"--seed {h.get('seed', 42)}",
        f"--num-generations {h.get('num_generations', 4)}",
        f"--temperature {h.get('temperature', 1.0)}",
        f"--results-dir {REMOTE_RESULTS}",
    ]
    return " ".join(parts)


def _poll_heartbeat(pod_id: str, agent_id: str, local_results_dir: str,
                    poll_interval_s: int = 90, max_missing: int = 5) -> None:
    """
    Every `poll_interval_s` seconds, SCP heartbeat.json from the pod and
    print its status. If heartbeat is missing `max_missing` times in a row,
    prints a warning (the Sentinel will handle intervention).
    """
    remote_hb = f"{REMOTE_RESULTS}/{agent_id}/heartbeat.json"
    local_hb = f"{local_results_dir}/{agent_id}/heartbeat.json"
    Path(f"{local_results_dir}/{agent_id}").mkdir(parents=True, exist_ok=True)
    missing = 0

    while True:
        time.sleep(poll_interval_s)
        try:
            scp_from_pod(pod_id, remote_hb, local_hb)
            with open(local_hb) as f:
                hb = json.load(f)
            status = hb.get("status", "?")
            reward = hb.get("current_reward", 0.0)
            steps = hb.get("steps_completed", 0)
            print(f"[runpod_agent][{agent_id}] heartbeat: status={status} "
                  f"steps={steps} reward={reward:.4f}")
            missing = 0
            if status in ("completed", "failed"):
                break
        except Exception as e:
            missing += 1
            print(f"[runpod_agent][{agent_id}] heartbeat missing ({missing}/{max_missing}): {e}")
            if missing >= max_missing:
                print(f"[runpod_agent][{agent_id}] WARNING: no heartbeat for "
                      f"{missing * poll_interval_s}s — Sentinel should intervene")
                break


@weave.op(name="RunPodAgent")
def run_grpo_on_runpod(
    entry: SpawnPlanEntry,
    local_results_dir: str = "results",
    terminate_after: bool = True,
) -> dict:
    """
    Full lifecycle for one GRPO SpawnPlanEntry on RunPod. Traced as a Weave op.

    Steps:
        1. Provision GPU pod
        2. Wait for RUNNING status
        3. Install dependencies into /workspace/venv
        4. Upload training code to /workspace/
        5. Launch train_grpo_countdown.py in background
        6. Poll heartbeat.json every 90s
        7. SCP eval_result.json back locally
        8. Terminate pod (unless terminate_after=False)

    Returns the eval_result dict, or {"status": "failed"} on error.
    """
    agent_id = entry.id
    pod_id = None

    try:
        # 1. Provision
        pod_id = create_training_pod(name=f"autorl-{agent_id}")

        # 2. Wait
        if not wait_for_pod(pod_id):
            raise RuntimeError(f"Pod {pod_id} failed to reach RUNNING state")

        # 3. Install deps
        install_dependencies(pod_id)

        # 4. Upload training code
        _upload_training_code(pod_id)

        # 5. Launch training (nohup so it survives SSH disconnection)
        train_cmd = _build_train_cmd(entry)
        log_path = f"{REMOTE_RESULTS}/{agent_id}/train.log"
        ssh_exec(pod_id, f"mkdir -p {REMOTE_RESULTS}/{agent_id}")
        ssh_exec(
            pod_id,
            f"nohup {train_cmd} > {log_path} 2>&1 &",
            timeout=30,  # just to launch; actual training runs async
        )
        print(f"[runpod_agent][{agent_id}] Training launched on pod {pod_id}")
        print(f"[runpod_agent][{agent_id}] Log: {log_path}")

        # 6. Poll heartbeat until training completes
        _poll_heartbeat(pod_id, agent_id, local_results_dir)

        # 7. SCP back eval_result.json
        remote_result = f"{REMOTE_RESULTS}/{agent_id}/eval_result.json"
        local_agent_dir = f"{local_results_dir}/{agent_id}"
        Path(local_agent_dir).mkdir(parents=True, exist_ok=True)
        local_result = f"{local_agent_dir}/eval_result.json"

        scp_from_pod(pod_id, remote_result, local_result)
        with open(local_result) as f:
            result = json.load(f)
        print(f"[runpod_agent][{agent_id}] Result: mean_return={result.get('mean_return', '?')}")
        return result

    except Exception as e:
        print(f"[runpod_agent][{agent_id}] FAILED: {e}")
        return {"agent_id": agent_id, "algo": "GRPO", "env": "Countdown",
                "status": "failed", "mean_return": 0.0, "error": str(e)}

    finally:
        if pod_id and terminate_after:
            try:
                terminate_pod(pod_id)
                print(f"[runpod_agent][{agent_id}] Pod {pod_id} terminated.")
            except Exception as e:
                print(f"[runpod_agent][{agent_id}] WARNING: could not terminate pod {pod_id}: {e}")


if __name__ == "__main__":
    # Init Weave tracing for the dispatcher itself
    _wandb_key = os.environ.get("WANDB_API_KEY")
    if _wandb_key and not os.environ.get("WEAVE_DISABLED"):
        try:
            weave.init(os.environ.get("WEAVE_PROJECT", "autorl"))
        except Exception as _e:
            print(f"[weave] init skipped ({_e})")

    # Quick smoke test: create one entry and run it
    entry = SpawnPlanEntry(
        id="agent_smoke",
        algo="GRPO",
        env="Countdown",
        exec="runpod",
        time_budget_min=20,
        hparams={
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "lr": 1e-6,
            "num_generations": 4,
            "temperature": 1.0,
            "seed": 42,
        },
    )
    result = run_grpo_on_runpod(entry, local_results_dir="results")
    print(json.dumps(result, indent=2))
