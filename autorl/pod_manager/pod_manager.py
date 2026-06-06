"""
RunPod pod lifecycle manager for AutoRL.

Provides: create, wait, SSH exec, verify deps, and terminate.
Used by agents/training_agent.py to run GRPO training remotely.
"""

import os
import subprocess
import time
import runpod

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

runpod.api_key = os.environ["RUNPOD_API_KEY"]

POD_ID: str | None = None


GPU_FALLBACK_ORDER = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A6000",
    "NVIDIA L40",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A5000",
]


def create_training_pod(name: str = "autorl-countdown") -> str:
    """
    Provision a GPU pod, trying multiple GPU types if preferred is unavailable.
    Returns pod ID.
    """
    global POD_ID
    for gpu in GPU_FALLBACK_ORDER:
        try:
            pod = runpod.create_pod(
                name=name,
                image_name="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
                gpu_type_id=gpu,
                gpu_count=1,
                container_disk_in_gb=40,
                volume_in_gb=40,
                ports="22/tcp",
            )
            POD_ID = pod["id"]
            print(f"[Pod] Created: {POD_ID} (GPU: {gpu})")
            return POD_ID
        except Exception as e:
            print(f"[Pod] {gpu} unavailable: {e}")
            continue
    raise RuntimeError("No GPU available. Check RunPod dashboard.")


def wait_for_pod(pod_id: str, timeout_s: int = 600, poll_s: int = 10) -> bool:
    """Block until pod is RUNNING. Returns True on success, False on timeout."""
    print(f"[Pod] Waiting for {pod_id}...")
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            pod = runpod.get_pod(pod_id)
            status = pod.get("desiredStatus") or pod.get("status", "")
            if status == "RUNNING":
                print(f"[Pod] {pod_id} is RUNNING")
                return True
            print(f"[Pod] Status: {status}")
        except Exception as e:
            print(f"[Pod] Poll error: {e}")
        time.sleep(poll_s)
    print(f"[Pod] Timeout after {timeout_s}s")
    return False


def get_pod_ssh_info(pod_id: str, retries: int = 18, delay: int = 10) -> tuple[str, int]:
    """
    Returns (host, port) for SSH. Retries until runtime is populated —
    RunPod reports RUNNING before network/ports are assigned.
    """
    for attempt in range(retries):
        pod = runpod.get_pod(pod_id)
        runtime = pod.get("runtime")
        if runtime and runtime.get("ports"):
            for entry in runtime["ports"]:
                if entry.get("privatePort") == 22:
                    ip = entry.get("ip") or runtime.get("ip")
                    port = entry.get("publicPort")
                    if ip and port:
                        return ip, port
        print(f"[Pod] Waiting for SSH info ({attempt + 1}/{retries})...")
        time.sleep(delay)
    raise RuntimeError(f"SSH not available for {pod_id} after {retries * delay}s")


def ssh_exec(pod_id: str, command: str, timeout: int = 7200) -> str:
    """
    SSH into pod, run command, return stdout. Raises on non-zero exit.
    For training commands, use VENV_PYTHON:
        ssh_exec(pod_id, f"{VENV_PYTHON} /workspace/training/train_grpo_countdown.py ...")
    """
    host, port = get_pod_ssh_info(pod_id)
    result = subprocess.run(
        [
            "ssh", "-p", str(port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=30",
            f"root@{host}", command,
        ],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"[Pod] SSH failed (exit {result.returncode}):\n{result.stderr}"
        )
    return result.stdout


def scp_from_pod(pod_id: str, remote_path: str, local_path: str) -> None:
    """Copy a file from pod to local machine."""
    host, port = get_pod_ssh_info(pod_id)
    subprocess.run(
        [
            "scp", "-P", str(port),
            "-o", "StrictHostKeyChecking=no",
            f"root@{host}:{remote_path}", local_path,
        ],
        check=True,
    )


VENV_PATH = "/workspace/venv"
VENV_PIP = f"{VENV_PATH}/bin/pip"
VENV_PYTHON = f"{VENV_PATH}/bin/python"


def install_dependencies(pod_id: str) -> None:
    """
    Create a fresh venv on the pod and install all GRPO training deps.
    Uses a venv to isolate from the broken system packages in the Docker image.
    """
    print("[Pod] Creating virtual environment...")
    ssh_exec(pod_id, f"python -m venv {VENV_PATH}")

    print("[Pod] Installing torch + torchvision...")
    ssh_exec(pod_id, f"{VENV_PIP} install -q torch torchvision")

    print("[Pod] Installing training dependencies...")
    ssh_exec(pod_id, f"{VENV_PIP} install -q trl transformers datasets accelerate peft bitsandbytes weave pydantic")

    print("[Pod] Dependencies installed.")


def verify_dependencies(pod_id: str) -> bool:
    """Verify all required packages import cleanly in the pod venv."""
    print("[Pod] Verifying imports...")
    cmd = (
        f'{VENV_PYTHON} -c "'
        'import torch; '
        'import trl; '
        'import transformers; '
        'import datasets; '
        'import accelerate; '
        'import peft; '
        'import bitsandbytes; '
        'import weave; '
        'import pydantic; '
        "print('ALL_DEPS_OK')\""
    )
    try:
        output = ssh_exec(pod_id, cmd)
        if "ALL_DEPS_OK" in output:
            print("[Pod] All dependencies verified.")
            return True
        print(f"[Pod] Unexpected output:\n{output}")
        return False
    except RuntimeError as e:
        print(f"[Pod] Verification failed:\n{e}")
        return False


def terminate_pod(pod_id: str) -> None:
    """Terminate the pod. Only call after submission is confirmed."""
    runpod.terminate_pod(pod_id)
    print(f"[Pod] {pod_id} terminated.")


if __name__ == "__main__":
    pod_id = create_training_pod()

    if not wait_for_pod(pod_id):
        print("Pod failed to start. Check RunPod dashboard.")
        raise SystemExit(1)

    install_dependencies(pod_id)

    if verify_dependencies(pod_id):
        print("Setup complete. Terminating pod to save cost.")
        terminate_pod(pod_id)
        print("Re-run create_training_pod() when ready for GRPO training.")
    else:
        print(f"Verification failed — pod {pod_id} left running for manual debug.")
        print("Terminate manually when done: terminate_pod(pod_id)")
