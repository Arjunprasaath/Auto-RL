"""
RunPod pod lifecycle manager for AutoRL.

Provides: create, wait, SSH exec, install deps, verify, and terminate.
Used by pod_manager/runpod_agent.py to run GRPO training remotely.
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

IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

GPU_FALLBACK_ORDER = [
    "NVIDIA H100 80GB HBM3",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A6000",
    "NVIDIA L40",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A5000",
]

VENV_PATH = "/workspace/venv"
VENV_PIP = f"{VENV_PATH}/bin/pip"
VENV_PYTHON = f"{VENV_PATH}/bin/python"


def create_training_pod(name: str = "autorl-countdown") -> str:
    """Provision a GPU pod, trying multiple GPU types if preferred is unavailable."""
    for gpu in GPU_FALLBACK_ORDER:
        try:
            pod = runpod.create_pod(
                name=name,
                image_name=IMAGE,
                gpu_type_id=gpu,
                gpu_count=1,
                container_disk_in_gb=40,
                volume_in_gb=40,
                ports="22/tcp",
            )
            pod_id = pod["id"]
            print(f"[Pod] Created: {pod_id} (GPU: {gpu})")
            return pod_id
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
    """SSH into pod, run command, return stdout. Raises on non-zero exit."""
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


def install_dependencies(pod_id: str) -> None:
    """
    Create a clean venv and install a CUDA-enabled torch explicitly from the
    cu128 wheel index (matching the image's CUDA 12.8.1), then the rest of the
    training deps. We do NOT use --system-site-packages because pip would then
    shadow the GPU torch with a CPU-only wheel, silently running training on CPU.
    """
    print("[Pod] Creating virtual environment...")
    ssh_exec(pod_id, f"python3 -m venv {VENV_PATH}")

    print("[Pod] Installing CUDA torch (cu128)...")
    ssh_exec(pod_id, (
        f"{VENV_PIP} install -q torch torchvision "
        "--index-url https://download.pytorch.org/whl/cu128"
    ))

    print("[Pod] Installing training dependencies...")
    ssh_exec(pod_id, (
        f"{VENV_PIP} install -q "
        "trl transformers peft datasets accelerate bitsandbytes "
        "weave wandb pydantic huggingface-hub[hf_transfer]"
    ))

    print("[Pod] Dependencies installed.")


def verify_dependencies(pod_id: str) -> bool:
    """
    Verify all required packages import cleanly AND that CUDA is available.
    Fails if torch can't see the GPU (which would silently run training on CPU).
    """
    print("[Pod] Verifying imports and CUDA...")
    cmd = (
        f'{VENV_PYTHON} -c "'
        'import torch, trl, transformers, datasets, accelerate, peft, '
        'bitsandbytes, weave, wandb, pydantic; '
        'assert torch.cuda.is_available(), \\"CUDA NOT AVAILABLE\\"; '
        'print(f\\"torch={torch.__version__} cuda={torch.cuda.is_available()} '
        'device={torch.cuda.get_device_name(0)}\\"); '
        "print('ALL_DEPS_OK')\""
    )
    try:
        output = ssh_exec(pod_id, cmd)
        if "ALL_DEPS_OK" in output:
            print(f"[Pod] All dependencies verified. {output.strip()}")
            return True
        print(f"[Pod] Unexpected output:\n{output}")
        return False
    except RuntimeError as e:
        print(f"[Pod] Verification failed:\n{e}")
        return False


def terminate_pod(pod_id: str) -> None:
    """Terminate the pod."""
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
    else:
        print(f"Verification failed — pod {pod_id} left running for manual debug.")
