"""Resolve compute device for local training (MPS on Apple Silicon)."""

import os


def mps_available() -> bool:
    try:
        import torch

        return torch.backends.mps.is_available()
    except ImportError:
        return False


def resolve_sb3_device() -> str:
    """MuJoCo SB3 (MlpPolicy): CPU — MPS lacks float64; SB3 recommends CPU for MLP."""
    return os.environ.get("AUTORL_SB3_DEVICE", "cpu")


def resolve_grpo_device() -> str:
    """Countdown GRPO: MPS when available (LLM benefits from Apple GPU)."""
    if override := os.environ.get("AUTORL_GRPO_DEVICE"):
        return override
    if mps_available():
        return "mps"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def resolve_device() -> str:
    """Default device (GRPO path). Prefer AUTORL_DEVICE override."""
    if override := os.environ.get("AUTORL_DEVICE"):
        return override
    return resolve_grpo_device()


def is_mps() -> bool:
    return mps_available()


def subprocess_env() -> dict[str, str]:
    return {
        **os.environ,
        "AUTORL_SB3_DEVICE": resolve_sb3_device(),
        "AUTORL_GRPO_DEVICE": resolve_grpo_device(),
        "AUTORL_DEVICE": resolve_grpo_device(),
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    }
