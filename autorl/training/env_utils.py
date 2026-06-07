"""Env creation helpers shared by all training scripts and the renderer.

Handles special cases:
  - Tuple obs spaces (Blackjack-v1): wrapped with FlattenObservation → Box
  - Image obs spaces (CarRacing-v3): auto-promotes MlpPolicy → CnnPolicy
  - WorldModel-v0: creates WorldModelEnv from WORLD_MODEL_CHECKPOINT /
    WORLD_MODEL_META environment variables (set by training_agent.py)

Both training (no render_mode) and inference (render_mode="rgb_array") use
the same wrapper stack so observations match the saved model's expectation.
"""

from __future__ import annotations

import os

import gymnasium
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation

_WORLD_MODEL_ENV_ID = "WorldModel-v0"


def _make_world_model_env() -> gymnasium.Env:
    """Instantiate WorldModelEnv from env-var paths set by training_agent.py."""
    checkpoint = os.environ.get("WORLD_MODEL_CHECKPOINT", "")
    meta_path  = os.environ.get("WORLD_MODEL_META", "")
    if not checkpoint or not meta_path:
        raise RuntimeError(
            "WorldModel-v0 requires WORLD_MODEL_CHECKPOINT and WORLD_MODEL_META "
            "environment variables to be set."
        )
    # Lazy imports — only needed when actually using the world model
    from agents.dataset_inspector_agent import DatasetMeta
    from training.world_model_env import WorldModelEnv

    meta = DatasetMeta.model_validate_json(open(meta_path).read())
    return WorldModelEnv(checkpoint, meta)


def make_env(env_id: str, render_mode: str | None = None) -> gymnasium.Env:
    """Create a Gymnasium env with any required compatibility wrappers applied.

    Args:
        env_id: Gymnasium environment id string.
        render_mode: Passed to ``gymnasium.make`` (use ``"rgb_array"`` for
            video rendering, ``None`` for training).
    """
    if env_id == _WORLD_MODEL_ENV_ID:
        return _make_world_model_env()

    kwargs: dict = {}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    try:
        env = gymnasium.make(env_id, **kwargs)
    except gymnasium.error.DependencyNotInstalled as exc:
        raise RuntimeError(
            f"Environment '{env_id}' requires an optional package: {exc}\n"
            "Install it (e.g. 'pip install swig && pip install gymnasium[box2d]') and retry."
        ) from exc

    # Tuple obs (e.g. Blackjack-v1): flatten to a 1-D Box so SB3 can handle it.
    # IMPORTANT: must be applied identically at training time AND inference time
    # so the model's expected obs shape matches what the env produces.
    if isinstance(env.observation_space, spaces.Tuple):
        env = FlattenObservation(env)

    return env


def resolve_policy(env: gymnasium.Env, requested: str) -> str:
    """Return the appropriate SB3 policy class name for this env.

    If the caller already explicitly set CnnPolicy (or MultiInputPolicy),
    honour that. Otherwise auto-promote MlpPolicy to CnnPolicy when the
    observation space is an image (3-D Box with channel dim ≤ 4).
    """
    if requested != "MlpPolicy":
        return requested

    obs_space = env.observation_space
    if (
        isinstance(obs_space, spaces.Box)
        and obs_space.shape is not None
        and len(obs_space.shape) == 3
        and obs_space.shape[2] in (1, 3, 4)
    ):
        print(
            f"[env_utils] image obs detected {obs_space.shape} — "
            "using CnnPolicy instead of MlpPolicy"
        )
        return "CnnPolicy"

    return requested
