"""Apply a user-designed reward_fn on top of a Gymnasium environment."""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np


def load_reward_fn(path: str):
    """Load ``reward_fn`` from a Python file written by the reward designer."""
    ns: dict = {"np": np}
    with open(path) as f:
        code = f.read()
    exec(compile(code, path, "exec"), ns)
    fn = ns.get("reward_fn")
    if not callable(fn):
        raise RuntimeError(f"{path} must define callable reward_fn(...)")
    return fn


class CustomRewardWrapper(gym.Wrapper):
    """Replace step reward with reward_fn(obs, action, reward, terminated, truncated, info)."""

    def __init__(self, env: gym.Env, reward_fn):
        super().__init__(env)
        self._reward_fn = reward_fn

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        try:
            shaped = float(
                self._reward_fn(
                    np.asarray(obs, dtype=np.float32),
                    action,
                    float(reward),
                    bool(terminated),
                    bool(truncated),
                    info if isinstance(info, dict) else {},
                )
            )
        except Exception as exc:
            raise RuntimeError(f"custom reward_fn failed: {exc}") from exc
        if not np.isfinite(shaped):
            shaped = float(reward)
        return obs, shaped, terminated, truncated, info


def maybe_wrap_reward(env: gym.Env) -> gym.Env:
    """Wrap env if AUTORL_REWARD_FN_PATH points to an existing reward file."""
    path = os.environ.get("AUTORL_REWARD_FN_PATH", "")
    if path and os.path.isfile(path):
        print(f"[reward_wrapper] applying custom reward from {path}")
        return CustomRewardWrapper(env, load_reward_fn(path))
    return env
