"""Gymnasium environment backed by a trained neural world model.

Provides a standard gym.Env interface so SB3 (PPO, SAC, A2C) can train
inside the learned simulator without ever touching a real environment.

Architecture
────────────
WorldModelMLP: MLP with three output heads
  Input  : [obs (obs_dim), action_encoded]
  Outputs: next_obs (obs_dim), reward (scalar), done_logit (scalar)

WorldModelEnv: wraps the MLP in a gym.Env
  reset()  → sample a random initial state from the stored initial-states parquet
  step(a)  → one forward pass through the MLP
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces


# ── Neural network ─────────────────────────────────────────────────────────────


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "elu":  nn.ELU,
}


class WorldModelMLP(nn.Module):
    """MLP dynamics model: f(obs, action) → (next_obs, reward, done_logit)."""

    def __init__(
        self,
        in_dim: int,
        obs_dim: int,
        hidden_sizes: list[int],
        activation: str = "silu",
        dropout: float = 0.0,
    ):
        super().__init__()
        act_cls = _ACTIVATIONS.get(activation.lower(), nn.SiLU)
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), act_cls()]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.next_obs_head = nn.Linear(prev, obs_dim)
        self.reward_head   = nn.Linear(prev, 1)
        self.done_head     = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return (
            self.next_obs_head(h),
            self.reward_head(h).squeeze(-1),
            self.done_head(h).squeeze(-1),
        )


# ── Gymnasium environment ─────────────────────────────────────────────────────


class WorldModelEnv(gym.Env):
    """Gym env backed by a learned dynamics model checkpoint."""

    metadata = {"render_modes": []}

    def __init__(self, checkpoint_path: str, meta):
        """
        Args:
            checkpoint_path: path to ``wm_checkpoint.pt`` saved by train_world_model.py
            meta: DatasetMeta (or dict with the same fields)
        """
        super().__init__()

        # Accept DatasetMeta or plain dict
        if hasattr(meta, "model_dump"):
            meta_dict = meta.model_dump()
        else:
            meta_dict = dict(meta)

        self._meta_dict = meta_dict
        self._obs_dim   = meta_dict["obs_dim"]
        self._act_type  = meta_dict["act_type"]
        self._act_n     = meta_dict.get("act_n")
        self._act_dim   = meta_dict["act_dim"]

        self._model, self._in_dim = self._load_model(checkpoint_path)
        self._initial_states      = self._load_initial_states(meta_dict["initial_states_path"])

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_dim,), dtype=np.float32,
        )
        if self._act_type == "discrete" and self._act_n:
            self.action_space = spaces.Discrete(self._act_n)
        else:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(self._act_dim,), dtype=np.float32,
            )

        self._obs = np.zeros(self._obs_dim, dtype=np.float32)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _load_model(self, path: str):
        ckpt         = torch.load(path, map_location="cpu", weights_only=False)
        in_dim       = ckpt["in_dim"]
        meta         = ckpt["meta"]
        # Prefer the hidden_sizes actually used during training (may differ from
        # meta["hidden_sizes"] when the planner overrode the default architecture).
        hidden_sizes = ckpt.get("hidden_sizes", meta["hidden_sizes"])
        model        = WorldModelMLP(
            in_dim,
            meta["obs_dim"],
            hidden_sizes,
            activation=ckpt.get("activation", "silu"),
            # Reconstruct identical architecture; model.eval() disables dropout at inference
            dropout=ckpt.get("dropout", 0.0),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, in_dim

    def _load_initial_states(self, path: str | None) -> np.ndarray:
        if path and os.path.exists(path):
            try:
                import pandas as pd
                return pd.read_parquet(path).values.astype(np.float32)
            except Exception:
                pass
        return np.zeros((1, self._obs_dim), dtype=np.float32)

    def _encode_action(self, action) -> np.ndarray:
        if self._act_type == "discrete" and self._act_n:
            oh = np.zeros(self._act_n, dtype=np.float32)
            oh[int(action)] = 1.0
            return oh
        return np.array(action, dtype=np.float32).flatten()

    # ── Gym interface ────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        idx       = self.np_random.integers(0, len(self._initial_states))
        self._obs = self._initial_states[idx].copy()
        return self._obs.copy(), {}

    def step(self, action):
        act_enc = self._encode_action(action)
        x       = np.concatenate([self._obs, act_enc]).astype(np.float32)

        with torch.no_grad():
            xt            = torch.from_numpy(x).unsqueeze(0)
            p_obs, p_rew, p_done = self._model(xt)

        next_obs   = p_obs.squeeze(0).numpy()
        reward     = float(p_rew.item())
        done_prob  = float(torch.sigmoid(p_done).item())
        terminated = done_prob > 0.5

        # Prevent observation runaway in early training
        next_obs   = np.clip(next_obs, -10.0, 10.0)
        self._obs  = next_obs

        return next_obs.copy(), reward, terminated, False, {}

    def render(self):
        return None
