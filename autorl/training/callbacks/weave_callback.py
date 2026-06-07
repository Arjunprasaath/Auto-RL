"""
WeaveLogCallback — Stable-Baselines3 callback that logs episode returns.

Imported by all local MuJoCo training scripts:
    train_ppo.py, train_sac.py, train_a2c.py

Tracks episode returns and prints a rolling average every `log_freq` steps.
The `ep_returns` list is read by training scripts to get the latest reward
for heartbeat updates.

Usage:
    cb = WeaveLogCallback(agent_id="agent_1")
    model.learn(total_timesteps=5000, callback=cb, reset_num_timesteps=False)
    last_reward = cb.ep_returns[-1] if cb.ep_returns else 0.0
"""

import weave
from stable_baselines3.common.callbacks import BaseCallback


class WeaveLogCallback(BaseCallback):
    def __init__(self, agent_id: str, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.agent_id = agent_id
        self.log_freq = log_freq
        self.ep_returns: list[float] = []
        self.ep_lengths: list[float] = []

    def _on_step(self) -> bool:
        # SB3 populates infos with episode stats when an episode ends
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_return = float(info["episode"]["r"])
                self.ep_returns.append(ep_return)
                self.ep_lengths.append(float(info["episode"]["l"]))

        if self.num_timesteps % self.log_freq == 0 and self.ep_returns:
            window = self.ep_returns[-10:]
            mean_r = sum(window) / len(window)
            print(
                f"[{self.agent_id}] "
                f"step={self.num_timesteps:>8,} "
                f"ep_return={mean_r:>8.1f} "
                f"(last {len(window)} eps)"
            )
            self._log_to_weave(mean_r)

        return True

    def get_sb3_metrics(self) -> dict:
        """Read latest SB3 training metrics from the logger (PPO / A2C / SAC)."""
        if self.logger is None:
            return {}
        nv = getattr(self.logger, "name_to_value", {}) or {}
        out: dict = {}
        if "train/explained_variance" in nv:
            out["explained_variance"] = float(nv["train/explained_variance"])
        if "train/entropy_loss" in nv:
            out["entropy_loss"] = float(nv["train/entropy_loss"])
        return out

    def _log_to_weave(self, mean_return: float):
        """Push the rolling mean return as a Weave online eval data point."""
        try:
            weave.log({
                "agent_id": self.agent_id,
                "step": self.num_timesteps,
                "mean_return": mean_return,
            })
        except Exception:
            # Weave logging is best-effort — never crash training
            pass
