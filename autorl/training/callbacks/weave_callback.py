"""
WeaveLogCallback — SB3 callback that tracks episode returns, episode lengths,
and SB3 logger metrics for the Doom Loop Sentinel.

Usage:
    cb = WeaveLogCallback(agent_id="agent_1")
    model.learn(total_timesteps=5000, callback=cb, reset_num_timesteps=False)

    # After each model.learn():
    last_reward   = cb.ep_returns[-1] if cb.ep_returns else 0.0
    sb3_metrics   = cb.get_sb3_metrics()   # explained_variance, entropy, etc.
    ep_lengths    = cb.ep_lengths           # full history, for regression detection
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
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_returns.append(float(info["episode"]["r"]))
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
        """Read SB3 logger values written during the last model.learn() call.

        PPO/A2C keys: train/explained_variance, train/entropy_loss, train/approx_kl
        SAC keys:     train/critic_loss, train/actor_loss, train/ent_coef
        Returns None for keys not present (algo-dependent).
        """
        lv: dict = {}
        try:
            lv = self.model.logger.name_to_value  # type: ignore[union-attr]
        except AttributeError:
            pass
        return {
            "explained_variance": lv.get("train/explained_variance"),
            "entropy_loss":       lv.get("train/entropy_loss"),
            "approx_kl":          lv.get("train/approx_kl"),
            "value_loss":         lv.get("train/value_loss"),
            "critic_loss":        lv.get("train/critic_loss"),
        }

    def _log_to_weave(self, mean_return: float) -> None:
        try:
            ep_len_window = self.ep_lengths[-10:]
            payload: dict = {
                "agent_id":       self.agent_id,
                "step":           self.num_timesteps,
                "mean_return":    mean_return,
                "ep_length_mean": sum(ep_len_window) / max(1, len(ep_len_window)),
            }
            # Include all SB3 logger metrics that are available for this algo
            # (PPO/A2C: explained_variance, entropy_loss, approx_kl, value_loss;
            #  SAC: critic_loss, actor_loss, ent_coef)
            payload.update(
                {k: v for k, v in self.get_sb3_metrics().items() if v is not None}
            )
            weave.log(payload)
        except Exception:
            pass
