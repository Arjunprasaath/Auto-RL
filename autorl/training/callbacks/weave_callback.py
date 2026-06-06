"""
WeaveLogCallback — Stable-Baselines3 callback that tracks episode returns and
logs progress (intended to push to Weave).

NOTE: This is a Person A local stub so that train_ppo.py / train_sac.py are
testable end-to-end before Person B's real implementation lands. It mirrors the
exact interface defined in the Person B Build Guide (Phase 1.2):

    cb = WeaveLogCallback(agent_id)
    model.learn(total_timesteps=CHUNK, callback=cb, reset_num_timesteps=False)
    last_r = cb.ep_returns[-1] if cb.ep_returns else 0.0

Swapping in Person B's version requires no changes to the training scripts.
"""

from stable_baselines3.common.callbacks import BaseCallback


class WeaveLogCallback(BaseCallback):
    def __init__(self, agent_id: str, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.agent_id = agent_id
        self.log_freq = log_freq
        self.ep_returns: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_returns.append(info["episode"]["r"])

        if self.num_timesteps % self.log_freq == 0 and self.ep_returns:
            recent = self.ep_returns[-10:]
            mean_r = sum(recent) / len(recent)
            print(f"[{self.agent_id}] step={self.num_timesteps} return={mean_r:.1f}")
        return True
