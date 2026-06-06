"""PPO training on MuJoCo. Run: python training/train_ppo.py --agent-id a1 ..."""

import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(_PKG_ROOT, ".env"))

from stable_baselines3 import PPO

from training.runner import parse_args, run_training


def train_ppo(agent_id, env_id, time_budget, lr, seed, results_dir):
    def make_model(tb_log):
        return PPO("MlpPolicy", env_id, learning_rate=lr, seed=seed, verbose=0,
                   tensorboard_log=tb_log)

    return run_training("PPO", make_model, agent_id, env_id, time_budget, lr, seed, results_dir)


if __name__ == "__main__":
    a = parse_args()
    train_ppo(a.agent_id, a.env_id, a.time_budget, a.lr, a.seed, a.results_dir)
