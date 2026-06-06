"""A2C training on MuJoCo. Synchronous actor-critic; lands between PPO and SAC."""

import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(_PKG_ROOT, ".env"))

from stable_baselines3 import A2C

from training.runner import parse_args, run_training


def train_a2c(agent_id, env_id, time_budget, lr, seed, results_dir):
    def make_model(tb_log):
        return A2C("MlpPolicy", env_id, learning_rate=lr, seed=seed, verbose=0,
                   tensorboard_log=tb_log)

    return run_training("A2C", make_model, agent_id, env_id, time_budget, lr, seed, results_dir)


if __name__ == "__main__":
    a = parse_args()
    train_a2c(a.agent_id, a.env_id, a.time_budget, a.lr, a.seed, a.results_dir)
