"""
A2C training script — runs locally on MuJoCo (Person B).

Synchronous actor-critic. Typically lands between PPO and SAC on MuJoCo
locomotion tasks. Third racer alongside Person A's PPO and SAC.

Time-budgeted training loop that:
  - emits a heartbeat every 60s (via HeartbeatWriter) for the Sentinel
  - honours Sentinel nudges (results/{agent_id}/nudge.json) mid-run
  - writes results/{agent_id}/eval_result.json on completion (EvalResult schema)

Run from the autorl/ package root, e.g.:
    python training/train_a2c.py --agent-id test_a2c --env-id HalfCheetah-v5 \
        --time-budget 120 --lr 3e-4 --seed 42
"""

import argparse
import json
import os
import sys
import time

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(_PKG_ROOT, ".env"))

import weave
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

from training.callbacks.heartbeat_writer import HeartbeatWriter
from training.callbacks.weave_callback import WeaveLogCallback


def init_weave():
    """Initialize Weave/W&B tracing. Non-fatal if keys are missing."""
    if os.environ.get("WEAVE_DISABLED"):
        print("[weave] tracing disabled via WEAVE_DISABLED")
        return None
    if not os.environ.get("WANDB_API_KEY"):
        print("[weave] WANDB_API_KEY not set — tracing skipped")
        return None
    project = os.environ.get("WEAVE_PROJECT", "autorl")
    try:
        client = weave.init(project)
        print(f"[weave] tracing to project '{project}'")
        return client
    except Exception as e:
        print(f"[weave] init skipped ({e})")
        return None


@weave.op(name="A2C_Training")
def train_a2c(agent_id, env_id, time_budget, lr, seed, results_dir):
    """Time-budgeted A2C training. Traced as a Weave op when tracing is on."""
    os.makedirs(f"{results_dir}/{agent_id}", exist_ok=True)

    hb = HeartbeatWriter(agent_id, results_dir)
    hb.start()

    model = A2C("MlpPolicy", env_id, learning_rate=lr, seed=seed, verbose=0)
    cb = WeaveLogCallback(agent_id)

    start = time.time()
    total_steps = 0
    CHUNK = 5000

    while time.time() - start < time_budget:
        model.learn(total_timesteps=CHUNK, callback=cb, reset_num_timesteps=False)
        total_steps += CHUNK

        last_r = cb.ep_returns[-1] if cb.ep_returns else 0.0
        hb.update(total_steps, last_r, loss=None)

        nudge = hb.check_nudge()
        if nudge:
            new_lr = nudge.get("lr", lr)
            model.policy.optimizer.param_groups[0]["lr"] = new_lr
            print(f"[{agent_id}] Nudged: lr={new_lr}")

    mean_r, std_r = evaluate_policy(model, model.get_env(), n_eval_episodes=20)

    ckpt = f"{results_dir}/{agent_id}/model.zip"
    model.save(ckpt)

    weave_run_id = ""
    try:
        call = weave.get_current_call()
        if call is not None:
            weave_run_id = str(call.id)
    except Exception:
        pass

    result = {
        "agent_id": agent_id,
        "algo": "A2C",
        "env": env_id,
        "status": "completed",
        "mean_return": float(mean_r),
        "std_return": float(std_r),
        "steps_trained": total_steps,
        "wall_time_s": time.time() - start,
        "weave_run_id": weave_run_id,
        "checkpoint_path": ckpt,
    }
    with open(f"{results_dir}/{agent_id}/eval_result.json", "w") as f:
        json.dump(result, f)

    hb.stop("completed")
    print(f"[{agent_id}] done: mean_return={mean_r:.1f} ±{std_r:.1f}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--env-id", default="HalfCheetah-v5")
    parser.add_argument("--time-budget", type=int, default=600)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="./results")
    args = parser.parse_args()

    init_weave()
    train_a2c(
        agent_id=args.agent_id,
        env_id=args.env_id,
        time_budget=args.time_budget,
        lr=args.lr,
        seed=args.seed,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
