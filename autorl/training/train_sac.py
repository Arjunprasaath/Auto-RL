"""SAC training on MuJoCo (continuous-action envs only). Off-policy; replay warmup."""

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

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

from training.callbacks.heartbeat_writer import HeartbeatWriter
from training.callbacks.weave_callback import WeaveLogCallback
from training.wandb_setup import finish_wandb_run, start_wandb_run

ALGO = "SAC"
CHUNK = 5000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent-id", required=True)
    p.add_argument("--env-id", default="HalfCheetah-v5")
    p.add_argument("--time-budget", type=int, default=600)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--device", default=os.environ.get("AUTORL_SB3_DEVICE", "cpu"))
    a = p.parse_args()

    os.makedirs(f"{a.results_dir}/{a.agent_id}", exist_ok=True)
    hb = HeartbeatWriter(a.agent_id, a.results_dir)
    hb.start()
    run, tb_log, wandb_cb = start_wandb_run(a.agent_id, ALGO, a.env_id, a.lr, a.seed, a.results_dir)

    model = SAC("MlpPolicy", a.env_id, learning_rate=a.lr, buffer_size=100_000,
                learning_starts=1000, seed=a.seed, verbose=0, tensorboard_log=tb_log,
                device=a.device)
    cb = WeaveLogCallback(a.agent_id)
    callback = CallbackList([cb, wandb_cb]) if wandb_cb else cb

    start = time.time()
    steps = 0
    try:
        while time.time() - start < a.time_budget:
            model.learn(CHUNK, callback=callback, reset_num_timesteps=False, tb_log_name=ALGO)
            steps += CHUNK
            hb.update(steps, cb.ep_returns[-1] if cb.ep_returns else 0.0)
            nudge = hb.check_nudge()
            if nudge:
                model.policy.optimizer.param_groups[0]["lr"] = nudge.get("lr", a.lr)
                print(f"[{a.agent_id}] nudged lr={nudge.get('lr')}")
    except Exception as exc:
        print(f"[{a.agent_id}] training error: {exc}")
        hb.update(steps, cb.ep_returns[-1] if cb.ep_returns else 0.0, loss=float("nan"))
        hb.stop("failed")
        finish_wandb_run(run)
        sys.exit(1)

    mean_r, std_r = evaluate_policy(model, model.get_env(), n_eval_episodes=20)
    ckpt = f"{a.results_dir}/{a.agent_id}/model.zip"
    model.save(ckpt)

    with open(f"{a.results_dir}/{a.agent_id}/eval_result.json", "w") as f:
        json.dump({
            "agent_id": a.agent_id, "algo": ALGO, "env": a.env_id, "status": "completed",
            "mean_return": float(mean_r), "std_return": float(std_r), "steps_trained": steps,
            "wall_time_s": time.time() - start, "weave_run_id": run.id if run else "",
            "checkpoint_path": ckpt,
        }, f)

    hb.stop("completed")
    finish_wandb_run(run)
    print(f"[{a.agent_id}] done: mean_return={mean_r:.1f} ±{std_r:.1f}")


if __name__ == "__main__":
    main()
