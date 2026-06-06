"""Shared training loop for the local MuJoCo scripts (PPO / SAC).

Handles: heartbeat thread, W&B per-step charts, Sentinel nudges, time-budgeted
training, evaluation, and writing eval_result.json (EvalResult schema).
"""

import argparse
import json
import os
import time

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

from training.callbacks.heartbeat_writer import HeartbeatWriter
from training.callbacks.weave_callback import WeaveLogCallback
from training.wandb_setup import finish_wandb_run, start_wandb_run

CHUNK = 5000


def run_training(algo, make_model, agent_id, env_id, time_budget, lr, seed, results_dir):
    """Run `make_model(tb_log)` for `time_budget` seconds and write results."""
    os.makedirs(f"{results_dir}/{agent_id}", exist_ok=True)

    hb = HeartbeatWriter(agent_id, results_dir)
    hb.start()
    run, tb_log, wandb_cb = start_wandb_run(agent_id, algo, env_id, lr, seed, results_dir)

    model = make_model(tb_log)
    cb = WeaveLogCallback(agent_id)
    callback = CallbackList([cb, wandb_cb]) if wandb_cb else cb

    start = time.time()
    steps = 0
    while time.time() - start < time_budget:
        model.learn(CHUNK, callback=callback, reset_num_timesteps=False, tb_log_name=algo)
        steps += CHUNK
        hb.update(steps, cb.ep_returns[-1] if cb.ep_returns else 0.0)
        nudge = hb.check_nudge()
        if nudge:
            model.policy.optimizer.param_groups[0]["lr"] = nudge.get("lr", lr)
            print(f"[{agent_id}] nudged lr={nudge.get('lr')}")

    mean_r, std_r = evaluate_policy(model, model.get_env(), n_eval_episodes=20)
    ckpt = f"{results_dir}/{agent_id}/model.zip"
    model.save(ckpt)

    result = {
        "agent_id": agent_id, "algo": algo, "env": env_id, "status": "completed",
        "mean_return": float(mean_r), "std_return": float(std_r), "steps_trained": steps,
        "wall_time_s": time.time() - start, "weave_run_id": run.id if run else "",
        "checkpoint_path": ckpt,
    }
    with open(f"{results_dir}/{agent_id}/eval_result.json", "w") as f:
        json.dump(result, f)

    hb.stop("completed")
    finish_wandb_run(run)
    print(f"[{agent_id}] done: mean_return={mean_r:.1f} ±{std_r:.1f}")
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent-id", required=True)
    p.add_argument("--env-id", default="HalfCheetah-v5")
    p.add_argument("--time-budget", type=int, default=600)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", default="./results")
    return p.parse_args()
