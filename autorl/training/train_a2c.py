"""A2C training on MuJoCo. Synchronous actor-critic; lands between PPO and SAC."""

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

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

from training.callbacks.heartbeat_writer import HeartbeatWriter
from training.callbacks.weave_callback import WeaveLogCallback
from training.env_utils import make_env, resolve_policy
from training.wandb_setup import (
    find_warm_start_checkpoint,
    finish_wandb_run,
    log_model_artifact,
    start_wandb_run,
)

ALGO = "A2C"
CHUNK = 5000

STAGNATION_LIMIT = 8
DROPOUT_SCHEDULE = [(0.33, 0.20), (0.66, 0.40)]
DROPOUT_WARMUP_CHUNKS = 4


def _race_dropout_check(
    coordinator,
    run_id: str,
    agent_id: str,
    progress: float,
    current_reward: float,
    checked: dict,
) -> bool:
    for threshold, peer_fraction in DROPOUT_SCHEDULE:
        if checked.get(threshold):
            continue
        if progress >= threshold:
            checked[threshold] = True
            try:
                best_peer = coordinator.get_best_peer_reward(run_id, agent_id)
                if best_peer is not None and best_peer > 0 and current_reward < peer_fraction * best_peer:
                    print(
                        f"[{agent_id}] race dropout at {threshold:.0%} budget: "
                        f"my_reward={current_reward:.1f} vs best_peer={best_peer:.1f} "
                        f"(threshold={peer_fraction:.0%})"
                    )
                    return True
            except Exception:  # noqa: BLE001
                pass
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent-id", required=True)
    p.add_argument("--env-id", default="HalfCheetah-v5")
    p.add_argument("--time-budget", type=int, default=600)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--device",   default=os.environ.get("AUTORL_SB3_DEVICE", "cpu"))
    p.add_argument("--n-steps",  type=int,   default=5)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--gamma",    type=float, default=0.99)
    p.add_argument("--policy",   type=str,   default="MlpPolicy")
    a = p.parse_args()

    os.makedirs(f"{a.results_dir}/{a.agent_id}", exist_ok=True)
    hb = HeartbeatWriter(a.agent_id, a.results_dir)
    hb.start()
    run, tb_log, wandb_cb = start_wandb_run(a.agent_id, ALGO, a.env_id, a.lr, a.seed, a.results_dir)

    env = make_env(a.env_id)
    policy = resolve_policy(env, a.policy)

    # (3) Create model with desired hparams; optionally warm-start policy weights
    model = A2C(
        policy, env,
        learning_rate=a.lr, n_steps=a.n_steps,
        ent_coef=a.ent_coef, gamma=a.gamma,
        seed=a.seed, verbose=0,
        tensorboard_log=tb_log, device=a.device,
    )
    warm_ckpt = find_warm_start_checkpoint(ALGO, a.env_id, a.results_dir)
    if warm_ckpt:
        try:
            warm_model = A2C.load(warm_ckpt, env=env, device=a.device)
            model.policy.load_state_dict(warm_model.policy.state_dict())
            del warm_model
            print(f"[{a.agent_id}] warm-started policy weights from {warm_ckpt}")
        except Exception as e:  # noqa: BLE001
            print(f"[{a.agent_id}] warm-start failed ({e}), training from scratch")

    cb = WeaveLogCallback(a.agent_id)
    callback = CallbackList([cb, wandb_cb]) if wandb_cb else cb

    ckpt = f"{a.results_dir}/{a.agent_id}/model.zip"
    best_reward = -float("inf")
    SAVE_EVERY = 10_000

    try:
        from coordination.redis_coordinator import coordinator as _coord
    except Exception:
        _coord = None
    run_id = os.path.basename(a.results_dir)
    dropout_checked: dict = {}
    stagnation_counter = 0
    dropout_reason: str | None = None

    start = time.time()
    steps = 0
    try:
        while time.time() - start < a.time_budget:
            model.learn(CHUNK, callback=callback, reset_num_timesteps=False, tb_log_name=ALGO)
            steps += CHUNK
            current_reward = cb.ep_returns[-1] if cb.ep_returns else 0.0
            hb.update(steps, current_reward)

            nudge = hb.check_nudge()
            if nudge:
                model.policy.optimizer.param_groups[0]["lr"] = nudge.get("lr", a.lr)
                print(f"[{a.agent_id}] nudged lr={nudge.get('lr')}")

            if current_reward > best_reward:
                best_reward = current_reward
                stagnation_counter = 0
                if steps % SAVE_EVERY == 0 or steps <= CHUNK * 2:
                    model.save(ckpt)
                    print(f"[{a.agent_id}] ✓ best checkpoint @ step {steps}: reward={current_reward:.1f}")
            else:
                stagnation_counter += 1

            # (1) Early stopping
            if steps >= DROPOUT_WARMUP_CHUNKS * CHUNK and stagnation_counter >= STAGNATION_LIMIT:
                print(f"[{a.agent_id}] early stopping: no improvement for {STAGNATION_LIMIT} chunks")
                dropout_reason = "early_stopped"
                break

            # (4/5) Hyperband / race dropout
            if _coord is not None and steps >= DROPOUT_WARMUP_CHUNKS * CHUNK:
                elapsed = time.time() - start
                progress = elapsed / a.time_budget
                if _race_dropout_check(_coord, run_id, a.agent_id, progress, current_reward, dropout_checked):
                    dropout_reason = "race_dropout"
                    break

    except Exception as exc:
        print(f"[{a.agent_id}] training error: {exc}")
        hb.update(steps, cb.ep_returns[-1] if cb.ep_returns else 0.0, loss=float("nan"))
        hb.stop("failed")
        finish_wandb_run(run)
        sys.exit(1)

    mean_r, std_r = evaluate_policy(model, model.get_env(), n_eval_episodes=20)
    if mean_r > best_reward:
        model.save(ckpt)

    final_status = dropout_reason if dropout_reason else "completed"

    log_model_artifact(run, ckpt, a.agent_id, {
        "algo": ALGO, "env": a.env_id, "lr": a.lr, "seed": a.seed,
        "mean_return": float(mean_r), "steps_trained": steps,
        "status": final_status,
    })

    with open(f"{a.results_dir}/{a.agent_id}/eval_result.json", "w") as f:
        json.dump({
            "agent_id": a.agent_id, "algo": ALGO, "env": a.env_id,
            "status": final_status,
            "mean_return": float(mean_r), "std_return": float(std_r),
            "steps_trained": steps,
            "wall_time_s": time.time() - start,
            "weave_run_id": run.id if run else "",
            "checkpoint_path": ckpt,
            "warm_started": warm_ckpt is not None,
        }, f)

    hb.stop(final_status)
    finish_wandb_run(run, mean_return=float(mean_r), std_return=float(std_r), checkpoint=ckpt)
    print(f"[{a.agent_id}] done ({final_status}): mean_return={mean_r:.1f} ±{std_r:.1f}")


if __name__ == "__main__":
    main()
