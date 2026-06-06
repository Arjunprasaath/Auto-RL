"""
SAC training script — runs locally on MuJoCo (Person A).

Off-policy actor-critic with a replay buffer. Typically outperforms PPO on
MuJoCo locomotion within the same time budget, so it should win the local race.

Same data contracts as train_ppo.py:
  - heartbeat.json (via HeartbeatWriter, owned by Person B) for the Sentinel
  - honours Sentinel nudges (results/{agent_id}/nudge.json) mid-run
  - writes results/{agent_id}/eval_result.json on completion (EvalResult schema)

Note: SAC is continuous-action only, so it stays on MuJoCo / Box-action envs.

Run from the autorl/ package root, e.g.:
    python training/train_sac.py --agent-id test_sac --env-id HalfCheetah-v5 \
        --time-budget 120 --lr 3e-4 --seed 42
"""

import argparse
import json
import os
import sys
import time

# Allow `from training...` / `from orchestrator...` regardless of CWD by putting
# the package root (autorl/) on sys.path when run as a script.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

# Load autorl/.env so WANDB_API_KEY / WEAVE_PROJECT are available to weave.init.
from dotenv import load_dotenv

load_dotenv(os.path.join(_PKG_ROOT, ".env"))

import weave
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

# HeartbeatWriter is owned/provided by Person B (training/callbacks/heartbeat_writer.py).
# This import is the agreed contract; the script runs once their module is present.
from training.callbacks.heartbeat_writer import HeartbeatWriter
from training.callbacks.weave_callback import WeaveLogCallback


def init_weave():
    """Initialize Weave/W&B tracing.

    Reads WEAVE_PROJECT (default "autorl") and authenticates via WANDB_API_KEY
    (both loaded from autorl/.env). Set WEAVE_DISABLED=1 to skip tracing for
    quick local runs. Failures are non-fatal so training still proceeds.
    """
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
    except Exception as e:  # noqa: BLE001 - local resilience
        print(f"[weave] init skipped ({e})")
        return None


@weave.op(name="SAC_Training")
def train_sac(agent_id, env_id, time_budget, lr, seed, results_dir):
    """Time-budgeted SAC training. Traced as a Weave op when tracing is on."""
    os.makedirs(f"{results_dir}/{agent_id}", exist_ok=True)

    hb = HeartbeatWriter(agent_id, results_dir)
    hb.start()

    # SAC is off-policy: replay buffer + warmup before learning begins.
    model = SAC(
        "MlpPolicy",
        env_id,
        learning_rate=lr,
        buffer_size=100_000,
        learning_starts=1000,  # reward ~0 for first ~1000 steps — expected
        seed=seed,
        verbose=0,
    )
    cb = WeaveLogCallback(agent_id)

    start = time.time()
    total_steps = 0
    CHUNK = 5000

    while time.time() - start < time_budget:
        model.learn(total_timesteps=CHUNK, callback=cb, reset_num_timesteps=False)
        total_steps += CHUNK

        last_r = cb.ep_returns[-1] if cb.ep_returns else 0.0
        hb.update(total_steps, last_r, loss=None)

        # Check for a Sentinel nudge and apply new hyperparameters live.
        nudge = hb.check_nudge()
        if nudge:
            new_lr = nudge.get("lr", lr)
            model.policy.optimizer.param_groups[0]["lr"] = new_lr
            print(f"[{agent_id}] Nudged: lr={new_lr}")

    mean_r, std_r = evaluate_policy(model, model.get_env(), n_eval_episodes=20)

    ckpt = f"{results_dir}/{agent_id}/model.zip"
    model.save(ckpt)

    # Record the Weave call id (if tracing) so the Evaluator can link back.
    weave_run_id = ""
    try:
        call = weave.get_current_call()
        if call is not None:
            weave_run_id = str(call.id)
    except Exception:  # noqa: BLE001
        pass

    result = {
        "agent_id": agent_id,
        "algo": "SAC",
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
    parser.add_argument("--time-budget", type=int, default=600)  # seconds
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="./results")
    args = parser.parse_args()

    init_weave()
    train_sac(
        agent_id=args.agent_id,
        env_id=args.env_id,
        time_budget=args.time_budget,
        lr=args.lr,
        seed=args.seed,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
