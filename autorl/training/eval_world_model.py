"""Evaluate a trained world model on held-out validation transitions.

Runs one-step prediction metrics (obs MSE, reward MAE, done accuracy) and
collects a short open-loop rollout for UI visualization.

Output: {results_dir}/{agent_id}/wm_eval.json

Run:
    python training/eval_world_model.py \
        --checkpoint runs/.../wm_trainer/wm_checkpoint.pt \
        --meta-path    runs/.../wm_trainer/dataset_meta.json \
        --dataset-path /path/to/dataset.parquet \
        --results-dir  runs/.../
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import numpy as np
import torch
import torch.nn as nn

from agents.dataset_inspector_agent import DatasetMeta, load_from_file
from training.train_world_model import build_tensors, SPLIT_SEED, VAL_FRACTION
from training.world_model_env import WorldModelMLP

AGENT_ID = "wm_trainer"
MAX_ROLLOUT_STEPS = 60
MAX_OBS_PLOT = 4


def _val_indices(n: int, val_fraction: float = VAL_FRACTION, seed: int = SPLIT_SEED) -> np.ndarray:
    n_val = max(1, int(n * val_fraction))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).numpy()
    return perm[:n_val]


def evaluate(
    checkpoint_path: str,
    meta_path: str,
    dataset_path: str,
    agent_id: str = AGENT_ID,
    results_dir: str | None = None,
) -> dict:
    meta = DatasetMeta.model_validate_json(open(meta_path).read())
    df = load_from_file(dataset_path)
    X, Y_obs, Y_rew, Y_done = build_tensors(df, meta)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    in_dim = ckpt["in_dim"]
    hidden_sizes = ckpt.get("hidden_sizes", meta.hidden_sizes)
    model = WorldModelMLP(
        in_dim,
        meta.obs_dim,
        hidden_sizes,
        activation=ckpt.get("activation", "silu"),
        dropout=ckpt.get("dropout", 0.0),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    n = len(X)
    i_val = _val_indices(n)
    X_v, obs_v, rew_v, done_v = X[i_val], Y_obs[i_val], Y_rew[i_val], Y_done[i_val]

    mse_fn = nn.MSELoss(reduction="none")
    bce_fn = nn.BCEWithLogitsLoss(reduction="none")

    with torch.no_grad():
        p_obs, p_rew, p_done = model(X_v)
        obs_mse = float(mse_fn(p_obs, obs_v).mean().item())
        rew_mse = float(mse_fn(p_rew, rew_v).mean().item())
        rew_mae = float((p_rew - rew_v).abs().mean().item())
        done_prob = torch.sigmoid(p_done)
        done_acc = float(((done_prob > 0.5) == (done_v > 0.5)).float().mean().item())
        val_loss = obs_mse + rew_mse + float(bce_fn(p_done, done_v).mean().item())

    # One-step rollout slice from validation set (contiguous segment for plotting)
    n_plot = min(MAX_ROLLOUT_STEPS, len(i_val))
    start = len(i_val) // 4
    end = start + n_plot
    n_obs = min(MAX_OBS_PLOT, meta.obs_dim)

    one_step = []
    for j in range(start, end):
        with torch.no_grad():
            po, pr, pd = model(X_v[j : j + 1])
        one_step.append({
            "step": j - start,
            "true_reward": float(rew_v[j].item()),
            "pred_reward": float(pr.item()),
            "true_done": float(done_v[j].item()),
            "pred_done": float(torch.sigmoid(pd).item()),
            "true_obs": obs_v[j, :n_obs].tolist(),
            "pred_obs": po[0, :n_obs].tolist(),
        })

    # Open-loop rollout: predicted state feeds next step (actions from val data)
    open_loop = []
    obs_state = obs_v[start].clone()
    for t in range(n_plot):
        j = start + t
        act_part = X_v[j, meta.obs_dim:]
        x = torch.cat([obs_state.unsqueeze(0), act_part.unsqueeze(0)], dim=1)
        with torch.no_grad():
            po, pr, pd = model(x)
        open_loop.append({
            "step": t,
            "true_reward": float(rew_v[j].item()),
            "pred_reward": float(pr.item()),
            "obs_mse_step": float(mse_fn(po, obs_v[j : j + 1]).mean().item()),
            "true_obs": obs_v[j, :n_obs].tolist(),
            "pred_obs": po[0, :n_obs].tolist(),
        })
        obs_state = po[0].clamp(-10.0, 10.0)

    result = {
        "agent_id": agent_id,
        "n_val_samples": int(len(i_val)),
        "obs_mse": obs_mse,
        "reward_mse": rew_mse,
        "reward_mae": rew_mae,
        "done_accuracy": done_acc,
        "val_loss": val_loss,
        "obs_dims_plotted": n_obs,
        "one_step_rollout": one_step,
        "open_loop_rollout": open_loop,
    }

    if results_dir:
        out_dir = os.path.join(results_dir, agent_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "wm_eval.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[{agent_id}] wm_eval saved → {out_path}")
        print(
            f"[{agent_id}] val: obs_mse={obs_mse:.4f}  rew_mae={rew_mae:.4f}  "
            f"done_acc={done_acc:.1%}  n={len(i_val)}"
        )

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--meta-path", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--agent-id", default=AGENT_ID)
    a = p.parse_args()

    evaluate(
        a.checkpoint,
        a.meta_path,
        a.dataset_path,
        agent_id=a.agent_id,
        results_dir=a.results_dir,
    )


if __name__ == "__main__":
    main()
