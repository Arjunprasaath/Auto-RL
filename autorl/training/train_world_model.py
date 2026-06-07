"""Train a neural world model on a user-supplied transition dataset.

Learns f(s, a) → (s′, r, done) via supervised learning on logged data.
Writes heartbeat.json (one per 5 epochs) and eval_result.json.
Saves the best checkpoint to {results_dir}/{agent_id}/wm_checkpoint.pt.

Run:
    python training/train_world_model.py \
        --agent-id wm_trainer \
        --dataset-path /path/to/dataset.parquet \
        --meta-path    /path/to/dataset_meta.json \
        --results-dir  runs/2024-01-01/
"""

from __future__ import annotations

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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from agents.dataset_inspector_agent import DatasetMeta, load_from_file
from training.callbacks.heartbeat_writer import HeartbeatWriter
from training.world_model_env import WorldModelMLP

AGENT_ID = "wm_trainer"


# ── Data preparation ──────────────────────────────────────────────────────────


def _encode_action(actions: np.ndarray, act_type: str, act_n: int | None) -> np.ndarray:
    if act_type == "discrete" and act_n:
        idx = actions.astype(int).flatten()
        oh  = np.eye(act_n, dtype=np.float32)[idx]
        return oh
    return actions.astype(np.float32)


def build_tensors(df, meta: DatasetMeta):
    obs     = df[meta.obs_cols].values.astype(np.float32)
    act_raw = df[meta.act_cols].values
    act_enc = _encode_action(act_raw, meta.act_type, meta.act_n)

    # next_obs: use dedicated columns if available, else shift obs by 1 row
    if meta.next_obs_cols and meta.next_obs_cols != meta.obs_cols:
        try:
            next_obs = df[meta.next_obs_cols].values.astype(np.float32)
        except KeyError:
            next_obs = np.roll(obs, -1, axis=0)
    else:
        next_obs = np.roll(obs, -1, axis=0)
    next_obs[-1] = obs[-1]  # last row wraps — use same obs as fallback

    rew  = df[meta.reward_col].values.astype(np.float32)
    done = df[meta.done_col].astype(float).values.astype(np.float32)

    X = np.concatenate([obs, act_enc], axis=1)
    return (
        torch.from_numpy(X),
        torch.from_numpy(next_obs),
        torch.from_numpy(rew),
        torch.from_numpy(done),
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent-id",      default=AGENT_ID)
    p.add_argument("--dataset-path",  required=True)
    p.add_argument("--meta-path",     required=True)
    p.add_argument("--results-dir",   required=True)
    p.add_argument("--time-budget",   type=int,   default=300, help="seconds")
    p.add_argument("--epochs",        type=int,   default=200)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--batch-size",    type=int,   default=256)
    # Architecture args — override meta.hidden_sizes when provided by the planner
    p.add_argument("--hidden-sizes",  nargs="+",  type=int,   default=None,
                   help="Override world model hidden layer sizes (e.g. 256 256 128)")
    p.add_argument("--activation",    default=None,
                   help="Activation function: relu | silu | tanh | elu")
    p.add_argument("--dropout",       type=float, default=0.0,
                   help="Dropout probability (0 = disabled)")
    a = p.parse_args()

    agent_dir = os.path.join(a.results_dir, a.agent_id)
    os.makedirs(agent_dir, exist_ok=True)

    # Load metadata & dataset
    meta = DatasetMeta.model_validate_json(open(a.meta_path).read())
    hb   = HeartbeatWriter(a.agent_id, a.results_dir)
    hb.start()

    hb.set_extra(total_epochs=a.epochs)   # lets the UI show a real progress bar

    print(f"[{a.agent_id}] loading dataset ({meta.n_samples:,} rows) …")
    df = load_from_file(a.dataset_path)
    X, Y_obs, Y_rew, Y_done = build_tensors(df, meta)

    # 80 / 20 train / val split
    n     = len(X)
    n_val = max(1, int(n * 0.2))
    perm  = torch.randperm(n)
    i_tr, i_val = perm[n_val:], perm[:n_val]
    tr_dl = DataLoader(TensorDataset(X[i_tr], Y_obs[i_tr], Y_rew[i_tr], Y_done[i_tr]),
                       batch_size=a.batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(X[i_val], Y_obs[i_val], Y_rew[i_val], Y_done[i_val]),
                        batch_size=a.batch_size)

    act_enc_dim  = (meta.act_n if (meta.act_type == "discrete" and meta.act_n) else meta.act_dim)
    in_dim       = meta.obs_dim + act_enc_dim
    hidden_sizes = a.hidden_sizes if a.hidden_sizes else meta.hidden_sizes
    activation   = a.activation  if a.activation  else "silu"
    dropout      = a.dropout

    print(f"[{a.agent_id}] arch: hidden={hidden_sizes}  act={activation}  dropout={dropout}")
    model     = WorldModelMLP(in_dim, meta.obs_dim, hidden_sizes, activation, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    mse_fn    = nn.MSELoss()
    bce_fn    = nn.BCEWithLogitsLoss()

    ckpt_path = os.path.join(agent_dir, "wm_checkpoint.pt")
    best_val  = float("inf")
    epoch     = 0
    start     = time.time()

    for epoch in range(1, a.epochs + 1):
        if time.time() - start > a.time_budget:
            print(f"[{a.agent_id}] time budget reached at epoch {epoch}")
            break

        # Training pass
        model.train()
        for xb, yb_obs, yb_rew, yb_done in tr_dl:
            p_obs, p_rew, p_done = model(xb)
            loss = mse_fn(p_obs, yb_obs) + mse_fn(p_rew, yb_rew) + bce_fn(p_done, yb_done)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation pass
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb_obs, yb_rew, yb_done in val_dl:
                p_obs, p_rew, p_done = model(xb)
                val_loss += (
                    mse_fn(p_obs, yb_obs) + mse_fn(p_rew, yb_rew) + bce_fn(p_done, yb_done)
                ).item()
        val_loss /= max(1, len(val_dl))

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "meta":        meta.model_dump(),
                "in_dim":      in_dim,
                "hidden_sizes": hidden_sizes,
                "activation":   activation,
                "dropout":      dropout,
            }, ckpt_path)

        # Heartbeat every 5 epochs: use –val_loss as "reward" so the UI shows improvement
        if epoch % 5 == 0:
            hb.update(epoch, -val_loss, loss=val_loss)
            print(f"[{a.agent_id}] epoch {epoch}/{a.epochs}: val_loss={val_loss:.4f}")

    wall = time.time() - start
    hb.stop("completed")

    result = {
        "agent_id":       a.agent_id,
        "algo":           "WORLD_MODEL",
        "env":            "WorldModel-v0",
        "status":         "completed",
        "mean_return":    -best_val,       # −val_loss: higher is better
        "std_return":     0.0,
        "steps_trained":  epoch,
        "wall_time_s":    wall,
        "weave_run_id":   "",
        "checkpoint_path": ckpt_path,
    }
    with open(os.path.join(agent_dir, "eval_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"[{a.agent_id}] done — best_val_loss={best_val:.4f}  checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
