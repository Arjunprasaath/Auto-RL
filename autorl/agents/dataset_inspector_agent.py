"""Dataset Inspector Agent — LLM-powered analysis of RL transition datasets.

Given a path to a CSV / JSON / parquet file containing transition data
(obs, action, reward, next_obs, done), inspects the schema and returns a
DatasetMeta describing observation/action dimensions, reward range, and
recommended world-model architecture.

Also handles HuggingFace dataset downloads via the `datasets` library.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Literal

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(_PKG_ROOT, ".env"))

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel

_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


# ── Schema ────────────────────────────────────────────────────────────────────


class DatasetMeta(BaseModel):
    obs_cols: list[str]
    act_cols: list[str]
    reward_col: str
    next_obs_cols: list[str]
    done_col: str
    obs_dim: int
    act_dim: int
    act_type: Literal["discrete", "continuous"]
    act_n: int | None = None          # number of discrete actions (discrete only)
    reward_min: float
    reward_max: float
    n_samples: int
    hidden_sizes: list[int]
    dataset_path: str
    initial_states_path: str | None = None
    source_env: str | None = None   # original gym env id (e.g. "Ant-v5"); set when known


# ── File loading ──────────────────────────────────────────────────────────────


def load_from_file(path: str) -> pd.DataFrame:
    """Load CSV, JSONL, JSON, or parquet into a flat DataFrame."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if path.endswith(".json"):
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)
    return pd.read_csv(path)


# ── HuggingFace download ──────────────────────────────────────────────────────


def download_from_huggingface(
    dataset_name: str,
    split: str = "train",
    out_dir: str = "/tmp",
    config_name: str | None = None,
) -> str:
    """Download a HuggingFace dataset, flatten it, and save as parquet.

    Returns the local parquet path.
    """
    from datasets import load_dataset  # type: ignore

    print(f"[inspector] downloading HF dataset: {dataset_name} split={split}")
    kwargs: dict = {}
    if config_name:
        kwargs["name"] = config_name

    ds = load_dataset(dataset_name, split=split, **kwargs)
    df = _flatten_hf_dataset(ds)
    out_path = os.path.join(out_dir, "dataset.parquet")
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[inspector] saved {len(df):,} rows → {out_path}")
    return out_path


def _flatten_hf_dataset(ds) -> pd.DataFrame:
    """Convert a HuggingFace Dataset to a flat step-level transition DataFrame.

    Handles two common formats:
    - Step-level: each row is one transition, array columns like 'observations'
      are expanded to obs_0, obs_1, …
    - Episode-level: each row is a full episode (T, D) — unrolled into T rows,
      deriving next_obs and done automatically (JAT, D4RL trajectory format).
    """
    import numpy as np

    try:
        df = ds.to_pandas()
    except Exception:
        df = pd.DataFrame(list(ds))

    if len(df) == 0:
        return df

    # Detect episode format: any column whose first cell is a 1-D object array
    # whose elements are themselves arrays (JAT / trajectory format).
    # np.stack(cell) can reconstruct the (T, D) episode matrix in this case.
    is_episode_format = False
    for _col in df.columns:
        _cell = df[_col].iloc[0]
        try:
            _arr = np.array(_cell, dtype=object)
            if _arr.ndim == 1 and len(_arr) > 0 and hasattr(_arr[0], "__len__") and len(_arr[0]) > 0:
                is_episode_format = True
                break
        except Exception:
            pass

    if is_episode_format:
        return _unroll_episodes(df)

    # ── Step-level: expand vector columns ──────────────────────────────────
    flat: dict[str, list] = {}
    drop: list[str] = []

    for col in df.columns:
        sample = df[col].iloc[0] if len(df) > 0 else None
        if sample is None or isinstance(sample, str):
            continue
        try:
            arr = np.array(df[col].tolist())
            if arr.ndim == 2:
                prefix = _col_prefix(col)
                for i in range(arr.shape[1]):
                    flat[f"{prefix}_{i}"] = arr[:, i]
                drop.append(col)
        except Exception:
            pass

    df = df.drop(columns=drop, errors="ignore")
    for k, v in flat.items():
        df[k] = v

    return df


def _episode_to_2d(raw) -> np.ndarray:
    """Convert a raw episode cell (list / object-array of sub-arrays) to (T, D) float32."""
    import numpy as np
    arr = np.array(raw, dtype=object)
    if arr.ndim == 1 and len(arr) > 0 and hasattr(arr[0], "__len__"):
        return np.stack(arr).astype(np.float32)
    return np.array(raw, dtype=np.float32)


def _unroll_episodes(df) -> pd.DataFrame:
    """Unroll an episode-format dataset into individual transition rows (vectorized).

    Derives next_obs (obs shifted by 1 within each episode) and
    done (1 only on the last step of each episode).
    """
    import numpy as np

    obs_key = act_key = rew_key = done_key = None
    for col in df.columns:
        lc, prefix = col.lower(), _col_prefix(col)
        if prefix in ("obs", "observation", "observations") or "obs" in lc:
            obs_key = col
        elif prefix in ("act", "action", "actions") or "act" in lc:
            act_key = col
        elif "reward" in lc or "rew" in lc:
            rew_key = col
        elif "done" in lc or "terminal" in lc or "truncat" in lc:
            done_key = col

    if not obs_key or not act_key or not rew_key:
        raise ValueError(
            f"Could not identify obs/act/reward columns in episode dataset. "
            f"Columns: {list(df.columns)}"
        )

    obs_segs, act_segs, rew_segs, next_obs_segs, done_segs = [], [], [], [], []

    for _, episode in df.iterrows():
        obs_ep = _episode_to_2d(episode[obs_key])            # (T, obs_dim)
        act_ep = _episode_to_2d(episode[act_key])            # (T, act_dim) or (T,)
        rew_ep = np.array(episode[rew_key], dtype=np.float32)
        if act_ep.ndim == 1:
            act_ep = act_ep[:, None]
        T = len(rew_ep)

        next_obs_ep        = np.empty_like(obs_ep)
        next_obs_ep[:-1]   = obs_ep[1:]
        next_obs_ep[-1]    = obs_ep[-1]

        done_ep = (
            np.array(episode[done_key], dtype=np.float32) if done_key
            else np.zeros(T, dtype=np.float32)
        )
        done_ep[-1] = 1.0

        obs_segs.append(obs_ep)
        act_segs.append(act_ep)
        rew_segs.append(rew_ep[:, None])
        next_obs_segs.append(next_obs_ep)
        done_segs.append(done_ep[:, None])

    obs_arr      = np.vstack(obs_segs)
    act_arr      = np.vstack(act_segs)
    rew_arr      = np.vstack(rew_segs)
    next_obs_arr = np.vstack(next_obs_segs)
    done_arr     = np.vstack(done_segs)

    obs_dim, act_dim = obs_arr.shape[1], act_arr.shape[1]
    obs_cols      = [f"obs_{i}"      for i in range(obs_dim)]
    act_cols      = [f"act_{i}"      for i in range(act_dim)]
    next_obs_cols = [f"next_obs_{i}" for i in range(obs_dim)]

    return pd.DataFrame(
        np.hstack([obs_arr, act_arr, rew_arr, next_obs_arr, done_arr]).astype(np.float32),
        columns=obs_cols + act_cols + ["reward"] + next_obs_cols + ["done"],
    )


def _col_prefix(col: str) -> str:
    mapping = {
        "observations": "obs", "observation": "obs",
        "next_observations": "next_obs", "next_observation": "next_obs",
        "actions": "act", "action": "act",
        "rewards": "rew",
    }
    return mapping.get(col, col)


# ── Core inspection ───────────────────────────────────────────────────────────


def inspect(dataset_path: str, initial_states_path: str) -> DatasetMeta:
    """Inspect a transition dataset and return DatasetMeta.

    The LLM maps arbitrary column names to obs/act/reward/next_obs/done
    so the world model knows how to slice the DataFrame.
    """
    df = load_from_file(dataset_path)

    sample_rows = df.head(20).to_dict(orient="records")
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    prompt = f"""You are analyzing a reinforcement learning transition dataset.

Dataset info:
- Total rows: {len(df)}
- Columns and dtypes: {json.dumps(dtypes)}
- First 20 rows (sample): {json.dumps(sample_rows, default=str)}

Identify which columns correspond to:
1. obs_cols: current observation/state columns (list of strings, can be multiple)
2. act_cols: action column(s) (list of strings)
3. reward_col: reward column (single string)
4. next_obs_cols: next-state observation columns (same length as obs_cols)
5. done_col: episode termination column (done / terminated / truncated)
6. act_type: "discrete" if action values are integers, "continuous" if floats
7. act_n: number of unique discrete actions (integer if discrete, null if continuous)
8. reward_min / reward_max: approximate min/max reward from the sample

Rules:
- obs columns may be named: obs_0, observation_0, state_0, s_0, o_0, etc.
- next_obs columns: next_obs_0, next_observation_0, next_state_0, etc.
- If next_obs columns don't exist, use obs_cols (world model will shift rows).
- An action is discrete if its dtype is int64 / int32 / int8.

Return ONLY a valid JSON object (no markdown, no explanation):
{{
  "obs_cols": ["obs_0", "obs_1"],
  "act_cols": ["action"],
  "reward_col": "reward",
  "next_obs_cols": ["next_obs_0", "next_obs_1"],
  "done_col": "done",
  "act_type": "discrete",
  "act_n": 4,
  "reward_min": -1.0,
  "reward_max": 1.0
}}"""

    resp = _client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = json.loads(resp.choices[0].message.content)

    obs_cols      = raw["obs_cols"]
    act_cols      = raw["act_cols"]
    next_obs_cols = raw.get("next_obs_cols") or obs_cols
    obs_dim       = len(obs_cols)
    act_dim       = len(act_cols)
    act_type      = raw["act_type"]
    act_n         = raw.get("act_n")

    # Refine reward range from actual data
    try:
        rc = raw["reward_col"]
        reward_min = float(df[rc].min())
        reward_max = float(df[rc].max())
    except Exception:
        reward_min = float(raw.get("reward_min", -1.0))
        reward_max = float(raw.get("reward_max", 1.0))

    # Recommend hidden sizes proportional to obs_dim
    if obs_dim <= 8:
        hidden_sizes = [64, 64]
    elif obs_dim <= 32:
        hidden_sizes = [128, 128]
    else:
        hidden_sizes = [256, 256]

    if initial_states_path:
        _save_initial_states(df, obs_cols, raw["done_col"], initial_states_path)

    return DatasetMeta(
        obs_cols=obs_cols,
        act_cols=act_cols,
        reward_col=raw["reward_col"],
        next_obs_cols=next_obs_cols,
        done_col=raw["done_col"],
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_type=act_type,
        act_n=act_n,
        reward_min=reward_min,
        reward_max=reward_max,
        n_samples=len(df),
        hidden_sizes=hidden_sizes,
        dataset_path=dataset_path,
        initial_states_path=initial_states_path,
    )


def _save_initial_states(
    df: pd.DataFrame, obs_cols: list[str], done_col: str, out_path: str
) -> None:
    """Extract episode-initial observations and save to parquet."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    try:
        done_series = df[done_col].astype(bool)
        # Row is initial if the *previous* row was done (or it's the first row)
        is_initial = done_series.shift(1, fill_value=True)
        initial_df = df.loc[is_initial, obs_cols].reset_index(drop=True)
        if len(initial_df) == 0:
            raise ValueError("no initial states found")
    except Exception:
        # Fallback: first 20% of rows as candidate starting states
        initial_df = df[obs_cols].head(max(1, len(df) // 5)).reset_index(drop=True)

    initial_df.to_parquet(out_path, index=False)
    print(f"[inspector] saved {len(initial_df)} initial states → {out_path}")
