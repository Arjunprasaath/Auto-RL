"""Reward Designer Agent — LLM-powered reward function design via multi-turn chat.

The agent receives the DatasetMeta (obs/action dims, current reward stats) and
a growing conversation history.  Each turn it proposes or refines a Python reward
function with this exact signature:

    def reward_fn(obs, action, next_obs, done, original_reward):
        # obs, next_obs : numpy array (obs_dim,)
        # action        : numpy array (act_dim,) or scalar for discrete
        # done          : bool
        # original_reward : float from the dataset
        return float(...)

Public API
----------
design_reward(meta, history, user_msg)  → RewardDesign   # one LLM turn
apply_reward_fn(dataset_path, meta, code, out_path) → str  # rewrite dataset
"""

from __future__ import annotations

import json
import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PKG_ROOT, ".env"))

import numpy as np
from openai import OpenAI
from pydantic import BaseModel

from agents.dataset_inspector_agent import DatasetMeta, load_from_file

_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL   = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


# ── Schema ─────────────────────────────────────────────────────────────────────


class RewardDesign(BaseModel):
    code: str           # complete reward_fn definition
    explanation: str    # one-liner shown under code
    message: str        # conversational response shown in chat


# ── Chat agent ─────────────────────────────────────────────────────────────────


def design_reward(
    meta: DatasetMeta,
    history: list[dict],   # [{"role": "user"|"assistant", "content": str}]
    user_msg: str,
) -> RewardDesign:
    """One turn of the reward-design conversation.

    Args:
        meta:     DatasetMeta from the dataset inspector
        history:  previous turns (passed straight through to the LLM)
        user_msg: latest message from the user (or "" to trigger first suggestion)
    """
    obs_preview = meta.obs_cols[:4]
    if len(meta.obs_cols) > 4:
        obs_preview.append("…")

    system = f"""You are an expert RL reward-function designer.

DATASET CONTEXT
  obs_dim    : {meta.obs_dim}   obs columns (sample): {obs_preview}
  act_type   : {meta.act_type}  act_dim={meta.act_dim}{f'  act_n={meta.act_n}' if meta.act_n else ''}
  reward col : "{meta.reward_col}"  range=[{meta.reward_min:.3f}, {meta.reward_max:.3f}]
  n_samples  : {meta.n_samples:,}

Your job: help the user design (or approve) a Python reward function.

REQUIRED FUNCTION SIGNATURE (do NOT change it):
    def reward_fn(obs, action, next_obs, done, original_reward):
        # obs, next_obs : np.ndarray shape ({meta.obs_dim},)
        # action        : np.ndarray shape ({meta.act_dim},){' or scalar int' if meta.act_type == 'discrete' else ''}
        # done          : bool
        # original_reward : float (from dataset)
        return float(...)   # must return a float

RULES
- Always return ONLY valid JSON — no markdown, no text outside the JSON object.
- Always include a complete, runnable `reward_fn` in "code".
- `np` (numpy) is available; no other imports are allowed.
- Default baseline: just return original_reward unchanged.
- When the user says "looks good", "approve", "done", or similar, echo back the same
  code unchanged and confirm.

RESPONSE FORMAT:
{{
  "message": "Conversational reply shown to the user (what you are doing / why).",
  "code": "def reward_fn(obs, action, next_obs, done, original_reward):\\n    return original_reward",
  "explanation": "One-line human summary of what this reward incentivises."
}}"""

    msgs: list[dict] = [{"role": "system", "content": system}]
    msgs += history
    msgs.append({
        "role": "user",
        "content": user_msg or "Please analyse my dataset and suggest a starting reward function.",
    })

    resp = _client.chat.completions.create(
        model=MODEL,
        messages=msgs,
        response_format={"type": "json_object"},
        temperature=0.4,
    )
    raw = json.loads(resp.choices[0].message.content)
    return RewardDesign(
        code=raw.get("code", "def reward_fn(obs, action, next_obs, done, original_reward):\n    return original_reward"),
        explanation=raw.get("explanation", ""),
        message=raw.get("message", ""),
    )


# ── Apply ──────────────────────────────────────────────────────────────────────


def apply_reward_fn(
    dataset_path: str,
    meta: DatasetMeta,
    code: str,
    out_path: str,
) -> str:
    """Execute the approved reward_fn over every row and save the result.

    Returns the path to the rewritten parquet file.
    Raises ValueError if the code is unsafe or doesn't compile.
    """
    # Compile and validate
    ns: dict = {"np": np}
    try:
        exec(compile(code, "<reward_fn>", "exec"), ns)
    except SyntaxError as exc:
        raise ValueError(f"reward_fn has a syntax error: {exc}") from exc
    if "reward_fn" not in ns or not callable(ns["reward_fn"]):
        raise ValueError("Code must define a callable named `reward_fn`.")
    fn = ns["reward_fn"]

    df = load_from_file(dataset_path)
    obs  = df[meta.obs_cols].values.astype(np.float32)
    act  = df[meta.act_cols].values.astype(np.float32)
    done = df[meta.done_col].values.astype(bool)
    orig = df[meta.reward_col].values.astype(np.float32)

    # next_obs — use dedicated columns when available, else shift
    if (
        meta.next_obs_cols
        and meta.next_obs_cols != meta.obs_cols
        and set(meta.next_obs_cols).issubset(df.columns)
    ):
        nobs = df[meta.next_obs_cols].values.astype(np.float32)
    else:
        nobs = np.roll(obs, -1, axis=0)

    new_rewards = np.array([
        float(fn(obs[i], act[i], nobs[i], bool(done[i]), float(orig[i])))
        for i in range(len(df))
    ], dtype=np.float32)

    df[meta.reward_col] = new_rewards
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[reward_designer] wrote {len(df):,} rows → {out_path}")
    return out_path
