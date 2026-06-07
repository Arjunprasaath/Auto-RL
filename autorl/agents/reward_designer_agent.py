"""Reward Designer Agent — LLM chat to shape rewards for spawned Gym agents.

Proposes a Python reward wrapper applied on top of the environment's native
reward during PPO / SAC / A2C training:

    def reward_fn(obs, action, reward, terminated, truncated, info):
        return float(...)

Public API
----------
design_spawn_reward(task, plan, history, user_msg) → RewardDesign
validate_reward_code(code) → None  (raises ValueError on bad code)
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

_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL   = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


class RewardDesign(BaseModel):
    code: str
    explanation: str
    message: str


def validate_reward_code(code: str) -> None:
    """Compile-check that code defines a callable ``reward_fn``."""
    ns: dict = {"np": np}
    try:
        exec(compile(code, "<reward_fn>", "exec"), ns)
    except SyntaxError as exc:
        raise ValueError(f"reward_fn syntax error: {exc}") from exc
    if "reward_fn" not in ns or not callable(ns["reward_fn"]):
        raise ValueError("Code must define a callable named `reward_fn`.")


def design_spawn_reward(
    task: str,
    plan: list[dict],
    history: list[dict],
    user_msg: str,
) -> RewardDesign:
    """One turn of reward-design chat for a Gym spawn plan."""
    agents_desc = "\n".join(
        f"  - {e.get('id', '?')}: {e.get('algo', '?')} on {e.get('env', '?')}"
        for e in plan
    ) or "  (no agents yet)"

    system = f"""You are an expert RL reward-shaping designer for Gymnasium environments.

TASK
  {task.strip() or "(not specified)"}

SPAWN PLAN (agents that will train with your reward)
{agents_desc}

Your job: help the user design (or approve) a Python reward-shaping function that
wraps each environment's native step reward during SB3 training.

REQUIRED FUNCTION SIGNATURE (do NOT change it):
    def reward_fn(obs, action, reward, terminated, truncated, info):
        # obs       : np.ndarray (env observation, any shape — use np.asarray)
        # action    : int or np.ndarray (discrete or continuous action)
        # reward    : float — the environment's original step reward
        # terminated, truncated : bool
        # info      : dict from env.step
        return float(...)   # shaped reward passed to the RL algorithm

RULES
- Return ONLY valid JSON — no markdown outside the JSON object.
- Always include a complete runnable `reward_fn` in "code".
- Only `np` (numpy) is available; no other imports.
- Default baseline: `return float(reward)` unchanged.
- Keep rewards finite; avoid runaway values.
- When the user says "looks good", "approve", or "done", echo the same code back.

RESPONSE FORMAT:
{{
  "message": "Conversational reply to the user.",
  "code": "def reward_fn(obs, action, reward, terminated, truncated, info):\\n    return float(reward)",
  "explanation": "One-line summary of what this reward incentivises."
}}"""

    msgs: list[dict] = [{"role": "system", "content": system}]
    msgs += history
    msgs.append({
        "role": "user",
        "content": user_msg or "Suggest a starting reward function for this task and agent lineup.",
    })

    resp = _client.chat.completions.create(
        model=MODEL,
        messages=msgs,
        response_format={"type": "json_object"},
        temperature=0.4,
    )
    raw = json.loads(resp.choices[0].message.content)
    design = RewardDesign(
        code=raw.get(
            "code",
            "def reward_fn(obs, action, reward, terminated, truncated, info):\n    return float(reward)",
        ),
        explanation=raw.get("explanation", ""),
        message=raw.get("message", ""),
    )
    validate_reward_code(design.code)
    return design
