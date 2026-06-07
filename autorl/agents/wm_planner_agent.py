"""Multi-agent planning coordinator for the World Model RL pipeline.

Runs a genuine multi-turn conversation where each sub-agent's output is
fed as accumulated context into the next agent's turn — so every decision
is informed by all previous ones.

Agent chain (executed in order):
  0. dataset_size_agent  — called before download; decides how many
                           rows / episodes to request from HuggingFace
  1. arch_search_agent   — recommends WM architecture (hidden_sizes,
                           activation, dropout)
  2. algo_selector_agent — picks the best 2-3 SB3 algorithms (PPO /
                           SAC / A2C) to compare inside the world model
  3. hparam_agent        — suggests hyperparameters for each algorithm,
                           including a deliberately high-LR "doom-loop
                           sentinel" agent to test stability

All decisions are logged incrementally to {run_dir}/agent_log.json so
the UI can display live reasoning as planning unfolds.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(_PKG_ROOT, ".env"))

from openai import OpenAI
from pydantic import BaseModel

from agents.dataset_inspector_agent import DatasetMeta

_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
PLANNER_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


# ── Schemas ────────────────────────────────────────────────────────────────────


class AgentLogEntry(BaseModel):
    agent: str        # "dataset_size" | "arch_search" | "algo_selector" | "hparam"
    decision: str     # short one-liner shown in UI
    reasoning: str    # longer explanation
    timestamp: float


class ArchDecision(BaseModel):
    hidden_sizes: list[int]
    activation: str   # "relu" | "silu" | "tanh" | "elu"
    dropout: float
    reasoning: str


class AlgoEntry(BaseModel):
    algo: str       # "PPO" | "SAC" | "A2C"
    agent_id: str   # "agent_1", "agent_2", …
    reasoning: str


class HparamEntry(BaseModel):
    algo: str
    agent_id: str
    hparams: dict
    reasoning: str


class PlannerDecisions(BaseModel):
    arch: ArchDecision
    algos: list[AlgoEntry]
    hparams: list[HparamEntry]
    agent_log: list[AgentLogEntry]


class DatasetSizeRecommendation(BaseModel):
    split: str              # e.g. "train[:100]" or "train[:50000]"
    n_samples_estimate: int
    reasoning: str


# ── Dataset size agent ────────────────────────────────────────────────────────


def recommend_dataset_size(
    dataset_name: str,
    config_name: str | None = None,
) -> DatasetSizeRecommendation:
    """Call the dataset_size_agent to decide how many rows/episodes to download.

    Fetches HuggingFace metadata (card, file sizes, column structure) and
    asks the LLM to pick an appropriate download split for world model training.
    """
    print(f"[planner/dataset_size] fetching HF info for {dataset_name!r} …")
    hf_info = _fetch_hf_info(dataset_name, config_name)

    prompt = f"""You are the DATASET SIZE AGENT for a world-model RL pipeline.

HuggingFace dataset: {dataset_name}
Config: {config_name or "default"}
Dataset info (from HF API + peek):
{json.dumps(hf_info, default=str)[:2500]}

Decide how many samples to download. Your goal: enough data to train a
meaningful neural dynamics model, but not so much that download takes > 5 min.

Rules:
- For step-level datasets (each row = 1 transition):
    target 50,000–200,000 rows; use "train[:N]" syntax
- For episode-level datasets (each row = a full episode of T steps):
    target ~50,000 transitions total, so download ≈ 50000 / T episodes
    if is_episode_format is true and episode_length_estimate is available,
    compute n_episodes = max(10, min(all_episodes, ceil(50000 / episode_length_estimate)))
- Never download less than 1 episode or 1,000 rows
- If total dataset is < 10,000 rows, just use "train" (download all)

Return ONLY valid JSON (no markdown, no explanation):
{{
  "split": "train[:100]",
  "n_samples_estimate": 100000,
  "reasoning": "Episode format with ~1000 steps/episode. 100 episodes = ~100K transitions; good training signal without excessive download time."
}}"""

    resp = _client.chat.completions.create(
        model=PLANNER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = json.loads(resp.choices[0].message.content)
    rec = DatasetSizeRecommendation(**raw)
    print(f"[planner/dataset_size] → split={rec.split!r}  "
          f"~{rec.n_samples_estimate:,} transitions")
    return rec


def _fetch_hf_info(dataset_name: str, config_name: str | None) -> dict:
    """Fetch HuggingFace dataset metadata for the size agent."""
    info: dict = {"dataset_name": dataset_name, "config_name": config_name}

    # HF hub dataset card
    try:
        from huggingface_hub import dataset_info as hf_dataset_info
        di = hf_dataset_info(dataset_name)
        info["tags"] = di.tags or []
        info["downloads_last_month"] = getattr(di, "downloads", None)
        siblings = getattr(di, "siblings", []) or []
        info["total_bytes"] = sum(getattr(s, "size", 0) or 0 for s in siblings)
        info["n_files"] = len(siblings)
        card = getattr(di, "cardData", None)
        if card:
            info["card_snippet"] = str(card)[:400]
    except Exception as e:
        info["card_fetch_error"] = str(e)

    # Peek at the first 3 rows to detect format
    try:
        from datasets import load_dataset
        kwargs: dict = {}
        if config_name:
            kwargs["name"] = config_name
        ds_peek = load_dataset(dataset_name, split="train[:3]", **kwargs)
        info["columns"] = ds_peek.column_names
        info["n_peek_rows"] = len(ds_peek)

        # Detect episode format
        if len(ds_peek) > 0:
            row0 = ds_peek[0]
            for col in ds_peek.column_names:
                val = row0[col]
                if isinstance(val, (list, tuple)):
                    import numpy as np
                    arr = np.array(val, dtype=object)
                    if arr.ndim == 1 and len(arr) > 10:
                        info["is_episode_format"] = True
                        info["episode_length_estimate"] = len(arr)
                        break
            if "is_episode_format" not in info:
                info["is_episode_format"] = False

        # Try to get total training set size (may be slow for huge datasets)
        try:
            info["n_train_total"] = ds_peek.info.splits["train"].num_examples
        except Exception:
            pass
    except Exception as e:
        info["peek_error"] = str(e)

    return info


# ── Multi-turn planning chain ──────────────────────────────────────────────────


def run_planning(
    meta: DatasetMeta,
    run_dir: str,
    on_log: Callable[[AgentLogEntry], None] | None = None,
) -> PlannerDecisions:
    """Run the full multi-turn planning chain.

    Each agent turn appends to `messages`, so all subsequent agents
    have full context from previous decisions.

    Writes {run_dir}/agent_log.json after every agent step so the UI
    can display reasoning as it streams in.

    Args:
        meta:     DatasetMeta produced by the inspector agent
        run_dir:  directory where agent_log.json will be written
        on_log:   optional callback invoked after each agent finishes
    """
    log_path = Path(run_dir) / "agent_log.json"
    agent_log: list[AgentLogEntry] = []

    def _append_log(entry: AgentLogEntry) -> None:
        agent_log.append(entry)
        log_path.write_text(
            json.dumps([e.model_dump() for e in agent_log], indent=2)
        )
        if on_log:
            on_log(entry)
        print(f"[planner/{entry.agent}] {entry.decision}")

    # ── Shared system context ────────────────────────────────────────────────
    system_msg = {
        "role": "system",
        "content": (
            "You are an expert RL system architect coordinating a multi-agent pipeline. "
            "You will make a series of decisions about how to build a world-model-based "
            "RL training system for a custom dataset. Each decision builds on the previous ones. "
            "Return ONLY valid JSON when asked — no markdown, no explanation text outside the JSON."
        ),
    }
    messages: list[dict] = [system_msg]

    # Dataset summary injected into every turn for context
    meta_summary = (
        f"Dataset: {Path(meta.dataset_path).name}\n"
        f"  obs_dim={meta.obs_dim}  act_type={meta.act_type}  act_dim={meta.act_dim}"
        f"{'  act_n=' + str(meta.act_n) if meta.act_n else ''}\n"
        f"  reward=[{meta.reward_min:.2f}, {meta.reward_max:.2f}]\n"
        f"  n_samples={meta.n_samples:,}"
    )

    # ── Turn 1 : arch_search_agent ───────────────────────────────────────────
    messages.append({
        "role": "user",
        "content": f"""{meta_summary}

As the ARCHITECTURE AGENT, design the optimal neural world model.
The model is an MLP: input = obs + encoded_action → outputs = next_obs, reward, done_logit.

Consider:
- obs_dim={meta.obs_dim} → input is {'tiny (<= 8)' if meta.obs_dim <= 8 else 'small' if meta.obs_dim <= 24 else 'medium' if meta.obs_dim <= 64 else 'large'}
- n_samples={meta.n_samples:,} → {'small dataset: use dropout ≥ 0.1 to regularise' if meta.n_samples < 50_000 else 'medium dataset: light dropout 0.0–0.05' if meta.n_samples < 200_000 else 'large dataset: dropout 0.0 ok'}
- act_type={meta.act_type}: {'continuous outputs → SiLU / ELU work well' if meta.act_type == 'continuous' else 'discrete actions → RELU is safe'}
- Reward range [{meta.reward_min:.2f}, {meta.reward_max:.2f}]: {'wide range → deeper network' if abs(meta.reward_max - meta.reward_min) > 5 else 'narrow range → shallower ok'}

Return ONLY valid JSON:
{{
  "hidden_sizes": [256, 256, 128],
  "activation": "silu",
  "dropout": 0.05,
  "reasoning": "..."
}}""",
    })

    arch_resp = _llm(messages)
    arch_raw = json.loads(arch_resp)
    arch = ArchDecision(**arch_raw)
    messages.append({"role": "assistant", "content": arch_resp})
    _append_log(AgentLogEntry(
        agent="arch_search",
        decision=(
            f"MLP {' → '.join(str(h) for h in arch.hidden_sizes)} "
            f"| {arch.activation.upper()} | dropout={arch.dropout}"
        ),
        reasoning=arch.reasoning,
        timestamp=time.time(),
    ))

    # ── Turn 2 : algo_selector_agent ─────────────────────────────────────────
    available_algos = (
        ["PPO", "SAC", "A2C"]
        if meta.act_type == "continuous"
        else ["PPO", "A2C"]
    )

    n_avail = len(available_algos)
    messages.append({
        "role": "user",
        "content": f"""Good. Architecture decided: {arch.model_dump_json()}.

As the ALGORITHM SELECTION AGENT, choose {n_avail} algorithm(s) to race inside
the world model.

Available algorithms for act_type="{meta.act_type}": {available_algos}

Algorithm characteristics:
- PPO: on-policy, stable, both action types — always include as baseline
- SAC: off-policy, sample-efficient, entropy regularisation — continuous only
- A2C: on-policy, fast, less stable than PPO — both action types

Strategy: pick algorithms with DIVERSE learning strategies for meaningful comparison.
Assign agent_id values as "agent_1", "agent_2", … in order (one per algo).

Return ONLY valid JSON (include only the algos you are using):
{{
  "algos": [
    {{"algo": "PPO",  "agent_id": "agent_1", "reasoning": "..."}},
    {{"algo": "SAC",  "agent_id": "agent_2", "reasoning": "..."}}
  ],
  "reasoning": "..."
}}""",
    })

    algo_resp = _llm(messages)
    algo_raw = json.loads(algo_resp)
    algos = [AlgoEntry(**a) for a in algo_raw["algos"]]
    # Validate: only use supported algos
    algos = [a for a in algos if a.algo.upper() in {"PPO", "SAC", "A2C"}]
    if not algos:
        algos = [AlgoEntry(algo="PPO", agent_id="agent_1", reasoning="fallback")]
    messages.append({"role": "assistant", "content": algo_resp})
    _append_log(AgentLogEntry(
        agent="algo_selector",
        decision=f"Competing: {' vs '.join(a.algo for a in algos)}",
        reasoning=algo_raw.get("reasoning", ""),
        timestamp=time.time(),
    ))

    # ── Turn 3 : hparam_agent ────────────────────────────────────────────────
    algo_list_str = json.dumps([{"algo": a.algo, "agent_id": a.agent_id} for a in algos])

    messages.append({
        "role": "user",
        "content": f"""Good. Selected algorithms: {algo_list_str}.

As the HYPERPARAMETER AGENT, suggest concrete hyperparameters for each algorithm.

Context:
- n_samples={meta.n_samples:,} → {'small: small batch, high LR' if meta.n_samples < 30_000 else 'medium' if meta.n_samples < 150_000 else 'large: larger batch ok'}
- obs_dim={meta.obs_dim}, act_dim={meta.act_dim}
- World-model episodes are typically 200–1000 steps

Typical ranges:
  PPO:  lr 1e-4–5e-4, n_steps 256–2048, gamma 0.99
  SAC:  lr 1e-4–3e-4, batch_size 256, gamma 0.99
  A2C:  lr 3e-4–7e-4, n_steps 128–512, gamma 0.99

Important: one agent MUST have a deliberately HIGH lr (≥ 0.5) to act as
the doom-loop sentinel (it will likely crash, triggering the recovery system).
This should be the last agent in the list.

Return ONLY valid JSON:
{{
  "hparams": [
    {{
      "algo": "PPO",
      "agent_id": "agent_1",
      "hparams": {{"lr": 0.0003, "gamma": 0.99, "n_steps": 512, "seed": 42}},
      "reasoning": "..."
    }},
    {{
      "algo": "A2C",
      "agent_id": "agent_3",
      "hparams": {{"lr": 1.0, "gamma": 0.99, "n_steps": 256, "seed": 99}},
      "reasoning": "High LR sentinel to test stability"
    }}
  ],
  "reasoning": "..."
}}""",
    })

    hp_resp = _llm(messages)
    hp_raw = json.loads(hp_resp)
    hparams = [HparamEntry(**h) for h in hp_raw["hparams"]]
    messages.append({"role": "assistant", "content": hp_resp})
    _append_log(AgentLogEntry(
        agent="hparam",
        decision=(
            f"Hparams set for {len(hparams)} agents — "
            f"sentinel LR={max((h.hparams.get('lr', 0) for h in hparams), default=0):.2g}"
        ),
        reasoning=hp_raw.get("reasoning", ""),
        timestamp=time.time(),
    ))

    return PlannerDecisions(
        arch=arch,
        algos=algos,
        hparams=hparams,
        agent_log=agent_log,
    )


# ── LLM helper ────────────────────────────────────────────────────────────────


def _llm(messages: list[dict]) -> str:
    """Make a single structured LLM call and return the raw JSON string."""
    resp = _client.chat.completions.create(
        model=PLANNER_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return resp.choices[0].message.content
