"""Phase 4.3 — Evaluator agent.

Ranks completed training runs using OpenAI.  Handles MuJoCo (SB3) and
Countdown (GRPO) result groups separately, since their return scales differ.

Also integrates Weave Evaluation (Phase 4.4):
  - ReturnScorer measures mean return and stability per result.
  - A weave.Evaluation dataset is created from the result list and evaluated
    with an identity predictor so every result gets scored and logged.

Usage (standalone, for testing):
    cd autorl
    python evaluator/evaluator_agent.py --run-dir runs/latest
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(_PKG_ROOT, ".env"))

import openai
import weave

from orchestrator.orchestrator_agent import EvalResult


# ── Config ────────────────────────────────────────────────────────────────────

OPENAI_MODEL = os.environ.get("EVALUATOR_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))

_client: openai.OpenAI | None = None


def _openai() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


# ── Phase 4.4 — Weave Evaluation ──────────────────────────────────────────────


class ReturnScorer(weave.Scorer):
    """Score a single EvalResult dict by return magnitude and stability."""

    @weave.op
    def score(self, output: dict, expected: dict | None = None) -> dict:  # type: ignore[override]
        mean_r = output.get("mean_return", 0.0)
        std_r  = max(output.get("std_return", 1.0), 0.01)
        valid_statuses = {"completed", "early_stopped", "race_dropout"}
        return {
            "return":    float(mean_r),
            "stability": float(1.0 / std_r),
            "completed": output.get("status") in valid_statuses,
        }


async def _push_weave_evaluation(results: list[EvalResult]) -> None:
    """Log results through weave.Evaluation so they appear in the Weave UI.

    Weave unpacks each dataset row as kwargs keyed by column name, so the
    predictor must accept those column names (not a single 'result' arg).
    """
    try:
        dataset = [r.model_dump() for r in results]

        @weave.op
        async def _identity(  # noqa: RUF029
            agent_id: str = "",
            algo: str = "",
            env: str = "",
            status: str = "",
            mean_return: float = 0.0,
            std_return: float = 0.0,
            steps_trained: int = 0,
            wall_time_s: float = 0.0,
            weave_run_id: str = "",
            checkpoint_path: str = "",
        ) -> dict:
            return {
                "agent_id": agent_id,
                "algo": algo,
                "env": env,
                "status": status,
                "mean_return": mean_return,
                "std_return": std_return,
                "steps_trained": steps_trained,
                "wall_time_s": wall_time_s,
                "weave_run_id": weave_run_id,
                "checkpoint_path": checkpoint_path,
            }

        evaluation = weave.Evaluation(
            name="AutoRL_EvalResults",
            dataset=dataset,
            scorers=[ReturnScorer()],
        )
        await evaluation.evaluate(_identity)
    except Exception as e:  # noqa: BLE001
        print(f"[evaluator] weave evaluation skipped ({e})")


# ── LLM ranking ───────────────────────────────────────────────────────────────


def _rank_group(group_name: str, group: list[EvalResult]) -> list[dict]:
    """Ask OpenAI to rank a result group and return a list of rank dicts."""
    prompt = f"""Rank these RL training results for {group_name}:

{json.dumps([r.model_dump() for r in group], indent=2)}

Consider (in order of importance):
1. mean_return — higher is better
2. stability — lower std_return relative to mean is better
3. status — prefer "completed" > "early_stopped" > "race_dropout" > "restarted" > "timed_out" > "failed"
4. sample efficiency — more steps_trained for the same return indicates lower efficiency

Note any Sentinel interventions (status="restarted") — a restarted agent that
ultimately completes is a success story, not a failure.
"early_stopped" means training stagnated but evaluate_policy was still run — treat the
mean_return as valid. "race_dropout" means the agent was significantly behind peers at
a midpoint check — also has a valid mean_return.

Output ONLY a JSON array, no other text:
[{{"rank": 1, "agent_id": "...", "algo": "...", "mean_return": 0.0, "rationale": "..."}}]"""

    response = _openai().chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if the model wrapped the JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw)


# ── Main evaluator ────────────────────────────────────────────────────────────


async def evaluate_results(results: list[EvalResult], run_dir: str) -> dict:
    """Rank all results and write rankings.json to run_dir."""

    mujoco_results   = [r for r in results if r.env != "Countdown"]
    countdown_results = [r for r in results if r.env == "Countdown"]

    rankings: dict[str, list[dict]] = {}

    for group_name, group in [("MuJoCo", mujoco_results), ("Countdown", countdown_results)]:
        if not group:
            continue
        print(f"[evaluator] ranking {len(group)} {group_name} result(s) via {OPENAI_MODEL}")
        try:
            rankings[group_name] = _rank_group(group_name, group)
        except Exception as e:  # noqa: BLE001
            print(f"[evaluator] LLM ranking failed for {group_name}: {e} — using fallback")
            # Fallback: sort by mean_return descending
            sorted_group = sorted(group, key=lambda r: r.mean_return, reverse=True)
            rankings[group_name] = [
                {
                    "rank": i + 1,
                    "agent_id": r.agent_id,
                    "algo": r.algo,
                    "mean_return": r.mean_return,
                    "rationale": f"Sorted by mean_return (LLM ranking unavailable: {e})",
                }
                for i, r in enumerate(sorted_group)
            ]

    # ── Weave Evaluation (Phase 4.4) ─────────────────────────────────────────
    await _push_weave_evaluation(results)

    # ── Persist rankings ──────────────────────────────────────────────────────
    rankings_path = os.path.join(run_dir, "rankings.json")
    with open(rankings_path, "w") as f:
        json.dump(rankings, f, indent=2)
    print(f"[evaluator] rankings written → {rankings_path}")

    return rankings


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="Run evaluator on existing results")
    p.add_argument("--run-dir", default=os.path.join(_PKG_ROOT, "runs", "latest"),
                   help="Run directory containing eval_result.json files per agent")
    args = p.parse_args()

    run_dir = os.path.realpath(args.run_dir)
    if not os.path.isdir(run_dir):
        sys.exit(f"[evaluator] run dir not found: {run_dir}")

    # Collect all eval_result.json files in the run directory
    results: list[EvalResult] = []
    for agent_dir in sorted(os.scandir(run_dir), key=lambda e: e.name):
        if not agent_dir.is_dir():
            continue
        r_path = os.path.join(agent_dir.path, "eval_result.json")
        if os.path.exists(r_path):
            with open(r_path) as f:
                results.append(EvalResult.model_validate(json.load(f)))

    if not results:
        sys.exit(f"[evaluator] no eval_result.json files found in {run_dir}")

    print(f"[evaluator] found {len(results)} result(s) in {run_dir}")

    if os.environ.get("WANDB_API_KEY") and not os.environ.get("WEAVE_DISABLED"):
        try:
            weave.init(os.environ.get("WEAVE_PROJECT", "autorl"))
        except Exception as e:  # noqa: BLE001
            print(f"[weave] init skipped ({e})")

    rankings = asyncio.run(evaluate_results(results, run_dir))

    for group, entries in rankings.items():
        print(f"\n{group}:")
        for e in entries:
            print(f"  {e.get('rank')}. {e.get('algo')} ({e.get('agent_id')}) "
                  f"return={e.get('mean_return', '?')} — {e.get('rationale', '')[:80]}")


if __name__ == "__main__":
    main()
