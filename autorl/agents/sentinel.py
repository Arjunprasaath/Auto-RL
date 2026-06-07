"""Doom Loop Sentinel — LLM-based agent (Phase 3.1).

Detection is rule-based (read heartbeat.json every 30 s — fast and reliable).
Intervention is LLM-driven: when a failure is detected the Sentinel calls GPT
with the full failure context + spawn_plan + history of prior interventions and
asks it to suggest a new hyperparameter configuration. That config is then
launched as a replacement agent via agents.training_agent.

All interventions are appended to sentinel_log.json in the run directory so
the full history of "what was tried and what happened" persists after the run.
Each intervention also writes sentinel_alert.json so the UI can render it.

Failure modes handled
─────────────────────
  nan_loss                  weights exploded → kill + LLM-restart once
  critic_diverged           EV < -0.5 after 10k steps → kill + LLM-restart once
  plateau                   reward stuck → LLM nudge → LLM-restart if still stuck
  entropy_collapsed         PPO entropy near 0 before 50k steps → LLM nudge
  episode_length_regression agent survived then forgot → LLM nudge
  stale_heartbeat           frozen >120 s → LLM nudge
  stale_after_nudge         still frozen >240 s → LLM-restart once
  second_failure            any failure after one restart → kill permanently
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone

import weave
from agents import Agent, AgentOutputSchema, Runner
from pydantic import BaseModel

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

CHECK_INTERVAL_S = 30
NUDGE_THRESHOLD_S = 120
KILL_THRESHOLD_S = 240

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini-2026-03-17")

# ─── LLM agent definition ────────────────────────────────────────────────────

_SENTINEL_SYSTEM = """\
You are the AutoRL Doom Loop Sentinel — an AI agent that recovers failing RL
training runs.

You receive:
  1. The original spawn-plan entry for a failed agent (algo, env, hparams).
  2. The failure reason (one of the types below).
  3. A history of all prior sentinel interventions on this run.

Failure reasons and recommended responses
─────────────────────────────────────────
  nan_loss                  — weights exploded. Lower lr dramatically (1e-4 – 5e-4).
  critic_diverged           — value function worse than mean-predictor. Lower lr, raise n_steps.
  plateau                   — reward stuck. Try different seed, adjust ent_coef or gamma.
  entropy_collapsed         — policy has lost all exploration. Raise ent_coef (0.01 – 0.1).
  episode_length_regression — agent learned to survive then forgot. Lower lr, raise ent_coef.
  stale_heartbeat_nudge     — process frozen. New seed; optionally adjust n_steps.
  stale_after_nudge         — still frozen after nudge. New seed + lower lr.

Rules
─────
  • Keep the same algo and env as the original entry.
  • NEVER suggest lr >= 0.1 (high lr causes NaN).
  • NEVER repeat an lr that has already been tried and failed.
  • Be creative — vary multiple hparams if prior same-algo attempts all failed.
  • Always include a short "rationale" string (1–2 sentences) explaining why you
    chose these values. This is shown to the user in the UI.

Output ONLY a JSON object — no other text:
  {"lr": 0.0003, "seed": 9999, "n_steps": 1024, "ent_coef": 0.01, "gamma": 0.99,
   "rationale": "Original lr=1.0 caused NaN. 3e-4 is the standard safe range for PPO on MuJoCo."}
"""


class SentinelHparams(BaseModel):
    lr: float
    seed: int
    n_steps: int | None = None
    ent_coef: float | None = None
    gamma: float | None = None
    rationale: str = ""


_sentinel_agent = Agent(
    name="DoomLoopSentinelLLM",
    instructions=_SENTINEL_SYSTEM,
    model=OPENAI_MODEL,
    output_type=AgentOutputSchema(SentinelHparams, strict_json_schema=False),
)

# ─── Helpers ─────────────────────────────────────────────────────────────────


def _read_heartbeat(hb_path: str) -> dict | None:
    try:
        with open(hb_path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _load_log(log_path: str) -> list[dict]:
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                return json.load(f)
        except Exception:  # noqa: BLE001
            pass
    return []


def _append_log(log_path: str, entry: dict) -> None:
    log = _load_log(log_path)
    log.append(entry)
    tmp = log_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(log, f, indent=2)
    os.replace(tmp, log_path)


def _write_nudge(results_dir: str, agent_id: str, hparams: dict) -> None:
    nudge_path = os.path.join(results_dir, agent_id, "nudge.json")
    tmp = nudge_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(hparams, f)
    os.replace(tmp, nudge_path)
    print(f"[sentinel] nudged {agent_id} → {hparams}")


def _write_alert(results_dir: str, agent_id: str, anomaly: str,
                 failed_hparams: dict, new_hparams: dict, action: str) -> None:
    """Write sentinel_alert.json for the CopilotKit UI to render."""
    alert = {
        "agent_id":      agent_id,
        "anomaly":       anomaly,
        "action":        action,   # "kill_restart" | "nudge"
        "failed_hparams": failed_hparams,
        "new_hparams":   {k: v for k, v in new_hparams.items() if k != "rationale"},
        "rationale":     new_hparams.get("rationale", ""),
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }
    alert_path = os.path.join(results_dir, "sentinel_alert.json")
    tmp = alert_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(alert, f, indent=2)
    os.replace(tmp, alert_path)
    print(f"[sentinel] alert written → {alert_path}")


# ─── LLM call ────────────────────────────────────────────────────────────────


@weave.op(name="SentinelLLM")
async def _llm_suggest_hparams(
    entry_dict: dict,
    failure_reason: str,
    prior_interventions: list[dict],
) -> dict:
    """Ask the LLM to suggest recovery hparams for a failed agent."""
    prompt = (
        f"Agent failed: {failure_reason}\n\n"
        f"Original config:\n{json.dumps(entry_dict, indent=2)}\n\n"
        f"Prior interventions on this run:\n"
        + (json.dumps(prior_interventions, indent=2) if prior_interventions else "None yet.")
        + "\n\nSuggest new hparams to recover this agent."
    )
    result = await Runner.run(_sentinel_agent, prompt)
    hparams = {k: v for k, v in result.final_output.model_dump().items() if v is not None}
    print(f"[sentinel] LLM suggests for {entry_dict['id']}: {hparams}")
    return hparams


# ─── Main sentinel loop ───────────────────────────────────────────────────────


@weave.op(name="DoomLoopSentinel")
async def run_sentinel(
    agent_ids: list[str],
    results_dir: str = "./results",
    stop_event: asyncio.Event | None = None,
) -> None:
    """Monitor all agents and use an LLM to decide recovery actions.

    Called by swarm_runner.run_swarm as a concurrent asyncio task.
    """
    from agents.training_agent import kill_training_agent, run_training_agent
    from orchestrator.orchestrator_agent import SpawnPlanEntry

    nudged: dict[str, datetime] = {}
    restarted: set[str] = set()
    killed: set[str] = set()

    log_path = os.path.join(results_dir, "sentinel_log.json")

    # Load spawn plan for agent context.
    plan_by_id: dict[str, SpawnPlanEntry] = {}
    plan_path = os.path.join(results_dir, "spawn_plan.json")
    if os.path.exists(plan_path):
        try:
            with open(plan_path) as f:
                for e in json.load(f):
                    plan_by_id[e["id"]] = SpawnPlanEntry.model_validate(e)
        except Exception as exc:  # noqa: BLE001
            print(f"[sentinel] could not load spawn_plan.json: {exc}")

    print(f"[sentinel] LLM sentinel watching {len(agent_ids)} agents in {results_dir}")

    # ── intervention helpers ──────────────────────────────────────────────────

    async def _intervene(agent_id: str, failure_reason: str, hb: dict) -> None:
        """Kill the agent, ask the LLM for a new config, launch a replacement."""
        entry = plan_by_id.get(agent_id)
        if entry is None:
            print(f"[sentinel] {agent_id}: no plan entry — killing permanently")
            await kill_training_agent(agent_id)
            killed.add(agent_id)
            return

        prior = _load_log(log_path)
        print(f"[sentinel] {agent_id}: {failure_reason} — asking LLM for recovery config")

        try:
            new_hparams = await _llm_suggest_hparams(
                entry_dict=entry.model_dump(),
                failure_reason=failure_reason,
                prior_interventions=prior,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[sentinel] LLM call failed ({exc}), falling back to lr=3e-4")
            new_hparams = {"lr": 3e-4, "seed": entry.hparams.get("seed", 42) + 1000,
                           "rationale": f"LLM call failed ({exc}); safe fallback lr=3e-4."}

        # Record the decision before launching.
        log_entry: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "failure_reason": failure_reason,
            "failed_hparams": entry.hparams,
            "heartbeat_at_failure": hb,
            "llm_suggested_hparams": new_hparams,
            "rationale": new_hparams.get("rationale", ""),
            "outcome": "pending",
        }
        _append_log(log_path, log_entry)
        log_idx = len(_load_log(log_path)) - 1

        _write_alert(results_dir, agent_id, failure_reason,
                     entry.hparams, new_hparams, "kill_restart")

        print(
            f"[sentinel] {agent_id}: killing and restarting with LLM config "
            f"lr={new_hparams.get('lr')} seed={new_hparams.get('seed')} "
            f"— {new_hparams.get('rationale', '')}"
        )
        await kill_training_agent(agent_id)

        # Run the replacement and update the log with outcome.
        async def _run_and_log() -> None:
            code = await run_training_agent(entry, results_dir, hparams_override=new_hparams)
            outcome = "completed" if code == 0 else "failed_again"
            current_log = _load_log(log_path)
            if log_idx < len(current_log):
                current_log[log_idx]["outcome"] = outcome
                eval_path = os.path.join(results_dir, agent_id, "eval_result.json")
                if os.path.exists(eval_path):
                    try:
                        with open(eval_path) as f:
                            current_log[log_idx]["eval_result"] = json.load(f)
                    except Exception:  # noqa: BLE001
                        pass
                tmp = log_path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(current_log, f, indent=2)
                os.replace(tmp, log_path)
            print(f"[sentinel] {agent_id}: restart outcome={outcome}")

        asyncio.create_task(_run_and_log())
        restarted.add(agent_id)

    async def _nudge_with_llm(agent_id: str, hb: dict,
                              reason: str = "stale_heartbeat_nudge") -> None:
        """Write nudge.json with LLM-suggested hparams; do not kill the agent."""
        entry = plan_by_id.get(agent_id)
        if entry is None:
            return

        prior = _load_log(log_path)
        print(f"[sentinel] {agent_id}: {reason} — asking LLM for nudge config")

        try:
            new_hparams = await _llm_suggest_hparams(
                entry_dict=entry.model_dump(),
                failure_reason=reason,
                prior_interventions=prior,
            )
        except Exception as exc:  # noqa: BLE001
            current_lr = hb.get("current_lr") or entry.hparams.get("lr", 3e-4)
            new_hparams = {"lr": current_lr / 2, "seed": entry.hparams.get("seed", 42) + 500,
                           "rationale": f"LLM call failed ({exc}); halved lr as fallback."}
            print(f"[sentinel] LLM nudge failed ({exc}), using halved lr={new_hparams['lr']:.2e}")

        _write_nudge(results_dir, agent_id, new_hparams)
        _write_alert(results_dir, agent_id, reason,
                     entry.hparams, new_hparams, "nudge")
        nudged[agent_id] = datetime.now(timezone.utc)

        _append_log(log_path, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "failure_reason": reason,
            "failed_hparams": entry.hparams,
            "heartbeat_at_failure": hb,
            "llm_suggested_hparams": new_hparams,
            "rationale": new_hparams.get("rationale", ""),
            "outcome": "nudge_sent",
        })

    # ── check loop ────────────────────────────────────────────────────────────

    async def _check_all() -> None:
        now = datetime.now(timezone.utc)

        for agent_id in agent_ids:
            if agent_id in killed:
                continue

            hb_path = os.path.join(results_dir, agent_id, "heartbeat.json")
            if not os.path.exists(hb_path):
                continue

            hb = _read_heartbeat(hb_path)
            if hb is None:
                continue

            try:
                ts = datetime.fromisoformat(hb["timestamp"])
                age_s = (now - ts).total_seconds()
            except Exception:  # noqa: BLE001
                continue

            anomaly = hb.get("anomaly")

            # ── Kill-and-restart anomalies (checked before status skip) ───────
            # nan_loss and critic_diverged are severe enough to act on immediately.
            if anomaly in ("nan_loss", "critic_diverged"):
                if agent_id not in restarted:
                    await _intervene(agent_id, anomaly, hb)
                else:
                    print(f"[sentinel] {agent_id}: {anomaly} on restart — killing permanently")
                    await kill_training_agent(agent_id)
                    killed.add(agent_id)
                    _append_log(log_path, {
                        "timestamp": now.isoformat(),
                        "agent_id": agent_id,
                        "failure_reason": f"{anomaly}_second_failure",
                        "outcome": "killed_permanently",
                    })
                continue

            # Skip agents that finished cleanly.
            if hb.get("status") in ("completed", "failed"):
                continue

            # ── Nudge-only anomalies (soft interventions) ─────────────────────
            # plateau: nudge first; restart if still stuck after nudge
            if anomaly == "plateau":
                if agent_id not in nudged:
                    await _nudge_with_llm(agent_id, hb, reason="plateau")
                elif agent_id not in restarted:
                    await _intervene(agent_id, "plateau_after_nudge", hb)

            # entropy_collapsed and ep_length_regression: nudge only (don't kill)
            elif anomaly in ("entropy_collapsed", "episode_length_regression"):
                if agent_id not in nudged:
                    await _nudge_with_llm(agent_id, hb, reason=anomaly)

            # ── Stale heartbeat: LLM nudge at 120 s ──────────────────────────
            elif age_s > NUDGE_THRESHOLD_S and agent_id not in nudged:
                await _nudge_with_llm(agent_id, hb, reason="stale_heartbeat_nudge")

            # ── Still stale after nudge: LLM kill + restart at 240 s ─────────
            elif age_s > KILL_THRESHOLD_S and agent_id in nudged:
                if agent_id not in restarted:
                    await _intervene(agent_id, "stale_after_nudge", hb)
                else:
                    print(f"[sentinel] {agent_id}: still stale after restart — killing permanently")
                    await kill_training_agent(agent_id)
                    killed.add(agent_id)
                    _append_log(log_path, {
                        "timestamp": now.isoformat(),
                        "agent_id": agent_id,
                        "failure_reason": "stale_second_failure",
                        "outcome": "killed_permanently",
                    })

    # ── main loop ─────────────────────────────────────────────────────────────

    while not (stop_event and stop_event.is_set()):
        await _check_all()
        await asyncio.sleep(CHECK_INTERVAL_S)

    # Final sweep: catch anomalies written just before the swarm shuts down.
    await _check_all()
    print("[sentinel] stopped")
    print(f"[sentinel] intervention log → {log_path}")
