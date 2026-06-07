"""Doom Loop Sentinel — LLM-based agent (Phase 3.1).

Detection is rule-based. With Redis, heartbeats trigger _check_all() immediately
via pub/sub (NaN loss detected in seconds, not up to 30 s). Without Redis the
sentinel falls back to polling heartbeat.json every CHECK_INTERVAL_S seconds.
Intervention is LLM-driven: when a failure is detected the Sentinel calls GPT
with the full failure context + spawn_plan + history of prior interventions and
asks it to suggest a new hyperparameter configuration. That config is then
launched as a replacement agent via agents.training_agent.

All interventions are appended to sentinel_log.json in the run directory so
the full history of "what was tried and what happened" persists after the run.

Failure modes handled
─────────────────────
  nan_loss             weights exploded → kill immediately, LLM-restart once
  stale_heartbeat      agent frozen >120 s → LLM nudge (write nudge.json)
  stale_after_nudge    still frozen >240 s → LLM-restart once
  second_failure       any failure after one restart → kill permanently
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
NUDGE_THRESHOLD_S = 600   # 10 min — GRPO model loading + first batch takes 5-10 min
KILL_THRESHOLD_S = 1200   # 20 min — allow full pod provisioning + startup after nudge

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini-2026-03-17")

# ─── LLM agent definition ────────────────────────────────────────────────────

_SENTINEL_SYSTEM = """\
You are the AutoRL Doom Loop Sentinel — an AI agent that recovers failing RL
training runs.

You receive:
  1. The original spawn-plan entry for a failed agent (algo, env, hparams).
  2. The failure reason: "nan_loss", "stale_heartbeat_nudge", or "stale_after_nudge".
  3. A history of all prior sentinel interventions on this run.

Your job: suggest a NEW hyperparameter configuration that is likely to succeed,
given what has already been tried and what failed.

Rules
─────
  • Keep the same algo and env as the original entry.
  • NEVER suggest lr >= 0.1 (high lr causes NaN).
  • NEVER repeat an lr that has already been tried and failed.
  • Output ONLY a JSON object of numeric hyperparameters, no other text.

Rules for SB3 agents (algo = PPO, SAC, A2C):
  • For nan_loss: suggest a much lower lr (1e-4 – 5e-4).
  • For stale / frozen: try a different seed; optionally adjust n_steps or ent_coef.
  • Be creative — vary multiple hparams if prior same-algo attempts all failed.
  • Valid fields: lr, seed, n_steps, ent_coef, gamma.

Rules for GRPO agents (algo = GRPO, env = Countdown):
  • NEVER suggest n_steps, ent_coef, or gamma — these are SB3-only and ignored by GRPO.
  • For stale / frozen: LOWER num_generations (e.g. 8 → 4) to reduce VRAM usage,
    try a different seed, optionally lower temperature.
  • For nan_loss: halve the lr and reduce num_generations.
  • Valid fields: lr, seed, num_generations, temperature.

Output format — SB3 example (all fields optional except lr and seed):
  {"lr": 0.0003, "seed": 9999, "n_steps": 1024, "ent_coef": 0.01, "gamma": 0.99}

Output format — GRPO example (all fields optional except lr and seed):
  {"lr": 1e-6, "seed": 7777, "num_generations": 4, "temperature": 0.9}
"""


class SentinelHparams(BaseModel):
    lr: float
    seed: int
    n_steps: int | None = None
    ent_coef: float | None = None
    gamma: float | None = None
    num_generations: int | None = None
    temperature: float | None = None


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
    # Push intervention summary to W&B run so the dashboard reflects it
    _update_wandb_sentinel_summary(entry)


def _write_nudge(results_dir: str, agent_id: str, hparams: dict) -> None:
    run_id = os.path.basename(results_dir)
    # Redis-first: push nudge so the training script can atomically pop it
    try:
        from coordination.redis_coordinator import coordinator
        coordinator.push_nudge(run_id, agent_id, hparams)
        print(f"[sentinel] nudged {agent_id} via Redis → {hparams}")
    except Exception:  # noqa: BLE001
        pass
    # File fallback: always write so scripts without Redis still get the nudge
    nudge_path = os.path.join(results_dir, agent_id, "nudge.json")
    tmp = nudge_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(hparams, f)
    os.replace(tmp, nudge_path)
    print(f"[sentinel] nudged {agent_id} via file → {hparams}")


def _update_wandb_sentinel_summary(entry: dict) -> None:
    """Push intervention counts and last failure reason to the active W&B run summary."""
    try:
        import wandb
        if wandb.run is None:
            return
        agent_id = entry.get("agent_id", "unknown")
        reason   = entry.get("failure_reason", "unknown")
        key_n    = f"sentinel/{agent_id}/interventions"
        key_r    = f"sentinel/{agent_id}/last_reason"
        key_o    = f"sentinel/{agent_id}/last_outcome"
        wandb.run.summary[key_n] = int(wandb.run.summary.get(key_n) or 0) + 1
        wandb.run.summary[key_r] = reason
        wandb.run.summary[key_o] = entry.get("outcome", "")
    except Exception:  # noqa: BLE001
        pass


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
    restarted: dict[str, datetime] = {}
    killed: set[str] = set()
    RESTART_GRACE_S = 900  # 15 min grace after restart for pod provisioning + startup

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
            new_hparams = {"lr": 3e-4, "seed": entry.hparams.get("seed", 42) + 1000}

        # Record the decision before launching.
        log_entry: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "failure_reason": failure_reason,
            "failed_hparams": entry.hparams,
            "heartbeat_at_failure": hb,
            "llm_suggested_hparams": new_hparams,
            "outcome": "pending",
        }
        _append_log(log_path, log_entry)
        log_idx = len(_load_log(log_path)) - 1

        print(
            f"[sentinel] {agent_id}: killing and restarting with LLM config "
            f"lr={new_hparams.get('lr')} seed={new_hparams.get('seed')}"
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
        restarted[agent_id] = datetime.now(timezone.utc)

    async def _nudge_with_llm(agent_id: str, hb: dict) -> None:
        """Write nudge.json with LLM-suggested hparams; do not kill the agent."""
        entry = plan_by_id.get(agent_id)
        if entry is None:
            return

        prior = _load_log(log_path)
        print(f"[sentinel] {agent_id}: stale heartbeat — asking LLM for nudge config")

        try:
            new_hparams = await _llm_suggest_hparams(
                entry_dict=entry.model_dump(),
                failure_reason="stale_heartbeat_nudge",
                prior_interventions=prior,
            )
        except Exception as exc:  # noqa: BLE001
            current_lr = hb.get("current_lr") or entry.hparams.get("lr", 3e-4)
            new_hparams = {"lr": current_lr / 2, "seed": entry.hparams.get("seed", 42) + 500}
            print(f"[sentinel] LLM nudge failed ({exc}), using halved lr={new_hparams['lr']:.2e}")

        _write_nudge(results_dir, agent_id, new_hparams)
        nudged[agent_id] = datetime.now(timezone.utc)

        _append_log(log_path, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "failure_reason": "stale_heartbeat_nudge",
            "failed_hparams": entry.hparams,
            "heartbeat_at_failure": hb,
            "llm_suggested_hparams": new_hparams,
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

            # ── NaN loss: kill + LLM restart (checked before status skip) ────
            if hb.get("anomaly") == "nan_loss":
                if agent_id not in restarted:
                    await _intervene(agent_id, "nan_loss", hb)
                else:
                    print(f"[sentinel] {agent_id}: NaN on restart — killing permanently")
                    await kill_training_agent(agent_id)
                    killed.add(agent_id)
                    _append_log(log_path, {
                        "timestamp": now.isoformat(),
                        "agent_id": agent_id,
                        "failure_reason": "nan_loss_second_failure",
                        "outcome": "killed_permanently",
                    })
                continue

            # Skip agents that finished cleanly.
            if hb.get("status") in ("completed", "failed"):
                continue

            # ── Stale heartbeat: LLM nudge ────────────────────────────────────
            if age_s > NUDGE_THRESHOLD_S and agent_id not in nudged:
                await _nudge_with_llm(agent_id, hb)

            # ── Still stale after nudge: LLM kill + restart ─────────────────
            elif age_s > KILL_THRESHOLD_S and agent_id in nudged:
                if agent_id not in restarted:
                    await _intervene(agent_id, "stale_after_nudge", hb)
                else:
                    restart_age = (now - restarted[agent_id]).total_seconds()
                    if restart_age < RESTART_GRACE_S:
                        print(f"[sentinel] {agent_id}: restarted {restart_age:.0f}s ago, "
                              f"grace period {RESTART_GRACE_S}s — skipping")
                        continue
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

    # Redis wakeup: a background task subscribes to the run's heartbeat channel
    # and sets this event whenever any agent publishes a heartbeat.  When fired,
    # the sentinel runs _check_all() immediately instead of waiting up to 30 s.
    # Without Redis the event is never set and the sentinel falls back to the
    # existing CHECK_INTERVAL_S poll cadence — behaviour is identical.
    _redis_wakeup = asyncio.Event()
    run_id = os.path.basename(results_dir)

    async def _redis_subscriber() -> None:
        """Listen on the Redis heartbeat channel and wake the sentinel on each message."""
        try:
            from coordination.redis_coordinator import coordinator as _coord
            import redis.asyncio as _aredis

            redis_url = os.environ.get("REDIS_URL")
            if not redis_url:
                return

            r = _aredis.from_url(redis_url, decode_responses=True,
                                  socket_connect_timeout=3, socket_timeout=3)
            channel = f"autorl:heartbeat:{run_id}"
            pubsub  = r.pubsub()
            await pubsub.subscribe(channel)
            print(f"[sentinel] Redis subscriber active on {channel}")

            async for msg in pubsub.listen():
                if stop_event and stop_event.is_set():
                    break
                if msg and msg.get("type") == "message":
                    _redis_wakeup.set()   # wake the main loop immediately
        except Exception as exc:  # noqa: BLE001
            print(f"[sentinel] Redis subscriber unavailable ({exc}) — using file poll only")

    asyncio.create_task(_redis_subscriber())

    print("[sentinel] starting main loop")
    while not (stop_event and stop_event.is_set()):
        await _check_all()
        # Wait for whichever comes first: a Redis heartbeat event, the 30 s
        # timeout, or the stop signal.  All three paths are safe to handle.
        try:
            done, _ = await asyncio.wait(
                [
                    asyncio.create_task(stop_event.wait()),
                    asyncio.create_task(_redis_wakeup.wait()),
                ],
                timeout=CHECK_INTERVAL_S,
                return_when=asyncio.FIRST_COMPLETED,
            )
        except Exception:  # noqa: BLE001
            done = set()

        if stop_event and stop_event.is_set():
            print("[sentinel] stop_event was set, breaking loop")
            break

        # Reset the wakeup so the next heartbeat can fire it again
        _redis_wakeup.clear()

    # Final sweep: catch anomalies written just before the swarm shuts down.
    print("[sentinel] doing final sweep")
    await _check_all()
    print("[sentinel] stopped")
    print(f"[sentinel] intervention log → {log_path}")
