"""Phase 4.5 — Run report generator.

Formats rankings.json + pipeline_summary.json into run_report.md.
No LLM call — pure template fill.

Usage:
    python evaluator/reporter.py --run-dir runs/latest
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

WEAVE_BASE = "https://wandb.ai/{entity}/weave/calls"


def _sentinel_log(run_dir: str) -> list[dict]:
    """Read sentinel_log.json from the run root directory."""
    log_path = Path(run_dir) / "sentinel_log.json"
    if log_path.exists():
        try:
            data = json.loads(log_path.read_text())
            return data if isinstance(data, list) else [data]
        except Exception:
            pass
    return []


def generate_report(run_dir: str, out_path: str | None = None) -> str:
    """Generate markdown report from run_dir artifacts. Returns the report text."""
    run_dir = os.path.realpath(run_dir)

    # Load summary
    summary: dict = {}
    summary_path = os.path.join(run_dir, "pipeline_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    task = summary.get("task", "RL training run")
    video_path = summary.get("video_path") or ""
    n_agents = summary.get("n_agents", "?")
    n_results = summary.get("n_results", "?")

    # Load rankings
    rankings: dict = summary.get("rankings", {})
    if not rankings:
        rankings_path = os.path.join(run_dir, "rankings.json")
        if os.path.exists(rankings_path):
            with open(rankings_path) as f:
                rankings = json.load(f)

    # Weave + W&B URLs (best-effort)
    weave_project  = os.environ.get("WEAVE_PROJECT", "autorl")
    wandb_entity   = os.environ.get("WANDB_ENTITY", os.environ.get("WANDB_USERNAME", ""))
    project_url    = (
        f"https://wandb.ai/{wandb_entity}/{weave_project}"
        if wandb_entity
        else f"W&B project: `{weave_project}`"
    )
    weave_url = (
        f"https://wandb.ai/{wandb_entity}/weave/calls"
        if wandb_entity
        else None
    )

    lines: list[str] = [
        "# AutoRL Run Report",
        "",
        f"**Task:** {task}",
        f"**Agents:** {n_agents} spawned, {n_results} completed",
        f"**W&B project:** [{project_url}]({project_url})" if wandb_entity else f"**W&B project:** {project_url}",
    ]
    if weave_url:
        lines.append(f"**Weave traces:** [{weave_url}]({weave_url})")
    lines += [f"**Run dir:** `{run_dir}`", ""]

    # ── Rankings ──────────────────────────────────────────────────────────────
    best_agent_id: str | None = None
    for group, entries in rankings.items():
        if not entries:
            continue
        lines += [f"## {group} Results", ""]
        for e in entries:
            rank      = e.get("rank", "?")
            algo      = e.get("algo", "?")
            agent_id  = e.get("agent_id", "?")
            ret       = e.get("mean_return", 0.0)
            rationale = e.get("rationale", "")
            if rank == 1 and best_agent_id is None:
                best_agent_id = agent_id
            lines.append(
                f"{rank}. **{algo}** ({agent_id}) — mean_return: `{ret:.2f}` — {rationale}"
            )
        lines.append("")

    # ── Sentinel interventions ─────────────────────────────────────────────────
    sentinel_entries = _sentinel_log(run_dir)
    if sentinel_entries:
        lines += ["## Sentinel Interventions", ""]
        for e in sentinel_entries:
            agent_id     = e.get("agent_id", "?")
            reason       = e.get("failure_reason", "?")
            failed_hp    = e.get("failed_hparams", {})
            suggested_hp = e.get("llm_suggested_hparams", {})
            outcome      = e.get("outcome", "?")
            ts           = e.get("timestamp", "")
            lines.append(
                f"- **{agent_id}** ({ts[:19]}): `{reason}` "
                f"| failed hparams: `{failed_hp}` "
                f"→ LLM suggested: `{suggested_hp}` "
                f"| outcome: `{outcome}`"
            )
        lines.append("")
    else:
        lines += ["## Sentinel Interventions", "", "_No interventions recorded._", ""]

    # ── Video ─────────────────────────────────────────────────────────────────
    if video_path and os.path.exists(video_path):
        rel = os.path.relpath(video_path, run_dir)
        lines += ["## Model in Action", "", f"Best MuJoCo agent video: `{rel}`", ""]

    report = "\n".join(lines)

    out_path = out_path or os.path.join(run_dir, "run_report.md")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"[reporter] report written → {out_path}")

    # Push a summary entry to the active W&B run (best-effort)
    try:
        import wandb
        if wandb.run is not None:
            update: dict = {"report_path": out_path}
            if best_agent_id:
                update["best_agent"] = best_agent_id
            if wandb_entity:
                update["wandb_project_url"] = project_url
            wandb.run.summary.update(update)
    except Exception:  # noqa: BLE001
        pass

    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Generate run_report.md from a pipeline run dir")
    p.add_argument("--run-dir", default=os.path.join(_PKG_ROOT, "runs", "latest"),
                   help="Run directory (default: runs/latest)")
    p.add_argument("--output", default=None, help="Output path (default: <run-dir>/run_report.md)")
    args = p.parse_args()
    generate_report(args.run_dir, args.output)


if __name__ == "__main__":
    main()
