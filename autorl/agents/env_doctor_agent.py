"""Environment Doctor Agent — detects subprocess setup errors and auto-fixes them.

Called by run_training_agent when a subprocess exits with a non-zero code.

Workflow
--------
1. Match stderr against a table of known patterns (fast path, no LLM).
2. If no match, ask the LLM for fix commands (slow path).
3. Run each command sequentially; stop on first failure.
4. Write a log entry to sentinel_log.json (same file the Sentinel uses) so the
   UI can display the intervention without any extra plumbing.
5. Return DoctorResult.fixed — if True the caller can retry the agent.

Only shell commands that install or configure the environment are permitted.
The agent NEVER modifies source files.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PKG_ROOT, ".env"))

import weave
from openai import OpenAI

if os.environ.get("WANDB_API_KEY") and not os.environ.get("WEAVE_DISABLED"):
    try:
        weave.init(os.environ.get("WEAVE_PROJECT", "autorl"))
    except Exception:
        pass

_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL   = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Always use the same Python / pip that is running this process — ensures
# packages land in the correct venv, not the system Python.
_PIP = f"{sys.executable} -m pip install --quiet"

# ── Known error patterns → fix commands (no LLM needed) ───────────────────────
# Each entry: (regex, [commands])
# Commands are run in order; the first failure stops the chain.
_KNOWN: list[tuple[str, list[str]]] = [
    # Atari / ALE  (ale-py may be installed but needs explicit registration —
    # env_utils.py already does that; if we still get here, the package is missing)
    (r"Namespace ALE not found",                [f"{_PIP} gymnasium[atari] ale-py"]),
    (r"No module named 'ale_py'",               [f"{_PIP} gymnasium[atari] ale-py"]),
    # Box2D
    (r"No module named 'box2d'",                [f"{_PIP} swig", f"{_PIP} gymnasium[box2d]"]),
    (r"DependencyNotInstalled.*box2d",          [f"{_PIP} swig", f"{_PIP} gymnasium[box2d]"]),
    (r"No module named 'pygame'",               [f"{_PIP} pygame"]),
    # MuJoCo
    (r"No module named 'mujoco'",               [f"{_PIP} gymnasium[mujoco] mujoco"]),
    (r"DependencyNotInstalled.*mujoco",         [f"{_PIP} gymnasium[mujoco] mujoco"]),
    # imageio / ffmpeg
    (r"No module named 'imageio'",              [f"{_PIP} imageio imageio-ffmpeg"]),
    (r"RuntimeError.*FFmpeg",                   [f"{_PIP} imageio-ffmpeg"]),
    # matplotlib
    (r"No module named 'matplotlib'",           [f"{_PIP} matplotlib"]),
    # huggingface
    (r"No module named 'datasets'",             [f"{_PIP} datasets"]),
    (r"No module named 'huggingface_hub'",      [f"{_PIP} huggingface_hub"]),
    # torch / stable-baselines3
    (r"No module named 'stable_baselines3'",    [f"{_PIP} stable-baselines3"]),
    (r"No module named 'torch'",                [f"{_PIP} torch"]),
]


# ── Public result type ─────────────────────────────────────────────────────────


class DoctorResult:
    def __init__(self, fixed: bool, commands: list[str], reasoning: str):
        self.fixed     = fixed
        self.commands  = commands
        self.reasoning = reasoning

    def __repr__(self) -> str:
        return f"DoctorResult(fixed={self.fixed}, commands={self.commands})"


# ── Core logic ─────────────────────────────────────────────────────────────────


def _known_fix(stderr: str) -> list[str] | None:
    for pattern, cmds in _KNOWN:
        if re.search(pattern, stderr, re.IGNORECASE):
            return cmds
    return None


def _llm_fix(stderr: str) -> tuple[list[str], str]:
    """Ask the LLM for fix commands when no known pattern matched."""
    prompt = f"""You are an auto-fix agent for a Python RL training system running on macOS/Linux.
A training subprocess failed. Here is the end of its stderr output:

{stderr[-3000:]}

Diagnose the error and return ONLY the shell command(s) that will fix it.
Only return install / setup commands (pip install, conda install, etc.).
Never suggest modifying source code.

Return ONLY valid JSON:
{{"commands": ["pip install ...", "..."], "reasoning": "one-line explanation"}}"""
    try:
        resp = _client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = json.loads(resp.choices[0].message.content)
        # Rewrite bare "pip install" → venv pip so packages land in the right env
        cmds = [
            re.sub(r"^pip install", _PIP, c) if c.startswith("pip install") else c
            for c in raw.get("commands", [])
        ]
        return cmds, raw.get("reasoning", "LLM-suggested fix")
    except Exception as exc:
        print(f"[doctor] LLM call failed: {exc}")
        return [], f"LLM unavailable: {exc}"


def _run_cmd(cmd: str) -> bool:
    """Run a shell fix command. Returns True on success."""
    print(f"[doctor] ▶ {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=180,
        )
        if result.returncode == 0:
            print(f"[doctor] ✓ success")
        else:
            print(f"[doctor] ✗ failed (exit {result.returncode})")
            tail = (result.stdout + result.stderr).strip().splitlines()
            for line in tail[-5:]:
                print(f"[doctor]   {line}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[doctor] ✗ timeout (180 s)")
        return False
    except Exception as exc:
        print(f"[doctor] ✗ error: {exc}")
        return False


def _append_log(results_dir: str, agent_id: str, commands: list[str],
                success: bool, reasoning: str, stderr: str) -> None:
    """Append a doctor entry to sentinel_log.json for UI display."""
    log_path = Path(results_dir) / "sentinel_log.json"
    try:
        entries = json.loads(log_path.read_text()) if log_path.exists() else []
    except Exception:
        entries = []

    entries.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_id": agent_id,
        "failure_reason": "environment_setup_error",
        "failed_hparams": {},
        "llm_suggested_hparams": {"fix_commands": commands},
        "rationale": reasoning,
        "outcome": "fixed_retrying" if success else "fix_failed",
        "doctor_stderr": stderr[-400:],
    })

    tmp = log_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=2))
    os.replace(tmp, log_path)


def diagnose_and_fix(stderr: str, agent_id: str, results_dir: str) -> DoctorResult:
    """Diagnose a subprocess failure and attempt to fix it.

    Args:
        stderr:      captured stderr text from the failed subprocess
        agent_id:    e.g. "agent_1" (for logging)
        results_dir: run directory (sentinel_log.json written here)

    Returns:
        DoctorResult with .fixed=True if all fix commands succeeded.
    """
    return _diagnose_and_fix_impl(stderr, agent_id, results_dir)


@weave.op(name="EnvDoctor")
def _diagnose_and_fix_impl(stderr: str, agent_id: str, results_dir: str) -> DoctorResult:
    print(f"[doctor] diagnosing failure for {agent_id}")

    # Fast path: known pattern
    cmds = _known_fix(stderr)
    reasoning = "matched known error pattern"

    # Slow path: LLM
    if not cmds:
        print(f"[doctor] no known pattern — calling LLM")
        cmds, reasoning = _llm_fix(stderr)

    if not cmds:
        print(f"[doctor] no fix found for {agent_id}")
        _append_log(results_dir, agent_id, [], False, "no fix found", stderr)
        return DoctorResult(fixed=False, commands=[], reasoning="no fix found")

    print(f"[doctor] fix plan for {agent_id}: {cmds}  ({reasoning})")
    ran: list[str] = []
    for cmd in cmds:
        ok = _run_cmd(cmd)
        ran.append(cmd)
        if not ok:
            _append_log(results_dir, agent_id, ran, False, reasoning, stderr)
            return DoctorResult(fixed=False, commands=ran, reasoning=reasoning)

    print(f"[doctor] ✓ all fixes applied for {agent_id} — will retry")
    _append_log(results_dir, agent_id, ran, True, reasoning, stderr)
    return DoctorResult(fixed=True, commands=ran, reasoning=reasoning)
