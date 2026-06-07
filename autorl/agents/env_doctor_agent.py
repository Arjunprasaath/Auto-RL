"""Environment Doctor Agent — detects subprocess errors and iteratively fixes them.

Called by run_training_agent when a subprocess exits with a non-zero code.

Workflow
--------
1. Check if the error is a training-divergence (NaN loss, gradient explosion) — skip
   those because the Doom Loop Sentinel already owns that recovery path.
2. Match stderr against a table of known patterns (fast path, no LLM).
3. If no match, ask the LLM for fix commands (slow path), passing in any commands
   that were already tried so the LLM won't suggest them again.
4. Run each command sequentially in the real terminal; stop on first failure.
5. Return DoctorResult — the caller (training_agent.py) retries the subprocess and
   calls back again if it still fails, up to MAX_DOCTOR_ITERATIONS total rounds.
6. Each attempt is appended to sentinel_log.json so the UI shows the full history.

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

# ── Training-divergence patterns — owned by the Sentinel, skip the doctor ─────
# These are numerical training failures, not environment setup errors.
_DIVERGENCE_PATTERNS = [
    r"nan",
    r"inf(?:inity)?",
    r"gradient.*explod",
    r"loss.*overflow",
    r"value.*error.*nan",
    r"floating.?point.*overflow",
]


def is_training_divergence(stderr: str) -> bool:
    """Return True when stderr looks like a numerical training failure.

    These errors are handled by the Doom Loop Sentinel (NaN loss detection via
    heartbeat anomaly field).  The Doctor should skip them to avoid interfering.
    """
    lower = stderr.lower()
    return any(re.search(p, lower) for p in _DIVERGENCE_PATTERNS)


# ── Known error patterns → fix commands (no LLM needed) ───────────────────────
# Each entry: (regex, [commands])
# Commands are run in order; the first failure stops the chain.
_KNOWN: list[tuple[str, list[str]]] = [
    # Atari / ALE
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
    # package version conflicts
    (r"cannot import name.*from.*torch",        [f"{_PIP} --upgrade torch"]),
    (r"cannot import name.*from.*gymnasium",    [f"{_PIP} --upgrade gymnasium"]),
    (r"ImportError.*version",                   [f"{_PIP} --upgrade stable-baselines3 gymnasium"]),
    # OpenGL / display (headless rendering)
    (r"No module named 'OpenGL'",               [f"{_PIP} pyopengl"]),
    (r"pyglet.*display",                        [f"{_PIP} pyglet", "export DISPLAY=:0"]),
    (r"GLFW.*error",                            [f"{_PIP} glfw"]),
    # CUDA / device
    (r"CUDA.*not available",                    [f"{_PIP} torch --index-url https://download.pytorch.org/whl/cpu"]),
    (r"device.*cuda.*not found",                [f"{_PIP} torch --index-url https://download.pytorch.org/whl/cpu"]),
    # disk / permissions
    (r"No space left on device",                ["find /tmp -maxdepth 1 -mtime +1 -delete"]),
    (r"Permission denied",                      [f"chmod -R u+rwX {_PKG_ROOT}"]),
    # wandb version conflict (seen in logs)
    (r"wandb.*version.*incompatible",           [f"{_PIP} --upgrade wandb"]),
    (r"cannot import name.*wandb",              [f"{_PIP} --upgrade wandb"]),
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


def _llm_fix(stderr: str, already_tried: set[str]) -> tuple[list[str], str]:
    """Ask the LLM for fix commands when no known pattern matched.

    Passes already-tried commands so the LLM won't repeat them.
    """
    tried_block = ""
    if already_tried:
        tried_block = (
            "\nThese commands have already been tried and did NOT fix the problem — "
            "do NOT suggest them again:\n"
            + "\n".join(f"  - {c}" for c in sorted(already_tried))
            + "\n"
        )

    prompt = f"""You are an auto-fix agent for a Python RL training system running on macOS/Linux.
A training subprocess failed. Here is the end of its stderr output:

{stderr[-3000:]}
{tried_block}
Diagnose the error and return ONLY the shell command(s) that will fix it.
Only return install / setup commands (pip install, conda install, apt-get, etc.).
Never suggest modifying source code. Prefer the active venv's pip: {_PIP}

Return ONLY valid JSON:
{{"commands": ["{_PIP} ...", "..."], "reasoning": "one-line explanation"}}"""
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
        # Filter out any commands the caller already tried
        cmds = [c for c in cmds if c not in already_tried]
        return cmds, raw.get("reasoning", "LLM-suggested fix")
    except Exception as exc:
        print(f"[doctor] LLM call failed: {exc}")
        return [], f"LLM unavailable: {exc}"


def _run_cmd(cmd: str) -> tuple[bool, str]:
    """Run a shell fix command in the real terminal.

    Returns (success, combined_output).
    """
    print(f"[doctor] ▶ {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=180,
        )
        output = (result.stdout + result.stderr).strip()
        if result.returncode == 0:
            print(f"[doctor] ✓ success")
        else:
            print(f"[doctor] ✗ failed (exit {result.returncode})")
            for line in output.splitlines()[-5:]:
                print(f"[doctor]   {line}")
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        print(f"[doctor] ✗ timeout (180 s)")
        return False, "timeout"
    except Exception as exc:
        print(f"[doctor] ✗ error: {exc}")
        return False, str(exc)


def _append_log(
    results_dir: str,
    agent_id: str,
    commands: list[str],
    success: bool,
    reasoning: str,
    stderr: str,
    attempt: int = 1,
) -> None:
    """Append a doctor entry to sentinel_log.json for UI display."""
    log_path = Path(results_dir) / "sentinel_log.json"
    try:
        entries = json.loads(log_path.read_text()) if log_path.exists() else []
    except Exception:
        entries = []

    label = f"attempt_{attempt}" if attempt > 1 else "environment_setup_error"
    entries.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_id": agent_id,
        "failure_reason": label,
        "failed_hparams": {},
        "llm_suggested_hparams": {"fix_commands": commands},
        "rationale": reasoning,
        "outcome": "fixed_retrying" if success else "fix_failed",
        "doctor_stderr": stderr[-400:],
    })

    tmp = log_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, indent=2))
    os.replace(tmp, log_path)


def diagnose_and_fix(
    stderr: str,
    agent_id: str,
    results_dir: str,
    already_tried: set[str] | None = None,
    attempt: int = 1,
) -> DoctorResult:
    """Diagnose a subprocess failure and attempt to fix it.

    Args:
        stderr:        captured stderr from the failed subprocess
        agent_id:      e.g. "agent_1" (for logging)
        results_dir:   run directory (sentinel_log.json written here)
        already_tried: commands run in previous attempts — the LLM will not
                       repeat them; new commands are added to this set in-place
        attempt:       which iteration this is (logged to sentinel_log.json)

    Returns:
        DoctorResult with .fixed=True if all fix commands succeeded.
    """
    return _diagnose_and_fix_impl(stderr, agent_id, results_dir, already_tried or set(), attempt)


@weave.op(name="EnvDoctor")
def _diagnose_and_fix_impl(
    stderr: str,
    agent_id: str,
    results_dir: str,
    already_tried: set[str],
    attempt: int,
) -> DoctorResult:
    print(f"[doctor] diagnosing failure for {agent_id} (attempt {attempt})")

    # Fast path: known pattern — filter out already-tried commands
    raw_cmds = _known_fix(stderr)
    reasoning = "matched known error pattern"
    cmds = [c for c in (raw_cmds or []) if c not in already_tried] if raw_cmds else None

    # Slow path: LLM
    if not cmds:
        print(f"[doctor] no known pattern — calling LLM (attempt {attempt})")
        cmds, reasoning = _llm_fix(stderr, already_tried)

    if not cmds:
        print(f"[doctor] no fix found for {agent_id} (attempt {attempt})")
        _append_log(results_dir, agent_id, [], False,
                    "no fix found — exhausted all options", stderr, attempt)
        return DoctorResult(fixed=False, commands=[], reasoning="no fix found")

    print(f"[doctor] fix plan for {agent_id} attempt {attempt}: {cmds}  ({reasoning})")
    ran: list[str] = []
    for cmd in cmds:
        ok, output = _run_cmd(cmd)
        ran.append(cmd)
        already_tried.add(cmd)
        if not ok:
            _append_log(results_dir, agent_id, ran, False, reasoning, stderr, attempt)
            return DoctorResult(fixed=False, commands=ran, reasoning=reasoning)

    print(f"[doctor] ✓ all fixes applied for {agent_id} attempt {attempt} — will retry training")
    _append_log(results_dir, agent_id, ran, True, reasoning, stderr, attempt)
    return DoctorResult(fixed=True, commands=ran, reasoning=reasoning)
