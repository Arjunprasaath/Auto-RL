"""HuggingFace Hub upload for winning AutoRL models.

Uploads the model.zip checkpoint and a model card to a public HF repo so
anyone can download and run the model with a single code snippet.

Requires HF_TOKEN env var (a write-access HuggingFace token).
Safe no-op when HF_TOKEN is absent or huggingface_hub is not installed.
"""

from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime, timezone


def _hf_enabled() -> bool:
    return bool(os.environ.get("HF_TOKEN"))


def _slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].strip("-") or "model"


def _repo_name(model_name: str | None, algo: str, env_id: str, ts: str) -> str:
    """Build repo slug: user name + timestamp, or algo/env fallback."""
    if model_name and model_name.strip():
        return f"autorl-{_slugify(model_name.strip())}-{ts}"
    env_slug = _slugify(env_id)
    return f"autorl-{algo.lower()}-{env_slug}-{ts}"


def push_model_to_hub(
    model_path: str,
    agent_id: str,
    algo: str,
    env_id: str,
    mean_return: float,
    std_return: float = 0.0,
    steps_trained: int = 0,
    model_name: str | None = None,
    pushed_at: str | None = None,
) -> tuple[str, str]:
    """Upload the winning model checkpoint to HuggingFace Hub.

    Creates a repo named ``{username}/autorl-{name}-{timestamp}`` (or algo/env fallback)
    and uploads:
      - ``model.zip``   — the SB3 checkpoint
      - ``README.md``   — model card with performance + usage snippet

    Returns
    -------
    (repo_url, code_snippet)
        ``repo_url``      — https://huggingface.co/{username}/{repo_name}
        ``code_snippet``  — standalone Python code users can copy/paste
    """
    if not _hf_enabled():
        raise ValueError("HF_TOKEN not set — cannot push to HuggingFace")

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise RuntimeError("huggingface_hub not installed") from e

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)

    username = api.whoami()["name"]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    display_name = (model_name or "").strip() or f"{algo} on {env_id}"
    pushed_at = pushed_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    repo_name = _repo_name(model_name, algo, env_id, ts)
    repo_id = f"{username}/{repo_name}"

    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

    # Upload model checkpoint
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="model.zip",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"AutoRL: {algo} on {env_id} · mean_return={mean_return:.1f}",
    )

    # Build usage code snippet (returned to the UI as copy-paste text)
    code_snippet = _build_code_snippet(repo_id, algo, env_id)

    # Write and upload model card
    card = _build_model_card(
        repo_id=repo_id,
        display_name=display_name,
        algo=algo,
        env_id=env_id,
        mean_return=mean_return,
        std_return=std_return,
        steps_trained=steps_trained,
        pushed_at=pushed_at,
        code_snippet=code_snippet,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as fh:
        fh.write(card)
        readme_path = fh.name

    try:
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="AutoRL: update model card",
        )
    finally:
        os.unlink(readme_path)

    repo_url = f"https://huggingface.co/{repo_id}"
    print(f"[hf] model pushed → {repo_url}")
    return repo_url, code_snippet


# ── Template helpers ──────────────────────────────────────────────────────────


def _build_code_snippet(repo_id: str, algo: str, env_id: str) -> str:
    return f"""\
# pip install stable-baselines3 huggingface_hub gymnasium mujoco imageio
from huggingface_hub import hf_hub_download
from stable_baselines3 import {algo}
import gymnasium as gym
import imageio

# Load the winning model from HuggingFace
model_path = hf_hub_download(repo_id="{repo_id}", filename="model.zip")
model = {algo}.load(model_path)

# Record a video rollout
env = gym.make("{env_id}", render_mode="rgb_array")
obs, _ = env.reset(seed=0)
frames, done = [], False
while not done:
    frames.append(env.render())
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
env.close()

imageio.mimsave("rollout.mp4", frames, fps=30)
print("✓ Saved rollout.mp4")\
"""


def _build_model_card(
    repo_id: str,
    display_name: str,
    algo: str,
    env_id: str,
    mean_return: float,
    std_return: float,
    steps_trained: int,
    pushed_at: str,
    code_snippet: str,
) -> str:
    return f"""\
---
library_name: stable-baselines3
tags:
  - reinforcement-learning
  - autorl
  - {algo.lower()}
  - {env_id.lower().replace("/", "-")}
---

# {display_name}

Trained automatically by **[AutoRL](https://github.com/wandb/autorl)** — a
multi-agent RL training race powered by W&B, Weave, and OpenAI.

## Performance

| Metric | Value |
|--------|-------|
| Mean Return | **{mean_return:.2f}** ± {std_return:.2f} |
| Steps Trained | {steps_trained:,} |
| Algorithm | {algo} |
| Environment | {env_id} |
| Pushed At | {pushed_at} |
| Repo | `{repo_id}` |

## Usage

```python
{code_snippet}
```
"""
