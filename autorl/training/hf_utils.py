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


def _hf_enabled() -> bool:
    return bool(os.environ.get("HF_TOKEN"))


def push_model_to_hub(
    model_path: str,
    agent_id: str,
    algo: str,
    env_id: str,
    mean_return: float,
    std_return: float = 0.0,
    steps_trained: int = 0,
) -> tuple[str, str]:
    """Upload the winning model checkpoint to HuggingFace Hub.

    Creates (or updates) a repo named  ``{username}/autorl-{algo}-{env_slug}``
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

    # Build a clean repo name: autorl-ppo-hopper-v5
    env_slug = re.sub(r"[^a-z0-9]+", "-", env_id.lower()).strip("-")
    algo_slug = algo.lower()
    repo_name = f"autorl-{algo_slug}-{env_slug}"
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
        algo=algo,
        env_id=env_id,
        mean_return=mean_return,
        std_return=std_return,
        steps_trained=steps_trained,
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
    algo: str,
    env_id: str,
    mean_return: float,
    std_return: float,
    steps_trained: int,
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

# AutoRL — {algo} on {env_id}

Trained automatically by **[AutoRL](https://github.com/wandb/autorl)** — a
multi-agent RL training race powered by W&B, Weave, and OpenAI.

## Performance

| Metric | Value |
|--------|-------|
| Mean Return | **{mean_return:.2f}** ± {std_return:.2f} |
| Steps Trained | {steps_trained:,} |
| Algorithm | {algo} |
| Environment | {env_id} |

## Usage

```python
{code_snippet}
```
"""
