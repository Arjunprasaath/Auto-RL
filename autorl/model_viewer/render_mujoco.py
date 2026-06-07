# Force pygame (used by Toy Text / Box2D envs) to render offscreen.
# Must be set BEFORE any gymnasium / pygame import to avoid SDL2 library
# conflicts with cv2 on macOS (both ship their own libSDL2).
import os
os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

"""
Agent video renderer — handles all Gymnasium environments.

Rolls out a trained SB3 policy (PPO / SAC / A2C) in rgb_array mode and
writes an MP4. Works across all supported env families:
  - MuJoCo continuous control (gymnasium[mujoco])
  - Classic Control (CartPole, Pendulum, MountainCar, Acrobot)
  - Toy Text / Grid World (FrozenLake, Taxi, CliffWalking)
  - Box2D (LunarLander, BipedalWalker) — if gymnasium[box2d] is installed

FPS is chosen automatically per env family so videos look natural:
  - MuJoCo / Classic Control / Box2D → 30 fps
  - Toy Text / Grid World            →  4 fps  (steps are readable)

Run from the autorl/ package root, e.g.:
    python model_viewer/render_mujoco.py \
        --checkpoint results/agent_2/model.zip \
        --env-id HalfCheetah-v5 --algo SAC \
        --output results/best_mujoco.mp4

    python model_viewer/render_mujoco.py \
        --checkpoint results/agent_1/model.zip \
        --env-id FrozenLake-v1 --algo PPO \
        --output results/frozen_lake.mp4 --n-episodes 5
"""

import argparse
import sys

import imageio
import numpy as np
from stable_baselines3 import A2C, PPO, SAC

# Allow imports from the autorl package root
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from training.env_utils import make_env  # noqa: E402

ALGO_MAP = {"PPO": PPO, "SAC": SAC, "A2C": A2C}

# Env families detected by env_id prefix / substring
_SLOW_FPS_ENVS = {
    "FrozenLake", "Taxi", "CliffWalking", "Blackjack",
}

# Default step counts per env family — enough to capture interesting behaviour
_DEFAULT_STEPS: dict[str, int] = {
    "mujoco":   1000,  # MuJoCo: ~33 s at 30 fps
    "classic":   500,  # CartPole / Pendulum: a few episodes
    "box2d":     800,  # LunarLander / BipedalWalker
    "toytext":   200,  # FrozenLake / Taxi: many short episodes
}


def _detect_family(env_id: str) -> str:
    """Heuristic env family detection from the env_id string."""
    eid = env_id.lower()
    if any(k.lower() in eid for k in _SLOW_FPS_ENVS):
        return "toytext"
    if any(k in eid for k in ("halfcheetah", "hopper", "ant", "walker", "swimmer",
                               "humanoid", "reacher", "pusher", "invertedpendulum",
                               "mujoco")):
        return "mujoco"
    if any(k in eid for k in ("lunarlander", "bipedalwalker", "carracing")):
        return "box2d"
    return "classic"


def _fps_for_env(env_id: str) -> int:
    return 4 if _detect_family(env_id) == "toytext" else 30


def _default_steps(env_id: str) -> int:
    return _DEFAULT_STEPS[_detect_family(env_id)]


def render_video(
    checkpoint_path: str,
    env_id: str,
    algo: str,
    output_path: str,
    n_steps: int | None = None,
    n_episodes: int | None = None,
) -> str:
    """Roll out a policy and save an MP4.

    Either n_steps (hard step cap) or n_episodes (collect N complete episodes)
    can be specified. If both are given, whichever limit is hit first wins.
    If neither is given, n_steps defaults based on env family.
    """
    if algo.upper() not in ALGO_MAP:
        raise ValueError(f"Unknown algo '{algo}'. Supported: {list(ALGO_MAP)}")

    # make_env applies FlattenObservation for Tuple obs spaces (Blackjack),
    # matching the wrapper stack used during training.
    env = make_env(env_id, render_mode="rgb_array")

    model = ALGO_MAP[algo.upper()].load(checkpoint_path)

    step_limit = n_steps if n_steps is not None else _default_steps(env_id)
    ep_limit   = n_episodes  # None means no episode cap

    obs, _ = env.reset()
    frames: list = []
    steps_done = 0
    episodes_done = 0

    while steps_done < step_limit:
        # Discrete/toy-text envs return a 0-d numpy scalar; SB3 predict()
        # requires at least a 1-D array — reshape silently when needed.
        obs_in = np.atleast_1d(np.asarray(obs))
        action, _ = model.predict(obs_in, deterministic=True)
        # Toy-text / Discrete envs (FrozenLake, Taxi…) index P[state][action]
        # as a plain Python int — unwrap the single-element array SB3 returns.
        if isinstance(action, np.ndarray) and action.size == 1:
            action = action.item()
        obs, _reward, terminated, truncated, _info = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        steps_done += 1

        if terminated or truncated:
            episodes_done += 1
            if ep_limit is not None and episodes_done >= ep_limit:
                break
            obs, _ = env.reset()  # obs will be normalised on next loop iter

    env.close()

    if not frames:
        raise RuntimeError(
            f"No frames captured for '{env_id}'. "
            "Check that render_mode='rgb_array' is supported by this environment."
        )

    fps = _fps_for_env(env_id)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)
    print(
        f"Video saved: {output_path} "
        f"({len(frames)} frames, {fps} fps, "
        f"{steps_done} steps, {episodes_done} episodes)"
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Render a trained SB3 agent to MP4 for any Gymnasium environment."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model.zip saved by SB3")
    parser.add_argument("--env-id",  default="HalfCheetah-v5",
                        help="Gymnasium environment id")
    parser.add_argument("--algo",    default="SAC", choices=list(ALGO_MAP),
                        help="SB3 algorithm used to train the checkpoint")
    parser.add_argument("--output",  default="results/best_agent.mp4",
                        help="Output MP4 path")
    parser.add_argument("--n-steps", type=int, default=None,
                        help="Max steps to record (default: auto per env family)")
    parser.add_argument("--n-episodes", type=int, default=None,
                        help="Stop after this many complete episodes")
    args = parser.parse_args()

    render_video(
        checkpoint_path=args.checkpoint,
        env_id=args.env_id,
        algo=args.algo,
        output_path=args.output,
        n_steps=args.n_steps,
        n_episodes=args.n_episodes,
    )


if __name__ == "__main__":
    main()
