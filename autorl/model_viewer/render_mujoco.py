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

import gymnasium as gym
from gymnasium import spaces

from training.env_utils import make_env  # noqa: E402

ALGO_MAP = {"PPO": PPO, "SAC": SAC, "A2C": A2C}


class ObsPadWrapper(gym.ObservationWrapper):
    """Pad or truncate flat observations to match the policy's expected obs_dim.

    Used when rendering a world-model-trained policy in the original real env:
    the policy was trained on padded observations (e.g. JAT pads to 105-dim)
    but the real env only emits the native observation (e.g. Ant-v5 → 27-dim).
    Padding with zeros restores the format the policy learned from.
    """

    def __init__(self, env: gym.Env, target_dim: int):
        super().__init__(env)
        self.target_dim = target_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(target_dim,), dtype=np.float32,
        )

    def observation(self, obs) -> np.ndarray:
        obs = np.array(obs, dtype=np.float32).flatten()
        if obs.shape[0] < self.target_dim:
            obs = np.pad(obs, (0, self.target_dim - obs.shape[0]))
        else:
            obs = obs[: self.target_dim]
        return obs

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


def _render_world_model_video(
    checkpoint_path: str,
    env_id: str,
    algo: str,
    output_path: str,
    n_steps: int = 500,
    n_episodes: int = 3,
) -> str:
    """Render a WorldModel-v0 policy as a matplotlib trajectory animation.

    Since the world model has no visual output (it's a neural network, not a
    real physics sim), we visualise the agent's behaviour as a scrolling
    time-series: reward per step + the first few observation dimensions.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.gridspec import GridSpec

    env   = make_env(env_id)
    model = ALGO_MAP[algo.upper()].load(checkpoint_path)

    obs_dim = env.observation_space.shape[0]
    n_obs_plot = min(obs_dim, 6)   # plot at most 6 obs dims

    # Collect trajectories
    all_rewards: list[float] = []
    all_obs: list[list[float]] = []
    ep_bounds: list[int] = [0]     # step indices where episodes start

    obs, _ = env.reset()
    steps_done = episodes_done = 0

    while steps_done < n_steps:
        action, _ = model.predict(np.atleast_1d(np.asarray(obs)), deterministic=True)
        if isinstance(action, np.ndarray) and action.size == 1:
            action = action.item()
        obs, reward, terminated, truncated, _ = env.step(action)
        all_rewards.append(float(reward))
        all_obs.append(list(obs[:n_obs_plot]))
        steps_done += 1
        if terminated or truncated:
            episodes_done += 1
            ep_bounds.append(steps_done)
            if episodes_done >= n_episodes:
                break
            obs, _ = env.reset()
    env.close()

    rewards_arr = np.array(all_rewards)
    obs_arr     = np.array(all_obs)        # (T, n_obs_plot)
    T           = len(rewards_arr)

    # Build animation: scroll a window of 80 steps across the data
    WINDOW = min(80, T)
    fig    = plt.figure(figsize=(10, 5), dpi=100, facecolor="#111111")
    fig.patch.set_facecolor("#111111")
    gs     = GridSpec(2, 1, figure=fig, hspace=0.4)

    ax_rew = fig.add_subplot(gs[0])
    ax_obs = fig.add_subplot(gs[1])

    colors = ["#6ee7b7", "#60a5fa", "#f472b6", "#fbbf24", "#a78bfa", "#34d399"]

    for ax in (ax_rew, ax_obs):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#9ca3af", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#374151")

    ax_rew.set_ylabel("Reward", color="#9ca3af", fontsize=9)
    ax_obs.set_ylabel("Obs dims", color="#9ca3af", fontsize=9)
    ax_obs.set_xlabel("Step", color="#9ca3af", fontsize=9)
    fig.suptitle(f"{algo} in World Model — {T} steps, {episodes_done} episodes",
                 color="#e5e7eb", fontsize=11, fontweight="bold")

    rew_line,  = ax_rew.plot([], [], color=colors[0], lw=1.5)
    obs_lines  = [ax_obs.plot([], [], color=colors[i % len(colors)],
                               lw=1.2, label=f"obs[{i}]")[0]
                  for i in range(n_obs_plot)]
    ax_obs.legend(loc="upper right", fontsize=7, facecolor="#111111",
                  labelcolor="#9ca3af", framealpha=0.6)

    # Episode boundary lines
    for b in ep_bounds[1:]:
        ax_rew.axvline(b, color="#6b7280", lw=0.8, ls="--", alpha=0.5)
        ax_obs.axvline(b, color="#6b7280", lw=0.8, ls="--", alpha=0.5)

    def _init():
        rew_line.set_data([], [])
        for ln in obs_lines:
            ln.set_data([], [])
        return [rew_line, *obs_lines]

    def _update(frame: int):
        end   = frame + 1
        start = max(0, end - WINDOW)
        xs    = np.arange(start, end)
        rew_line.set_data(xs, rewards_arr[start:end])
        ax_rew.set_xlim(start, start + WINDOW)
        ax_rew.set_ylim(rewards_arr.min() - 0.1, rewards_arr.max() + 0.1)
        for i, ln in enumerate(obs_lines):
            ln.set_data(xs, obs_arr[start:end, i])
        ax_obs.set_xlim(start, start + WINDOW)
        obs_min, obs_max = obs_arr.min(), obs_arr.max()
        pad = max(0.1, (obs_max - obs_min) * 0.1)
        ax_obs.set_ylim(obs_min - pad, obs_max + pad)
        return [rew_line, *obs_lines]

    ani = animation.FuncAnimation(
        fig, _update, frames=T, init_func=_init,
        blit=True, interval=50,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = animation.FFMpegWriter(fps=20, bitrate=800)
    ani.save(output_path, writer=writer)
    plt.close(fig)

    print(f"World model video saved: {output_path} ({T} steps, {episodes_done} episodes)")
    return output_path


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

    # World model envs have no visual output — use trajectory animation instead
    if env_id == "WorldModel-v0":
        return _render_world_model_video(
            checkpoint_path, env_id, algo, output_path,
            n_steps=n_steps or 500,
            n_episodes=n_episodes or 3,
        )

    # make_env applies FlattenObservation for Tuple obs spaces (Blackjack),
    # matching the wrapper stack used during training.
    env = make_env(env_id, render_mode="rgb_array")

    model = ALGO_MAP[algo.upper()].load(checkpoint_path)

    # Obs-shape mismatch: e.g. JAT-trained policy expects (105,) but Ant-v5 gives (27,).
    # Pad the real env's observations to match the policy's expected shape.
    env_obs_dim    = env.observation_space.shape[0] if env.observation_space.shape else None
    model_obs_dim  = model.observation_space.shape[0] if model.observation_space.shape else None
    if env_obs_dim and model_obs_dim and env_obs_dim != model_obs_dim:
        print(
            f"[render] obs shape mismatch: env={env_obs_dim} model={model_obs_dim} "
            f"→ wrapping with ObsPadWrapper(target_dim={model_obs_dim})"
        )
        env = ObsPadWrapper(env, model_obs_dim)

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
