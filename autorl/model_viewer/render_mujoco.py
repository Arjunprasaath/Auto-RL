"""
MuJoCo video render (Person A).

After the Evaluator picks the best MuJoCo agent, the Orchestrator calls this
script with the winning checkpoint. It rolls out the policy and writes an MP4
whose path is sent to Person B's ModelViewer component.

Run from the autorl/ package root, e.g.:
    python model_viewer/render_mujoco.py --checkpoint results/agent_2/model.zip \
        --env-id HalfCheetah-v5 --algo SAC --output results/best_mujoco.mp4
"""

import argparse
import os

import gymnasium
import imageio
from stable_baselines3 import A2C, PPO, SAC

ALGO_MAP = {"PPO": PPO, "SAC": SAC, "A2C": A2C}


def render_video(
    checkpoint_path: str,
    env_id: str,
    algo: str,
    output_path: str,
    n_steps: int = 500,
):
    env = gymnasium.make(env_id, render_mode="rgb_array")
    model = ALGO_MAP[algo].load(checkpoint_path)

    obs, _ = env.reset()
    frames = []

    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            obs, _ = env.reset()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    imageio.mimsave(output_path, frames, fps=30)
    env.close()
    print(f"Video saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-id", default="HalfCheetah-v5")
    parser.add_argument("--algo", default="SAC", choices=list(ALGO_MAP))
    parser.add_argument("--output", default="results/best_mujoco.mp4")
    parser.add_argument("--n-steps", type=int, default=500)
    args = parser.parse_args()
    render_video(args.checkpoint, args.env_id, args.algo, args.output, args.n_steps)


if __name__ == "__main__":
    main()
