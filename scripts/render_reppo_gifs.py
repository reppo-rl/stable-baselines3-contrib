
import warnings
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3.common.env_util import make_vec_env

from sb3_contrib import REPPO

GIF_DIR = Path(__file__).parent / "gifs"
GIF_DIR.mkdir(exist_ok=True)


def train(env_id: str, policy_kwargs: dict, total_timesteps: int,
          n_envs: int = 4, n_steps: int = 1024, batch_size: int = 128,
          n_epochs: int = 8, seed: int = 42) -> REPPO:
    env = make_vec_env(env_id, n_envs=n_envs, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = REPPO(
            "MlpPolicy", env,
            n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
            seed=seed, policy_kwargs=policy_kwargs,
        )
    print(f"  Training {total_timesteps:,} steps ...", flush=True)
    model.learn(total_timesteps=total_timesteps)
    env.close()
    return model


def render_gif(model: REPPO, env_id: str, path: Path,
               n_episodes: int = 3, fps: int = 30) -> None:
    env = gym.make(env_id, render_mode="rgb_array")
    frames = []
    ep_returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        ep_return = 0.0
        done = False
        while not done:
            frame = env.render()
            frames.append(frame)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
        ep_returns.append(ep_return)

    env.close()
    imageio.mimsave(str(path), frames, fps=fps, loop=0)
    mean_r = np.mean(ep_returns)
    print(f"  Saved {path.name}  ({len(frames)} frames, "
          f"{n_episodes} eps, mean return={mean_r:.1f})")




CONFIGS = [
    {
        "label": "Pendulum-v1",
        "env_id": "Pendulum-v1",
        "gif": GIF_DIR / "pendulum_reppo.gif",
        "total_timesteps": 200_000,
        "n_envs": 4,
        "n_steps": 1024,
        "batch_size": 128,
        "n_epochs": 8,
        "policy_kwargs": dict(
            vmin=-1600.0, vmax=0.0,
            net_arch={"pi": [256, 256], "qf": [256, 256]},
        ),
        "n_episodes": 2,
        "fps": 30,
    },
    {
        "label": "MountainCarContinuous-v0",
        "env_id": "MountainCarContinuous-v0",
        "gif": GIF_DIR / "mountaincar_reppo.gif",
        "total_timesteps": 300_000,
        "n_envs": 4,
        "n_steps": 1024,
        "batch_size": 128,
        "n_epochs": 10,
        "policy_kwargs": dict(
            vmin=-150.0, vmax=150.0,
            net_arch={"pi": [256, 256], "qf": [256, 256]},
        ),
        "n_episodes": 2,
        "fps": 30,
    },
    {
        "label": "BipedalWalker-v3",
        "env_id": "BipedalWalker-v3",
        "gif": GIF_DIR / "bipedal_reppo.gif",
        "total_timesteps": 1_000_000,
        "n_envs": 4,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 8,
        "policy_kwargs": dict(
            vmin=-300.0, vmax=300.0,
            net_arch={"pi": [256, 256, 256], "qf": [256, 256, 256]},
        ),
        "n_episodes": 2,
        "fps": 50,
    },
]

for cfg in CONFIGS:
    print(f"\n{'='*50}")
    print(f"  {cfg['label']}")
    print(f"{'='*50}")
    model = train(
        env_id=cfg["env_id"],
        policy_kwargs=cfg["policy_kwargs"],
        total_timesteps=cfg["total_timesteps"],
        n_envs=cfg["n_envs"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
    )
    render_gif(model, cfg["env_id"], cfg["gif"],
               n_episodes=cfg["n_episodes"], fps=cfg["fps"])

print(f"\nAll GIFs written to {GIF_DIR}/")
