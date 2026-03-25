from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback

from sb3_contrib import REPPO
from sb3_contrib.common.envs.mjx_playground_wrapper import (
    MjxPlaygroundGymWrapper,
    MjxPlaygroundVecEnv,
)

CONFIGS = [
    {
        "env_name": "CheetahRun",
        "num_envs": 16384,
        "device": "cuda",
        "seed": 42,
        "max_episode_steps": 1000,
        "total_timesteps": 100_000_000,
        "n_steps": 32,
        "batch_size": 16384,
        "n_epochs": 4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "learning_rate": 3e-4,
        "critic_learning_rate": 3e-4,
        "ent_target_mult": -0.5,
        "desired_kl": 0.2,
        "kl_samples": 8,
        "vmin": -1000.0,
        "vmax": 1000.0,
        "net_arch": {"pi": [256, 256, 256], "qf": [256, 256, 256]},
    },
]


def make_env(cfg: dict) -> MjxPlaygroundVecEnv:
    gym_env = MjxPlaygroundGymWrapper(
        env_name=cfg["env_name"],
        num_envs=cfg["num_envs"],
        seed=cfg["seed"],
        device=cfg["device"],
        max_episode_steps=cfg["max_episode_steps"],
        config_overrides={"impl": "mjx"},
    )
    return MjxPlaygroundVecEnv(gym_env)


def train(cfg: dict) -> REPPO:
    transitions_per_update = cfg["num_envs"] * cfg["n_steps"]
    num_updates = cfg["total_timesteps"] // transitions_per_update

    run = wandb.init(
        project="reppo-mjx",
        name=f"{cfg['env_name']}-{cfg['num_envs']}envs",
        config={
            **cfg,
            "transitions_per_update": transitions_per_update,
            "num_updates": num_updates,
        },
        sync_tensorboard=True,
        save_code=True,
        reinit=True,
    )
    tb_log_dir = f"runs/{run.id}"

    print(f"\nEnvironment          : {cfg['env_name']}")
    print(f"Num envs             : {cfg['num_envs']}  (GPU-vectorized via JAX/MJX)")
    print(f"Device               : {cfg['device']}")
    print(f"Transitions/update   : {transitions_per_update:,}")
    print(f"Num updates          : {num_updates:,}")
    print(f"Total timesteps      : {cfg['total_timesteps']:,}")
    print(f"WandB run            : {run.url}\n")

    vec_env = make_env(cfg)

    model = REPPO(
        "MlpPolicy",
        vec_env,
        learning_rate=cfg["learning_rate"],
        critic_learning_rate=cfg["critic_learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        ent_target_mult=cfg["ent_target_mult"],
        desired_kl=cfg["desired_kl"],
        kl_samples=cfg["kl_samples"],
        aux_coef=1.0,
        max_grad_norm=1.0,
        policy_kwargs=dict(
            net_arch=cfg["net_arch"],
            vmin=cfg["vmin"],
            vmax=cfg["vmax"],
            num_critic_bins=201,
            state_dependent_std=True,
        ),
        verbose=1,
        seed=cfg["seed"],
        device=cfg["device"],
        tensorboard_log=tb_log_dir,
    )

    t0 = time.time()
    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=WandbCallback(verbose=0),
    )
    elapsed = time.time() - t0
    sps = cfg["total_timesteps"] / elapsed
    print(f"\nDone in {elapsed:.0f}s  ({sps:,.0f} steps/s)")

    wandb.summary["steps_per_second"] = sps
    wandb.summary["total_time_s"] = elapsed

    vec_env.close()

    os.makedirs("models", exist_ok=True)
    save_path = f"models/reppo_{cfg['env_name']}_{run.id}"
    model.save(save_path)
    print(f"Model saved → {save_path}")

    os.makedirs("scripts/gifs", exist_ok=True)
    gif_path = f"scripts/gifs/reppo_{cfg['env_name']}_{run.id}.gif"
    mean_return = render_gif(model, cfg, gif_path, n_episodes=3, fps=30)
    wandb.summary["eval_mean_return"] = mean_return
    wandb.log({"eval/gif": wandb.Video(gif_path, fps=30, format="gif")})

    run.finish()
    return model


def render_gif(model: REPPO, cfg: dict, gif_path: str, n_episodes: int = 3, fps: int = 30) -> None:
    render_env = MjxPlaygroundGymWrapper(
        env_name=cfg["env_name"],
        num_envs=1,
        seed=cfg["seed"] + 99,
        device=cfg["device"],
        max_episode_steps=cfg["max_episode_steps"],
        render_mode="rgb_array",
    )

    frames = []
    ep_returns = []

    for ep in range(n_episodes):
        obs_np, _ = render_env.reset(seed=ep)
        obs = obs_np[0]
        ep_return = 0.0
        done = False

        while not done:
            frame = render_env.render()
            if frame is not None:
                frames.append(frame)

            action, _ = model.predict(obs, deterministic=True)
            import torch
            action_t = torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)
            obs_np, reward, terminated, truncated, _ = render_env.step(action_t)
            obs = obs_np[0]
            ep_return += float(reward[0].item())
            done = bool((terminated | truncated)[0].item())

        ep_returns.append(ep_return)
        print(f"  Episode {ep + 1}: return = {ep_return:.1f}")

    render_env.close()

    os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    mean_return = float(np.mean(ep_returns))
    print(f"GIF saved → {gif_path}  ({len(frames)} frames, mean return={mean_return:.1f})")
    return mean_return


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--wandb-project", default="reppo-mjx")
    args = parser.parse_args()
    return {k: v for k, v in vars(args).items() if v is not None}


if __name__ == "__main__":
    overrides = parse_args()
    for cfg in CONFIGS:
        if "env" in overrides:
            cfg["env_name"] = overrides["env"]
        if "num_envs" in overrides:
            cfg["num_envs"] = overrides["num_envs"]
        if "device" in overrides:
            cfg["device"] = overrides["device"]
        if "timesteps" in overrides:
            cfg["total_timesteps"] = overrides["timesteps"]
        train(cfg)
