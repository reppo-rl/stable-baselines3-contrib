from __future__ import annotations

import os
import time
import warnings

import gymnasium as gym
import imageio
import numpy as np
import wandb
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback

from sb3_contrib import REPPO

warnings.filterwarnings("ignore", category=DeprecationWarning)

RUNS = [
    {
        "env_id": "HalfCheetah-v4",
        "timesteps": 3_000_000,
        "n_envs": 8,
        "seed": 42,
        "vmin": -500.0, "vmax": 500.0,
        "n_steps": 2048, "batch_size": 256, "n_epochs": 8,
        "gamma": 0.99, "gae_lambda": 0.95,
        "ent_target_mult": -0.5, "desired_kl": 0.2, "kl_samples": 16,
        "net_arch": {"pi": [256, 256, 256], "qf": [256, 256, 256]},
        "solve_threshold": 4000.0,
    },
    {
        "env_id": "BipedalWalker-v3",
        "timesteps": 5_000_000,
        "n_envs": 8,
        "seed": 42,
        "vmin": -300.0, "vmax": 300.0,
        "n_steps": 2048, "batch_size": 256, "n_epochs": 8,
        "gamma": 0.99, "gae_lambda": 0.95,
        "ent_target_mult": -0.5, "desired_kl": 0.2, "kl_samples": 16,
        "net_arch": {"pi": [256, 256, 256], "qf": [256, 256, 256]},
        "solve_threshold": 300.0,
    },
]


def train(cfg: dict) -> REPPO:
    env_id = cfg["env_id"]
    total_timesteps = cfg["timesteps"]

    run = wandb.init(
        project="reppo",
        name=env_id,
        config=cfg,
        sync_tensorboard=True,
        save_code=False,
        reinit=True,
    )
    tb_log_dir = f"runs/{run.id}"

    train_env = make_vec_env(env_id, n_envs=cfg["n_envs"], seed=cfg["seed"])
    model = REPPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        critic_learning_rate=3e-4,
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        target_entropy=None,
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
        verbose=0,
        seed=cfg["seed"],
        tensorboard_log=tb_log_dir,
    )

    t0 = time.time()
    print(f"\nTraining REPPO on {env_id}  ({total_timesteps:,} steps, {cfg['n_envs']} envs)")
    model.learn(total_timesteps=total_timesteps, callback=WandbCallback(verbose=0))
    print(f"Done in {time.time() - t0:.0f}s")
    train_env.close()

    eval_env = gym.make(env_id)
    returns = []
    for _ in range(10):
        obs, _ = eval_env.reset()
        ep_return, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            ep_return += reward
            done = terminated or truncated
        returns.append(ep_return)
    eval_env.close()
    mean, std = float(np.mean(returns)), float(np.std(returns))
    solved = mean >= cfg["solve_threshold"]
    print(f"Eval: mean={mean:.1f}  std={std:.1f}  solved={solved}")

    wandb.summary["mean_return"] = mean
    wandb.summary["std_return"] = std
    wandb.summary["solved"] = solved
    run.finish()

    os.makedirs("models", exist_ok=True)
    model.save(f"models/reppo_{env_id.replace('/', '_')}")

    os.makedirs("scripts/gifs", exist_ok=True)
    gif_path = f"scripts/gifs/reppo_{env_id.replace('/', '_')}.gif"
    render_env = gym.make(env_id, render_mode="rgb_array")
    frames = []
    for ep in range(3):
        obs, _ = render_env.reset(seed=ep)
        done = False
        while not done:
            frames.append(render_env.render())
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = render_env.step(action)
            done = terminated or truncated
    render_env.close()
    imageio.mimsave(gif_path, frames, fps=30, loop=0)
    print(f"GIF saved → {gif_path}")

    return model


for cfg in RUNS:
    train(cfg)
