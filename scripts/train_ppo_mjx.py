from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import torch as th
import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib.common.envs.mjx_playground_wrapper import (
    MjxPlaygroundGymWrapper,
    MjxPlaygroundVecEnv,
)

CONFIGS = [
    {
        "env_name": "CheetahRun",
        "num_envs": 1024,
        "device": "cuda",
        "seed": 42,
        "max_episode_steps": 1000,
        "total_timesteps": 500000000,
        "n_steps": 128,
        "batch_size": 2048,
        "n_epochs": 8,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "net_arch": {"pi": [512, 512, 512], "vf": [512, 512, 512]},
    },
]


def make_env(cfg: dict) -> MjxPlaygroundVecEnv:
    gym_env = MjxPlaygroundGymWrapper(
        env_name=cfg["env_name"],
        num_envs=cfg["num_envs"],
        seed=cfg["seed"],
        device=cfg["device"],
        max_episode_steps=cfg["max_episode_steps"],
        config_overrides={"impl": "jax"},
    )
    vec_env = MjxPlaygroundVecEnv(gym_env)
    return VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)


def train(cfg: dict) -> PPO:
    transitions_per_update = cfg["num_envs"] * cfg["n_steps"]
    num_updates = cfg["total_timesteps"] // transitions_per_update

    run = wandb.init(
        project=cfg.get("wandb_project", "ppo-mjx"),
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

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        normalize_advantage=True,
        policy_kwargs=dict(
            net_arch=cfg["net_arch"],
            optimizer_class=th.optim.Adam,
            optimizer_kwargs={"betas": (0.9, 0.999)},
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

    os.makedirs("models", exist_ok=True)
    save_path = f"models/ppo_{cfg['env_name']}_{run.id}"
    model.save(save_path)
    print(f"Model saved → {save_path}")

    vec_env.close()
    run.finish()
    return model


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--wandb-project", default="ppo-mjx")
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
        if "wandb_project" in overrides:
            cfg["wandb_project"] = overrides["wandb_project"]
        train(cfg)
