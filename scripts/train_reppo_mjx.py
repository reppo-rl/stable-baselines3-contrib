from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import torch as th
import wandb
from wandb.integration.sb3 import WandbCallback

from sb3_contrib import REPPO
from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib.common.envs.mjx_playground_wrapper import (
    MjxPlaygroundGymWrapper,
    MjxPlaygroundVecEnv,
)

CONFIGS = [
    {
        "env_name": "CartpoleBalance",
        "num_envs": 1024,
        "device": "cuda",
        "seed": 226,
        "max_episode_steps": 1000,
        "total_timesteps": 3_000_000,
        "n_steps": 128,
        "num_mini_batches": 128,
        "n_epochs": 4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "learning_rate": 3e-4,
        "critic_learning_rate": 3e-4,
        "anneal_lr": False,
        "max_grad_norm": 0.5,
        "ent_target_mult": 0.5,
        "desired_kl": 0.05,
        "kl_start": 0.01,
        "ent_start": 0.01,
        "kl_samples": 16,
        "vmin": 0.0,
        "vmax": 100.0,
        "aux_loss_coeff": 1.0,
        "jit": True,
        "net_arch": {"pi": [512, 512], "qf": [512, 512]},
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


def train(cfg: dict) -> REPPO:
    transitions_per_update = cfg["num_envs"] * cfg["n_steps"]
    num_updates = cfg["total_timesteps"] // transitions_per_update
    batch_size = transitions_per_update // cfg["num_mini_batches"]

    run = wandb.init(
        project=cfg.get("wandb_project", "reppo-mjx"),
        name=f"{cfg['env_name']}-{cfg['num_envs']}envs",
        config={
            **cfg,
            "transitions_per_update": transitions_per_update,
            "num_updates": num_updates,
            "batch_size": batch_size,
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
    print(f"Mini-batch size      : {batch_size:,}")
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
        batch_size=batch_size,
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        ent_target_mult=cfg["ent_target_mult"],
        desired_kl=cfg["desired_kl"],
        kl_samples=cfg["kl_samples"],
        aux_coef=cfg["aux_loss_coeff"],
        max_grad_norm=cfg["max_grad_norm"],
        policy_kwargs=dict(
            net_arch=cfg["net_arch"],
            vmin=cfg["vmin"],
            vmax=cfg["vmax"],
            num_critic_bins=151,
            state_dependent_std=True,
            alpha_temp_init=cfg["ent_start"],
            alpha_kl_init=cfg["kl_start"],
            optimizer_class=th.optim.Adam,
            optimizer_kwargs={"betas": (0.9, 0.999)},
        ),
        stats_window_size=1000,
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
    save_path = f"models/reppo_{cfg['env_name']}_{run.id}"
    model.save(save_path)
    vec_env.save(f"{save_path}_vecnormalize.pkl")
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
        if "wandb_project" in overrides:
            cfg["wandb_project"] = overrides["wandb_project"]
        train(cfg)
