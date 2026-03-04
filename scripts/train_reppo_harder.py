"""Train and evaluate REPPO on environments harder than Pendulum-v1.

Environment priority (first available wins):
  1. BipedalWalker-v3    -- requires: pip install swig "gymnasium[box2d]"
  2. HalfCheetah-v4      -- requires: pip install "gymnasium[mujoco]"
  3. MountainCarContinuous-v0  -- always available (sparse reward, hard exploration)

All three are substantially harder than Pendulum-v1 in different ways:
  - BipedalWalker:   24D obs / 4D act, complex locomotion, solving reward ≥ 300
  - HalfCheetah:     17D obs / 6D act, high-dim MuJoCo locomotion
  - MountainCar:     2D obs / 1D act, very sparse reward (hard exploration problem)

Usage:
    python scripts/train_reppo_harder.py
    python scripts/train_reppo_harder.py --env HalfCheetah-v4 --timesteps 500000
    python scripts/train_reppo_harder.py --reppo-only --n-envs 8
"""

from __future__ import annotations

import argparse
import time
import warnings
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from sb3_contrib import REPPO

warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Environment catalogue
# ---------------------------------------------------------------------------

ENV_CONFIGS: dict[str, dict[str, Any]] = {
    "BipedalWalker-v3": {
        "description": "4-DOF bipedal locomotion (24D obs, 4D act)",
        "solve_threshold": 300.0,
        "reppo": dict(
            vmin=-300.0, vmax=300.0,
            n_steps=2048, batch_size=256, n_epochs=8,
            gamma=0.99, gae_lambda=0.95,
            ent_target_mult=-0.5, desired_kl=0.2, kl_samples=16,
            net_arch={"pi": [256, 256, 256], "qf": [256, 256, 256]},
        ),
        "ppo": dict(
            n_steps=2048, batch_size=256, n_epochs=8,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.0, vf_coef=0.5,
            net_arch=[256, 256, 256],
        ),
        "default_timesteps": 1_000_000,
    },
    "HalfCheetah-v4": {
        "description": "MuJoCo half-cheetah locomotion (17D obs, 6D act)",
        "solve_threshold": 4000.0,
        "reppo": dict(
            vmin=-500.0, vmax=500.0,
            n_steps=2048, batch_size=256, n_epochs=8,
            gamma=0.99, gae_lambda=0.95,
            ent_target_mult=-0.5, desired_kl=0.2, kl_samples=16,
            net_arch={"pi": [256, 256, 256], "qf": [256, 256, 256]},
        ),
        "ppo": dict(
            n_steps=2048, batch_size=256, n_epochs=8,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.0, vf_coef=0.5,
            net_arch=[256, 256, 256],
        ),
        "default_timesteps": 1_000_000,
    },
    "MountainCarContinuous-v0": {
        "description": "Sparse-reward mountain car (2D obs, 1D act)",
        "solve_threshold": 90.0,
        "reppo": dict(
            vmin=-150.0, vmax=150.0,
            n_steps=1024, batch_size=128, n_epochs=10,
            gamma=0.99, gae_lambda=0.95,
            ent_target_mult=-0.5, desired_kl=0.2, kl_samples=16,
            net_arch={"pi": [256, 256], "qf": [256, 256]},
        ),
        "ppo": dict(
            n_steps=1024, batch_size=128, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.001, vf_coef=0.5,
            net_arch=[256, 256],
        ),
        "default_timesteps": 300_000,
    },
}


def detect_env() -> str:
    """Return the first environment in the priority list that can be instantiated."""
    for env_id in ENV_CONFIGS:
        try:
            env = gym.make(env_id)
            env.reset()
            env.close()
            return env_id
        except Exception:
            continue
    raise RuntimeError("No suitable environment found. Install box2d or mujoco.")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class ProgressCallback:
    """Lightweight logging — avoids pulling in the full SB3 callback stack."""

    def __init__(self, total_timesteps: int, log_every: int = 10):
        self.total = total_timesteps
        self.log_every = log_every
        self._start = time.time()
        self._last_log = 0
        self._ep_returns: list[float] = []

    def on_step(self, locals_: dict[str, Any]) -> bool:
        for info in locals_.get("infos", []):
            if "episode" in info:
                self._ep_returns.append(info["episode"]["r"])

        n_steps = locals_.get("self").num_timesteps
        if n_steps - self._last_log >= self.log_every * locals_.get("self").n_steps:
            self._last_log = n_steps
            elapsed = time.time() - self._start
            fps = n_steps / elapsed
            pct = 100 * n_steps / self.total
            ep_str = "N/A"
            if self._ep_returns:
                recent = self._ep_returns[-20:]
                ep_str = f"{np.mean(recent):7.1f} ± {np.std(recent):.1f}"
            print(
                f"  [{pct:5.1f}%  {n_steps:>8,}/{self.total:,}]"
                f"  ep_return={ep_str}"
                f"  fps={fps:5.0f}"
            )
        return True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_reppo(
    env_id: str,
    cfg: dict[str, Any],
    total_timesteps: int,
    n_envs: int,
    seed: int,
    verbose: int,
    wandb_callback: Any = None,
    tb_log_dir: str | None = None,
) -> REPPO:
    reppo_cfg = cfg["reppo"]
    train_env = make_vec_env(env_id, n_envs=n_envs, seed=seed)

    model = REPPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        critic_learning_rate=3e-4,
        n_steps=reppo_cfg["n_steps"],
        batch_size=reppo_cfg["batch_size"],
        n_epochs=reppo_cfg["n_epochs"],
        gamma=reppo_cfg["gamma"],
        gae_lambda=reppo_cfg["gae_lambda"],
        target_entropy=None,
        ent_target_mult=reppo_cfg["ent_target_mult"],
        desired_kl=reppo_cfg["desired_kl"],
        kl_samples=reppo_cfg["kl_samples"],
        aux_coef=1.0,
        max_grad_norm=1.0,
        policy_kwargs=dict(
            net_arch=reppo_cfg["net_arch"],
            vmin=reppo_cfg["vmin"],
            vmax=reppo_cfg["vmax"],
            num_critic_bins=201,
            state_dependent_std=True,
        ),
        verbose=verbose,
        seed=seed,
        tensorboard_log=tb_log_dir,
    )

    t0 = time.time()
    print(f"\n>>> Training REPPO  ({total_timesteps:,} steps, {n_envs} envs)")
    print(f"    vmin={reppo_cfg['vmin']}  vmax={reppo_cfg['vmax']}"
          f"  batch={reppo_cfg['batch_size']}  epochs={reppo_cfg['n_epochs']}")

    model.learn(total_timesteps=total_timesteps, progress_bar=False, callback=wandb_callback)

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.0f}s  ({total_timesteps / elapsed:.0f} fps)")
    train_env.close()
    return model


def train_ppo(
    env_id: str,
    cfg: dict[str, Any],
    total_timesteps: int,
    n_envs: int,
    seed: int,
    verbose: int,
) -> PPO:
    ppo_cfg = cfg["ppo"]
    train_env = make_vec_env(env_id, n_envs=n_envs, seed=seed)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        ent_coef=ppo_cfg["ent_coef"],
        vf_coef=ppo_cfg["vf_coef"],
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=ppo_cfg["net_arch"]),
        verbose=verbose,
        seed=seed,
    )

    t0 = time.time()
    print(f"\n>>> Training PPO baseline  ({total_timesteps:,} steps, {n_envs} envs)")
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.0f}s  ({total_timesteps / elapsed:.0f} fps)")
    train_env.close()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, env_id: str, n_episodes: int) -> tuple[float, float]:
    """Run deterministic rollouts on a fresh unwrapped env."""
    eval_env = gym.make(env_id)
    returns = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        ep_return = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            ep_return += reward
            done = terminated or truncated
        returns.append(ep_return)
    eval_env.close()
    return float(np.mean(returns)), float(np.std(returns))


def render_gif(model, env_id: str, path: str, n_episodes: int = 3, fps: int = 30) -> None:
    """Render deterministic rollouts to a GIF. Requires imageio."""
    import imageio
    env = gym.make(env_id, render_mode="rgb_array")
    frames = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            frames.append(env.render())
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    env.close()
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"    GIF saved → {path}  ({len(frames)} frames)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="REPPO benchmark on environments harder than Pendulum-v1"
    )
    parser.add_argument(
        "--env", type=str, default=None,
        choices=list(ENV_CONFIGS.keys()),
        help="Environment to use (auto-detected if not specified)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Total training timesteps (default: per-env preset)",
    )
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Parallel environments (default: 4)")
    parser.add_argument("--n-eval-episodes", type=int, default=10,
                        help="Evaluation episodes (default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reppo-only", action="store_true",
                        help="Skip PPO baseline")
    parser.add_argument("--verbose", type=int, default=0,
                        help="SB3 verbosity (0=quiet)")
    parser.add_argument("--wandb", action="store_true",
                        help="Log to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="reppo",
                        help="W&B project name (default: reppo)")
    parser.add_argument("--save-model", action="store_true",
                        help="Save trained model to models/")
    parser.add_argument("--save-gif", action="store_true",
                        help="Render a GIF of the final policy to scripts/gifs/")
    args = parser.parse_args()

    # Resolve environment
    env_id = args.env or detect_env()
    cfg = ENV_CONFIGS[env_id]
    total_timesteps = args.timesteps or cfg["default_timesteps"]

    print(f"\n{'='*62}")
    print(f"  REPPO harder-env benchmark")
    print(f"  env             : {env_id}")
    print(f"  description     : {cfg['description']}")
    print(f"  solve_threshold : {cfg['solve_threshold']}")
    print(f"  timesteps       : {total_timesteps:,}")
    print(f"  n_envs          : {args.n_envs}")
    print(f"  seed            : {args.seed}")
    print(f"{'='*62}")

    results: dict[str, tuple[float, float]] = {}

    # W&B setup
    wandb_callback = None
    if args.wandb:
        if not WANDB_AVAILABLE:
            raise ImportError("wandb not installed. Run: pip install wandb")
        run = wandb.init(
            project=args.wandb_project,
            config={
                "env": env_id,
                "timesteps": total_timesteps,
                "n_envs": args.n_envs,
                "seed": args.seed,
                **cfg["reppo"],
            },
            sync_tensorboard=True,
            save_code=True,
        )
        wandb_callback = WandbCallback(verbose=0)
        tb_log_dir = f"runs/{run.id}"

    # REPPO
    reppo_model = train_reppo(
        env_id, cfg, total_timesteps, args.n_envs, args.seed, args.verbose,
        wandb_callback, tb_log_dir if args.wandb else None,
    )
    print(f"\n>>> Evaluating REPPO ({args.n_eval_episodes} episodes, deterministic) ...")
    m, s = evaluate(reppo_model, env_id, args.n_eval_episodes)
    results["REPPO"] = (m, s)
    print(f"    mean={m:.1f}  std={s:.1f}")

    if args.save_model:
        import os; os.makedirs("models", exist_ok=True)
        save_path = f"models/reppo_{env_id.replace('/', '_')}_{total_timesteps}"
        reppo_model.save(save_path)
        print(f"    Model saved → {save_path}.zip")

    if args.save_gif:
        import os; os.makedirs("scripts/gifs", exist_ok=True)
        gif_path = f"scripts/gifs/reppo_{env_id.replace('/', '_')}.gif"
        print(f"\n>>> Rendering GIF ...")
        render_gif(reppo_model, env_id, gif_path)

    # PPO baseline
    if not args.reppo_only:
        ppo_model = train_ppo(
            env_id, cfg, total_timesteps, args.n_envs, args.seed, args.verbose
        )
        print(f"\n>>> Evaluating PPO ({args.n_eval_episodes} episodes, deterministic) ...")
        m, s = evaluate(ppo_model, env_id, args.n_eval_episodes)
        results["PPO"] = (m, s)
        print(f"    mean={m:.1f}  std={s:.1f}")

    # Summary
    solve = cfg["solve_threshold"]
    print(f"\n{'='*62}")
    print(f"  RESULTS  —  {env_id}  ({total_timesteps:,} timesteps)")
    print(f"{'='*62}")
    print(f"  {'Algorithm':<12} {'Mean return':>14} {'Std':>8}  Solved?")
    print(f"  {'-'*48}")
    for name, (m, s) in results.items():
        solved = "YES" if m >= solve else "no"
        print(f"  {name:<12} {m:>14.1f} {s:>8.1f}  {solved}")
    print(f"\n  Solving threshold : {solve}")
    print(f"{'='*62}\n")

    if args.wandb and WANDB_AVAILABLE:
        for name, (m, s) in results.items():
            wandb.summary[f"{name}/mean_return"] = m
            wandb.summary[f"{name}/std_return"] = s
        run.finish()


if __name__ == "__main__":
    main()
