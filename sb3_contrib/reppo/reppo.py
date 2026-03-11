"""REPPO (Relative Entropy Pathwise Policy Optimization) for Stable-Baselines3.

REPPO is an advanced RL algorithm that combines:
- Distributional value functions using HL-Gauss encoding
- Entropy-regularized policy optimization
- KL-divergence based policy updates
- Separate actor and critic update phases

Paper: https://arxiv.org/abs/2507.11019
Reference: https://github.com/cvoelcker/reppo
"""

from typing import Any, ClassVar, TypeVar
import warnings

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch as th
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecEnv

from sb3_contrib.reppo.policies import ActorQPolicy, CnnPolicy, MlpPolicy, MultiInputPolicy
from sb3_contrib.common.hl_gauss import embed_targets

SelfREPPO = TypeVar("SelfREPPO", bound="REPPO")


class REPPO(OnPolicyAlgorithm):
    """Relative Entropy Pathwise Policy Optimization (REPPO).

    Paper: https://arxiv.org/pdf/2507.11019

    :param policy: The policy model to use (ActorQPolicy, MlpPolicy)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param critic_learning_rate: Learning rate for critic optimizer.
        If None, uses the same as learning_rate.
    :param n_steps: The number of steps to run for each environment per update
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for TD-lambda returns
    :param target_entropy: Target entropy for automatic temperature adjustment.
        If None, will use dim(action_space) * ent_target_mult as default.
    :param ent_target_mult: Multiplier for default target entropy (default: -0.5).
        Target entropy = action_dim * ent_target_mult. Only used when target_entropy is None.
        Should be negative so the dual-gradient temperature update converges.
    :param desired_kl: Desired KL divergence threshold for policy updates
    :param kl_samples: Number of action samples from old policy for Monte Carlo KL estimation
    :param aux_coef: Coefficient for auxiliary self-prediction loss
    :param max_grad_norm: The maximum value for the gradient clipping
    :param rollout_buffer_class: Rollout buffer class to use.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer
    :param stats_window_size: Window size for the rollout logging
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level
    :param seed: Seed for the pseudo random generators
    :param device: Device on which the code should be run
    :param _init_setup_model: Whether to build the network at creation
    """

    policy_aliases: ClassVar[dict[str, type[ActorQPolicy]]] = {
        "ActorQPolicy": ActorQPolicy,
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: str | type[ActorQPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        critic_learning_rate: float | None = None,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 8,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        target_entropy: float | None = None,
        ent_target_mult: float = -0.5,
        desired_kl: float = 0.2,
        kl_samples: int = 16,
        aux_coef: float = 1.0,
        max_grad_norm: float = 1.0,
        rollout_buffer_class: type[RolloutBuffer] | None = None,
        rollout_buffer_kwargs: dict[str, Any] | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ):
        if target_entropy is not None:
            self.target_entropy = target_entropy
        elif env is None:
            # env is None during load(), target_entropy will be set from saved data
            self.target_entropy = None
        elif isinstance(env, str):
            temp_env = gym.make(env)
            self.target_entropy = float(np.prod(temp_env.action_space.shape).item()) * ent_target_mult
            temp_env.close()
        else:
            self.target_entropy = float(np.prod(env.action_space.shape).item()) * ent_target_mult

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=0.0,  # Not used in REPPO (handled by temperature)
            vf_coef=0.0,   # Not used (separate critic update)
            max_grad_norm=max_grad_norm,
            use_sde=False,
            sde_sample_freq=-1,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
            supported_action_spaces=(gym.spaces.Box,),
        )

        assert (
            batch_size > 1
        ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1, (
                f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            )
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.desired_kl = desired_kl
        self.kl_samples = kl_samples
        self.aux_coef = aux_coef
        self.critic_learning_rate = critic_learning_rate
        self.ent_target_mult = ent_target_mult

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self.old_policy = None
        self._create_optimizers()

    def _create_optimizers(self) -> None:
        actor_params = list(self.policy.q_net.actor.parameters())
        if not self.policy.q_net.state_dependent_std and hasattr(self.policy.q_net, 'log_std'):
            actor_params.append(self.policy.q_net.log_std)
        actor_params.extend([
            self.policy.q_net.log_alpha_temp,
            self.policy.q_net.log_alpha_kl,
        ])

        critic_params = (
            list(self.policy.q_net.critic_features.parameters())
            + list(self.policy.q_net.critic_final.parameters())
            + list(self.policy.q_net.critic_norm.parameters())
            + list(self.policy.q_net.critic_embedding.parameters())
            + list(self.policy.q_net.predictor.parameters())
        )

        lr = self.lr_schedule(1)
        critic_lr = self.critic_learning_rate if self.critic_learning_rate is not None else lr
        opt_cls = self.policy.optimizer_class
        opt_kwargs = self.policy.optimizer_kwargs

        self.actor_optimizer = opt_cls(actor_params, lr=lr, **opt_kwargs)
        self.critic_optimizer = opt_cls(critic_params, lr=critic_lr, **opt_kwargs)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        self._rollout_dones = np.zeros((n_rollout_steps, env.num_envs), dtype=np.float32)
        self._rollout_truncations = np.zeros((n_rollout_steps, env.num_envs), dtype=np.float32)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device).float()
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    actions= self.policy.unscale_action(actions)
                else:
                    actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(actions)
            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            truncations = np.zeros_like(dones, dtype=np.float32)
            for idx, done in enumerate(dones):
                if done and infos[idx].get("TimeLimit.truncated", False):
                    truncations[idx] = 1.0

            self._rollout_dones[n_steps - 1] = dones.astype(np.float32)
            self._rollout_truncations[n_steps - 1] = truncations

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device).float())

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)

        lr = self.lr_schedule(self._current_progress_remaining)
        critic_lr = self.critic_learning_rate if self.critic_learning_rate is not None else lr
        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = lr
        for param_group in self.critic_optimizer.param_groups:
            param_group["lr"] = critic_lr

        if self.old_policy is None:
            self.old_policy = type(self.policy)(
                self.observation_space,
                self.action_space,
                lambda x: self.lr_schedule(1),
                **(self.policy_kwargs or {})
            )
            self.old_policy.to(self.device)

        self.old_policy.load_state_dict(self.policy.state_dict())
        self.old_policy.eval()

        self._compute_returns_and_advantage()

        buf = self.rollout_buffer
        n_envs = buf.observations.shape[1] if buf.observations.ndim > 2 else 1
        total_size = buf.buffer_size * n_envs

        obs_flat = th.as_tensor(buf.observations, device=self.device, dtype=th.float32).reshape(total_size, -1)
        act_flat = th.as_tensor(buf.actions, device=self.device, dtype=th.float32).reshape(total_size, -1)
        ret_flat = th.as_tensor(buf.returns, device=self.device, dtype=th.float32).reshape(total_size)
        target_embed_flat = self._target_embeddings
        aux_mask_flat = self._aux_masks
        trunc_mask_flat = self._truncation_masks

        critic_losses, aux_losses, actor_losses, entropies, kl_divergences, value_prediction_errors = [], [], [], [], [], []

        for epoch in range(self.n_epochs):
            indices = th.randperm(total_size, device=self.device)

            for start in range(0, total_size, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]

                batch_obs = obs_flat[batch_idx]
                batch_act = act_flat[batch_idx]
                batch_ret = ret_flat[batch_idx]
                batch_target_embed = target_embed_flat[batch_idx]
                batch_aux_mask = aux_mask_flat[batch_idx]
                batch_trunc_mask = trunc_mask_flat[batch_idx]

                critic_metrics = self._update_critic(
                    batch_obs, batch_act, batch_ret,
                    batch_target_embed, batch_aux_mask,
                    batch_trunc_mask,
                )
                critic_losses.append(critic_metrics["critic_loss"])
                aux_losses.append(critic_metrics["aux_loss"])
                value_prediction_errors.append(critic_metrics["value_prediction_error"])

                actor_metrics = self._update_actor(batch_obs)
                actor_losses.append(actor_metrics["actor_loss"])
                entropies.append(actor_metrics["entropy"])
                kl_divergences.append(actor_metrics["kl_divergence"])

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten()
        )

        self.logger.record("train/entropy", np.mean(entropies))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/aux_loss", np.mean(aux_losses))
        self.logger.record("train/kl_divergence", np.mean(kl_divergences))
        self.logger.record("train/value_prediction_error", np.mean(value_prediction_errors))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/alpha_temp", self.policy.q_net.alpha_temp.item())
        self.logger.record("train/alpha_kl", self.policy.q_net.alpha_kl.item())
        self.logger.record("train/target_entropy", self.target_entropy)

        if hasattr(self.policy.q_net, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.q_net.log_std).mean().item())

    def _update_critic(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        returns: th.Tensor,
        target_embeddings: th.Tensor,
        aux_mask: th.Tensor,
        truncation_mask: th.Tensor,
    ) -> dict[str, float]:
        value_pred, value_logits, pred_embed = self.policy.q_net.critic_forward_full(obs, actions)

        embedded_returns = embed_targets(
            returns.view(-1),
            min_value=self.policy.q_net.vmin,
            max_value=self.policy.q_net.vmax,
            num_bins=self.policy.q_net.num_critic_bins,
        ).detach()

        value_loss = -(
            truncation_mask.unsqueeze(-1)
            * embedded_returns * th.log_softmax(value_logits, dim=-1)
        ).sum(-1).mean()

        aux_loss = (
            aux_mask.unsqueeze(-1)
            * (pred_embed - target_embeddings.detach()) ** 2
        ).mean()

        total_loss = value_loss + self.aux_coef * aux_loss

        self.critic_optimizer.zero_grad()
        total_loss.backward()
        th.nn.utils.clip_grad_norm_(
            self.critic_optimizer.param_groups[0]["params"], self.max_grad_norm
        )
        self.critic_optimizer.step()

        value_prediction_error = (value_pred.view(-1) - returns.view(-1)).abs().mean().item()

        return {
            "critic_loss": value_loss.item(),
            "aux_loss": aux_loss.item(),
            "value_prediction_error": value_prediction_error,
        }

    def _update_actor(self, obs: th.Tensor) -> dict[str, float]:
        current_dist = self.policy.q_net.get_action_dist(obs)
        predicted_actions = current_dist.sample()
        on_policy_values = self.policy.q_net.evaluate_actions(obs, predicted_actions)
        entropy = -current_dist.log_prob(predicted_actions).sum(-1)

        with th.no_grad():
            old_dist = self.old_policy.q_net.get_action_dist(obs)
            old_pi_actions = old_dist.sample((self.kl_samples,))
            old_log_probs = old_dist.log_prob(old_pi_actions).sum(-1).mean(0)

        new_log_probs = current_dist.log_prob(old_pi_actions.detach()).sum(-1).mean(0)
        kl_per_sample = old_log_probs.detach() - new_log_probs
        kl_divergence = kl_per_sample.mean()

        normal_loss = -on_policy_values.view(-1) - self.policy.q_net.alpha_temp.detach() * entropy
        policy_loss = th.where(
            kl_per_sample < self.desired_kl,
            normal_loss,
            kl_per_sample * self.policy.q_net.alpha_kl.detach(),
        ).mean()

        temp_target_loss = self.policy.q_net.alpha_temp * (self.target_entropy + entropy.mean()).detach()
        kl_target_loss = -self.policy.q_net.alpha_kl * (kl_divergence - self.desired_kl).detach()
        actor_loss = policy_loss + temp_target_loss + kl_target_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        th.nn.utils.clip_grad_norm_(
            self.actor_optimizer.param_groups[0]["params"], self.max_grad_norm
        )
        self.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "entropy": entropy.mean().item(),
            "kl_divergence": kl_divergence.item(),
        }

    def _compute_returns_and_advantage(self) -> None:
        buf = self.rollout_buffer
        n_envs = buf.observations.shape[1] if buf.observations.ndim > 2 else 1

        dones = self._rollout_dones
        truncations = self._rollout_truncations
        terminals = dones * (1.0 - truncations)

        with th.no_grad():
            all_obs = th.as_tensor(buf.observations, device=self.device, dtype=th.float32)
            all_act = th.as_tensor(buf.actions, device=self.device, dtype=th.float32)

            obs_2d = all_obs.reshape(self.n_steps * n_envs, -1)
            act_2d = all_act.reshape(self.n_steps * n_envs, -1)

            dist = self.policy.q_net.get_action_dist(obs_2d)
            sampled_actions = dist.sample()
            lp = dist.log_prob(sampled_actions)
            if lp.dim() > 1:
                lp = lp.sum(-1)
            all_log_probs = lp.cpu().numpy().reshape(self.n_steps, n_envs)
            all_q_values = self.policy.q_net.evaluate_actions(obs_2d, sampled_actions).cpu().numpy().reshape(self.n_steps, n_envs)
            enc_flat = self.policy.q_net.get_encoder_features(obs_2d, act_2d)
            feat_3d = enc_flat.reshape(self.n_steps, n_envs, -1)

            last_obs = th.as_tensor(self._last_obs, device=self.device, dtype=th.float32)
            last_dist = self.policy.q_net.get_action_dist(last_obs)
            last_actions = last_dist.sample()
            last_lp = last_dist.log_prob(last_actions)
            if last_lp.dim() > 1:
                last_lp = last_lp.sum(-1)
            last_log_probs = last_lp.cpu().numpy()
            last_values = self.policy.q_net.evaluate_actions(last_obs, last_actions).cpu().numpy()
            last_feat = self.policy.q_net.get_encoder_features(last_obs, last_actions)

        encoder_stack = th.cat([feat_3d, last_feat.unsqueeze(0)], dim=0)
        target_embeddings = encoder_stack[1:]

        truncation_masks = 1.0 - truncations
        aux_masks = 1.0 - dones

        total_size = self.n_steps * n_envs
        self._target_embeddings = target_embeddings.reshape(total_size, -1)
        self._aux_masks = th.as_tensor(aux_masks, device=self.device, dtype=th.float32).reshape(total_size)
        self._truncation_masks = th.as_tensor(truncation_masks, device=self.device, dtype=th.float32).reshape(total_size)

        alpha_temp = self.policy.q_net.alpha_temp.item()

        next_values = np.concatenate([all_q_values[1:], last_values[np.newaxis]], axis=0)
        next_log_probs = np.concatenate([all_log_probs[1:], last_log_probs[np.newaxis]], axis=0)

        soft_rewards = buf.rewards - self.gamma * alpha_temp * next_log_probs

        trunc = truncations.copy()
        trunc[-1] = 1.0

        returns = np.zeros_like(buf.rewards)
        last_gve = np.zeros((n_envs,), dtype=np.float32)

        for step in reversed(range(self.n_steps)):
            lambda_sum = (
                self.gae_lambda * last_gve
                + (1.0 - self.gae_lambda) * next_values[step]
            )
            delta = self.gamma * np.where(
                trunc[step].astype(bool),
                next_values[step],
                (1.0 - terminals[step]) * lambda_sum,
            )
            last_gve = soft_rewards[step] + delta
            returns[step] = last_gve

        buf.returns = returns
        buf.advantages = returns - buf.values

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + [  # noqa: RUF005
            "old_policy",
            "_rollout_dones",
            "_rollout_truncations",
            "_target_embeddings",
            "_aux_masks",
            "_truncation_masks",
        ]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor_optimizer", "critic_optimizer"]
        return state_dicts, []

    def learn(
        self: SelfREPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "REPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfREPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
