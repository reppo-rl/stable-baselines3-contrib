from __future__ import annotations
from typing import Any

import gymnasium as gym
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, constraints
from torch.distributions.transforms import Transform
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule

from sb3_contrib.common.hl_gauss import HLGaussTransform


class TanhTransform(Transform):
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size: int = 1):
        super().__init__(cache_size=cache_size)

    def _call(self, x: th.Tensor) -> th.Tensor:
        return x.tanh()

    def _inverse(self, y: th.Tensor) -> th.Tensor:
        return th.atanh(y)

    def log_abs_det_jacobian(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        log2 = th.log(th.tensor(2.0, device=x.device, dtype=x.dtype))
        return 2.0 * (log2 - x - F.softplus(-2.0 * x))


def _make_mlp(
    in_features: int,
    out_features: int,
    hidden_dims: list[int],
    act: type[nn.Module] = nn.SiLU,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_features
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.RMSNorm(h), act()]
        prev = h
    layers.append(nn.Linear(prev, out_features))
    return nn.Sequential(*layers)


class ActorQNetwork(nn.Module):
    """Combined actor + distributional critic network."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        net_arch: dict[str, list[int]],
        activation_fn: type[nn.Module] = nn.SiLU,
        num_critic_bins: int = 151,
        vmin: float = -300.0,
        vmax: float = 300.0,
        min_std: float = 0.1,
        alpha_temp_init: float = 0.01,
        alpha_kl_init: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_critic_bins = num_critic_bins
        self.vmin = vmin
        self.vmax = vmax
        self.min_std = min_std

        pi_arch = net_arch.get("pi", [512, 512, 512])
        qf_arch = net_arch.get("qf", [512, 512, 512])
        encoder_dim = qf_arch[-1]

        self.actor = _make_mlp(observation_dim, 2 * action_dim, pi_arch, activation_fn)

        self.critic_encoder = _make_mlp(
            observation_dim + action_dim, encoder_dim, qf_arch[:-1], activation_fn
        )

        self.critic_head = nn.Sequential(
            activation_fn(),
            nn.Linear(encoder_dim, encoder_dim),
            nn.RMSNorm(encoder_dim),
            activation_fn(),
            nn.Linear(encoder_dim, num_critic_bins),
        )

        self.register_buffer("values", th.linspace(vmin, vmax, num_critic_bins))

        hl = HLGaussTransform(vmin, vmax, num_critic_bins)
        zero_target = hl.embed_targets(th.zeros(1))
        self.zero_dist = nn.Parameter(zero_target.squeeze(0).clone())

        self.pred_head_act = activation_fn()
        self.predictor = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.RMSNorm(encoder_dim),
            activation_fn(),
            nn.Linear(encoder_dim, encoder_dim),
        )

        self.log_alpha_temp = nn.Parameter(th.log(th.tensor(float(alpha_temp_init))))
        self.log_alpha_kl = nn.Parameter(th.log(th.tensor(float(alpha_kl_init))))

    @property
    def alpha_temp(self) -> th.Tensor:
        return self.log_alpha_temp.exp()

    @property
    def alpha_kl(self) -> th.Tensor:
        return self.log_alpha_kl.exp()

    def get_action_dist(self, obs: th.Tensor) -> TransformedDistribution:
        x = self.actor(obs)
        mean, log_std = x.chunk(2, dim=-1)
        std = log_std.exp() + self.min_std
        return TransformedDistribution(
            Normal(mean, std, validate_args=False),
            [TanhTransform(cache_size=1)],
        )

    def forward_actor(self, obs: th.Tensor):
        x = self.actor(obs)
        mean, log_std = x.chunk(2, dim=-1)
        std = log_std.exp() + self.min_std
        dist = TransformedDistribution(
            Normal(mean, std, validate_args=False),
            [TanhTransform(cache_size=1)],
        )
        return dist, th.tanh(mean), self.alpha_temp, self.alpha_kl

    def _encode(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        return self.critic_encoder(th.cat([obs, actions], dim=-1))

    def _to_logits(self, features: th.Tensor) -> th.Tensor:
        return self.critic_head(features) + 40.9 * self.zero_dist

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        features = self._encode(obs, actions)
        logits = self._to_logits(features)
        return (th.softmax(logits, dim=-1) * self.values).sum(dim=-1)

    def critic_forward_full(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self._encode(obs, actions)
        logits = self._to_logits(features)
        value = (th.softmax(logits, dim=-1) * self.values).sum(dim=-1)
        pred = self.predictor(self.pred_head_act(features))
        return value, logits, pred

    def get_encoder_features(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        return self._encode(obs, actions)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        dist, det, _, _ = self.forward_actor(obs)
        return det if deterministic else dist.rsample()


class ActorQPolicy(BasePolicy):
    """SB3-compatible policy wrapping ActorQNetwork."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.SiLU,
        num_critic_bins: int = 151,
        vmin: float = -300.0,
        vmax: float = 300.0,
        min_std: float = 0.1,
        alpha_temp_init: float = 0.01,
        alpha_kl_init: float = 0.01,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        use_sde: bool = False,
        **kwargs,
    ):
        if net_arch is None:
            net_arch = {"pi": [512, 512], "qf": [512, 512]}
        elif isinstance(net_arch, list):
            net_arch = {"pi": net_arch, "qf": net_arch}

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.num_critic_bins = num_critic_bins
        self.vmin = vmin
        self.vmax = vmax
        self.min_std = min_std
        self.alpha_temp_init = alpha_temp_init
        self.alpha_kl_init = alpha_kl_init

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        obs_dim = self.observation_space.shape[0]
        act_dim = self.action_space.shape[0]

        self.q_net = ActorQNetwork(
            observation_dim=obs_dim,
            action_dim=act_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            num_critic_bins=self.num_critic_bins,
            vmin=self.vmin,
            vmax=self.vmax,
            min_std=self.min_std,
            alpha_temp_init=self.alpha_temp_init,
            alpha_kl_init=self.alpha_kl_init,
        )

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.q_net(observation, deterministic=deterministic)

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        obs = obs.float()
        dist, det_actions, _, _ = self.q_net.forward_actor(obs)
        actions = det_actions if deterministic else dist.rsample()
        values = self.q_net.evaluate_actions(obs, actions).unsqueeze(-1)
        log_probs = dist.log_prob(actions.clamp(-1 + 1e-6, 1 - 1e-6))
        if log_probs.dim() > 1:
            log_probs = log_probs.sum(-1)
        return actions, values, log_probs

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        with th.no_grad():
            obs = obs.float()
            _, det_actions, _, _ = self.q_net.forward_actor(obs)
            return self.q_net.evaluate_actions(obs, det_actions).unsqueeze(-1)

    def get_distribution(self, obs: th.Tensor) -> TransformedDistribution:
        return self.q_net.get_action_dist(obs)


MlpPolicy = ActorQPolicy
CnnPolicy = ActorQPolicy
MultiInputPolicy = ActorQPolicy
