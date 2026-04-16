from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvObs, VecEnvStepReturn


import jax
import jax.numpy as jnp



def jax_to_torch_gpu(x) -> torch.Tensor:
    return torch.from_dlpack(x)


def jax_to_torch_cpu(x) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x))


def torch_to_jax_gpu(x: torch.Tensor):
    return jax.dlpack.from_dlpack(x.detach())


def torch_to_jax_cpu(x: torch.Tensor):
    return jnp.asarray(x.detach().numpy())


class MjxPlaygroundGymWrapper(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        env_name: str,
        num_envs: int = 1,
        seed: int = 0,
        device: str = "cuda",
        config_overrides: dict[str, Any] | None = None,
        max_episode_steps: int = 1000,
        render_mode: str | None = None,
        camera_resolution: tuple[int, int] = (640, 480),
    ):
        super().__init__()


        try:
            from mujoco_playground import registry  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "mujoco_playground is required for MjxPlaygroundGymWrapper. "
                "Install it with: pip install mujoco-playground"
            ) from e

        self._env_name = env_name
        self._num_envs = num_envs
        self._seed = seed
        self._device_str = device
        self._max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self._camera_resolution = camera_resolution

        env_cfg = registry.get_default_config(env_name)
        if config_overrides:
            for k, v in config_overrides.items():
                setattr(env_cfg, k, v)
        self._env = registry.load(env_name, config=env_cfg)

        use_gpu = device != "cpu"
        self._jax_to_torch = jax_to_torch_gpu if use_gpu else jax_to_torch_cpu
        self._torch_to_jax = torch_to_jax_gpu if use_gpu else torch_to_jax_cpu
        self._torch_device = torch.device(device)

        self._key = jax.random.PRNGKey(seed)

        self._v_reset = jax.jit(jax.vmap(self._env.reset))
        self._v_step = jax.jit(jax.vmap(self._env.step))

        self._state = None
        self._step_count: np.ndarray | None = None

        self._setup_spaces()

        if render_mode is not None:
            self._setup_renderer()
        else:
            self._renderer = None
            self._viewer = None
            self._mj_model = None
            self._mj_data = None

    def _setup_spaces(self) -> None:
        act_size = self._env.action_size
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._num_envs, act_size), dtype=np.float32
        )

        dummy_key = jax.random.PRNGKey(0)
        dummy_state = self._env.reset(dummy_key)
        obs = dummy_state.obs

        if isinstance(obs, dict):
            obs_spaces = {}
            for k, v in obs.items():
                obs_spaces[k] = spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(self._num_envs,) + v.shape,
                    dtype=np.float32,
                )
            self.observation_space = spaces.Dict(obs_spaces)
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self._num_envs,) + obs.shape,
                dtype=np.float32,
            )

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[torch.Tensor | dict, dict]:
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)

        env_idx = options.get("env_idx") if options else None

        if env_idx is None or self._state is None:
            keys = jax.random.split(self._key, self._num_envs + 1)
            self._key, subkeys = keys[0], keys[1:]
            self._state = self._v_reset(subkeys)
            self._step_count = np.zeros(self._num_envs, dtype=np.int32)
        else:
            idx = jnp.asarray(env_idx)
            keys = jax.random.split(self._key, len(env_idx) + 1)
            self._key, subkeys = keys[0], keys[1:]
            new_states = self._v_reset(subkeys)
            self._state = self._update_state_at_indices(self._state, new_states, idx)
            self._step_count[env_idx] = 0

        obs = self._convert_observation(self._state.obs)
        info = self._convert_info(self._state.info if hasattr(self._state, "info") else {})
        return obs, info

    def step(
        self,
        action: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor | dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        action = action.to(self._torch_device)
        jax_action = self._torch_to_jax(action)

        self._state = self._v_step(self._state, jax_action)
        self._step_count += 1

        obs = self._convert_observation(self._state.obs)
        reward = self._jax_to_torch(self._state.reward).to(dtype=torch.float32)
        terminated = self._jax_to_torch(self._state.done).to(dtype=torch.bool)
        truncated = torch.tensor(
            self._step_count >= self._max_episode_steps,
            dtype=torch.bool,
            device=self._torch_device,
        )
        info = self._convert_info(self._state.info if hasattr(self._state, "info") else {})

        return obs, reward, terminated, truncated, info

    def _convert_observation(self, obs: jax.Array | dict) -> torch.Tensor | dict:
        if isinstance(obs, dict):
            return {k: self._jax_to_torch(v).to(dtype=torch.float32) for k, v in obs.items()}
        tensor = self._jax_to_torch(obs).to(dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _convert_info(self, info: dict) -> dict:
        out = {}
        for k, v in info.items():
            if isinstance(v, jax.Array):
                out[k] = self._jax_to_torch(v)
            elif isinstance(v, dict):
                out[k] = self._convert_info(v)
            else:
                out[k] = v
        return out

    @staticmethod
    def _update_state_at_indices(state, new_states, indices: jax.Array):
        def _update_leaf(old_leaf, new_leaf):
            return old_leaf.at[indices].set(new_leaf)

        return jax.tree_util.tree_map(_update_leaf, state, new_states)

    def _setup_renderer(self) -> None:
        try:
            import mujoco
            import mujoco.viewer
        except ImportError as e:
            raise ImportError("mujoco is required for rendering.") from e

        mj_model = None
        for attr in ("mj_model", "model", "sys"):
            candidate = getattr(self._env, attr, None)
            if candidate is not None and isinstance(candidate, mujoco.MjModel):
                mj_model = candidate
                break
        if mj_model is None:
            raise RuntimeError("Cannot find a mujoco.MjModel on the wrapped environment for rendering.")
        self._mj_model = mj_model
        self._mj_data = mujoco.MjData(mj_model)

        w, h = self._camera_resolution
        if self.render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(mj_model, height=h, width=w)
            self._viewer = None
            self._tracking_cam = mujoco.MjvCamera()
            self._tracking_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self._tracking_cam.trackbodyid = 1
            self._tracking_cam.distance = 3.0
            self._tracking_cam.elevation = -10.0
        else:
            self._renderer = None
            self._viewer = mujoco.viewer.launch_passive(mj_model, self._mj_data)

    def _sync_mjx_to_mujoco(self) -> None:
        import mujoco
        pipeline_state = getattr(self._state, "pipeline_state", None) or getattr(self._state, "data", None)
        if pipeline_state is None:
            return
        try:
            from mujoco import mjx
            mjx.get_data_into(self._mj_data, self._mj_model, jax.tree_util.tree_map(lambda x: x[0], pipeline_state))
        except Exception:
            qpos = np.asarray(pipeline_state.qpos[0])
            qvel = np.asarray(pipeline_state.qvel[0])
            self._mj_data.qpos[:] = qpos
            self._mj_data.qvel[:] = qvel
            mujoco.mj_forward(self._mj_model, self._mj_data)

    def render(self) -> np.ndarray | None:
        if self._state is None:
            return None
        self._sync_mjx_to_mujoco()
        if self.render_mode == "rgb_array" and self._renderer is not None:
            import mujoco
            self._renderer.update_scene(self._mj_data, camera=self._tracking_cam)
            return self._renderer.render()
        elif self.render_mode == "human" and self._viewer is not None:
            self._viewer.sync()
        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        if self._viewer is not None:
            self._viewer.close()

    @property
    def device(self) -> str:
        return self._device_str

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def unwrapped(self) -> "MjxPlaygroundGymWrapper":
        return self


class MjxPlaygroundStdWrapper(gym.Wrapper):
    def __init__(self, env: MjxPlaygroundGymWrapper):
        super().__init__(env)
        self._setup_observation_space()

    def _setup_observation_space(self) -> None:
        inner = self.env.observation_space
        if isinstance(inner, spaces.Dict):
            new_spaces: dict[str, spaces.Space] = {}
            for k, v in inner.spaces.items():
                if k in ("state", "observation", "obs"):
                    new_spaces["policy"] = v
                else:
                    new_spaces[k] = v
            if "policy" not in new_spaces:
                first_key = next(iter(inner.spaces))
                new_spaces["policy"] = inner.spaces[first_key]
            self.observation_space = spaces.Dict(new_spaces)
        else:
            self.observation_space = spaces.Dict({"policy": inner})

    def reset(self, **kwargs) -> tuple[dict, dict]:
        obs, info = self.env.reset(**kwargs)
        return self._process_observation(obs), info

    def step(self, action) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated | truncated

        done_indices = done.nonzero(as_tuple=False).squeeze(-1).tolist()
        if done_indices:
            reset_obs, _ = self.env.reset(options={"env_idx": done_indices})
            obs = self._merge_observations(obs, reset_obs, done_indices)

        return self._process_observation(obs), reward, terminated, truncated, info

    def _process_observation(self, obs: torch.Tensor | dict) -> dict:
        if isinstance(obs, dict):
            out: dict[str, torch.Tensor] = {}
            for k, v in obs.items():
                new_k = "policy" if k in ("state", "observation", "obs") else k
                out[new_k] = v
            if "policy" not in out:
                first_k = next(iter(out))
                out["policy"] = out.pop(first_k)
            return out
        return {"policy": obs}

    @staticmethod
    def _merge_observations(
        obs: torch.Tensor | dict,
        reset_obs: torch.Tensor | dict,
        indices: list[int],
    ) -> torch.Tensor | dict:
        idx = torch.tensor(indices, dtype=torch.long)
        if isinstance(obs, dict):
            merged = {}
            for k in obs:
                t = obs[k].clone()
                t[idx] = reset_obs[k][idx] if isinstance(reset_obs, dict) else reset_obs[idx]
                merged[k] = t
            return merged
        merged = obs.clone()
        merged[idx] = reset_obs[idx]
        return merged

    def seed(self, seed: int | None = None) -> None:
        self.env.reset(seed=seed)

    @property
    def unwrapped(self) -> MjxPlaygroundGymWrapper:
        return self.env.unwrapped

    @property
    def device(self) -> str:
        return self.env.device


class MjxPlaygroundVecEnv(VecEnv):
    def __init__(self, gym_wrapper: MjxPlaygroundGymWrapper):
        self._gym = gym_wrapper
        n = gym_wrapper.num_envs

        single_obs_space = self._strip_batch_dim(gym_wrapper.observation_space, n)
        single_act_space = self._strip_batch_dim(gym_wrapper.action_space, n)

        super().__init__(n, single_obs_space, single_act_space)

        self._actions: np.ndarray | None = None
        self._torch_device = torch.device(gym_wrapper.device)
        self._ep_rewards = np.zeros(n, dtype=np.float32)
        self._ep_lengths = np.zeros(n, dtype=np.int32)

    @staticmethod
    def _strip_batch_dim(space: spaces.Space, n: int) -> spaces.Space:
        if isinstance(space, spaces.Dict):
            return spaces.Dict({
                k: spaces.Box(low=v.low[0], high=v.high[0], dtype=v.dtype)
                for k, v in space.spaces.items()
            })
        return spaces.Box(low=space.low[0], high=space.high[0], dtype=space.dtype)

    def reset(self) -> VecEnvObs:
        obs, _ = self._gym.reset()
        self._ep_rewards[:] = 0.0
        self._ep_lengths[:] = 0
        return self._to_numpy(obs)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        action_tensor = torch.as_tensor(self._actions, dtype=torch.float32, device=self._torch_device)
        obs, rewards, terminated, truncated, raw_infos = self._gym.step(action_tensor)

        dones = (terminated | truncated).cpu().numpy()
        rewards_np = rewards.cpu().numpy()
        truncated_np = truncated.cpu().numpy()
        obs_np = self._to_numpy(obs)

        self._ep_rewards += rewards_np
        self._ep_lengths += 1

        done_indices = np.where(dones)[0].tolist()

        infos: list[dict] = [{} for _ in range(self.num_envs)]
        for i in done_indices:
            infos[i]["terminal_observation"] = (
                obs_np[i].copy() if isinstance(obs_np, np.ndarray)
                else {k: v[i].copy() for k, v in obs_np.items()}
            )
            infos[i]["TimeLimit.truncated"] = bool(truncated_np[i])
            infos[i]["episode"] = {"r": float(self._ep_rewards[i]), "l": int(self._ep_lengths[i])}
            self._ep_rewards[i] = 0.0
            self._ep_lengths[i] = 0

        if done_indices:
            reset_obs, _ = self._gym.reset(options={"env_idx": done_indices})
            reset_np = self._to_numpy(reset_obs)
            if isinstance(obs_np, np.ndarray):
                obs_np[done_indices] = reset_np[done_indices]
            else:
                for k in obs_np:
                    obs_np[k][done_indices] = reset_np[k][done_indices]

        return obs_np, rewards_np, dones, infos

    def close(self) -> None:
        self._gym.close()

    def seed(self, seed: int | None = None) -> list[int | None]:
        self._gym.reset(seed=seed)
        return [seed] * self.num_envs

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        return [getattr(self._gym, attr_name)] * self.num_envs

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        setattr(self._gym, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        result = getattr(self._gym, method_name)(*method_args, **method_kwargs)
        return [result] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices: VecEnvIndices = None) -> list[bool]:
        return [False] * self.num_envs

    def get_images(self) -> list[np.ndarray | None]:
        return [self._gym.render()]

    @staticmethod
    def _to_numpy(obs: torch.Tensor | dict) -> np.ndarray | dict:
        if isinstance(obs, torch.Tensor):
            return obs.cpu().numpy()
        return {k: v.cpu().numpy() for k, v in obs.items()}

    @property
    def device(self) -> str:
        return self._gym.device
