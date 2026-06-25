from __future__ import annotations

import copy
import os
import statistics
import time
import warnings
from collections import deque
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_obs_groups, store_code_state
from tensordict import TensorDict
from torch.distributions import Normal

import rsl_rl
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class HistoryRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """RSL-RL wrapper that appends a fixed-length proprioception history."""

    def __init__(self, env, clip_actions: float | None = None, history_length: int = 5, history_obs_key: str = "policy"):
        self.history_length = history_length
        self.history_obs_key = history_obs_key
        self._obs_history = None
        super().__init__(env, clip_actions=clip_actions)

    def reset(self) -> tuple[TensorDict, dict]:
        obs, extras = super().reset()
        self._reset_history(obs)
        return self._with_history(obs), extras

    def get_observations(self) -> TensorDict:
        obs = super().get_observations()
        if self._obs_history is None:
            self._reset_history(obs)
        else:
            self._append_history(obs)
        return self._with_history(obs)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        obs, rew, dones, extras = super().step(actions)
        self._append_history(obs)
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            self._obs_history[done_ids] = 0.0
        return self._with_history(obs), rew, dones, extras

    def _reset_history(self, obs: TensorDict) -> None:
        obs_dim = obs[self.history_obs_key].shape[-1]
        self._obs_history = torch.zeros(
            self.num_envs,
            self.history_length,
            obs_dim,
            device=obs[self.history_obs_key].device,
            dtype=obs[self.history_obs_key].dtype,
        )
        self._append_history(obs)

    def _append_history(self, obs: TensorDict) -> None:
        self._obs_history = torch.roll(self._obs_history, shifts=-1, dims=1)
        self._obs_history[:, -1, :] = obs[self.history_obs_key]

    def _with_history(self, obs: TensorDict) -> TensorDict:
        obs = obs.clone(False)
        obs["obs_history"] = self._obs_history.reshape(self.num_envs, -1)
        return obs


class ActorCriticDreamWaQ(nn.Module):
    """DreamWaQ-style actor-critic with a history encoder.

    The actor uses current policy observations plus a latent code inferred from
    proprioceptive history. The critic may still use privileged observations.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        history_key: str = "obs_history",
        latent_dim: int = 16,
        velocity_dim: int = 3,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        actor_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        critic_hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        velocity_target_start: int = 0,
        velocity_loss_weight: float = 1.0,
        reconstruct_loss_weight: float = 1.0,
        kl_loss_weight: float = 1.0e-4,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print("ActorCriticDreamWaQ.__init__ ignored unexpected args: " + str([key for key in kwargs]))
        super().__init__()
        self.obs_groups = obs_groups
        self.history_key = history_key
        self.latent_dim = latent_dim
        self.velocity_dim = velocity_dim
        self.velocity_target_start = velocity_target_start
        self.velocity_loss_weight = velocity_loss_weight
        self.reconstruct_loss_weight = reconstruct_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization

        self.activation_name = activation
        self.activation = _activation(activation)

        num_actor_obs = sum(obs[key].shape[-1] for key in obs_groups["policy"])
        num_critic_obs = sum(obs[key].shape[-1] for key in obs_groups["critic"])
        num_history_obs = obs[history_key].shape[-1]
        code_dim = velocity_dim + latent_dim
        self.actor_obs_dim = num_actor_obs

        self.encoder = _mlp(num_history_obs, encoder_hidden_dims[-1], encoder_hidden_dims[:-1], activation)
        self.encode_mean_latent = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        self.encode_logvar_latent = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        self.encode_mean_vel = nn.Linear(encoder_hidden_dims[-1], velocity_dim)
        self.encode_logvar_vel = nn.Linear(encoder_hidden_dims[-1], velocity_dim)
        self.decoder = _mlp(code_dim, num_actor_obs, (64, 128), activation)
        self.actor = _mlp(num_actor_obs + code_dim, num_actions, actor_hidden_dims, activation)
        self.critic = _mlp(num_critic_obs, 1, critic_hidden_dims, activation)

        if actor_obs_normalization or critic_obs_normalization:
            raise ValueError("DreamWaQ-lite currently keeps observation normalization disabled.")

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args(False)

        print(f"DreamWaQ encoder: {self.encoder}")
        print(f"DreamWaQ actor: {self.actor}")
        print(f"DreamWaQ critic: {self.critic}")

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def cenet_forward(self, obs_history: torch.Tensor, sample: bool = True):
        features = self.encoder(obs_history)
        mean_latent = self.encode_mean_latent(features)
        logvar_latent = self.encode_logvar_latent(features)
        mean_vel = self.encode_mean_vel(features)
        logvar_vel = self.encode_logvar_vel(features)
        latent = self._reparameterize(mean_latent, logvar_latent) if sample else mean_latent
        vel = self._reparameterize(mean_vel, logvar_vel) if sample else mean_vel
        code = torch.cat((vel, latent), dim=-1)
        decoded = self.decoder(code)
        return code, vel, decoded, mean_vel, logvar_vel, mean_latent, logvar_latent

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        code, _, _, _, _, _, _ = self.cenet_forward(obs[self.history_key], sample=True)
        self._update_distribution(torch.cat((code, actor_obs), dim=-1))
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        code, _, _, _, _, _, _ = self.cenet_forward(obs[self.history_key], sample=False)
        return self.actor(torch.cat((code, actor_obs), dim=-1))

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        return self.critic(self.get_critic_obs(obs))

    def auxiliary_loss(self, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        actor_obs = self.get_actor_obs(obs)
        critic_obs = self.get_critic_obs(obs)
        _, vel, decoded, _, _, mean_latent, logvar_latent = self.cenet_forward(obs[self.history_key], sample=True)
        vel_target = critic_obs[:, self.velocity_target_start : self.velocity_target_start + self.velocity_dim].detach()
        recon_target = actor_obs.detach()
        vel_loss = nn.functional.mse_loss(vel, vel_target)
        recon_loss = nn.functional.mse_loss(decoded, recon_target)
        kl_loss = -0.5 * torch.mean(1 + logvar_latent - mean_latent.pow(2) - logvar_latent.exp())
        total = (
            self.velocity_loss_weight * vel_loss
            + self.reconstruct_loss_weight * recon_loss
            + self.kl_loss_weight * kl_loss
        )
        return total, {"velocity": vel_loss.detach(), "reconstruction": recon_loss.detach(), "kl": kl_loss.detach()}

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[key] for key in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[key] for key in self.obs_groups["critic"]], dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        pass

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True

    @staticmethod
    def _reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    def _update_distribution(self, actor_input: torch.Tensor) -> None:
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, self.std.expand_as(mean))


class PPODreamWaQ(PPO):
    """PPO with DreamWaQ encoder auxiliary losses."""

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_aux_loss = 0.0
        mean_vel_loss = 0.0
        mean_recon_loss = 0.0
        mean_kl_loss = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            original_batch_size = obs_batch.batch_size[0]
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            aux_loss, aux_parts = self.policy.auxiliary_loss(obs_batch)
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + aux_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_aux_loss += aux_loss.item()
            mean_vel_loss += aux_parts["velocity"].item()
            mean_recon_loss += aux_parts["reconstruction"].item()
            mean_kl_loss += aux_parts["kl"].item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {
            "value_function": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "dreamwaq_aux": mean_aux_loss / num_updates,
            "dreamwaq_velocity": mean_vel_loss / num_updates,
            "dreamwaq_reconstruction": mean_recon_loss / num_updates,
            "dreamwaq_kl": mean_kl_loss / num_updates,
        }


class DreamWaQRunner:
    """Small OnPolicyRunner variant for DreamWaQ-lite."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self._configure_multi_gpu()
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        obs = self.env.get_observations()
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], ["critic"])
        self.alg = self._construct_algorithm(obs)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        self._prepare_logging_writer()
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations().to(self.device)
        self.train_mode()
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        if self.is_distributed:
            self.alg.broadcast_parameters()
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                collection_time = time.time() - start
                start = time.time()
                self.alg.compute_returns(obs)
            loss_dict = self.alg.update()
            learn_time = time.time() - start
            self.current_learning_iteration = it
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
                store_code_state(self.log_dir, self.git_status_repos)
        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", self.alg.policy.action_std.mean().item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])

        episode_scalars = {}
        for ep_info in locs["ep_infos"]:
            for key, value in ep_info.items():
                if value is None:
                    continue
                try:
                    if isinstance(value, torch.Tensor):
                        scalar = value.float().mean().item()
                    else:
                        scalar = float(value)
                except (TypeError, ValueError):
                    continue
                episode_scalars.setdefault(key, []).append(scalar)
        for key, values in episode_scalars.items():
            self.writer.add_scalar(key, statistics.mean(values), locs["it"])

        log_string = (
            f"{'#' * width}\n"
            f"{(' Learning iteration ' + str(locs['it']) + '/' + str(locs['tot_iter']) + ' ').center(width)}\n\n"
            f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, "
            f"learning {locs['learn_time']:.3f}s)\n"
            f"{'Mean action noise std:':>{pad}} {self.alg.policy.action_std.mean().item():.2f}\n"
        )
        for key, value in locs["loss_dict"].items():
            log_string += f"{f'Mean {key} loss:':>{pad}} {value:.4f}\n"
        if len(locs["rewbuffer"]) > 0:
            log_string += f"{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"
            log_string += f"{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"
        for key in (
            "Metrics/base_velocity/error_vel_xy",
            "Metrics/base_velocity/error_vel_yaw",
            "Episode_Reward/track_lin_vel_xy_exp",
            "Episode_Reward/track_ang_vel_z_exp",
            "Episode_Termination/terrain_out_of_bounds",
        ):
            if key in episode_scalars:
                log_string += f"{key + ':':>{pad}} {statistics.mean(episode_scalars[key]):.4f}\n"
        log_string += f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
        log_string += f"{'Iteration time:':>{pad}} {locs['collection_time'] + locs['learn_time']:.2f}s\n"
        log_string += f"{'Time elapsed:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(self.tot_time))}\n"
        print(log_string)

    def save(self, path: str, infos=None) -> None:
        torch.save(
            {
                "model_state_dict": self.alg.policy.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos", {})

    def warm_start_from_ppo(self, path: str, map_location: str | None = None) -> dict[str, int]:
        """Initialize actor/critic weights from a standard RSL-RL PPO checkpoint.

        The DreamWaQ actor has extra latent-code inputs. Some DreamWaQ-Gap
        tasks also append height-scan observations after the rough proprioceptive
        observations. For the first actor layer, copy the standard policy
        weights into the original proprioceptive slice and keep both the
        latent-code and appended terrain-observation columns at random init.
        """
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location or self.device)
        source = loaded_dict["model_state_dict"]
        target = self.alg.policy.state_dict()
        copied = 0
        partial = 0

        for key, value in source.items():
            if key not in target:
                continue
            if target[key].shape == value.shape:
                target[key].copy_(value.to(device=target[key].device, dtype=target[key].dtype))
                copied += 1
                continue
            if key == "actor.0.weight" and target[key].shape[0] == value.shape[0] and target[key].shape[1] > value.shape[1]:
                code_dim = self.alg.policy.velocity_dim + self.alg.policy.latent_dim
                source_cols = value.shape[1]
                end_col = code_dim + source_cols
                if end_col <= target[key].shape[1]:
                    target[key][:, code_dim:end_col].copy_(
                        value.to(device=target[key].device, dtype=target[key].dtype)
                    )
                else:
                    target[key][:, -source_cols:].copy_(
                        value.to(device=target[key].device, dtype=target[key].dtype)
                    )
                partial += 1

        self.alg.policy.load_state_dict(target)
        print(
            f"[INFO] Warm-started DreamWaQ-lite from {path}: "
            f"{copied} tensors copied, {partial} tensors partially copied."
        )
        return {"copied": copied, "partial": partial}

    def get_inference_policy(self, device: str | None = None) -> callable:
        self.eval_mode()
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference

    def train_mode(self) -> None:
        self.alg.policy.train()

    def eval_mode(self) -> None:
        self.alg.policy.eval()

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.git_status_repos.append(repo_file_path)

    def _construct_algorithm(self, obs: TensorDict) -> PPODreamWaQ:
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)
        if self.alg_cfg.get("rnd_cfg") is not None or self.alg_cfg.get("symmetry_cfg") is not None:
            raise ValueError("DreamWaQ-lite runner currently does not support RND or symmetry configs.")
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "empirical_normalization is deprecated and ignored by DreamWaQ-lite.",
                DeprecationWarning,
            )
        policy_cfg = copy.deepcopy(self.policy_cfg)
        policy_cfg.pop("class_name", None)
        actor_critic = ActorCriticDreamWaQ(obs, self.cfg["obs_groups"], self.env.num_actions, **policy_cfg).to(self.device)
        alg_cfg = copy.deepcopy(self.alg_cfg)
        alg_cfg.pop("class_name", None)
        alg = PPODreamWaQ(actor_critic, device=self.device, **alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)
        alg.init_storage("rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])
        return alg

    def _configure_multi_gpu(self) -> None:
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(f"Device '{self.device}' does not match local rank '{self.gpu_local_rank}'.")
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        torch.cuda.set_device(self.gpu_local_rank)

    def _prepare_logging_writer(self) -> None:
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self.logger_type = self.cfg.get("logger", "tensorboard").lower()
            if self.logger_type != "tensorboard":
                raise ValueError("DreamWaQ-lite currently supports tensorboard logging only.")
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)


class DreamWaQJitExporter(nn.Module):
    """TorchScript exporter that keeps an internal observation-history buffer."""

    def __init__(self, policy: ActorCriticDreamWaQ, history_length: int = 5):
        super().__init__()
        self.actor = copy.deepcopy(policy.actor).cpu()
        self.encoder = copy.deepcopy(policy.encoder).cpu()
        self.encode_mean_latent = copy.deepcopy(policy.encode_mean_latent).cpu()
        self.encode_mean_vel = copy.deepcopy(policy.encode_mean_vel).cpu()
        self.history_length = history_length
        self.obs_dim = policy.actor_obs_dim
        self.register_buffer("obs_history", torch.zeros(1, history_length, self.obs_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_history.shape[0] != obs.shape[0]:
            self.obs_history = torch.zeros(obs.shape[0], self.history_length, self.obs_dim, device=obs.device)
        self.obs_history = torch.roll(self.obs_history, shifts=-1, dims=1)
        self.obs_history[:, -1, :] = obs
        features = self.encoder(self.obs_history.reshape(obs.shape[0], -1))
        code = torch.cat((self.encode_mean_vel(features), self.encode_mean_latent(features)), dim=-1)
        return self.actor(torch.cat((code, obs), dim=-1))

    @torch.jit.export
    def reset(self):
        self.obs_history[:] = 0.0

    def export(self, path: str, filename: str = "policy.pt") -> None:
        os.makedirs(path, exist_ok=True)
        self.eval()
        torch.jit.script(self).save(os.path.join(path, filename))


def export_dreamwaq_policy_as_jit(policy: ActorCriticDreamWaQ, path: str, filename: str = "policy.pt") -> None:
    DreamWaQJitExporter(policy).export(path, filename)


def _activation(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "selu":
        return nn.SELU()
    if name == "lrelu":
        return nn.LeakyReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def _mlp(input_dim: int, output_dim: int, hidden_dims: tuple[int, ...] | list[int], activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(_activation(activation))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)
