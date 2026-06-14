from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _selected_joint_ids(asset: Articulation, asset_cfg: SceneEntityCfg) -> list[int]:
    if asset_cfg.joint_ids == slice(None):
        return list(range(len(asset.joint_names)))
    if isinstance(asset_cfg.joint_ids, slice):
        return list(range(len(asset.joint_names)))[asset_cfg.joint_ids]
    return list(asset_cfg.joint_ids)


def _target_joint_tensor(
    env: ManagerBasedRLEnv,
    asset: Articulation,
    asset_cfg: SceneEntityCfg,
    target_joint_pos: dict[str, float],
) -> tuple[list[int], torch.Tensor]:
    cache = getattr(env, "_stand_height_target_joint_pos_cache", {})
    joint_ids = _selected_joint_ids(asset, asset_cfg)
    cache_key = (asset_cfg.name, tuple(joint_ids), tuple(target_joint_pos.items()))
    if cache_key in cache:
        return joint_ids, cache[cache_key]

    target = torch.empty(len(joint_ids), device=env.device)
    for local_id, joint_id in enumerate(joint_ids):
        joint_name = asset.joint_names[joint_id]
        for pattern, value in target_joint_pos.items():
            if re.fullmatch(pattern, joint_name):
                target[local_id] = value
                break
        else:
            raise ValueError(f"No target joint position configured for joint '{joint_name}'.")

    cache[cache_key] = target
    env._stand_height_target_joint_pos_cache = cache
    return joint_ids, target


def stand_still_target_joint_pos_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_joint_pos: dict[str, float],
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize distance from a target pose when the command is near zero."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids, target = _target_joint_tensor(env, asset, asset_cfg, target_joint_pos)
    reward = torch.sum(torch.abs(asset.data.joint_pos[:, joint_ids] - target), dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def target_joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_joint_pos: dict[str, float],
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize distance from a target pose, with stronger scaling when standing still."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids, target = _target_joint_tensor(env, asset, asset_cfg, target_joint_pos)
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    target_error = torch.linalg.norm(asset.data.joint_pos[:, joint_ids] - target, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        target_error,
        stand_still_scale * target_error,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward
