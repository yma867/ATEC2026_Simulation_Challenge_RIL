# Reference: https://github.com/fan-ziqi/robot_lab

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


def terrain_height_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terrain height scan relative to the robot base height.

    IsaacLab's stock height_scan helper updates the observation scale to a
    tensor in some versions. This local variant keeps the manager config
    unchanged and returns only the clipped observation tensor.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2]


def raycast_depth_features(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
    data_type: str = "distance_to_camera",
    out_hw: tuple[int, int] = (12, 16),
    max_distance: float = 2.5,
) -> torch.Tensor:
    """Compressed ray-caster depth image for terrain-aware locomotion.

    This uses IsaacLab's RayCasterCamera output, not an RGB renderer. It is much
    cheaper than camera rendering while preserving the forward terrain geometry
    cue used by parkour-style policies.
    """
    sensor = env.scene[sensor_cfg.name]
    depth = sensor.data.output[data_type]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)

    depth = torch.nan_to_num(depth, nan=max_distance, posinf=max_distance, neginf=0.0)
    depth = torch.clamp(depth, 0.0, max_distance) / max_distance - 0.5
    if out_hw is not None:
        depth = F.interpolate(depth.unsqueeze(1), size=out_hw, mode="bilinear", align_corners=False).squeeze(1)
    return depth.flatten(start_dim=1)


def taskd_lidar_height_features(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("taskd_lidar_sensor"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    out_dim: int = 651,
) -> torch.Tensor:
    """TaskD-compatible LiDAR height scan compressed to a fixed actor vector.

    TaskD exposes a 16-channel, 360-degree LiDAR height scan as the extero group
    (5760 values).  The deployment adapter uses the same flatten-then-resample
    transform, so this term is the training-side source of truth.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    ray_heights = sensor.data.ray_hits_w[..., 2]
    base_z = asset.data.root_pos_w[:, 2].view(-1, *([1] * (ray_heights.ndim - 1)))
    heights = base_z - ray_heights
    heights = torch.nan_to_num(heights, nan=2.0, posinf=2.0, neginf=-2.0)
    heights = torch.clamp(heights.flatten(start_dim=1), -2.0, 2.0)
    if heights.shape[-1] != out_dim:
        heights = F.interpolate(heights.unsqueeze(1), size=out_dim, mode="linear", align_corners=False).squeeze(1)
    return heights


def taskd_depth_features(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
    data_type: str = "distance_to_camera",
    out_hw: tuple[int, int] = (12, 16),
    max_distance: float = 2.5,
    neutral_value: float = 0.5,
    dropout_prob: float = 0.35,
) -> torch.Tensor:
    """TaskD-compatible compressed depth feature with training-time dropout.

    At deployment, TaskD may run without camera rendering.  Dropout teaches the
    locomotion policy that a neutral open-depth map is valid, while still letting
    it use real depth when available.
    """
    depth = raycast_depth_features(env, sensor_cfg=sensor_cfg, data_type=data_type, out_hw=out_hw, max_distance=max_distance)
    if dropout_prob > 0.0:
        mask = torch.rand((depth.shape[0], 1), device=depth.device) < dropout_prob
        neutral = torch.full_like(depth, neutral_value)
        depth = torch.where(mask, neutral, depth)
    return depth
