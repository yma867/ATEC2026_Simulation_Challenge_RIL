from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import Camera, ContactSensor, RayCaster
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg, RewardTermCfg


PARKOUR_NUM_PROP = 53
PARKOUR_NUM_SCAN = 132
PARKOUR_NUM_PRIV_EXPLICIT = 9
PARKOUR_NUM_PRIV_LATENT = 29
PARKOUR_HISTORY_LENGTH = 10
PARKOUR_OBS_DIM = (
    PARKOUR_NUM_PROP
    + PARKOUR_NUM_SCAN
    + PARKOUR_NUM_PRIV_EXPLICIT
    + PARKOUR_NUM_PRIV_LATENT
    + PARKOUR_HISTORY_LENGTH * PARKOUR_NUM_PROP
)


@dataclass(frozen=True)
class CrossingWaypointCfg:
    """TaskD crossing waypoints in world coordinates.

    The default lane follows the box placement used by the existing TaskD state
    machine after the box has been pushed into the pit.
    """

    lane_y: float = 0.0
    w0_x: float = -1.30
    w1_x: float = -0.70
    w3_x: float = 0.80
    w4_x: float = 2.00
    reach_distance: float = 0.35
    target_speed: float = 0.45


def _as_env_ids(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return env_ids.to(device=env.device)


def _fit_feature_dim(x: torch.Tensor, out_dim: int) -> torch.Tensor:
    x = x.flatten(start_dim=1)
    if x.shape[-1] == out_dim:
        return x
    if x.shape[-1] > out_dim:
        return F.interpolate(x.unsqueeze(1), size=out_dim, mode="linear", align_corners=False).squeeze(1)
    pad = torch.zeros((x.shape[0], out_dim - x.shape[-1]), device=x.device, dtype=x.dtype)
    return torch.cat((x, pad), dim=-1)


def _task_to_parkour_leg_order(leg_tensor: torch.Tensor) -> torch.Tensor:
    """TaskD B2Piper leg order FR,FL,RR,RL -> Parkour joint-type/native order."""
    perm = torch.tensor((3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8), device=leg_tensor.device)
    return leg_tensor.index_select(dim=-1, index=perm)


def _parkour_to_task_leg_order(leg_tensor: torch.Tensor) -> torch.Tensor:
    """Parkour joint-type/native order -> TaskD B2Piper leg order FR,FL,RR,RL."""
    perm = torch.tensor((1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10), device=leg_tensor.device)
    return leg_tensor.index_select(dim=-1, index=perm)


def _get_box_xy(env: ManagerBasedEnv, default_lane_y: float) -> torch.Tensor:
    if "box" not in env.scene.rigid_objects:
        return torch.tensor((0.0, default_lane_y), device=env.device).repeat(env.num_envs, 1)
    box: RigidObject = env.scene["box"]
    return box.data.root_pos_w[:, :2]


def _local_xy(env: ManagerBasedEnv, xy_w: torch.Tensor) -> torch.Tensor:
    del env
    return xy_w


def _local_x(env: ManagerBasedEnv, x_w: torch.Tensor) -> torch.Tensor:
    del env
    return x_w


def crossing_waypoints(
    env: ManagerBasedEnv,
    waypoint_cfg: CrossingWaypointCfg | None = None,
) -> torch.Tensor:
    """Return waypoints shaped (num_envs, 5, 2)."""
    cfg = waypoint_cfg or CrossingWaypointCfg()
    box_xy = _local_xy(env, _get_box_xy(env, cfg.lane_y))
    lane_y = torch.full((env.num_envs,), cfg.lane_y, device=env.device, dtype=box_xy.dtype)
    # Use the actual box center only for the middle waypoint.  The route lane is
    # kept mostly fixed so the policy sees the same crossing corridor at train
    # and deploy time.
    mid_left_x = torch.full_like(lane_y, -0.35)
    mid_right_x = torch.full_like(lane_y, 0.35)
    return torch.stack(
        (
            torch.stack((torch.full_like(lane_y, cfg.w0_x), lane_y), dim=-1),
            torch.stack((torch.full_like(lane_y, -1.00), lane_y), dim=-1),
            torch.stack((torch.full_like(lane_y, cfg.w1_x), lane_y), dim=-1),
            torch.stack((mid_left_x, lane_y), dim=-1),
            box_xy,
            torch.stack((mid_right_x, lane_y), dim=-1),
            torch.stack((torch.full_like(lane_y, cfg.w3_x), lane_y), dim=-1),
            torch.stack((torch.full_like(lane_y, 1.10), lane_y), dim=-1),
            torch.stack((torch.full_like(lane_y, 1.40), lane_y), dim=-1),
            torch.stack((torch.full_like(lane_y, 1.70), lane_y), dim=-1),
            torch.stack((torch.full_like(lane_y, cfg.w4_x), lane_y), dim=-1),
        ),
        dim=1,
    )


def _major_waypoint_indices(num_waypoints: int) -> torch.Tensor:
    # For the default 11-point route:
    # [W0, micro, W1, micro, W2/box, micro, W3, micro, micro, micro, W4]
    if num_waypoints == 11:
        return torch.tensor((0, 2, 4, 6, 10), dtype=torch.long)
    return torch.linspace(0, num_waypoints - 1, 5).round().to(torch.long)


def current_waypoint_indices(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    waypoint_cfg: CrossingWaypointCfg | None = None,
) -> torch.Tensor:
    cfg = waypoint_cfg or CrossingWaypointCfg()
    robot: Articulation = env.scene[asset_cfg.name]
    waypoints = crossing_waypoints(env, cfg)
    root_xy = _local_xy(env, robot.data.root_pos_w[:, :2])
    dists = torch.linalg.norm(waypoints - root_xy.unsqueeze(1), dim=-1)
    reached = dists < cfg.reach_distance
    # Count reached prefix; also advance monotonically by x progress for the
    # mostly one-dimensional TaskD crossing route.
    idx_from_distance = torch.cumsum(reached.to(torch.long), dim=1).argmax(dim=1)
    x_thresholds = waypoints[:, :, 0] - cfg.reach_distance * 0.5
    max_idx = waypoints.shape[1] - 1
    idx_from_x = torch.sum(_local_x(env, robot.data.root_pos_w[:, 0:1]) > x_thresholds, dim=1).clamp(max=max_idx)
    return torch.maximum(idx_from_distance, idx_from_x).clamp(min=0, max=max_idx)


def current_and_next_waypoints(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    waypoint_cfg: CrossingWaypointCfg | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg = waypoint_cfg or CrossingWaypointCfg()
    waypoints = crossing_waypoints(env, cfg)
    idx = current_waypoint_indices(env, asset_cfg, cfg)
    cur = waypoints.gather(1, idx.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
    next_idx = torch.clamp(idx + 1, max=waypoints.shape[1] - 1)
    nxt = waypoints.gather(1, next_idx.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
    return cur, nxt, idx


def _target_yaw_from_xy(root_xy: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
    delta = target_xy - root_xy
    return torch.atan2(delta[:, 1], delta[:, 0])


def _robot_yaw(robot: Articulation) -> torch.Tensor:
    _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
    return yaw


class TaskDParkourObservations(ManagerTermBase):
    """Build the 753-dim Parkour/RMA observation from TaskD-native state."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.contact_sensor: ContactSensor | None = env.scene.sensors.get(cfg.params["contact_sensor_cfg"].name)
        self.contact_sensor_cfg: SceneEntityCfg = cfg.params["contact_sensor_cfg"]
        self.lidar_sensor: RayCaster | None = env.scene.sensors.get(cfg.params["lidar_sensor_cfg"].name)
        self.lidar_sensor_cfg: SceneEntityCfg = cfg.params["lidar_sensor_cfg"]
        self.waypoint_cfg = CrossingWaypointCfg(**cfg.params.get("waypoint", {}))
        self.history_length = int(cfg.params.get("history_length", PARKOUR_HISTORY_LENGTH))
        self.contact_default = float(cfg.params.get("contact_default", 0.5))
        self._obs_history = torch.zeros(env.num_envs, self.history_length, PARKOUR_NUM_PROP, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._obs_history.zero_()
        else:
            self._obs_history[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        contact_sensor_cfg: SceneEntityCfg,
        lidar_sensor_cfg: SceneEntityCfg,
        waypoint: dict | None = None,
        history_length: int = PARKOUR_HISTORY_LENGTH,
        contact_default: float = 0.5,
    ) -> torch.Tensor:
        del asset_cfg, contact_sensor_cfg, lidar_sensor_cfg, waypoint, history_length, contact_default
        prop = self._build_prop(env)
        scan = self._height_scan()
        priv_explicit = self._priv_explicit()
        priv_latent = self._priv_latent()
        obs = torch.cat(
            (
                prop,
                scan,
                priv_explicit,
                priv_latent,
                self._obs_history.reshape(env.num_envs, -1),
            ),
            dim=-1,
        )
        prop_for_history = prop.clone()
        prop_for_history[:, 6:8] = 0.0
        self._obs_history = torch.where(
            (env.episode_length_buf <= 1).view(-1, 1, 1),
            torch.stack([prop_for_history] * self.history_length, dim=1),
            torch.cat((self._obs_history[:, 1:], prop_for_history.unsqueeze(1)), dim=1),
        )
        return obs

    def _build_prop(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        robot = self.robot
        roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
        root_xy = _local_xy(env, robot.data.root_pos_w[:, :2])
        cur, nxt, _ = current_and_next_waypoints(env, self.asset_cfg, self.waypoint_cfg)
        delta_yaw = wrap_to_pi(_target_yaw_from_xy(root_xy, cur) - yaw)
        delta_next_yaw = wrap_to_pi(_target_yaw_from_xy(root_xy, nxt) - yaw)
        joint_pos = robot.data.joint_pos[:, self.asset_cfg.joint_ids] - robot.data.default_joint_pos[:, self.asset_cfg.joint_ids]
        joint_vel = robot.data.joint_vel[:, self.asset_cfg.joint_ids]
        joint_pos = _task_to_parkour_leg_order(joint_pos)
        joint_vel = _task_to_parkour_leg_order(joint_vel)
        try:
            last_action = env.action_manager.get_term("joint_leg").raw_actions
        except Exception:
            last_action = torch.zeros_like(joint_pos)
        last_action = _task_to_parkour_leg_order(last_action[:, :12])
        batch = robot.data.root_pos_w.shape[0]
        prop = torch.cat(
            (
                robot.data.root_ang_vel_b * 0.25,
                torch.stack((wrap_to_pi(roll), wrap_to_pi(pitch)), dim=-1),
                torch.zeros((batch, 1), device=env.device),
                delta_yaw.unsqueeze(-1),
                delta_next_yaw.unsqueeze(-1),
                torch.zeros((batch, 2), device=env.device),
                torch.full((batch, 1), self.waypoint_cfg.target_speed, device=env.device),
                torch.ones((batch, 1), device=env.device),
                torch.zeros((batch, 1), device=env.device),
                joint_pos,
                joint_vel * 0.05,
                last_action,
                self._contact_fill(),
            ),
            dim=-1,
        )
        if prop.shape[-1] != PARKOUR_NUM_PROP:
            raise RuntimeError(f"TaskD Parkour prop dim mismatch: got {prop.shape[-1]}, expected {PARKOUR_NUM_PROP}")
        return prop

    def _contact_fill(self) -> torch.Tensor:
        batch = self.robot.data.root_pos_w.shape[0]
        if self.contact_sensor is None or self.contact_sensor_cfg.body_ids is None:
            return torch.full((batch, 4), self.contact_default, device=self.device)
        forces = self.contact_sensor.data.net_forces_w_history[:, 0, self.contact_sensor_cfg.body_ids]
        prev_forces = self.contact_sensor.data.net_forces_w_history[:, -1, self.contact_sensor_cfg.body_ids]
        contact = torch.logical_or(torch.norm(forces, dim=-1) > 2.0, torch.norm(prev_forces, dim=-1) > 2.0)
        return contact.float() - 0.5

    def _height_scan(self) -> torch.Tensor:
        batch = self.robot.data.root_pos_w.shape[0]
        if self.lidar_sensor is None:
            return torch.zeros((batch, PARKOUR_NUM_SCAN), device=self.device)
        hits = self.lidar_sensor.data.ray_hits_w
        pos_z = self.lidar_sensor.data.pos_w[:, 2].unsqueeze(-1)
        heights = torch.nan_to_num(pos_z - hits[..., 2] - 0.3, nan=1.0, posinf=1.0, neginf=-1.0)
        heights = torch.clamp(heights, -1.0, 1.0)
        return _fit_feature_dim(heights, PARKOUR_NUM_SCAN)

    def _priv_explicit(self) -> torch.Tensor:
        base_lin_vel = self.robot.data.root_lin_vel_b
        return torch.cat((base_lin_vel * 2.0, 0.0 * base_lin_vel, 0.0 * base_lin_vel), dim=-1)

    def _priv_latent(self) -> torch.Tensor:
        robot = self.robot
        body_mass = robot.root_physx_view.get_masses()[:, 0].to(self.device).unsqueeze(-1)
        body_com = robot.data.com_pos_b[:, 0, :].to(self.device)
        mass_params = torch.cat((body_mass, body_com), dim=-1)
        friction = robot.root_physx_view.get_material_properties()[:, 0, 0].to(self.device).unsqueeze(-1)
        joint_ids = self.asset_cfg.joint_ids
        stiffness = robot.data.joint_stiffness[:, joint_ids].to(self.device)
        default_stiffness = torch.clamp(robot.data.default_joint_stiffness[:, joint_ids].to(self.device), min=1.0e-6)
        damping = robot.data.joint_damping[:, joint_ids].to(self.device)
        default_damping = torch.clamp(robot.data.default_joint_damping[:, joint_ids].to(self.device), min=1.0e-6)
        latent = torch.cat((mass_params, friction, stiffness / default_stiffness - 1.0, damping / default_damping - 1.0), dim=-1)
        return _fit_feature_dim(latent, PARKOUR_NUM_PRIV_LATENT)


class TaskDDepthCameraObservation(ManagerTermBase):
    """Return TaskD head depth in Parkour depth encoder shape: (N, 58, 87)."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.camera: Camera = env.scene[cfg.params["sensor_cfg"].name]
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.out_hw = tuple(cfg.params.get("out_hw", (58, 87)))
        self.max_distance = float(cfg.params.get("max_distance", 2.0))

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        out_hw: tuple[int, int] = (58, 87),
        max_distance: float = 2.0,
    ) -> torch.Tensor:
        del env, sensor_cfg, out_hw, max_distance
        depth = self.camera.data.output.get("depth")
        if depth is None:
            depth = self.camera.data.output.get("distance_to_camera")
        if depth is None:
            return torch.full((self.num_envs, *self.out_hw), 0.5, device=self.device)
        depth = depth.squeeze(-1) if depth.ndim == 4 else depth
        # Match Parkour crop convention: remove small border and bottom rows.
        if depth.shape[-2] > 4 and depth.shape[-1] > 8:
            depth = depth[:, :-2, 4:-4]
        depth = torch.nan_to_num(depth, nan=self.max_distance, posinf=self.max_distance, neginf=0.0)
        depth = torch.clamp(depth, 0.0, self.max_distance) / self.max_distance - 0.5
        depth = F.interpolate(depth.unsqueeze(1), size=self.out_hw, mode="bilinear", align_corners=False).squeeze(1)
        return depth


class TaskDDeltaYawOk(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.waypoint_cfg = CrossingWaypointCfg(**cfg.params.get("waypoint", {}))

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float = 0.6,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        waypoint: dict | None = None,
    ) -> torch.Tensor:
        del waypoint
        robot: Articulation = env.scene[asset_cfg.name]
        cur, _, _ = current_and_next_waypoints(env, asset_cfg, self.waypoint_cfg)
        yaw_err = wrap_to_pi(_target_yaw_from_xy(_local_xy(env, robot.data.root_pos_w[:, :2]), cur) - _robot_yaw(robot))
        return (torch.abs(yaw_err) < threshold).unsqueeze(-1)


def reward_tracking_goal_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    waypoint: dict | None = None,
) -> torch.Tensor:
    cfg = CrossingWaypointCfg(**(waypoint or {}))
    robot: Articulation = env.scene[asset_cfg.name]
    cur, _, _ = current_and_next_waypoints(env, asset_cfg, cfg)
    delta = cur - _local_xy(env, robot.data.root_pos_w[:, :2])
    direction = delta / (torch.linalg.norm(delta, dim=-1, keepdim=True) + 1.0e-6)
    projected_vel = torch.sum(robot.data.root_lin_vel_w[:, :2] * direction, dim=-1)
    return torch.clamp(projected_vel / max(cfg.target_speed, 1.0e-3), min=-1.0, max=1.0)


def reward_tracking_yaw(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    waypoint: dict | None = None,
) -> torch.Tensor:
    cfg = CrossingWaypointCfg(**(waypoint or {}))
    robot: Articulation = env.scene[asset_cfg.name]
    cur, _, _ = current_and_next_waypoints(env, asset_cfg, cfg)
    yaw_err = torch.abs(wrap_to_pi(_target_yaw_from_xy(_local_xy(env, robot.data.root_pos_w[:, :2]), cur) - _robot_yaw(robot)))
    return torch.exp(-yaw_err)


def reward_lateral_deviation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    waypoint: dict | None = None,
    deadband: float = 0.08,
) -> torch.Tensor:
    """Penalize drifting sideways away from the TaskD crossing lane."""
    cfg = CrossingWaypointCfg(**(waypoint or {}))
    robot: Articulation = env.scene[asset_cfg.name]
    lateral_error = torch.abs(_local_xy(env, robot.data.root_pos_w[:, :2])[:, 1] - cfg.lane_y)
    return torch.clamp(lateral_error - deadband, min=0.0)


class reward_waypoint_reached(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.waypoint_cfg = CrossingWaypointCfg(**cfg.params.get("waypoint", {}))
        num_waypoints = crossing_waypoints(env, self.waypoint_cfg).shape[1]
        self.reached = torch.zeros((env.num_envs, num_waypoints), dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self.reached.zero_()
        else:
            self.reached[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        reward_values: tuple[float, ...] = (),
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        waypoint: dict | None = None,
    ) -> torch.Tensor:
        del waypoint
        robot: Articulation = env.scene[asset_cfg.name]
        waypoints = crossing_waypoints(env, self.waypoint_cfg)
        dists = torch.linalg.norm(waypoints - _local_xy(env, robot.data.root_pos_w[:, :2]).unsqueeze(1), dim=-1)
        reached_now = dists < self.waypoint_cfg.reach_distance
        trigger = reached_now & (~self.reached)
        self.reached |= reached_now
        if len(reward_values) == waypoints.shape[1]:
            values = torch.tensor(reward_values, device=env.device, dtype=robot.data.root_pos_w.dtype)
        else:
            values = torch.full((waypoints.shape[1],), 0.35, device=env.device, dtype=robot.data.root_pos_w.dtype)
            major_idx = _major_waypoint_indices(waypoints.shape[1]).to(env.device)
            major_values = torch.tensor((0.5, 2.0, 3.0, 4.0, 5.0), device=env.device, dtype=robot.data.root_pos_w.dtype)
            values[major_idx] = major_values
        return (trigger.float() * values.unsqueeze(0)).sum(dim=1)


class reward_cross_x_milestone(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        thresholds = cfg.params.get("thresholds", (-0.7, 0.0, 0.8, 2.0))
        self.thresholds = torch.tensor(thresholds, device=env.device, dtype=torch.float32)
        self.given = torch.zeros((env.num_envs, len(thresholds)), dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self.given.zero_()
        else:
            self.given[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        thresholds: tuple[float, ...] = (-0.7, 0.0, 0.8, 2.0),
        reward_values: tuple[float, ...] = (2.0, 5.0, 8.0, 12.0),
    ) -> torch.Tensor:
        del thresholds
        robot: Articulation = env.scene[asset_cfg.name]
        crossed = _local_x(env, robot.data.root_pos_w[:, 0:1]) > self.thresholds.view(1, -1)
        trigger = crossed & (~self.given)
        self.given |= crossed
        values = torch.tensor(reward_values, device=env.device, dtype=robot.data.root_pos_w.dtype)
        return (trigger.float() * values.view(1, -1)).sum(dim=1)


def reward_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    obstacle_x_range: tuple[float, float] | None = None,
    obstacle_scale: float = 1.0,
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    rew = torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1)
    if obstacle_x_range is not None:
        x = _local_x(env, robot.data.root_pos_w[:, 0])
        in_obstacle = (x > float(obstacle_x_range[0])) & (x < float(obstacle_x_range[1]))
        rew = torch.where(in_obstacle, rew * float(obstacle_scale), rew)
    return rew


def reward_lin_vel_z(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    obstacle_x_range: tuple[float, float] | None = None,
    obstacle_scale: float = 1.0,
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    rew = torch.square(robot.data.root_lin_vel_b[:, 2])
    if obstacle_x_range is not None:
        x = _local_x(env, robot.data.root_pos_w[:, 0])
        in_obstacle = (x > float(obstacle_x_range[0])) & (x < float(obstacle_x_range[1]))
        rew = torch.where(in_obstacle, rew * float(obstacle_scale), rew)
    return rew


def reward_ang_vel_xy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.root_ang_vel_b[:, :2]), dim=1)


def reward_feet_stumble(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Parkour-style stumble penalty: large horizontal foot force relative to vertical force."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    stumble = torch.any(torch.norm(forces[:, :, :2], dim=2) > 4.0 * torch.abs(forces[:, :, 2]), dim=1)
    return stumble.to(dtype=forces.dtype)


def reward_feet_edge(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    edge_x_ranges: tuple[tuple[float, float], ...] = ((-0.90, -0.55), (0.55, 0.90)),
    edge_y_abs: float = 2.9,
) -> torch.Tensor:
    """TaskD-safe equivalent of Parkour's feet-edge penalty.

    The original project uses terrain-generator edge masks.  TaskD's final
    terrain is fixed, so we encode the trench lip regions directly in authored
    TaskD coordinates.  This keeps the reward method while avoiding any new
    deployment input or Parkour-specific terrain-mask dependency.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_xy = _local_xy(env, robot.data.body_pos_w[:, asset_cfg.body_ids, :2].reshape(-1, 2)).reshape(
        env.num_envs, len(asset_cfg.body_ids), 2
    )
    foot_x = foot_xy[:, :, 0]
    foot_y = torch.abs(foot_xy[:, :, 1])
    on_x_edge = torch.zeros_like(foot_x, dtype=torch.bool)
    for lo, hi in edge_x_ranges:
        on_x_edge |= (foot_x > float(lo)) & (foot_x < float(hi))
    on_y_edge = foot_y > float(edge_y_abs)
    forces = sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    prev_forces = sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids]
    contact = (torch.norm(forces, dim=-1) > 2.0) | (torch.norm(prev_forces, dim=-1) > 2.0)
    return torch.sum((on_x_edge | on_y_edge) & contact, dim=-1).to(dtype=forces.dtype)


def reward_base_height_window(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_height: float = 0.38,
    max_height: float = 1.05,
) -> torch.Tensor:
    """Reward keeping the base in a reasonable walking height window."""
    robot: Articulation = env.scene[asset_cfg.name]
    z = robot.data.root_pos_w[:, 2]
    return ((z > min_height) & (z < max_height)).to(z.dtype)


def reward_near_fall(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_height: float = 0.18,
    max_tilt: float = 0.85,
) -> torch.Tensor:
    """Penalize states that are close to falling before the terminal fires."""
    robot: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    low = robot.data.root_pos_w[:, 2] < min_height
    tilted = (torch.abs(wrap_to_pi(roll)) > max_tilt) | (torch.abs(wrap_to_pi(pitch)) > max_tilt)
    return (low | tilted).to(robot.data.root_pos_w.dtype)


class reward_action_rate(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.prev = torch.zeros(env.num_envs, 12, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self.prev.zero_()
        else:
            self.prev[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        del asset_cfg
        try:
            action = env.action_manager.get_term("joint_leg").raw_actions[:, :12]
        except Exception:
            action = torch.zeros_like(self.prev)
        rew = torch.norm(action - self.prev, dim=1)
        self.prev = action.detach().clone()
        return rew


class reward_dof_acc(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params["asset_cfg"]
        self.prev_vel = torch.zeros(env.num_envs, len(self.asset_cfg.joint_ids), device=env.device)
        self.dt = env.step_dt

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self.prev_vel.zero_()
        else:
            self.prev_vel[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        robot: Articulation = env.scene[asset_cfg.name]
        vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
        acc = (vel - self.prev_vel) / self.dt
        self.prev_vel = vel.detach().clone()
        return torch.sum(torch.square(acc), dim=1)


def reward_dof_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1)


def reward_hip_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1)


def reward_torques(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def reward_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    return torch.sum((torch.norm(forces, dim=-1) > 0.1).float(), dim=1)


def reward_box_displacement(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    target_xy: tuple[float, float] = (-0.45, -0.90),
    tolerance: float = 0.12,
) -> torch.Tensor:
    if asset_cfg.name not in env.scene.rigid_objects:
        return torch.zeros(env.num_envs, device=env.device)
    box: RigidObject = env.scene[asset_cfg.name]
    target = torch.tensor(target_xy, device=env.device, dtype=box.data.root_pos_w.dtype).view(1, 2)
    err = torch.linalg.norm(_local_xy(env, box.data.root_pos_w[:, :2]) - target, dim=-1)
    return torch.clamp(err - tolerance, min=0.0)


def final_cross_success(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    x_threshold: float = 2.0,
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return _local_x(env, robot.data.root_pos_w[:, 0]) > x_threshold


def bad_crossing_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    limit_angle: float = 1.2,
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    return (torch.abs(wrap_to_pi(roll)) > limit_angle) | (torch.abs(wrap_to_pi(pitch)) > limit_angle)


def reset_crossing_box(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    pose_range: dict[str, tuple[float, float]] | None = None,
    curriculum: dict | None = None,
):
    if asset_cfg.name not in env.scene.rigid_objects:
        return
    asset: RigidObject = env.scene[asset_cfg.name]
    env_ids = _as_env_ids(env, env_ids)
    curriculum = curriculum or {}
    target_x = float(curriculum.get("target_x", 0.0))
    target_y = float(curriculum.get("target_y", 0.0))
    target_z = float(curriculum.get("target_z", -0.30))
    max_x_offset = float(curriculum.get("max_x_offset", 0.12))
    max_y_offset = float(curriculum.get("max_y_offset", 0.18))
    max_yaw = float(curriculum.get("max_yaw", 0.25))
    stage_delta = float(curriculum.get("stage_delta", 0.20))
    success_threshold = float(curriculum.get("success_threshold", 0.65))
    failure_threshold = float(curriculum.get("failure_threshold", 0.25))
    ema_alpha = float(curriculum.get("ema_alpha", 0.05))
    success_x = float(curriculum.get("success_x", 2.0))

    if not hasattr(env, "_taskd_box_curriculum_level"):
        env._taskd_box_curriculum_level = 0.0
        env._taskd_box_success_ema = 0.0

    robot: Articulation | None = env.scene.articulations.get("robot")
    if robot is not None and len(env_ids) > 0:
        robot_x = _local_x(env, robot.data.root_pos_w[:, 0])[env_ids]
        success = (robot_x > success_x).float().mean().item()
        env._taskd_box_success_ema = (1.0 - ema_alpha) * env._taskd_box_success_ema + ema_alpha * success
        if env._taskd_box_success_ema > success_threshold:
            env._taskd_box_curriculum_level = min(1.0, env._taskd_box_curriculum_level + stage_delta)
            env._taskd_box_success_ema = 0.5
        elif env._taskd_box_success_ema < failure_threshold:
            env._taskd_box_curriculum_level = max(0.0, env._taskd_box_curriculum_level - 0.5 * stage_delta)

    level = float(env._taskd_box_curriculum_level)
    pose_range = pose_range or {
        "x": (target_x - level * max_x_offset, target_x + level * max_x_offset),
        "y": (target_y - level * max_y_offset, target_y + level * max_y_offset),
        "z": (target_z - 0.02, target_z + 0.02),
        "yaw": (-level * max_yaw, level * max_yaw),
    }
    def r(key: str, default: tuple[float, float]) -> torch.Tensor:
        lo, hi = pose_range.get(key, default)
        return math_utils.sample_uniform(lo, hi, (len(env_ids),), device=env.device)

    # Positions above are TaskD-authored crossing/world coordinates.  The pit
    # center is x=0, so do not add env.scene.env_origins here.
    positions_local = torch.stack(
        (r("x", (target_x, target_x)), r("y", (target_y, target_y)), r("z", (target_z, target_z))),
        dim=-1,
    )
    positions = positions_local
    yaw = r("yaw", (0.0, 0.0))
    zeros = torch.zeros_like(yaw)
    quat = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    vel = torch.zeros((len(env_ids), 6), device=env.device)
    asset.write_root_pose_to_sim(torch.cat((positions, quat), dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(vel, env_ids=env_ids)


def reset_crossing_robot_root(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pose: tuple[float, float, float] = (-1.80, 0.0, 0.80),
):
    """Reset robot root to a deterministic crossing start pose.

    TaskD's generated terrain is centered in the world frame after generation,
    while IsaacLab still exposes terrain origins as spawn hints.  This reset
    follows the same convention as the viewer/debug scene: the crossing start is
    the authored task/world pose, not randomized per episode.
    """
    if asset_cfg.name not in env.scene.articulations:
        return
    asset: Articulation = env.scene[asset_cfg.name]
    env_ids = _as_env_ids(env, env_ids)
    root_states = asset.data.default_root_state[env_ids].clone()
    positions = torch.tensor(pose, device=env.device, dtype=root_states.dtype).view(1, 3).repeat(len(env_ids), 1)
    root_states[:, 0:3] = positions
    root_states[:, 7:13] = 0.0
    asset.write_root_pose_to_sim(root_states[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_states[:, 7:13], env_ids=env_ids)
