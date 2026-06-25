from __future__ import annotations

from copy import deepcopy

from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass

from .env_cfg import TaskDEnvB2Cfg
from .mdp import crossing as crossing_mdp

LEG_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]


def _materialize_taskd_crossing_observations(obs_cfg):
    obs_cfg.policy = obs_cfg.PolicyCfg()
    obs_cfg.policy.taskd_crossing_observations = deepcopy(obs_cfg.policy.taskd_crossing_observations)
    obs_cfg.depth_camera = obs_cfg.DepthCameraCfg()
    obs_cfg.depth_camera.depth_cam = deepcopy(obs_cfg.depth_camera.depth_cam)
    obs_cfg.delta_yaw_ok = obs_cfg.DeltaYawOkCfg()
    obs_cfg.delta_yaw_ok.delta_yaw_ok = deepcopy(obs_cfg.delta_yaw_ok.delta_yaw_ok)
    return obs_cfg


@configclass
class TaskDCrossingObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        taskd_crossing_observations = ObsTerm(
            func=crossing_mdp.TaskDParkourObservations,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True),
                "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_foot"),
                "lidar_sensor_cfg": SceneEntityCfg("lidar_sensor"),
                "waypoint": {},
                "history_length": crossing_mdp.PARKOUR_HISTORY_LENGTH,
                "contact_default": 0.5,
            },
            clip=(-100.0, 100.0),
        )

    @configclass
    class DepthCameraCfg(ObsGroup):
        depth_cam = ObsTerm(
            func=crossing_mdp.TaskDDepthCameraObservation,
            params={
                "sensor_cfg": SceneEntityCfg("head_camera"),
                "out_hw": (58, 87),
                "max_distance": 2.0,
            },
        )

    @configclass
    class DeltaYawOkCfg(ObsGroup):
        delta_yaw_ok = ObsTerm(
            func=crossing_mdp.TaskDDeltaYawOk,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "waypoint": {},
                "threshold": 0.6,
            },
        )

    policy: PolicyCfg = PolicyCfg()
    depth_camera: DepthCameraCfg = DepthCameraCfg()
    delta_yaw_ok: DeltaYawOkCfg = DeltaYawOkCfg()


@configclass
class TaskDCrossingRewardsCfg:
    reward_tracking_goal_vel = RewTerm(
        func=crossing_mdp.reward_tracking_goal_vel,
        weight=1.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "waypoint": {}},
    )
    reward_tracking_yaw = RewTerm(
        func=crossing_mdp.reward_tracking_yaw,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "waypoint": {}},
    )
    reward_lateral_deviation = RewTerm(
        func=crossing_mdp.reward_lateral_deviation,
        weight=-0.6,
        params={"asset_cfg": SceneEntityCfg("robot"), "waypoint": {}, "deadband": 0.12},
    )
    reward_waypoint_reached = RewTerm(
        func=crossing_mdp.reward_waypoint_reached,
        weight=3.0,
        params={
            "waypoint": {},
            "reward_values": (1.0, 0.6, 3.0, 0.8, 6.0, 0.8, 8.0, 0.8, 0.8, 0.8, 12.0),
        },
    )
    reward_cross_x_milestone = RewTerm(
        func=crossing_mdp.reward_cross_x_milestone,
        weight=2.0,
        params={
            "thresholds": (-1.2, -0.7, -0.35, 0.0, 0.35, 0.8, 1.4, 2.0),
            "reward_values": (1.0, 1.5, 2.0, 4.0, 4.0, 6.0, 8.0, 12.0),
        },
    )
    reward_orientation = RewTerm(
        func=crossing_mdp.reward_orientation,
        weight=-0.8,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "obstacle_x_range": (-0.95, 1.45),
            "obstacle_scale": 0.0,
        },
    )
    reward_lin_vel_z = RewTerm(
        func=crossing_mdp.reward_lin_vel_z,
        weight=-0.8,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "obstacle_x_range": (-0.95, 1.45),
            "obstacle_scale": 0.35,
        },
    )
    reward_ang_vel_xy = RewTerm(
        func=crossing_mdp.reward_ang_vel_xy,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_feet_stumble = RewTerm(
        func=crossing_mdp.reward_feet_stumble,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_foot")},
    )
    reward_feet_edge = RewTerm(
        func=crossing_mdp.reward_feet_edge,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_foot"),
            "edge_x_ranges": ((-0.90, -0.55), (0.55, 0.90)),
            "edge_y_abs": 2.9,
        },
    )
    reward_base_height_window = RewTerm(
        func=crossing_mdp.reward_base_height_window,
        weight=0.8,
        params={"asset_cfg": SceneEntityCfg("robot"), "min_height": 0.38, "max_height": 1.05},
    )
    reward_near_fall = RewTerm(
        func=crossing_mdp.reward_near_fall,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "min_height": 0.12, "max_tilt": 1.35},
    )
    reward_action_rate = RewTerm(
        func=crossing_mdp.reward_action_rate,
        weight=-0.06,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_dof_acc = RewTerm(
        func=crossing_mdp.reward_dof_acc,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
    )
    reward_dof_error = RewTerm(
        func=crossing_mdp.reward_dof_error,
        weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
    )
    reward_hip_pos = RewTerm(
        func=crossing_mdp.reward_hip_pos,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    )
    reward_torques = RewTerm(
        func=crossing_mdp.reward_torques,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
    )
    reward_collision = RewTerm(
        func=crossing_mdp.reward_collision,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["base_link"])},
    )
    reward_box_displacement = RewTerm(
        func=crossing_mdp.reward_box_displacement,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("box"), "target_xy": (0.0, 0.0), "tolerance": 0.12},
    )

@configclass
class TaskDCrossingEnvCfg(TaskDEnvB2Cfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2048
        self.episode_length_s = 12.0
        self.scene.box.init_state.pos = (0.0, 0.0, -0.30)
        self.scene.box.spawn.rigid_props.disable_gravity = False
        self.scene.box.spawn.mass_props.mass = 8.0
        self.scene.box.spawn.physics_material.static_friction = 0.9
        self.scene.box.spawn.physics_material.dynamic_friction = 0.8
        self.scene.box.spawn.physics_material.restitution = 0.0
        self.scene.robot.init_state.pos = (-1.80, 0.0, 0.80)
        self.actions.joint_wheel = None
        self.actions.joint_arm = None
        self.actions.joint_leg.scale = 0.25
        self.actions.joint_leg.clip = {".*": (-4.8, 4.8)}
        if self.scene.lidar_sensor is not None:
            # Match Isaaclab_Parkour's teacher input: 132 local terrain-height
            # samples from a yaw-aligned grid in front of the robot.  The
            # default TaskD lidar is a 360-degree multi-channel range scan; its
            # semantics are wrong for the Parkour scan encoder.
            self.scene.lidar_sensor = MultiMeshRayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/base_link",
                update_period=self.sim.dt * self.decimation,
                offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.375, 0.0, 20.0)),
                attach_yaw_only=True,
                pattern_cfg=patterns.GridPatternCfg(resolution=0.15, size=[1.65, 1.5]),
                max_distance=100.0,
                debug_vis=False,
                mesh_prim_paths=[
                    "/World/ground",
                    MultiMeshRayCasterCfg.RaycastTargetCfg(
                        prim_expr="{ENV_REGEX_NS}/Box",
                        is_shared=True,
                        track_mesh_transforms=True,
                    ),
                ],
            )
        if self.scene.head_camera is not None:
            self.scene.head_camera.update_period = self.sim.dt * self.decimation
        self.scene.contact_sensor.update_period = self.sim.dt * self.decimation
        self.observations = _materialize_taskd_crossing_observations(TaskDCrossingObservationsCfg())
        self.rewards = TaskDCrossingRewardsCfg()
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True),
                "position_range": (0.95, 1.05),
                "velocity_range": (0.0, 0.0),
            },
        )
        self.events.reset_robot_root = EventTerm(
            func=crossing_mdp.reset_crossing_robot_root,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "pose": (-1.80, 0.0, 0.80),
            },
        )
        self.events.reset_box = EventTerm(
            func=crossing_mdp.reset_crossing_box,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("box"),
                "curriculum": {
                    "target_x": 0.0,
                    "target_y": 0.0,
                    "target_z": -0.30,
                    "max_x_offset": 0.12,
                    "max_y_offset": 0.18,
                    "max_yaw": 0.25,
                    "stage_delta": 0.20,
                    "success_threshold": 0.65,
                    "failure_threshold": 0.25,
                    "success_x": 2.0,
                },
            },
        )
        self.events.physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.6, 2.0),
                "dynamic_friction_range": (0.6, 2.0),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        )
        self.events.randomize_rigid_body_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "mass_distribution_params": (-1.0, 3.0),
                "operation": "add",
            },
        )
        self.events.randomize_rigid_body_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
            },
        )
        self.events.push_by_setting_velocity = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(8.0, 8.0),
            is_global_time=True,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
            },
        )
        self.rewards.reward_action_rate.weight = -0.06
        self.rewards.reward_dof_acc.weight = -2.5e-7
        self.rewards.reward_torques.weight = -1.0e-5
        self.terminations.fall.params["minimum_height"] = -0.45
        self.terminations.x_reached.params["x_threshold"] = 2.2


@configclass
class TaskDCrossingEasyEnvCfg(TaskDCrossingEnvCfg):
    """Stage-1 crossing curriculum: same TaskD policy contract, easier trench."""

    pit_width_range: tuple[float, float] = (0.65, 0.75)
    platform_height_range: tuple[float, float] = (0.70, 0.80)


@configclass
class TaskDCrossingMidEnvCfg(TaskDCrossingEnvCfg):
    """Stage-2 crossing curriculum: intermediate trench before full TaskD."""

    pit_width_range: tuple[float, float] = (0.95, 1.05)
    platform_height_range: tuple[float, float] = (0.85, 0.95)


@configclass
class TaskDCrossingHardEnvCfg(TaskDCrossingEnvCfg):
    """Stage-3 crossing curriculum: near-final TaskD geometry."""

    pit_width_range: tuple[float, float] = (1.15, 1.25)
    platform_height_range: tuple[float, float] = (0.95, 1.05)
