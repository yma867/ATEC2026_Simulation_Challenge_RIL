from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp.events import apply_external_force_torque, randomize_rigid_body_mass

from parkour_isaaclab.envs.mdp import events, observations, rewards
from parkour_tasks.extreme_parkour_task.config.go2.parkour_mdp_cfg import (
    ActionsCfg,
    CommandsCfg,
    EventCfg,
    ParkourEventsCfg,
    StudentRewardsCfg,
    TeacherObservationsCfg,
    TeacherRewardsCfg,
    TerminationsCfg,
)

B2_BASE_LINK_NAME = "base_link"
B2_FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
B2_COLLISION_BODY_NAMES = [B2_BASE_LINK_NAME, ".*_calf", ".*_thigh"]


@configclass
class B2TeacherObservationsCfg(TeacherObservationsCfg):
    @configclass
    class PolicyCfg(ObsGroup):
        extreme_parkour_observations = ObsTerm(
            func=observations.ExtremeParkourObservations,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "parkour_name": "base_parkour",
                "history_length": 10,
                "base_body_name": B2_BASE_LINK_NAME,
            },
            clip=(-100, 100),
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class B2StudentObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        extreme_parkour_observations = ObsTerm(
            func=observations.ExtremeParkourObservations,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "parkour_name": "base_parkour",
                "history_length": 10,
                "base_body_name": B2_BASE_LINK_NAME,
            },
            clip=(-100, 100),
        )

    @configclass
    class DepthCameraPolicyCfg(ObsGroup):
        depth_cam = ObsTerm(
            func=observations.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "resize": (58, 87),
                "buffer_len": 2,
                "debug_vis": False,
            },
        )

    @configclass
    class DeltaYawOkPolicyCfg(ObsGroup):
        deta_yaw_ok = ObsTerm(
            func=observations.obervation_delta_yaw_ok,
            params={
                "parkour_name": "base_parkour",
                "threshold": 0.6,
            },
        )

    policy: PolicyCfg = PolicyCfg()
    depth_camera: DepthCameraPolicyCfg = DepthCameraPolicyCfg()
    delta_yaw_ok: DeltaYawOkPolicyCfg = DeltaYawOkPolicyCfg()


@configclass
class B2TeacherRewardsCfg(TeacherRewardsCfg):
    reward_collision = RewTerm(
        func=rewards.reward_collision,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=B2_COLLISION_BODY_NAMES),
        },
    )
    reward_feet_edge = RewTerm(
        func=rewards.reward_feet_edge,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names=B2_FOOT_NAMES),
            "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=".*_foot"),
            "parkour_name": "base_parkour",
            "base_body_name": B2_BASE_LINK_NAME,
        },
    )


@configclass
class B2StudentRewardsCfg(StudentRewardsCfg):
    reward_collision = RewTerm(
        func=rewards.reward_collision,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=B2_COLLISION_BODY_NAMES),
        },
    )


@configclass
class B2EventCfg(EventCfg):
    randomize_rigid_body_mass = EventTerm(
        func=randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=B2_BASE_LINK_NAME),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    randomize_rigid_body_com = EventTerm(
        func=events.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=B2_BASE_LINK_NAME),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )
    base_external_force_torque = EventTerm(
        func=apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=B2_BASE_LINK_NAME),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
