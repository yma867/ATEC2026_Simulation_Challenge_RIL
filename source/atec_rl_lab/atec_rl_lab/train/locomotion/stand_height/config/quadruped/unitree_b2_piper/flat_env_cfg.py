from isaaclab.utils import configclass

import atec_rl_lab.train.locomotion.stand_height.mdp as stand_height_mdp  # noqa: F401
from atec_rl_lab.train.locomotion.velocity.config.quadruped.unitree_b2_piper.flat_env_cfg import (
    UnitreeB2PiperFlatEnvCfg,
)

TARGET_BASE_HEIGHT = 0.2
BASE_HEIGHT_L2_WEIGHT = -5.0

# Target crouch pose used to anchor the standing posture.
CROUCH_JOINT_POS = {
    ".*R_hip_joint": -0.3,
    ".*L_hip_joint": 0.3,
    "F[L,R]_thigh_joint": 0.6,
    "R[L,R]_thigh_joint": 1.0,
    ".*_calf_joint": -2.0,
}


@configclass
class UnitreeB2PiperStandHeightFlatEnvCfg(UnitreeB2PiperFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Reuse the existing velocity command term, but pin it to zero for pure standing.
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_height = None

        # The commands are constant or unused, so they do not need to be observed.
        self.observations.policy.velocity_commands = None
        self.observations.critic.velocity_commands = None
        self.observations.policy.base_height_command = None
        self.observations.critic.base_height_command = None

        # Tighten reset conditions and remove pushes for a pure standing-height task.
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.02),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None

        # Replace the default tracked command with a fixed target base height.
        self.rewards.base_height_l2.weight = BASE_HEIGHT_L2_WEIGHT
        self.rewards.base_height_l2.params["command_name"] = None
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.rewards.base_height_l2.params["target_height"] = TARGET_BASE_HEIGHT
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Keep the zero-velocity task active so drifting backward is explicitly worse.
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.25
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25

        # Keep a posture anchor for standing. Without this, the policy can satisfy
        # height/zero-velocity rewards by folding the front legs backward and pitching
        # the body forward.
        self.rewards.upward.weight = 1.0
        self.rewards.flat_orientation_l2.weight = -3.0
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.5
        self.rewards.stand_still.func = stand_height_mdp.stand_still_target_joint_pos_l1
        self.rewards.stand_still.weight = -2.0
        self.rewards.joint_pos_penalty.func = stand_height_mdp.target_joint_pos_penalty
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.joint_names
        self.rewards.stand_still.params["target_joint_pos"] = CROUCH_JOINT_POS
        self.rewards.joint_pos_penalty.params["target_joint_pos"] = CROUCH_JOINT_POS

        # Reinstate the contact/stability terms that discourage body collapse.
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Disable gait rewards that are not meaningful for a fixed crouch.
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_contact.weight = 0.0
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_gait.weight = 0.0
        self.rewards.feet_height.weight = 0.0
        self.rewards.feet_height_body.weight = -2.0
        self.rewards.feet_height_body.params["target_height"] = -0.35
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]

        if self.__class__.__name__ == "UnitreeB2PiperStandHeightFlatEnvCfg":
            self.disable_zero_weight_rewards()
